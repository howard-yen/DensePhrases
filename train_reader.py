import json
import argparse
import torch
import os
import random
import numpy as np
import requests
import logging
import math
import copy
import string
import faiss
import wandb

from time import time
from tqdm import tqdm, trange

from densephrases.utils.single_utils import set_seed
from densephrases.utils.eval_utils import normalize_answer, f1_score, exact_match_score, drqa_exact_match_score, \
        drqa_regex_match_score, drqa_metric_max_over_ground_truths, drqa_normalize
from densephrases.utils.open_utils import load_query_encoder, load_phrase_index, get_query2vec, load_qa_pairs
from eval_phrase_retrieval import evaluate_results
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from transformers import (
    MODEL_MAPPING,
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    AutoModelForQuestionAnswering,
    AdamW,
    get_linear_schedule_with_warmup,
)

from reader import Reader
from utils import process_sample, process_reader_input

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def train_reader(args):
    #wandb setup
    wandb.init(project="DensePhrases-Reader-Training", entity='howard-yen', mode="online" if args.wandb else "disabled")
    wandb.config.update(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or not args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        args.n_gpu = 0 if not args.cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    torch.cuda.empty_cache()

    logger.info('Training BERT model used for reading')
    config = AutoConfig.from_pretrained(args.config_name if args.config_name else args.pretrained_name_or_path, cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.pretrained_name_or_path, cache_dir=args.cache_dir if args.cache_dir else None)

    model = Reader(config=config)
    model.to(device)

    # Check if saved model states exist
    if args.load_dir:
        if os.path.isfile(os.path.join(args.load_dir, "reader_model_load")):
            # Load in saved model states
            model = model.from_pretrained(os.path.join(args.load_dir, "reader_model_load"))
            #model.load_state_dict(torch.load(os.path.join(args.load_dir, "reader_model.pt")))
            logger.info(f'model loaded from {args.load_dir}')

    train_dataset = process_reader_input(tokenizer, args.train_file, max_query_length=args.max_query_length, max_seq_length=args.max_seq_length, top_k=args.top_k)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)


    # Optimizer setting
    def is_train_param(name):
        if name.endswith("bert_start.embeddings.word_embeddings.weight") or \
            name.endswith("bert_end.embeddings.word_embeddings.weight") or \
            name.endswith("bert_q_start.embeddings.word_embeddings.weight") or \
            name.endswith("bert_q_end.embeddings.word_embeddings.weight"):
            logger.info(f'freezing {name}')
            return False
        return True

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [{
            "params": [
                p for n, p in model.named_parameters() \
                    if not any(nd in n for nd in no_decay) and is_train_param(n)
            ],
            "weight_decay": args.weight_decay,
        }, {
            "params": [
                p for n, p in model.named_parameters() \
                    if any(nd in n for nd in no_decay) and is_train_param(n)
            ],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if args.load_dir:
        if os.path.isfile(os.path.join(args.load_dir, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.load_dir, "scheduler.pt")
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(args.load_dir, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(args.load_dir, "scheduler.pt")))
            logger.info(f'optimizer and scheduler loaded from {args.load_dir}')

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.load_dir:
        try:
            # set global_step to global_step of last saved checkpoint from model path
            checkpoint_suffix = args.load_dir.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)

    for ep_idx, _ in enumerate(train_iterator):
        logger.info(f"\n[Epoch {ep_idx+1}]")

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()

            torch.cuda.empty_cache()

            batch = tuple(t.to(device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "token_type_ids": batch[1],
                "attention_mask": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
                "is_impossible": batch[5],
                "rank": batch[6],
                #"input_ids_": batch[8],
                #"attention_mask_": batch[9],
                #"token_type_ids_": batch[10],
            }

            # skip this batch if none of the questions have a good passage
            if inputs['rank'].max() == -1:
                continue

            N = inputs['is_impossible'].size(0)
            # in batch negatives
            #for im_idx, im in enumerate(inputs['is_impossible']):
            #    logger.info('doing in-batch negatives')
            #    # can fill with negative example if didn't have passage
            #    # can change to only have one positive passage during training
            #    replace_count = torch.numel(im[im==-1])
            #    if replace_count == 0:
            #        continue
            #    # for now, assume batch_size >= top_k
            #    replacements = torch.randperm(N-1)
            #    replacements[replacements >= im_idx] += 1

            #    input_temp = inputs['input_ids'][im_idx][0]
            #    type_temp = inputs['token_type_ids'][im_idx][0]
            #    question_str = input_temp[type_temp == 0]
            #    question_str = tokenizer.decode(question_str.tolist())

            #    for rep_idx, rep in enumerate(replacements):
            #        if rep_idx >= replace_count:
            #            break

            #        input_temp = inputs['input_ids'][rep][0]
            #        type_temp = inputs['token_type_ids'][rep][0]
            #        passage_str = input_temp[type_temp == 1]
            #        passage_str = tokenizer.decode(passage_str.tolist())
            #        new_encoded = tokenizer.encode_plus(question_str, text_pair=question_str, max_length=args.max_seq_length, pad_to_max_length=True, return_token_type_ids=True, return_attention_mask=True)
            #        inputs['input_ids'][im_idx][args.top_k - replace_count + rep_idx] = torch.tensor(new_encoded['input_ids']).to(device)
            #        inputs['token_type_ids'][im_idx][args.top_k - replace_count + rep_idx] = torch.tensor(new_encoded['token_type_ids']).to(device)
            #        inputs['attention_mask'][im_idx][args.top_k - replace_count + rep_idx] = torch.tensor(new_encoded['attention_mask']).to(device)

            loss = model(**inputs)
            epoch_iterator.set_description(f"Loss={loss.item():.3f}")

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        # Validation acc
                        logger.setLevel(logging.WARNING)
                        results, _ = eval_result_bert(args, model, tokenizer, config) #, prefix=global_step)
                        wandb.log(
                            {"Eval EM": results['exact'], "Eval F1": results['f1']}, step=global_step,
                        )
                        logger.setLevel(logging.INFO)

                    wandb.log(
                        {"lr": scheduler.get_lr()[0], "loss": (tr_loss - logging_loss) / args.logging_steps},
                        step=global_step
                    )
                    logging_loss = tr_loss

                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    #model_to_save.save_pretrained(output_dir)

                    #torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'reader_model.pt'))
                    model_to_save.save_pretrained(os.path.join(output_dir, 'reader_model'))

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            batch = tuple(t.to('cpu') for t in batch)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # QueryEncoder
    parser.add_argument('--model_type', default='bert', type=str)
    parser.add_argument("--pretrained_name_or_path", default='SpanBERT/spanbert-base-cased', type=str)
    parser.add_argument("--config_name", default="", type=str)
    parser.add_argument("--tokenizer_name", default="", type=str)
    parser.add_argument("--do_lower_case", default=False, action='store_true')
    parser.add_argument('--max_query_length', default=64, type=int)
    parser.add_argument("--cache_dir", default=None, type=str)
    parser.add_argument("--query_encoder_path", default='', type=str)
    parser.add_argument("--query_port", default='-1', type=str)

    # PhraseIndex
    parser.add_argument('--dump_dir', default='dump')
    parser.add_argument('--phrase_dir', default='phrase')
    parser.add_argument('--index_dir', default='256_flat_SQ4')
    parser.add_argument('--index_name', default='index.faiss')
    parser.add_argument('--idx2id_name', default='idx2id.hdf5')
    parser.add_argument('--index_port', default='-1', type=str)

    # These can be dynamically changed.
    parser.add_argument('--max_answer_length', default=10, type=int)
    parser.add_argument('--top_k', default=10, type=int)
    parser.add_argument('--nprobe', default=256, type=int)
    parser.add_argument('--truecase', default=False, action='store_true')
    parser.add_argument("--truecase_path", default='truecase/english_with_questions.dist', type=str)

    # KILT
    parser.add_argument('--is_kilt', default=False, action='store_true')
    parser.add_argument('--kilt_gold_path', default='kilt/trex/trex-dev-kilt.jsonl')
    parser.add_argument('--title2wikiid_path', default='wikidump/title2wikiid.json')

    # Serving options
    parser.add_argument('--examples_path', default='examples.txt')

    # Training query encoder
    parser.add_argument('--train_path', default=None)
    parser.add_argument('--per_gpu_train_batch_size', default=48, type=int)
    parser.add_argument('--num_train_epochs', default=10, type=float)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--warmup_steps", default=0.1, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--use_inbatch_neg", action="store_true", help="Whether to run with inb-neg.")
    parser.add_argument("--fp16", action="store_true", help="For fp16")
    parser.add_argument('--output_dir', default=None, type=str)


    # Training bert reranker
    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument("--append_title", action="store_true", help="Whether to append title in context.")
    parser.add_argument("--threads", type=int, default=20, help="multiple threads for converting example to features")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--load_dir",
        default=None,
        type=str,
        help="The load directory where the model checkpoints are saved. Set to output_dir if not specified.",
    )
    parser.add_argument("--logging_steps", type=int, default=5000, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=9999999999, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )

    # Evaluation
    parser.add_argument('--dev_path', default='open-qa/nq-open/dev_preprocessed.json')
    parser.add_argument('--test_path', default='open-qa/nq-open/test_preprocessed.json')
    parser.add_argument('--candidate_path', default=None)
    parser.add_argument('--regex', default=False, action='store_true')
    parser.add_argument('--eval_batch_size', default=10, type=int)

    # Run mode
    parser.add_argument('--run_mode', default='train_query')
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--draft', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--seed', default=1992, type=int)
    args = parser.parse_args()

    # Seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.run_mode == 'train_reader':
        train_reader(args)

    else:
        raise NotImplementedError
