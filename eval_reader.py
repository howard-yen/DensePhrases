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
from tqdm import tqdm

from densephrases.utils.eval_utils import normalize_answer, f1_score, exact_match_score, drqa_exact_match_score, \
        drqa_regex_match_score, drqa_metric_max_over_ground_truths, drqa_normalize
from densephrases.utils.open_utils import load_query_encoder, load_phrase_index, get_query2vec, load_qa_pairs
from densephrases.utils.kilt.eval import evaluate as kilt_evaluate
from densephrases.utils.kilt.kilt_utils import store_data as kilt_store_data
from eval_phrase_retrieval import evaluate_results

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
from utils import process_sample

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def eval_reader(args, model=None, tokenizer=None, config=None):
    #wandb setup
    wandb.init(project="DensePhrases-Reader-Evaluation", notes="", entity="howard-yen", mode="online" if args.wandb else "disabled")
    wandb.config.update(args)

    # Setup CUDA, GPU & distributed evaluation
    if args.local_rank == -1 or not args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        args.n_gpu = 0 if not args.cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)

        # in case this was called by the training method
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        args.n_gpu = 1
    args.device = device

    config = config if config is not None else AutoConfig.from_pretrained(args.config_name if args.config_name else args.pretrained_name_or_path, cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.pretrained_name_or_path, cache_dir=args.cache_dir if args.cache_dir else None)

    if model is not None:
        model = model
    else:
        model = Reader(config=config)
        if args.load_dir:
            if os.path.exists(os.path.join(args.load_dir, "pytorch_model.bin")):
                #model = Reader.from_pretrained(os.path.join(args.load_dir, "reader_model"))
                model.load_state_dict(torch.load(os.path.join(args.load_dir, "pytorch_model.bin")))
                logger.info(f'model loaded from {args.load_dir}')
            else:
                logger.info('missing reader model, exiting')
                exit()
        else:
            logger.info('missing reader model load dir, exiting')
            exit()

    model.to(device)
    model.eval()


    with open(args.test_path, encoding='utf-8') as f:
        data = json.load(f)

    qids = []
    predictions = []
    evidences = []
    titles = []
    scores = []
    answers = []
    questions = []
    new_predictions = []

    count = 0

    softmax = torch.nn.Softmax(dim=-1)

    SEP_INPUT_ID = 102

    with torch.no_grad():
        for qid in tqdm(data):
            if count >= 1e100000:
                break
            else:
                count += 1
            qids.append(qid)
            sample = data[qid]

            question = sample['question']
            questions.append(question)
            answer = sample['answer']
            prediction = sample['prediction']
            title = sample['title']
            evidence = sample['evidence']
            score = sample['score']

            processed = process_sample(tokenizer, question, answer, prediction, title, evidence, max_seq_length=args.max_seq_length)
            M = len(prediction)

            input_ids = torch.tensor(processed['input_ids']).to(device)
            attention_mask = torch.tensor(processed['attention_mask']).to(device)
            token_type_ids = torch.tensor(processed['token_type_ids']).to(device)

            #can also calculate loss for the eval set
            output = model(input_ids=input_ids.view(-1, M, args.max_seq_length), attention_mask=attention_mask.view(-1, M, args.max_seq_length), token_type_ids=token_type_ids.view(-1, M, args.max_seq_length))

            #start_logits = output.start_logits
            #end_logits = output.end_logits
            start_logits, end_logits, rank_logits = output

            start_prob = softmax(start_logits)
            end_prob = softmax(end_logits)
            rank_prob = softmax(rank_logits)

            # do not let the start or end be [SEP]
            start_prob[input_ids.view(1, M, -1)==SEP_INPUT_ID] = 0
            end_prob[input_ids.view(1, M, -1)==SEP_INPUT_ID] = 0

            span_prob = torch.bmm(start_prob.view(M, -1, 1), end_prob.view(M, 1, -1))
            span_prob = torch.triu(span_prob)

            # mask to limit the length of the span
            mask = torch.ones_like(span_prob)
            mask = torch.triu(mask, diagonal=args.max_answer_length)
            span_prob[mask==1] = 0
            # mask out the question
            span_prob[token_type_ids==0] = 0

            best_span_prob, best_span_idx = torch.max(span_prob.view(M, -1), dim=1)

            offset = processed['offset_mapping']
            new_prediction = [processed['backs'][idx][offset[idx][pos//args.max_seq_length][0]:offset[idx][pos%args.max_seq_length][1]] for idx, pos in enumerate(best_span_idx)]

            rank_prob = rank_prob.view(M)
            rank_prob *= best_span_prob

            _, indices = torch.sort(rank_prob, descending=True)

            new_prediction = [new_prediction[i] for i in indices]
            evidence = [evidence[i] for i in indices]
            title = [title[i] for i in indices]
            score = [score[i] for i in indices]


            input_ids = input_ids.to('cpu')
            attention_mask = attention_mask.to('cpu')
            token_type_ids = token_type_ids.to('cpu')

            new_predictions.append(new_prediction)
            predictions.append(prediction)
            evidences.append(evidence)
            titles.append(title)
            scores.append(score)
            answers.append(answer)

    return evaluate_results(new_predictions, qids, questions, answers, args, evidences, scores, titles)



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
    parser.add_argument('--aggregate', default=False, action='store_true')
    parser.add_argument('--agg_strat', default='opt1', type=str)
    parser.add_argument('--truecase', default=False, action='store_true')
    parser.add_argument("--truecase_path", default='truecase/english_with_questions.dist', type=str)

    # KILT
    parser.add_argument('--is_kilt', default=False, action='store_true')
    parser.add_argument('--kilt_gold_path', default='kilt/trex/trex-dev-kilt.jsonl')
    parser.add_argument('--title2wikiid_path', default='wikidump/title2wikiid.json')

    # Serving options
    parser.add_argument('--examples_path', default='examples.txt')

    # Evaluation
    parser.add_argument('--dev_path', default='open-qa/nq-open/dev_preprocessed.json')
    parser.add_argument('--test_path', default='open-qa/nq-open/test_preprocessed.json')
    parser.add_argument('--candidate_path', default=None)
    parser.add_argument('--regex', default=False, action='store_true')
    parser.add_argument('--eval_batch_size', default=10, type=int)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")

    # Run mode
    parser.add_argument('--run_mode', default='eval')
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--draft', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--save_pred', default=False, action='store_true')
    parser.add_argument('--seed', default=1992, type=int)
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument(
        "--load_dir",
        default=None,
        type=str,
        help="The load directory where the model checkpoints are saved. Set to output_dir if not specified.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    args = parser.parse_args()

    # Seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.run_mode == 'eval_reader':
        eval_reader(args)

    else:
        raise NotImplementedError
