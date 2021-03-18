import json
import argparse
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import os
import random
import numpy as np
import requests
import logging
import math
import copy
import wandb
import string
import re

from time import time
from tqdm import tqdm, trange

from densephrases.models import DensePhrases, MIPS, MIPSLight
from densephrases.utils.single_utils import backward_compat
from densephrases.utils.squad_utils import get_question_dataloader, TrueCaser
import densephrases.utils.squad_utils as squad_utils
from densephrases.utils.embed_utils import get_question_results
from densephrases.utils.eval_utils import normalize_answer, f1_score, exact_match_score, drqa_exact_match_score, \
        drqa_regex_match_score, drqa_metric_max_over_ground_truths, drqa_normalize
from densephrases.utils.kilt.eval import evaluate as kilt_evaluate
from densephrases.utils.kilt.kilt_utils import store_data as kilt_store_data
from densephrases.experiments.run_single import load_and_cache_examples
from reader import Reader

from transformers import (
    MODEL_MAPPING,
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    AutoModelForQuestionAnswering,
    AdamW,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def load_query_encoder(device, args):
    assert args.query_encoder_path

    # Configure paths for query encoder serving
    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.pretrained_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.pretrained_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    # Pre-trained DensePhrases
    model = DensePhrases(
        config=config,
        tokenizer=tokenizer,
        transformer_cls=MODEL_MAPPING[config.__class__],
    )
    try:
        model.load_state_dict(backward_compat(
            torch.load(os.path.join(args.query_encoder_path, 'pytorch_model.bin'), map_location=torch.device('cpu'))
        ))
    except Exception as e:
        print(e)
        model.load_state_dict(torch.load(os.path.join(args.query_encoder_path, 'pytorch_model.bin')), strict=False)
    model.to(device)

    logger.info(f'DensePhrases loaded from {args.query_encoder_path} having {MODEL_MAPPING[config.__class__]}')
    logger.info('Number of model parameters: {:,}'.format(sum(p.numel() for p in model.parameters())))
    return model, tokenizer


def get_query2vec(query_encoder, tokenizer, args, batch_size=64):
    device = 'cuda' if args.cuda else 'cpu'
    def query2vec(queries):
        question_dataloader, question_examples, query_features = get_question_dataloader(
            queries, tokenizer, args.max_query_length, batch_size=batch_size
        )
        question_results = get_question_results(
            question_examples, query_features, question_dataloader, device, query_encoder, batch_size=batch_size
        )
        if args.debug:
            logger.info(f"{len(query_features)} queries: {' '.join(query_features[0].tokens_)}")
        outs = []
        for qr_idx, question_result in enumerate(question_results):
            out = (
                question_result.start_vec.tolist(), question_result.end_vec.tolist(), query_features[qr_idx].tokens_
            )
            outs.append(out)
        return outs
    return query2vec


def load_phrase_index(args, load_light=False):
    # Configure paths for index serving
    phrase_dump_dir = os.path.join(args.dump_dir, args.phrase_dir)
    index_dir = os.path.join(args.dump_dir, args.index_dir)
    index_path = os.path.join(index_dir, args.index_name)
    idx2id_path = os.path.join(index_dir, args.idx2id_name)

    # Load mips
    mips_class = MIPS if not load_light else MIPSLight
    mips = mips_class(
        phrase_dump_dir=phrase_dump_dir,
        index_path=index_path,
        idx2id_path=idx2id_path,
        cuda=args.cuda,
        logging_level=logging.DEBUG if args.debug else logging.INFO
    )
    return mips


def embed_all_query(questions, args, query_encoder, tokenizer, batch_size=48):
    query2vec = get_query2vec(
        query_encoder=query_encoder, tokenizer=tokenizer, args=args, batch_size=batch_size
    )

    all_outs = []
    for q_idx in tqdm(range(0, len(questions), batch_size)):
        outs = query2vec(questions[q_idx:q_idx+batch_size])
        all_outs += outs
    start = np.concatenate([out[0] for out in all_outs], 0)
    end = np.concatenate([out[1] for out in all_outs], 0)
    query_vec = np.concatenate([start, end], 1)
    logger.info(f'Query reps: {query_vec.shape}')
    return query_vec


def load_qa_pairs(data_path, args, draft_num_examples=1000, shuffle=False):
    q_ids = []
    questions = []
    answers = []
    data = json.load(open(data_path))['data']
    for item in data:
        q_id = item['id']
        question = item['question']
        answer = item['answers']
        if len(answer) == 0:
            continue
        q_ids.append(q_id)
        questions.append(question)
        answers.append(answer)
    questions = [query[:-1] if query.endswith('?') else query for query in questions]

    if args.truecase:
        try:
            logger.info('Loading truecaser for queries')
            truecase = TrueCaser(os.path.join(os.environ['DPH_DATA_DIR'], args.truecase_path))
            questions = [truecase.get_true_case(query) if query == query.lower() else query for query in questions]
        except Exception as e:
            print(e)

    if args.do_lower_case:
        logger.info(f'Lowercasing queries')
        questions = [query.lower() for query in questions]

    if args.draft:
        q_ids = np.array(q_ids)[:draft_num_examples].tolist()
        questions = np.array(questions)[:draft_num_examples].tolist()
        answers = np.array(answers)[:draft_num_examples].tolist()

    if shuffle:
        qa_pairs = list(zip(q_ids, questions, answers))
        random.shuffle(qa_pairs)
        q_ids, questions, answers = zip(*qa_pairs)
        logger.info(f'Shuffling QA pairs')

    logger.info(f'Loading {len(questions)} questions from {data_path}')
    logger.info(f'Sample Q ({q_ids[0]}): {questions[0]}, A: {answers[0]}')
    return q_ids, questions, answers


def eval_inmemory(args, mips=None, query_encoder=None, tokenizer=None):
    # Load dataset and encode queries
    qids, questions, answers = load_qa_pairs(args.test_path, args)
    if query_encoder is None:
        print(f'Query encoder will be loaded from {args.query_encoder_path}')
        device = 'cuda' if args.cuda else 'cpu'
        query_encoder, tokenizer = load_query_encoder(device, args)
    query_vec = embed_all_query(questions, args, query_encoder, tokenizer)

    # Load MIPS
    if mips is None:
        mips = load_phrase_index(args)

    # Search
    step = args.eval_batch_size
    predictions = []
    evidences = []
    titles = []
    scores = []
    for q_idx in tqdm(range(0, len(questions), step)):
        result = mips.search(
            query_vec[q_idx:q_idx+step],
            q_texts=questions[q_idx:q_idx+step], nprobe=args.nprobe,
            top_k=args.top_k, max_answer_length=args.max_answer_length,
        )
        prediction = [[ret['answer'] for ret in out] if len(out) > 0 else [''] for out in result]
        evidence = [[ret['context'] for ret in out] if len(out) > 0 else [''] for out in result]
        title = [[ret['title'] for ret in out] if len(out) > 0 else [['']] for out in result]
        score = [[ret['score'] for ret in out] if len(out) > 0 else [-1e10] for out in result]
        predictions += prediction
        evidences += evidence
        titles += title
        scores += score

    logger.info(f"Avg. {sum(mips.num_docs_list)/len(mips.num_docs_list):.2f} number of docs per query")
    eval_fn = evaluate_results if not args.is_kilt else evaluate_results_kilt
    return eval_fn(predictions, qids, questions, answers, args, evidences, scores, titles)


def evaluate_results(predictions, qids, questions, answers, args, evidences, scores, titles, q_tokens=None):
    wandb.init(project="DensePhrases (open)", mode="online" if args.wandb else "disabled")
    wandb.config.update(args)

    # Filter if there's candidate
    if args.candidate_path is not None:
        candidates = set()
        with open(args.candidate_path) as f:
            for line in f:
                line = line.strip().lower()
                candidates.add(line)
        logger.info(f'{len(candidates)} candidates are loaded from {args.candidate_path}')
        topk_preds = [list(filter(lambda x: (x in candidates) or (x.lower() in candidates), a)) for a in predictions]
        topk_preds = [a[:args.top_k] if len(a) > 0 else [''] for a in topk_preds]
        predictions = topk_preds[:]
        top1_preds = [a[0] for a in topk_preds]
    else:
        predictions = [a[:args.top_k] if len(a) > 0 else [''] for a in predictions]
        top1_preds = [a[0] for a in predictions]
    no_ans = sum([a == '' for a in top1_preds])
    logger.info(f'no_ans/all: {no_ans}, {len(top1_preds)}')
    logger.info(f'Evaluating {len(top1_preds)} answers.')

    # Get em/f1
    f1s, ems = [], []
    for prediction, groundtruth in zip(top1_preds, answers):
        if len(groundtruth)==0:
            f1s.append(0)
            ems.append(0)
            continue
        f1s.append(max([f1_score(prediction, gt)[0] for gt in groundtruth]))
        ems.append(max([exact_match_score(prediction, gt) for gt in groundtruth]))
    final_f1, final_em = np.mean(f1s), np.mean(ems)
    logger.info('EM: %.2f, F1: %.2f'%(final_em * 100, final_f1 * 100))

    # Top 1/k em (or regex em)
    exact_match_topk = 0
    exact_match_top1 = 0
    f1_score_topk = 0
    f1_score_top1 = 0
    pred_out = {}
    for i in range(len(predictions)):
        # For debugging
        if i < 3:
            logger.info(f'{i+1}) {questions[i]}')
            logger.info(f'=> groundtruths: {answers[i]}, top 5 prediction: {predictions[i][:5]}')

        match_fn = drqa_regex_match_score if args.regex else drqa_exact_match_score
        em_topk = max([drqa_metric_max_over_ground_truths(
            match_fn, prediction, answers[i]
        ) for prediction in predictions[i][:args.top_k]])
        em_top1 = drqa_metric_max_over_ground_truths(
            match_fn, top1_preds[i], answers[i]
        )
        exact_match_topk += em_topk
        exact_match_top1 += em_top1

        f1_topk = 0
        f1_top1 = 0
        if not args.regex:
            match_fn = lambda x, y: f1_score(x, y)[0]
            f1_topk = max([drqa_metric_max_over_ground_truths(
                match_fn, prediction, answers[i]
            ) for prediction in predictions[i][:args.top_k]])
            f1_top1 = drqa_metric_max_over_ground_truths(
                match_fn, top1_preds[i], answers[i]
            )
            f1_score_topk += f1_topk
            f1_score_top1 += f1_top1

        pred_out[qids[i]] = {
                'question': questions[i],
                'answer': answers[i], 'prediction': predictions[i], 'score': scores[i], 'title': titles[i],
                'evidence': evidences[i] if evidences is not None else '',
                'em_top1': bool(em_top1), f'em_top{args.top_k}': bool(em_topk),
                'f1_top1': f1_top1, f'f1_top{args.top_k}': f1_topk,
                'q_tokens': q_tokens[i] if q_tokens is not None else ['']
        }

    total = len(predictions)
    exact_match_top1 = 100.0 * exact_match_top1 / total
    f1_score_top1 = 100.0 * f1_score_top1 / total
    logger.info({'exact_match_top1': exact_match_top1, 'f1_score_top1': f1_score_top1})
    exact_match_topk = 100.0 * exact_match_topk / total
    f1_score_topk = 100.0 * f1_score_topk / total
    logger.info({f'exact_match_top{args.top_k}': exact_match_topk, f'f1_score_top{args.top_k}': f1_score_topk})
    wandb.log(
        {"Top1 EM": exact_match_top1, "Top1 F1": f1_score_top1,
         "Topk EM": exact_match_topk, "Topk F1": f1_score_topk}
    )

    # Dump predictions
    if len(args.query_encoder_path) == 0:
        pred_dir = os.path.join(os.environ['DPH_SAVE_DIR'], 'pred')
    else:
        pred_dir = os.path.join(args.query_encoder_path, 'pred')
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    pred_path = os.path.join(
        pred_dir, os.path.splitext(os.path.basename(args.test_path))[0] + f'_{total}.pred'
    )
    logger.info(f'Saving prediction file to {pred_path}')
    with open(pred_path, 'w') as f:
        json.dump(pred_out, f)

    return exact_match_top1


def evaluate_results_kilt(predictions, qids, questions, answers, args, evidences, scores, titles):
    wandb.init(project="DensePhrases (KILT)", mode="online" if args.wandb else "disabled")
    wandb.config.update(args)
    total=len(predictions)

    # load title2id dict and convert predicted titles into wikipedia_ids
    with open(args.title2wikiid_path) as f:
        title2wikiid = json.load(f)
    pred_wikipedia_ids = [[[title2wikiid[t] for t in title_] for title_ in title] for title in titles]

    # dump official predictions
    if len(args.query_encoder_path) == 0:
        pred_dir = os.path.join(os.environ['DPH_SAVE_DIR'], 'pred-kilt')
    else:
        pred_dir = os.path.join(args.query_encoder_path, 'pred-kilt')
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    pred_official_path = os.path.join(
        pred_dir, f'{args.query_encoder_path.split("/")[-1]}_' +
        os.path.splitext(os.path.basename(args.test_path))[0] + f'_{total}.jsonl'
    )
    official_preds_to_save = []
    for prediction, question, pred_wikipedia_id, qid in zip(predictions, questions, pred_wikipedia_ids, qids):
        outputs = []
        for pred, pred_wid in zip(prediction, pred_wikipedia_id):
            outputs.append({
                'answer': pred,
                'provenance':[{'wikipedia_id':pred_wid_} for pred_wid_ in pred_wid]
            })

        official_preds_to_save.append({
            'id': qid,
            'input': question,
            'output': [outputs[0]]
        })

    logger.info(f'Saving official prediction file to {pred_official_path}')
    kilt_store_data(pred_official_path, official_preds_to_save)

    assert '.jsonl' in args.kilt_gold_path, "kilt_gold_path should be .jsonl"
    result = kilt_evaluate(
        gold=args.kilt_gold_path,
        guess=pred_official_path)

    # logging results
    result_to_logging = {
        'accuracy':result['downstream']['accuracy'],
        'f1':result['downstream']['f1'],
        'KILT-accuracy':result['kilt']['KILT-accuracy'],
        'KILT-f1':result['kilt']['KILT-f1'],
        'Rprec':result['retrieval']['Rprec'],
        'recall@5':result['retrieval']['recall@5']
    }

    logger.info(result_to_logging)
    wandb.log(result_to_logging)

    # make custom predictions
    pred_out = {}
    for i in range(len(predictions)):
        # For debugging
        if i < 3:
            logger.info(f'{i+1}) {questions[i]}')
            logger.info(f'=> groundtruths: {answers[i]}, top 5 prediction: {predictions[i][:5]}')

        guess_answer = predictions[i][0]
        gold_candidate_answers = answers[i]
        local_accuracy = 0
        if guess_answer in gold_candidate_answers:
            local_accuracy = 1

        pred_out[qids[i]] = {
                'question': questions[i],
                'answer': answers[i], 'prediction': predictions[i], 'score': scores[i], 'title': titles[i],
                'evidence': evidences[i] if evidences is not None else '',
                'em_top1': bool(local_accuracy),
        }

    # dump custom predictions
    pred_path = os.path.join(
        pred_dir, os.path.splitext(os.path.basename(args.test_path))[0] + f'_{total}.pred'
    )
    logger.info(f'Saving custom prediction file to {pred_path}')
    with open(pred_path, 'w') as f:
        json.dump(pred_out, f)

    return result['downstream']['accuracy']


def eval_results_bert(args, mips=None, query_encoder=None, tokenizer=None):
    qids, questions, answers = load_qa_pairs(args.test_path, args)

    if query_encoder is None:
        print(f'Query encoder will be loaded from {args.query_encoder_path}')
        device = 'cuda' if args.cuda else 'cpu'
        query_encoder, tokenizer = load_query_encoder(device, args)

    query_vec = embed_all_query(questions, args, query_encoder, tokenizer)

    if mips is None:
        mips = load_phrase_index(args)

    step = args.eval_batch_size
    predictions=[]
    evidences=[]
    titles=[]
    scores=[]

    for q_idx in tqdm(range(0, len(questions), step)):
        result = mips.search(
                query_vec[q_idx:q_idx+step],
                q_texts=questions[q_idx:q_idx+step], nprobe=args.nprobe,
                top_k=args.top_k, max_answer_length=args.max_answer_length,
        )
        # possible prediction for each qa pair in the batch
        # the actual phrase
        prediction = [[ret['answer'] for ret in out] if len(out) > 0 else [''] for out in result]
        # the passage that the phrase is found in
        evidence = [[ret['context'] for ret in out] if len(out) > 0 else [''] for out in result]
        # wikipedia title?
        title = [[ret['title'] for ret in out] if len(out) > 0 else [''] for out in result]
        # score of the phrase
        score = [[ret['score'] for ret in out] if len(out) > 0 else [-1e10] for out in result]
        predictions += prediction
        evidences += evidence
        titles += title
        scores += score

    logger.info(f'Avg. {sum(mips.num_docs_list)/len(mips.num_docs_list):.2f} number of docs per entry')

    logger.info(f'Reranking using BERT model')

    return

def train_bert(args, mips=None, query_encoder=None):
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

    logger.info('Training BERT model used for reranking')
    config = AutoConfig.from_pretrained('./outputs/spanbert-base-cased-sqdnq')
    tokenizer = AutoTokenizer.from_pretrained('./outputs/spanbert-base-cased-sqdnq')

    logger.info('Loaded the training examples')

    dataset, examples, features = load_and_cache_examples(args, tokenizer=tokenizer, output_examples=True)

    if mips is None:
        mips = load_phrase_index(args)

    if query_encoder is None:
        print(f'Query encoder will be loaded from {args.query_encoder_path}')
        device = 'cuda' if args.cuda else 'cpu'
        query_encoder, tokenizer = load_query_encoder(device, args)

    questions = examples[ex.question_text for ex in examples]
    query_vec = embed_all_query(questions, args, query_encoder, tokenizer)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    model = Reader(config=config, tokenizer=tokenizer)
    model.to(device)

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
    if args.output_dir:
        if os.path.isfile(os.path.join(args.output_dir, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.output_dir, "scheduler.pt")
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(args.output_dir, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(args.output_dir, "scheduler.pt")))
            logger.info(f'optimizer and scheduler loaded from {args.output_dir}')

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
    logger.info("  Num examples = %d", len(dataset))
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

        epoch_iterator = tqdm(dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
                "input_ids_": batch[8],
                "attention_mask_": batch[9],
                "token_type_ids_": batch[10],
            }

            ## need more input pre-processing here: distantly supervised labeling
            example_index = [] # need to figure out how to get this

            # get the top k phrases for each query in the batch
            queries = []
            q_texts = []
            answers = []
            for ex_idx in example_index:
                queries.append(query_vec[ex_idx])
                q_texts.append(questions[ex_idx])
                answer.append(examples[ex_idx].answer_text)

            result = mips.search(queries, q_texts=q_texts, nprobe=args.nprobe, top_k=args.top_k, max_answer_length=args.max_answer_length)

            prediction = [[ret['answer'] for ret in out] if len(out) > 0 else [''] for out in result]
            evidence = [[ret['context'] for ret in out] if len(out) > 0 else [''] for out in result]
            title = [[ret['title'] for ret in out] if len(out) > 0 else [['']] for out in result]
            score = [[ret['score'] for ret in out] if len(out) > 0 else [-1e10] for out in result]

            start_positions = []
            end_positions = []
            # label each of the evidence
            # each question can have multiple evidences and
            # each evidence can have multiple exact matches
            for answer in answers:
                starts = []
                ends = []
                for evi in evidence:
                    exact_matches = re.finditer(re.escape(answer), evi)
                    starts.append([m.start() for m in exact_matches])
                    ends.append([m.end() for m in exact_matches])
                start_positions.append(starts)
                end_positions.append(ends)

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
                        results, _ = evaluate(args, model, tokenizer, prefix=global_step)
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

                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def train_query_encoder(args, mips=None):
    # Freeze one for MIPS
    device = 'cuda' if args.cuda else 'cpu'
    logger.info("Loading pretrained encoder: this one is for MIPS (fixed)")
    pretrained_encoder, tokenizer = load_query_encoder(device, args)

    # Train another
    logger.info("Loading target encoder: this one is for training")
    target_encoder, _= load_query_encoder(device, args)

    # MIPS
    if mips is None:
        mips = load_phrase_index(args)

    # Optimizer setting
    def is_train_param(name):
        if name.endswith("bert_start.embeddings.word_embeddings.weight") or \
            name.endswith("bert_end.embeddings.word_embeddings.weight") or \
            name.endswith("bert_q_start.embeddings.word_embeddings.weight") or \
            name.endswith("bert_q_end.embeddings.word_embeddings.weight"):
            logger.info(f'freezing {name}')
            return False
        return True
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [{
            "params": [
                p for n, p in target_encoder.named_parameters() \
                    if not any(nd in n for nd in no_decay) and is_train_param(n)
            ],
            "weight_decay": 0.01,
        }, {
            "params": [
                p for n, p in target_encoder.named_parameters() \
                    if any(nd in n for nd in no_decay) and is_train_param(n)
            ],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    step_per_epoch = math.ceil(len(load_qa_pairs(args.train_path, args)[1]) / args.per_gpu_train_batch_size)
    t_total = int(step_per_epoch // args.gradient_accumulation_steps * args.num_train_epochs)
    logger.info(f"Train for {t_total} iterations")
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
     )
    eval_steps = math.ceil(len(load_qa_pairs(args.dev_path, args)[1]) / args.eval_batch_size)
    logger.info(f"Test takes {eval_steps} iterations")

    # Train arguments
    args.per_gpu_train_batch_size = int(args.per_gpu_train_batch_size / args.gradient_accumulation_steps)
    best_acc = -1000.0
    for ep_idx in range(int(args.num_train_epochs)):

        # Training
        total_loss = 0.0
        total_accs = []
        total_accs_k = []

        # Load training dataset
        _, questions, answers = load_qa_pairs(args.train_path, args, shuffle=True)
        pbar = tqdm(get_top_phrases(
            mips, questions, answers, pretrained_encoder, tokenizer,
            args.per_gpu_train_batch_size, args.train_path, args)
        )

        for step_idx, (questions, answers, outs) in enumerate(pbar):
            train_dataloader, _, _ = get_question_dataloader(
                questions, tokenizer, args.max_query_length, batch_size=args.per_gpu_train_batch_size
            )
            svs, evs, tgts = get_phrase_vecs(mips, questions, answers, outs, args)

            target_encoder.train()
            svs_t = torch.Tensor(svs).to(device)
            evs_t = torch.Tensor(evs).to(device)
            tgts_t = [torch.Tensor([tgt_ for tgt_ in tgt if tgt_ is not None]).to(device) for tgt in tgts]

            # Train query encoder
            assert len(train_dataloader) == 1
            for batch in train_dataloader:
                batch = tuple(t.to(device) for t in batch)
                loss, accs = target_encoder.train_query(
                    input_ids_=batch[0], attention_mask_=batch[1], token_type_ids_=batch[2],
                    start_vecs=svs_t,
                    end_vecs=evs_t,
                    targets=tgts_t
                )

                # Optimize, get acc and report
                if loss is not None:
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    if args.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    total_loss += loss.mean().item()
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(target_encoder.parameters(), args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    target_encoder.zero_grad()

                    pbar.set_description(
                        f"Ep {ep_idx+1} Tr loss: {loss.mean().item():.2f}, acc: {sum(accs)/len(accs):.3f}"
                    )

                if accs is not None:
                    total_accs += accs
                    total_accs_k += [len(tgt) > 0 for tgt in tgts_t]
                else:
                    total_accs += [0.0]*len(tgts_t)
                    total_accs_k += [0.0]*len(tgts_t)

        step_idx += 1
        logger.info(
            f"Avg train loss ({step_idx} iterations): {total_loss/step_idx:.2f} | train " +
            f"acc@1: {sum(total_accs)/len(total_accs):.3f} | acc@{args.top_k}: {sum(total_accs_k)/len(total_accs_k):.3f}"
        )

        # Evaluation
        logger.setLevel(logging.WARNING)
        new_args = copy.deepcopy(args)
        new_args.wandb = False
        new_args.top_k = 10
        new_args.test_path = args.dev_path
        dev_em = eval_inmemory(new_args, mips, target_encoder, tokenizer)
        logger.setLevel(logging.INFO)
        logger.info(f"Develoment set acc@1: {dev_em:.3f}")

        # Save best model
        if dev_em > best_acc:
            best_acc = dev_em
            save_path = args.output_dir
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            target_encoder.save_pretrained(save_path)
            logger.info(f"Saved best model with acc {best_acc:.3f} into {save_path}")

        if (ep_idx + 1) % 1 == 0:
            logger.info('Updating pretrained encoder')
            pretrained_encoder = target_encoder
            train_cache = []
            eval_cache = []

    print()
    logger.info(f"Best model has acc {best_acc:.3f} saved as {save_path}")


def get_top_phrases(mips, questions, answers, query_encoder, tokenizer, batch_size, path, args):
    # Search
    step = batch_size
    phrase_idxs = []
    search_fn = mips.search
    query2vec = get_query2vec(
        query_encoder=query_encoder, tokenizer=tokenizer, args=args, batch_size=batch_size
    )
    for q_idx in tqdm(range(0, len(questions), step)):
        outs = query2vec(questions[q_idx:q_idx+step])
        start = np.concatenate([out[0] for out in outs], 0)
        end = np.concatenate([out[1] for out in outs], 0)
        query_vec = np.concatenate([start, end], 1)

        outs = search_fn(
            query_vec,
            q_texts=questions[q_idx:q_idx+step], nprobe=args.nprobe,
            top_k=args.top_k, return_idxs=True,
            max_answer_length=args.max_answer_length,
        )
        yield questions[q_idx:q_idx+step], answers[q_idx:q_idx+step], outs


def get_phrase_vecs(mips, questions, answers, outs, args):
    assert mips is not None

    # Get phrase and vectors
    phrase_idxs = [[(out_['doc_idx'], out_['start_idx'], out_['end_idx'], out_['answer'],
        out_['start_vec'], out_['end_vec']) for out_ in out]
        for out in outs
    ]
    for b_idx, phrase_idx in enumerate(phrase_idxs):
        while len(phrase_idxs[b_idx]) < args.top_k * 2: # two separate top-k from start/end
            phrase_idxs[b_idx].append((-1, 0, 0, '', np.zeros((768)), np.zeros((768))))
        phrase_idxs[b_idx] = phrase_idxs[b_idx][:args.top_k*2]
    flat_phrase_idxs = [phrase for phrase_idx in phrase_idxs for phrase in phrase_idx]
    doc_idxs = [int(phrase_idx_[0]) for phrase_idx_ in flat_phrase_idxs]
    start_idxs = [int(phrase_idx_[1]) for phrase_idx_ in flat_phrase_idxs]
    end_idxs = [int(phrase_idx_[2]) for phrase_idx_ in flat_phrase_idxs]
    phrases = [phrase_idx_[3] for phrase_idx_ in flat_phrase_idxs]
    start_vecs = [phrase_idx_[4] for phrase_idx_ in flat_phrase_idxs]
    end_vecs = [phrase_idx_[5] for phrase_idx_ in flat_phrase_idxs]

    start_vecs = np.stack(
        # [mips.dequant(mips.offset, mips.scale, start_vec) # Use this for IVFSQ4
        [start_vec
         for start_vec, start_idx in zip(start_vecs, start_idxs)]
    )

    end_vecs = np.stack(
        # [mips.dequant(mips.offset, mips.scale, end_vec) # Use this for IVFSQ4
        [end_vec
         for end_vec, end_idx in zip(end_vecs, end_idxs)]
    )

    zero_mask = np.array([[1] if doc_idx >= 0 else [0] for doc_idx in doc_idxs])
    start_vecs = start_vecs * zero_mask
    end_vecs = end_vecs * zero_mask

    # Find targets based on exact string match
    match_fn = drqa_regex_match_score if args.regex else drqa_exact_match_score # Punctuation included
    targets = [[drqa_metric_max_over_ground_truths(match_fn, phrase[3], answer) for phrase in phrase_idx]
        for phrase_idx, answer in zip(phrase_idxs, answers)]
    targets = [[ii if val else None for ii, val in enumerate(target)] for target in targets]

    # Reshape
    batch_size = len(answers)
    start_vecs = np.reshape(start_vecs, (batch_size, args.top_k*2, -1))
    end_vecs = np.reshape(end_vecs, (batch_size, args.top_k*2, -1))
    return start_vecs, end_vecs, targets


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

    if args.run_mode == 'train_query':
        # Train
        mips = load_phrase_index(args)
        train_query_encoder(args, mips)

        # Eval
        args.query_encoder_path = args.output_dir
        logger.info(f"Evaluating {args.query_encoder_path}")
        args.top_k = 10
        eval_inmemory(args, mips)

    elif args.run_mode == 'eval_inmemory':
        eval_inmemory(args)

    elif args.run_mode == 'eval_results_bert':
        eval_results_bert(args)

    elif args.run_mode == 'train_bert':
        features, dataset = train_bert(args)

    else:
        raise NotImplementedError
