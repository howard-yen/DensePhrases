import torch
import json
from tqdm import tqdm
from torch.utils.data import TensorDataset

def process_sample(tokenizer, question, answer, prediction, title, evidence, max_query_length=50, max_seq_length=384, include_prediction=False, return_tensor=False):
    input_ids = []
    attention_mask = []
    token_type_ids = []
    M = len(prediction)
    offset_mappings = []
    backs = []

    # limit question to max_query_length
    q_tokens = tokenizer(question, return_offsets_mapping=True)
    if len(q_tokens['input_ids']) > max_query_length:
        question = question[0:q_tokens['offset_mapping'][max_query_length-1][1]]

    for idx in range(M):
        if include_prediction:
            back = prediction[idx] + " [SEP] " + title[idx][0] + " [SEP] " + evidence[idx]
        else:
            back = title[idx][0] + " [SEP] " + evidence[idx]

        encoded = tokenizer(question, text_pair=back, padding="max_length", max_length=max_seq_length, truncation=True, return_offsets_mapping=True, return_token_type_ids=True, return_attention_mask=True)

        input_ids.append(encoded['input_ids'])
        attention_mask.append(encoded['attention_mask'])
        token_type_ids.append(encoded['token_type_ids'])
        offset_mappings.append(encoded['offset_mapping'])
        backs.append(back)

    if return_tensor:
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        token_type_ids = torch.tensor(token_type_ids)

    ret = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'offset_mapping': offset_mappings,
        'backs': backs
    }

    return ret


def process_reranker_input(tokenizer, train_file, max_query_length, max_seq_length, top_k):
    logger.info(f'Loading reader input dataset from {train_file}')
    with open(train_file, encoding='utf-8') as f:
        data = json.load(f)['data']

    all_input_ids = []
    all_token_type_ids = []
    all_attention_mask = []
    all_is_impossible = []
    all_ems = []

    for sample_idx, sample in enumerate(tqdm(data)):

        question = sample['question']
        answer = sample['answer']
        predictions = sample['prediction']
        titles = sample['title']
        evidences = sample['evidence']
        scores = sample['score']
        f1s = sample['f1s']
        ems = sample['exact_matches']
        ems = [1 if em else 0 for em in ems]

        if max(ems) < 1:
            continue

        is_impossible = sample['is_impossible']
        is_impossible = [1 if imp else 0 for imp in is_impossible]


        s = process_sample(tokenizer=tokenizer, question=question, answer=answer, prediction=predictions, title=titles, evidence=evidences, max_query_length=max_query_length, max_seq_length=max_seq_length, include_prediction=True)

        input_ids = s['input_ids']
        token_type_ids = s['token_type_ids']
        attention_mask = s['attention_mask']

        # need to pad so ensure the same size in every dimension
        # replaced by negative sample in batch while training
        for extra in range(len(predictions), top_k):
            input_ids.append([0 for i in range(max_seq_length)])
            token_type_ids.append([0 for i in range(max_seq_length)])
            attention_mask.append([0 for i in range(max_seq_length)])
            is_impossible.append(-1)
            ems.append(-1)

        all_input_ids.append(input_ids)
        all_token_type_ids.append(token_type_ids)
        all_attention_mask.append(attention_mask)
        all_is_impossible.append(is_impossible)
        all_ems.append(ems)

    all_input_ids = torch.tensor(all_input_ids)
    all_token_type_ids = torch.tensor(all_token_type_ids)
    all_attention_mask = torch.tensor(all_attention_mask)
    all_is_impossible = torch.tensor(all_is_impossible)
    all_ems = torch.tensor(all_ems)

    dataset = TensorDataset(all_input_ids, all_token_type_ids, all_attention_mask, all_is_impossible, all_ems)

    return dataset


def process_reader_input(tokenizer, train_file, max_query_length, max_seq_length, top_k):
    with open(train_file, encoding='utf-8') as f:
        data = json.load(f)['data']

    all_input_ids = []
    all_token_type_ids = []
    all_attention_mask = []
    all_start_pos = []
    all_end_pos = []
    all_is_impossible = []
    all_best_passage = []

    for sample_idx, sample in enumerate(tqdm(data)):
        question = sample['question']
        answer = sample['answer']
        predictions = sample['prediction']
        titles = sample['title']
        evidences = sample['evidence']
        start_pos = sample['start_pos']
        end_pos = sample['end_pos']
        scores = sample['score']

        is_impossible = sample['is_impossible']
        is_impossible = [1 if imp else 0 for imp in is_impossible]

        input_ids = []
        token_type_ids = []
        attention_mask = []
        starts = []
        ends = []
        best = -1

        question_tokens = tokenizer.tokenize(question)
        if len(question_tokens) > max_query_length:
            question_tokens = question_tokens[0:max_query_length]

        for pred_idx, pred in enumerate(predictions):
            # [CLS] Question [SEP] Title [SEP] Passage
            title_tokens = tokenizer.tokenize(titles[pred_idx])
            front_tokens = question_tokens
            # CLS + 2 SEP
            before_passage_length = len(question_tokens) + len(title_tokens) + 3

            # get the start and ending token index for the answer
            if is_impossible[pred_idx]:
                back_tokens = tokenizer.tokenize(evidences[pred_idx])
                # CLS token is at index 0 and represents the no answer token
                starts.append(0)
                ends.append(0)
            else:
                before_answer = tokenizer.tokenize(evidences[pred_idx][0:start_pos[pred_idx]-1])
                answer_tokens = tokenizer.tokenize(evidences[pred_idx][start_pos[pred_idx]:end_pos[pred_idx]+1])
                after_answer = tokenizer.tokenize(evidences[pred_idx][end_pos[pred_idx]+1:])
                back_tokens = before_answer + answer_tokens + after_answer

                # before passage + before answer + answer token length must be smaller than max_seq_length or the reader won't be able to even see it
                if before_passage_length + len(before_answer) < max_seq_length:
                    starts.append(before_passage_length + len(before_answer))
                    _end_pos = before_passage_length + len(before_answer) + len(answer_tokens) - 1
                    ends.append(_end_pos if _end_pos < max_seq_length else max_seq_length-1)
                    if best == -1:
                        best = pred_idx
                else:
                    starts.append(0)
                    ends.append(0)
                    is_impossible[pred_idx] = 1

            back_tokens = title_tokens + ['[SEP]'] + back_tokens

            encoded = tokenizer(front_tokens, text_pair=back_tokens, max_length=max_seq_length, padding="max_length", truncation=True, return_offsets_mapping=True, return_token_type_ids=True, return_attention_mask=True, is_split_into_words=True)

            input_ids.append(encoded['input_ids'])
            token_type_ids.append(encoded['token_type_ids'])
            attention_mask.append(encoded['attention_mask'])

        # need to pad so ensure the same size in every dimension
        # replaced by negative sample in batch while training
        for extra in range(len(predictions), top_k):
            input_ids.append([0 for i in range(max_seq_length)])
            token_type_ids.append([0 for i in range(max_seq_length)])
            attention_mask.append([0 for i in range(max_seq_length)])
            starts.append(0)
            ends.append(0)
            is_impossible.append(-1)

        all_input_ids.append(input_ids)
        all_token_type_ids.append(token_type_ids)
        all_attention_mask.append(attention_mask)
        all_start_pos.append(starts)
        all_end_pos.append(ends)
        all_is_impossible.append(is_impossible)
        all_best_passage.append(best)

    all_input_ids = torch.tensor(all_input_ids)
    all_token_type_ids = torch.tensor(all_token_type_ids)
    all_attention_mask = torch.tensor(all_attention_mask)
    all_start_pos = torch.tensor(all_start_pos)
    all_end_pos = torch.tensor(all_end_pos)
    all_is_impossible = torch.tensor(all_is_impossible)
    all_best_passage = torch.tensor(all_best_passage)

    dataset = TensorDataset(all_input_ids, all_token_type_ids, all_attention_mask, all_start_pos, all_end_pos, all_is_impossible, all_best_passage)

    return dataset


