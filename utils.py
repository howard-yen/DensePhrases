import torch

def process_sample(tokenizer, question, answer, prediction, title, evidence, max_query_length=50, max_seq_length=384, include_prediction=False, return_tensor=False):
    input_ids = []
    attention_mask = []
    token_type_ids = []
    M = len(prediction)
    offset_mappings = []
    backs = []

    #TODO: modify question to have only max_query_length

    for idx in range(M):
        if include_prediction:
            back = prediction[idx] + " [SEP] " + title[idx][0] + " [SEP] " + evidence[idx]
        else:
            back = title[idx][0] + " [SEP] " + evidence[idx]

        encoded = tokenizer(question, text_pair=back, padding="max_length", max_length=max_seq_length, truncation=True, return_offsets_mapping=True, return_token_type_ids=True)

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
    #all_best_passage = []
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


        input_ids = []
        token_type_ids = []
        attention_mask = []
        best = 0

        question_tokens = tokenizer.tokenize(question)
        if len(question_tokens) > max_query_length:
            question_tokens = question_tokens[0:max_query_length]

        for pred_idx, pred in enumerate(predictions):
            title_tokens = tokenizer.tokenize(titles[pred_idx])
            pred_tokens = tokenizer.tokenize(pred)
            passage_tokens = tokenizer.tokenize(evidences[pred_idx])
            back_tokens = pred_tokens + ['[SEP]'] + title_tokens + ['[SEP]'] + passage_tokens

            encoded = tokenizer.encode_plus(question_tokens, text_pair=back_tokens, max_length=max_seq_length, padding='max_length', truncation=True, return_token_type_ids=True, return_attention_mask=True, is_split_into_words=True)

            input_ids.append(encoded['input_ids'])
            token_type_ids.append(encoded['token_type_ids'])
            attention_mask.append(encoded['attention_mask'])

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
        #all_best_passage.append(best)
        all_ems.append(ems)

    all_input_ids = torch.tensor(all_input_ids)
    all_token_type_ids = torch.tensor(all_token_type_ids)
    all_attention_mask = torch.tensor(all_attention_mask)
    all_is_impossible = torch.tensor(all_is_impossible)
    all_ems = torch.tensor(all_ems)
    #all_best_passage = torch.tensor(all_best_passage)

    dataset = TensorDataset(all_input_ids, all_token_type_ids, all_attention_mask, all_is_impossible, all_ems)

    return dataset


