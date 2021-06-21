import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoConfig, AutoModel
import json
from tqdm import tqdm
import os
from apex import amp
from transformers import PreTrainedModel, AdamW

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

class Reranker(PreTrainedModel):
    def __init__(self, config):
        super(Reranker, self).__init__(config)
        logger.info('loading model')
        # load the model used for reading
        self.model = AutoModel.from_pretrained('outputs/spanbert-base-cased-sqdnq', config=config, )

        #linear layer that rerankk the passages
        self.qa_classifier = nn.Linear(config.hidden_size, 1)

    # perform one forward step of the model that maps the input ids to a start and end position for each passage and also select one passage among all the passages in each example in the batch
    # size of the input should be N batch size x M candidates per batch x L length of each candidate (question + title + evidence passage)
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, is_impossible=None, ems=None):
        N, M, L = input_ids.size()

        # sequence_output = N*M x L x hidden_size
        # pooled_output = N*M x hidden_size
        # hidden_states = (N*M x L x hidden_size, N*M x L x hidden_size)
        #sequence_output, pooled_output = self.model(input_ids=input_ids.view(N * M, L), attention_mask=attention_mask.view(N * M, L), token_type_ids=token_type_ids.view(N * M, L))
        output = self.model(input_ids=input_ids.view(N * M, L), attention_mask=attention_mask.view(N * M, L), token_type_ids=token_type_ids.view(N * M, L))
        sequence_output = output.last_hidden_state

        # N*M x 1, only consider the first token([CLS]) to determine the rank
        ranks = self.qa_classifier(sequence_output[:, 0, :])
        ranks = ranks.squeeze(-1)

        softmax = nn.Softmax(dim=-1)
        rank_logits = softmax(ranks.view(N, M))
        print('logits:', rank_logits)

        #start_loss, end_loss, rank_loss = self.compute_loss(start_logits, end_logits, rank_logits, start_positions, end_positions, is_impossible, rank, N, M, L)
        if self.training:
            return self.compute_loss(rank_logits, is_impossible, ems, N, M, L)
            #return start_loss + end_loss + rank_loss

        return rank_logits.view(N, M)
        #return start_loss.view(N, M, L), end_loss.view(N, M, L), rank_loss.view(N, M)

    def compute_loss(self, rank_prob, is_impossible, ems, N, M, L):
        #print('em logits:', rank_logits[ems==1])
        loss = -torch.log(torch.sum(rank_prob[ems==1], dim=-1))
        #print('loss:', loss)
        loss = loss.mean()

        return loss

def process_reranker_input(tokenizer, train_file=None):
    logger.info(f'Loading reader input dataset from {train_file}')
    with open(train_file, encoding='utf-8') as f:
        data = json.load(f)['data']

    max_seq_length = 384
    all_input_ids = []
    all_token_type_ids = []
    all_attention_mask = []
    all_is_impossible = []
    #all_best_passage = []
    all_ems = []

    for sample_idx, sample in enumerate(tqdm(data)):
        if sample_idx > 1000:
            break

        question = sample['question']
        answer = sample['answer']
        predictions = sample['prediction']
        titles = sample['title']
        evidences = sample['evidence']
        scores = sample['score']
        f1s = sample['f1s']
        ems = sample['exact_matches']
        ems = [1 if em else 0 for em in ems]

        #print(answer)
        #print(predictions)
        #print(ems)
        #print('---------------')

        if max(ems) < 1:
            continue

        is_impossible = sample['is_impossible']
        is_impossible = [1 if imp else 0 for imp in is_impossible]


        input_ids = []
        token_type_ids = []
        attention_mask = []
        best = 0

        question_tokens = tokenizer.tokenize(question)
        if len(question_tokens) > 50:
            question_tokens = question_tokens[0:50]

        for pred_idx, pred in enumerate(predictions):
            title_tokens = tokenizer.tokenize(titles[pred_idx])
            pred_tokens = tokenizer.tokenize(pred)
            passage_tokens = tokenizer.tokenize(evidences[pred_idx])
            back_tokens = pred_tokens + ['[SEP]'] + title_tokens + ['[SEP]'] + passage_tokens

            #encoded = tokenizer.encode_plus(question, text_pair=back_tokens, max_length=args.max_seq_length, pad_to_max_length=True, return_token_type_ids=True, return_attention_mask=True)
            encoded = tokenizer.encode_plus(question_tokens, text_pair=back_tokens, max_length=max_seq_length, padding='max_length', truncation=True, return_token_type_ids=True, return_attention_mask=True, is_split_into_words=True)

            input_ids.append(encoded['input_ids'])
            token_type_ids.append(encoded['token_type_ids'])
            attention_mask.append(encoded['attention_mask'])

        # need to pad so ensure the same size in every dimension
        # replaced by negative sample in batch while training
        for extra in range(len(predictions), 10):
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


if __name__ == '__main__':
    logger.info('testing reranker')
    logger.info('loading config and tokenizer')
    config = AutoConfig.from_pretrained('outputs/spanbert-base-cased-sqdnq', output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained('outputs/spanbert-base-cased-sqdnq')

    device = 'cuda'
    model = Reranker(config=config)
    model.load_state_dict(torch.load('outputs/reranker/checkpoint-75000/pytorch_model.bin'))
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

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [{
            "params": [
                p for n, p in model.named_parameters() \
                    if not any(nd in n for nd in no_decay) and is_train_param(n)
            ],
            "weight_decay": 0.01,
        }, {
            "params": [
                p for n, p in model.named_parameters() \
                    if any(nd in n for nd in no_decay) and is_train_param(n)
            ],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-3, eps=1e-8)

    #model = amp.initialize(model)
    model, optimizer = amp.initialize(model, optimizer)

    logger.info('loading dataset')
    dataset = process_reranker_input(tokenizer, train_file='reranker_inputs.json')
    dataloader = DataLoader(dataset, batch_size=1)

    model.train()

    epoch = 5

    for ep in range(epoch):
        print(f'\n-----------epoch {ep}-------------')
        ep_loss = torch.tensor([0.0]).to(device)
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "token_type_ids": batch[1],
                "attention_mask": batch[2],
                "is_impossible": batch[3],
                "ems": batch[4],
                #"input_ids_": batch[8],
                #"attention_mask_": batch[9],
                #"token_type_ids_": batch[10],
            }
            loss = model(**inputs)
            #print('loss:',loss)
            ep_loss += loss
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            optimizer.step()

            batch = tuple(t.to('cpu') for t in batch)
            torch.cuda.empty_cache()
            #sequence_output, pooled_output, hidden_states = model.model(input_ids=inputs['input_ids'].view(-1, 384), attention_mask=inputs['attention_mask'].view(-1, 384), token_type_ids=inputs['token_type_ids'].view(-1, 384))

        print('ep loss', ep_loss/len(dataloader))

    #model.save_pretrained('temp_reranker_model')
