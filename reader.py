import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel, BertForMaskedLM, AutoTokenizer, AutoConfig, AutoModel
import json
from tqdm import tqdm
import os

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

class Reader(nn.Module):
    def __init__(self, config):
        super(Reader, self).__init__()
        logger.info('loading model')
        # load the model used for reading
        self.model = AutoModel.from_pretrained('outputs/spanbert-base-cased-sqdnq', config=config, )

        # linear layers that converts from the hidden states of the reader to starting and ending positions and rank the passages
        # can also import from spanbert, look at run_single.py
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.qa_classifier = nn.Linear(config.hidden_size, 1)

    # perform one forward step of the model that maps the input ids to a start and end position for each passage and also select one passage among all the passages in each example in the batch
    # size of the input should be N batch size x M candidates per batch x L length of each candidate (question + title + evidence passage)
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, start_positions=None, end_positions=None, is_impossible=None, rank=None):
        N, M, L = input_ids.size()

        # sequence_output = N*M x L x hidden_size
        # pooled_output = N*M x hidden_size
        # hidden_states = (N*M x L x hidden_size, N*M x L x hidden_size)
        sequence_output, pooled_output = self.model(input_ids=input_ids.view(N * M, L), attention_mask=attention_mask.view(N * M, L), token_type_ids=token_type_ids.view(N * M, L))

        # positions = N*M x L x 2
        positions = self.qa_outputs(sequence_output)
        start_pos, end_pos = positions.split(1, dim=-1)
        # N*M x L
        start_pos = start_pos.squeeze(-1)
        end_pos = end_pos.squeeze(-1)

        # N*M x 1, only consider the first token([CLS]) to determine the rank
        ranks = self.qa_classifier(sequence_output[:, 0, :])
        ranks = ranks.squeeze(-1)

        softmax = nn.LogSoftmax(dim=-1)
        start_logits = softmax(start_pos)
        end_logits = softmax(end_pos)
        rank_logits = softmax(ranks.view(N, M))

        if self.training:
            return self.compute_loss(start_logits, end_logits, rank_logits, start_positions, end_positions, is_impossible, rank, N, M, L)

        return start_logits.view(N, M, L), end_logits.view(N, M, L), rank_logits.view(N, M)

    def compute_loss(self, start_logits, end_logits, rank_logits, start_positions, end_positions, is_impossible, rank, N, M, L):
        ignore_index = -1
        nll = nn.NLLLoss(ignore_index=ignore_index)

        start_positions = start_positions.view(N * M)
        end_positions = end_positions.view(N * M)

        start_positions.clamp_(0, L-1)
        end_positions.clamp_(0, L-1)

        start_loss = nll(start_logits, start_positions)
        end_loss = nll(start_logits, start_positions)

        rank_logits = rank_logits.view(N, M)

        # questions with no good passage has -1, which is ignored
        rank_loss = nll(rank_logits, rank)

        return start_loss + end_loss + rank_loss

# for testing, delete later
def process_reader_input(tokenizer, train_file=None):
    logger.info(f'Loading reader input dataset from {train_file}')
    with open(train_file, encoding='utf-8') as f:
        data = json.load(f)['data'][0:100]

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
        if len(question_tokens) > 64:
            question_tokens = question_tokens[0:64]

        for pred_idx, pred in enumerate(predictions):
            title_tokens = tokenizer.tokenize(titles[pred_idx])
            front_tokens = question_tokens # + ['[SEP]'] + title_tokens
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

                # before passage + before answer + answer token length must be smaller than args.max_seq_length or the reader won't be able to even see it
                if before_passage_length + len(before_answer) + len(answer_tokens) <= 384:
                    starts.append(before_passage_length + len(before_answer) + 1)
                    ends.append(before_passage_length + len(before_answer) + len(answer_tokens))
                    if best == -1:
                        best = pred_idx
                else:
                    starts.append(0)
                    ends.append(0)
                    is_impossible[pred_idx] = 1

            back_tokens = title_tokens + ['[SEP]'] + back_tokens

            encoded = tokenizer.encode_plus(front_tokens, text_pair=back_tokens, max_length=384, pad_to_max_length=True, return_token_type_ids=True, return_attention_mask=True)

            input_ids.append(encoded['input_ids'])
            token_type_ids.append(encoded['token_type_ids'])
            attention_mask.append(encoded['attention_mask'])

        # need to pad so ensure the same size in every dimension
        # use negative samples from other passages
        for extra in range(len(predictions), 10):
            input_ids.append([0 for i in range(384)])
            token_type_ids.append([0 for i in range(384)])
            attention_mask.append([0 for i in range(384)])
            starts.append(-1)
            ends.append(-1)
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

if __name__ == '__main__':
    logger.info('testing reader')
    logger.info('loading config and tokenizer')
    config = AutoConfig.from_pretrained('outputs/spanbert-base-cased-sqdnq', output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained('outputs/spanbert-base-cased-sqdnq')

    device = 'cuda'
    model = Reader(config=config)
    model.to(device)
    args = {'train_file': 'tqa_ds_train.json'}

    logger.info('loading dataset')
    dataset = process_reader_input(tokenizer, train_file='tqa_ds_train.json')
    dataloader = DataLoader(dataset, batch_size=2)

    model.train()

    for batch in dataloader:
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
        #loss = model(**inputs)
        #print(loss)
        sequence_output, pooled_output, hidden_states = model.model(input_ids=inputs['input_ids'].view(-1, 384), attention_mask=inputs['attention_mask'].view(-1, 384), token_type_ids=inputs['token_type_ids'].view(-1, 384))
