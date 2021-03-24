import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel, BertForMaskedLM, AutoTokenizer, AutoConfig, AutoModel
import json
from tqdm import tqdm

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

class Reader(nn.Module):
    def __init__(self, config):
        super(Reader, self).__init__()

        logger.info('loading model')
        # load the model used for reading
        self.model = AutoModel.from_pretrained('outputs/spanbert-base-cased-sqdnq', config=config)

        # linear layers that converts from the hidden states of the reader to starting and ending positions and rank the passages
        # can also import from spanbert, look at run_single.py
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.qa_classifier = nn.Linear(config.hidden_size, 1)

    # perform one forward step of the model that maps the input ids to a start and end position for each passage and also select one passage among all the passages in each example in the batch
    # size of the input should be N batch size x M candidates per batch x L length of each candidate (question + title + evidence passage)
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, start_positions=None, end_positions=None, is_impossible=None):
        N, M, L = input_ids.size()
        # sequence_output = N*M x L x hidden_size
        # pooled_output = N*M x hidden_size
        # hidden_states = (N*M x L x hidden_size, N*M x L x hidden_size)
        sequence_output, pooled_output, hidden_states = self.model(input_ids=input_ids.view(N * M, L), attention_mask=attention_mask.view(N * M, L), token_type_ids=token_type_ids.view(N * M, L))

        # positions = N*M x L x 2
        positions = self.qa_outputs(sequence_output)
        start_pos, end_pos = positions.split(1, dim=-1)
        # N*M x L
        start_pos = start_pos.squeeze(-1)
        end_pos = end_pos.squeeze(-1)

        # clamp is just relu but set max to ignored_index
        ignored_index = start_pos.size(1)
        #start_pos.clamp_(0,ignored_index)
        #end_pos.clamp_(0, ignored_index)
        #relu = nn.ReLU()
        #start_pos = relu(start_pos)
        #end_pos = relu(end_pos)

        # N*M x 1, only consider the first token([CLS]) to determine the rank
        ranks = self.qa_classifier(sequence_output[:, 0, :])
        ranks = ranks.squeeze(-1)
        #ranks.clamp_(0, ignored_index)
        #ranks = relu(ranks)

        softmax = nn.Softmax(dim=-1)
        start_pos = softmax(start_pos)
        end_pos = softmax(end_pos)
        ranks = softmax(ranks.view(N, M))

        if self.training:
            return self.compute_loss(start_pos, end_pos, ranks, start_positions, end_positions, is_impossible, N, M, L)

        return start_pos, end_pos, ranks

    def compute_loss(self, start_pos, end_pos, ranks, start_positions, end_positions, is_impossible, N, M, L):
        logger.info('computing loss')
        #still not sure why dpr used ignored_index
        ignored_index = start_pos.size(1)
        #dpr uses crossentropy with L number of classes, but for now we use 1 for should select and 0 for shouldn't select
        cel = nn.BCELoss()

        #need to do to device
        start_zeroes = torch.zeros(start_pos.size())
        end_zeroes = torch.zeros(end_pos.size())
        for i in range(N):
            for j in range(M):
                start_zeroes[i*N + j][start_positions[i][j]] = 1
                end_zeroes[i*N + j][end_positions[i][j]] = 1

        start_loss = cel(start_pos, start_zeroes)
        end_loss = cel(end_pos, end_zeroes)

        # need to change because -1 means there wasn't a prediction
        is_impossible[is_impossible==-1] = 0
        is_impossible = is_impossible.to(dtype=torch.float)
        passage_loss = cel(ranks, is_impossible.view(ranks.size()))

        return start_loss + end_loss + passage_loss

# for testing, delete later
def process_reader_input(tokenizer, train_file=None):
    logger.info(f'Loading reader input dataset from {train_file}')
    with open(train_file, encoding='utf-8') as f:
        data = json.load(f)['data'][0:10]

    all_input_ids = []
    all_token_type_ids = []
    all_attention_mask = []
    all_start_pos = []
    all_end_pos = []
    all_is_impossible = []

    for sample in tqdm(data):
        question = sample['question']
        answer = sample['answer']
        predictions = sample['prediction']
        titles = sample['title']
        evidences = sample['evidence']
        start_pos = sample['start_pos']
        end_pos = sample['end_pos']
        is_impossible = sample['is_impossible']
        is_impossible = [1 if imp else 0 for imp in is_impossible]

        input_ids = []
        token_type_ids = []
        attention_mask = []
        starts = []
        ends = []

        question_tokens = tokenizer.tokenize(question)
        for pred_idx, pred in enumerate(predictions):
            title_tokens = tokenizer.tokenize(titles[pred_idx])
            front_tokens = question_tokens + ['[SEP]'] + title_tokens #should i be adding SEP between questiona and title?

            # get the start and ending token index for the answer
            if is_impossible[pred_idx]:
                back_tokens = tokenizer.tokenize(evidences[pred_idx])
                starts.append(-1)
                ends.append(-1)
            else:
                before_answer = tokenizer.tokenize(evidences[pred_idx][0:start_pos[pred_idx]-1])
                answer_tokens = tokenizer.tokenize(evidences[pred_idx][start_pos[pred_idx]:end_pos[pred_idx]+1])
                after_answer = tokenizer.tokenize(evidences[pred_idx][end_pos[pred_idx]+1:])
                back_tokens = before_answer + answer_tokens + after_answer
                starts.append(len(front_tokens) + len(before_answer) + 1)
                ends.append(len(front_tokens) + len(before_answer) + len(answer_tokens))

            encoded = tokenizer.encode_plus(front_tokens, text_pair=back_tokens, max_length=384, pad_to_max_length=True, return_token_type_ids=True, return_attention_mask=True)

            input_ids.append(encoded['input_ids'])
            token_type_ids.append(encoded['token_type_ids'])
            attention_mask.append(encoded['attention_mask'])

        # need to pad so ensure the same size in every dimension
        for extra in range(len(predictions), 10):
            # maybe can use the no answer token after implementing that
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

    device = 'cuda'
    all_input_ids = torch.tensor(all_input_ids)
    all_token_type_ids = torch.tensor(all_token_type_ids)
    all_attention_mask = torch.tensor(all_attention_mask)
    all_start_pos = torch.tensor(all_start_pos)
    all_end_pos = torch.tensor(all_end_pos)
    all_is_impossible = torch.tensor(all_is_impossible)

    dataset = TensorDataset(all_input_ids, all_token_type_ids, all_attention_mask, all_start_pos, all_end_pos, all_is_impossible)

    return dataset

if __name__ == '__main__':
    logger.info('testing reader')
    logger.info('loading config and tokenizer')
    config = AutoConfig.from_pretrained('outputs/spanbert-base-cased-sqdnq', output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained('outputs/spanbert-base-cased-sqdnq')

    model = Reader(config=config)
    args = {'train_file': 'tqa_ds_train.json'}

    logger.info('loading dataset')
    dataset = process_reader_input(tokenizer, train_file='tqa_ds_train.json')
    dataloader = DataLoader(dataset, batch_size=2)

    model.train()

    for batch in dataloader:
        b = batch

    logger.info('running on one batch')
    start_loss, end_loss, passage_loss = model(b[0], b[2], b[1], b[3], b[4], b[5])
