import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import BertTokenizer, BertModel, BertForMaskedLM, AutoTokenizer, AutoConfig, AutoModel
import json
import densephrases.utils.squad_utils as squad_utils

import logging
logging.basicConfig(level=logging.INFO)

class Reader(nn.Module):
    def __init__(self, config=None, tokenizer=None):
        super(Reader, self).__init__()

        if config is None:
            self.config = AutoConfig.from_pretrained('outputs/spanbert-base-cased-sqdnq', output_hidden_states=True)
        else:
            self.config = config

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained('outputs/spanbert-base-cased-sqdnq')
        else:
            self.tokenizer = tokenizer

        self.model = AutoModel.from_pretrained('outputs/spanbert-base-cased-sqdnq', config=self.config)

        # can also import from spanbert, look at run_single.py
        self.qa_outputs = nn.Linear(self.config.hidden_size, 2)
        self.qa_classifier = nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, start_truth=None, end_truth=None, passage_truth=None):
        sequence_output, pooled_output, hidden_states = self.model(input_ids=input_ids.view(-1, len(input_ids)), attention_mask=None)

        positions = self.qa_outputs(sequence_output)
        start_positions, end_positions = positions.split(1, dim=-1)
        start_positions = start_positions.squeeze(-1)
        end_positions = end_positions.squeeze(-1)

        softmax = nn.Softmax(dim=-1)
        start_positions = softmax(start_positions)
        end_positions = softmax(end_positions)

        ranks = self.qa_classifier(sequence_output[:, 0, :])
        ranks = softmax(ranks)

        if self.training:
            #self.compute_loss()
            return compute_loss(start_positions, end_positions, ranks, start_truth, end_truth, passage_truth)
            pass

        return start_positions, end_positions, ranks

    def compute_loss(self, start_positions, end_positions, ranks, start_truth, end_truth, passage_truth):
        # also refer to run_single.py
        # find the answer string in that passage, if contain then can annotate (weak/distant supervision)
        # if not then put special token that is "no answer", can use [CLS]
        cel = nn.CrossEntropyLoss()

        return

    def create_input(self, passage, title=None):
        # should get N questions x M passages per question
        if title:
            input_ids = torch.tensor(self.tokenizer.encode(title, text_pair=passage))
        else:
            input_ids = torch.tensor(self.tokenizer.encode(passage))

        return input_ids



if __name__ == '__main__':
    #reader = Reader()
    #reader.eval()

    passage = 'Obama was born in Honolulu, Hawaii. After graduating from Columbia University in 1983, he worked as a community organizer in Chicago.'

    #s_output, p_output, h_states = reader(passage)

    #train_sampler = RandomSampler(input_data)
    #train_dataloader = DataLoader(input_data, sampler=train_sampler, batch_size=25)
