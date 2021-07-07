import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import BertTokenizer, BertModel, BertForMaskedLM, AutoTokenizer, AutoConfig, AutoModel
import json
from tqdm import tqdm
import os
from transformers import PreTrainedModel, AdamW

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

class Reader(PreTrainedModel):
    def __init__(self, config):
        super(Reader, self).__init__(config)
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
        output = self.model(input_ids=input_ids.view(N * M, L), attention_mask=attention_mask.view(N * M, L), token_type_ids=token_type_ids.view(N * M, L))

        # positions = N*M x L x 2
        positions = self.qa_outputs(output.last_hidden_state)
        start_pos, end_pos = positions.split(1, dim=-1)
        # N*M x L
        start_pos = start_pos.squeeze(-1)
        end_pos = end_pos.squeeze(-1)

        # N*M x 1, only consider the first token([CLS]) to determine the rank
        ranks = self.qa_classifier(output.last_hidden_state[:, 0, :])
        ranks = ranks.squeeze(-1)

        softmax = nn.LogSoftmax(dim=-1)
        start_logits = softmax(start_pos)
        end_logits = softmax(end_pos)
        rank_logits = softmax(ranks.view(N, M))

        #start_loss, end_loss, rank_loss = self.compute_loss(start_logits, end_logits, rank_logits, start_positions, end_positions, is_impossible, rank, N, M, L)
        if self.training:
            return self.compute_loss(start_logits, end_logits, rank_logits, start_positions, end_positions, is_impossible, rank, N, M, L)
            #return start_loss + end_loss + rank_loss

        return start_logits.view(N, M, L), end_logits.view(N, M, L), rank_logits.view(N, M)
        #return start_loss.view(N, M, L), end_loss.view(N, M, L), rank_loss.view(N, M)

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
        #return start_loss, end_loss, rank_loss
