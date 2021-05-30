import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import os
import json

from tqdm import tqdm, trange
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForQuestionAnswering
from utils import process_sample

from densephrases.experiments.run_open import load_and_cache_examples


if __name__ == "__main__":
    test_path = 'outputs/dph-nqsqd-pb2_pq96-nq-10/pred/test_preprocessed_3610.pred'
    max_query_length = 50
    max_seq_length = 384
    max_answer_length = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad", cache_dir="./cache")
    model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad", cache_dir="./cache")
    model.to(device)
    model.eval()

    with open(test_path, encoding='utf-8') as f:
        data = json.load(f)

    qids = []
    predictions = []
    evidences = []
    answers = []
    questions = []
    titles = []
    new_predictions = []

    softmax = nn.Softmax(dim=-1)

    with torch.no_grad():
        for qid in tqdm(data):
            model.eval()
            torch.cuda.empty_cache()
            qids.append(qid)
            sample = data[qid]

            question = sample['question']
            answer = sample['answer']
            prediction = sample['prediction']
            title = sample['title']
            evidence = sample['evidence']

            questions.append(question)

            processed = process_sample(tokenizer, question, answer, prediction, title, evidence, max_seq_length=max_seq_length)
            M = len(prediction)

            input_ids = torch.tensor(processed['input_ids']).to(device)
            attention_mask = torch.tensor(processed['attention_mask']).to(device)
            token_type_ids = torch.tensor(processed['token_type_ids']).to(device)

            output = model(input_ids = input_ids.view(M, max_seq_length), attention_mask = attention_mask.view(M, max_seq_length), token_type_ids=token_type_ids.view(M, max_seq_length))

            start_logits = output.start_logits
            end_logits = output.end_logits

            start_prob = softmax(start_logits)
            end_prob = softmax(end_logits)

            span_prob = torch.bmm(start_prob.view(M, -1, 1), end_prob.view(M, 1, -1))
            span_prob = torch.triu(span_prob)

            # mask to limit the length of the span
            mask = torch.ones_like(span_prob)
            mask = torch.triu(mask, diagonal=max_answer_length)
            span_prob[mask==1] = 0

            # mask out the question
            span_prob[token_type_ids==0] = 0

            best_span = torch.argmax(span_prob.view(M, -1), dim=1)

            offset = processed['offset_mapping']
            new_prediction = [processed['backs'][idx][offset[idx][pos//max_seq_length][0]:offset[idx][pos%max_seq_length][1]] for idx, pos in enumerate(best_span)]
            new_predictions.append(new_prediction)
            #print('answer:',answer)
            #print('prediction:',prediction)
            #print('new prediction:',new_prediction)

            input_ids = input_ids.to('cpu')
            attention_mask = attention_mask.to('cpu')
            token_type_ids = token_type_ids.to('cpu')
            torch.cuda.empty_cache()

