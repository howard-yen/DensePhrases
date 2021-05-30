import json
import pdb
import re
import random
from tqdm import tqdm
import string
import argparse

#try:
#    from eval_utils import (
#        drqa_exact_match_score,
#        drqa_regex_match_score,
#        drqa_metric_max_over_ground_truths
#    )
#except ModuleNotFoundError:
#    import sys
#    import os
#    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
#    from eval_utils import (
#        drqa_exact_match_score,
#        drqa_regex_match_score,
#        drqa_metric_max_over_ground_truths
#    )

from densephrases.utils.eval_utils import drqa_exact_match_score, drqa_regex_match_score, drqa_metric_max_over_ground_truths, f1_score, exact_match_score

# fix random seed
random.seed(0)

def find_substring_and_return_first(substring, string):
    substring_idxs = [m.start() for m in re.finditer(re.escape(substring), string)]
    # substring_idx = random.choice(substring_idxs)
    substring_idx = substring_idxs[0]
    return substring_idx

def main(args):
    with open(args.input_path, encoding='utf-8') as f:
        data = json.load(f)

    output_data = []

    counter = 0
    print('processing input')
    for sample_id in tqdm(data):
        if counter > 10:
            #return
            pass
        counter += 1

        sample = data[sample_id]

        question = sample['question']
        answers = sample['answer']
        predictions = sample['prediction'][0:args.top_k]
        titles = sample['title'][0:args.top_k]
        evidences = sample['evidence'][0:args.top_k]
        scores = sample['score'][0:args.top_k]

        f1s = [max([f1_score(prediction, gt)[0] for gt in answers]) for prediction in predictions]
        ems = [max([exact_match_score(prediction, gt) for gt in answers]) for prediction in predictions]

        is_impossible = []
        start_pos = []
        end_pos = []

        match_fn = drqa_regex_match_score if args.regex else drqa_exact_match_score

        # check if prediction is matched in a golden answer in the answer list
        for pred_idx, pred in enumerate(predictions):
            if pred != "" and drqa_metric_max_over_ground_truths(match_fn, pred, answers):
                is_impossible.append(False)
                answer_start = find_substring_and_return_first(pred, evidences[pred_idx])
                start_pos.append(answer_start)
                end_pos.append(answer_start + len(pred)-1)
            else:
                is_impossible.append(True)
                start_pos.append(-1)
                end_pos.append(-1)

        # could add score in here if we somehow make the reader also take the score into account
        output_data.append({
                'question': question,
                'answer': answers,
                'prediction': predictions,
                'title': [title[0] for title in titles],
                'evidence': evidences,
                'is_impossible': is_impossible,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'score': scores,
                'f1s': f1s,
                'exact_matches': ems,
                })

        #print(output_data[-1])
        #print('-------')

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'data': output_data
        },f)

# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/n/fs/nlp-hyen/DensePhrases/outputs/dph-nqsqd-pb2_pq96-nq-10/pred/train_preprocessed_79168.pred')
    parser.add_argument('--output_path', type=str, default='reranker_inputs.json')
    parser.add_argument('--top_k', default=10, type=int)
    parser.add_argument('--regex', action='store_true')
    args = parser.parse_args()

    main(args)
