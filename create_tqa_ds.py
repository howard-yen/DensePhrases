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

from densephrases.utils.eval_utils import drqa_exact_match_score, drqa_regex_match_score, drqa_metric_max_over_ground_truths

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

    print('processing input')
    for sample_id in tqdm(data):
        sample = data[sample_id]

        question = sample['question']
        answers = sample['answer']
        predictions = sample['prediction'][0:args.top_k]
        titles = sample['title'][0:args.top_k]
        evidences = sample['evidence'][0:args.top_k]
        scores = sample['score'][0:args.top_k]

        # each question has at least one prediction
        if len(predictions) < args.top_k:
            count += 1
            if len(predictions) < min_pred:
                min_pred = len(predictions)
            if len(predictions) == 0:
                zeros += 1

        is_impossible = []
        start_pos = []
        end_pos = []

        match_fn = drqa_regex_match_score if args.regex else drqa_exact_match_score

        #answer_text = ""
        #ds_context = ""
        #ds_title = ""
        # is_from_context = False

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
                })

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'data': output_data
        },f)

# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, default='/home/pred/sbcd_sqdqgnqqg_inb64_s384_sqdnq_pinb2_0_20181220_concat_train_preprocessed_78785.pred')
    parser.add_argument('output_path', type=str, default='tqa_ds_train.json')
    parser.add_argument('--top_k', default=10, type=int)
    parser.add_argument('--regex', action='store_true')
    args = parser.parse_args()

    main(args)
