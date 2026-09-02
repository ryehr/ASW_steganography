"""Build instinwild_en.tsv, the training/validation/test split of InstructionWild.

Produces 12,500 training, 1,000 validation and 500 test question-answer pairs
(Section 7.1), keeping only answers between 20 and 512 tokens so that every
sample has enough length for the sliding window to move over.

instinwild_en.tsv is checked in; rerun this only to change the split.  Answers
come from the Hugging Face copy of the dataset by default, or from a local
instinwild_en.json with --source_json.

Example:
    python 0.Prepare_dataset.py
    python 0.Prepare_dataset.py --source_json instinwild_en.json --seed 0
"""

import argparse
import csv
import json
import random

from transformers import AutoTokenizer


def load_records(args):
    if args.source_json:
        with open(args.source_json, encoding='utf-8') as f:
            return json.load(f)
    from datasets import load_dataset
    dataset = load_dataset(args.source_dataset, split='train')
    return [{'instruction': r['instruction'], 'output': r['output']} for r in dataset]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_json', default=None, type=str,
                        help='local InstructionWild json; omit to pull from the Hub')
    parser.add_argument('--source_dataset', default='fuliucansheng/InstructionWild', type=str)
    parser.add_argument('--tokenizer', default='Qwen/Qwen2.5-7B-Instruct', type=str)
    parser.add_argument('--output', default='instinwild_en.tsv', type=str)
    parser.add_argument('--train_num', default=12500, type=int)
    parser.add_argument('--validation_num', default=1000, type=int)
    parser.add_argument('--test_num', default=500, type=int)
    parser.add_argument('--min_answer_tokens', default=20, type=int)
    parser.add_argument('--max_answer_tokens', default=512, type=int)
    parser.add_argument('--seed', default=0, type=int,
                        help='seeds the shuffle within each split')
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    data = load_records(args)
    print(f'{len(data)} source records')

    # Disjoint 60/20/20 pools, shuffled inside each pool, so a question can never
    # appear in two splits.
    n = len(data)
    pools = {
        'train': list(range(0, int(n * 0.6))),
        'validation': list(range(int(n * 0.6), int(n * 0.8))),
        'test': list(range(int(n * 0.8), n)),
    }
    targets = {'train': args.train_num, 'validation': args.validation_num,
               'test': args.test_num}
    for pool in pools.values():
        rng.shuffle(pool)

    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['part', 'new_id', 'question', 'answer'])
        for part, pool in pools.items():
            kept = 0
            for i in pool:
                if kept >= targets[part]:
                    break
                question = data[i]['instruction']
                answer = data[i]['output']
                if len(question) < 2:
                    continue
                length = tokenizer(answer, return_tensors='pt')['input_ids'].shape[1]
                if not args.min_answer_tokens <= length <= args.max_answer_tokens:
                    continue
                writer.writerow([part, kept, question, answer])
                kept += 1
            print(f'{part}: {kept}')
    print(f'written to {args.output}')
