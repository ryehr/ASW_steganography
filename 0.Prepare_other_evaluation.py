"""Build the two out-of-domain evaluation sets used in Section 7.5 and Appendix D.

    databricks-dolly.tsv   500 open_qa instructions from databricks-dolly-15k
    supernatural.tsv       500 instances from Super-NaturalInstructions

Both are prompt-only: the covertexts and stegotexts are generated from the
questions, so no reference answer is stored.

Example:
    python 0.Prepare_other_evaluation.py            # both
    python 0.Prepare_other_evaluation.py --which dolly
"""

import argparse
import csv

from datasets import load_dataset


def write_tsv(path, questions):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['part', 'new_id', 'question', 'answer'])
        for i, question in enumerate(questions):
            writer.writerow(['test', i, question, ''])
    print(f'{path}: {len(questions)} questions')


def collect(dataset, extract, num, keep=None):
    """Take the first ``num`` usable questions, skipping empty ones."""
    questions = []
    for record in dataset:
        if keep is not None and not keep(record):
            continue
        question = (extract(record) or '').strip()
        if len(question) < 2:
            continue
        questions.append(question)
        if len(questions) == num:
            break
    return questions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--which', default='both', choices=('both', 'dolly', 'supernatural'))
    parser.add_argument('--num', default=500, type=int)
    parser.add_argument('--dolly_dataset', default='databricks/databricks-dolly-15k', type=str)
    parser.add_argument('--supernatural_dataset',
                        default='andersonbcdefg/supernatural-instructions-2m', type=str)
    args = parser.parse_args()

    if args.which in ('both', 'dolly'):
        dataset = load_dataset(args.dolly_dataset, split='train')
        write_tsv('databricks-dolly.tsv', collect(
            dataset, lambda r: r['instruction'], args.num,
            keep=lambda r: r['category'] == 'open_qa'))

    if args.which in ('both', 'supernatural'):
        # Streamed: the full Super-NaturalInstructions dump is far larger than
        # the 500 instances needed here.
        dataset = load_dataset(args.supernatural_dataset, split='train', streaming=True)
        write_tsv('supernatural.tsv', collect(
            dataset, lambda r: r.get('prompt') or r.get('inputs') or r.get('input'),
            args.num))
