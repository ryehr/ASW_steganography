"""Text-quality and efficiency metrics for stegotexts (Table 2).

Compares each stegotext file against the covertexts from 3.Generation_normal.py
and reports the columns of Table 2:

    dPPL        |PPL(stegotext) - PPL(covertext)|, the paper's absolute
                difference rather than raw perplexity
    BLEU        2-gram, per Section 7.2.1
    ROUGE-L
    BERTScore   F1
    Capacity    embedded bits per token
    Time        seconds to embed one stegotext

Steganalysis accuracy, the fifth Table 2 column, comes from 6.Steganalysis.py.

Example:
    python 3.Stega_evaluation.py --model Qwen2.5-7B-Instruct --dataset instinwild_en
"""

import argparse
import functools
import glob
import json
import logging
import os

import evaluate
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import asw


@functools.lru_cache(maxsize=None)
def metric(name):
    """Load an evaluate metric once, then reuse it across candidate files."""
    # module_type disambiguates 'perplexity', which also exists as a measurement.
    return evaluate.load(name, module_type='metric')


@functools.lru_cache(maxsize=None)
def ppl_model(model_id, device):
    """The scorer for perplexity, loaded once."""
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


@torch.no_grad()
def perplexity(texts, args):
    """Mean per-text perplexity under ``--ppl_model``.

    Computed directly rather than through `evaluate`'s perplexity metric, which
    requires a transformers version below v5.  The arithmetic is the same:
    exponentiate each text's mean token NLL, then average over texts.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = ppl_model(args.ppl_model, device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    scores = []
    for start in range(0, len(texts), args.ppl_batch_size):
        batch = texts[start:start + args.ppl_batch_size]
        encoded = tokenizer(batch, return_tensors='pt', padding=True, truncation=True,
                            max_length=args.ppl_max_length - 1)
        input_ids = encoded['input_ids'].to(device)
        mask = encoded['attention_mask'].to(device)

        # A leading BOS gives the first real token something to condition on,
        # so it contributes to the score like every other token.
        bos = torch.full((input_ids.shape[0], 1), tokenizer.bos_token_id, device=device)
        input_ids = torch.cat([bos, input_ids], dim=1)
        mask = torch.cat([torch.ones_like(bos), mask], dim=1)

        logits = model(input_ids=input_ids, attention_mask=mask).logits
        nll = F.cross_entropy(
            logits[:, :-1].transpose(1, 2), input_ids[:, 1:], reduction='none')
        target_mask = mask[:, 1:].float()
        per_text = (nll * target_mask).sum(1) / target_mask.sum(1).clamp(min=1)
        scores.extend(per_text.exp().tolist())

    return float(np.mean(scores))


def evaluate_group(candidates, references, reference_ppl, args):
    scores = {}
    scores['BLEU'] = metric('bleu').compute(
        predictions=candidates, references=references, max_order=2)['bleu']
    scores['ROUGE-L'] = metric('rouge').compute(
        predictions=candidates, references=references)['rougeL']
    bert = metric('bertscore').compute(
        predictions=candidates, references=references,
        model_type=args.bertscore_model)
    scores['BERTScore'] = float(np.mean(bert['f1']))

    candidate_ppl = perplexity(candidates, args)
    scores['PPL'] = candidate_ppl
    # Table 2 reports the absolute gap to the covertexts rather than raw
    # perplexity, in either direction.
    scores['dPPL'] = abs(candidate_ppl - reference_ppl)
    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen2.5-7B-Instruct', type=str)
    parser.add_argument('--dataset', default='instinwild_en', type=str)
    parser.add_argument('--folder', default='3.Stega_data', type=str)
    parser.add_argument('--reference_file', default=None, type=str,
                        help='defaults to Normal_<model>_<dataset>.tsv in --folder')
    parser.add_argument('--ppl_model', default='gpt2', type=str)
    parser.add_argument('--ppl_batch_size', default=8, type=int)
    parser.add_argument('--ppl_max_length', default=512, type=int)
    parser.add_argument('--bertscore_model', default='distilbert-base-uncased', type=str)
    parser.add_argument('--output', default='3.Stega_data/evaluation.json', type=str)
    parser.add_argument('--log', default='experiment.log', type=str)
    args = parser.parse_args()

    logging.basicConfig(filename=args.log, level=logging.WARNING,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    reference_file = args.reference_file or os.path.join(
        args.folder, f'Normal_{args.model}_{args.dataset}.tsv')
    if not os.path.exists(reference_file):
        raise SystemExit(f'{reference_file} not found; run 3.Generation_normal.py first')

    reference_all = list(asw.read_stega_tsv(reference_file)['Text'].astype(str))

    # The covertexts are the same for every candidate file, so their perplexity
    # is computed once.
    reference_ppl = perplexity(reference_all, args)
    print(f'reference PPL ({os.path.basename(reference_file)}): {reference_ppl:.3f}')
    logging.warning(f'Reference PPL: {reference_ppl}')

    candidates = sorted(
        f for f in glob.glob(os.path.join(args.folder, '*.tsv'))
        if args.model in os.path.basename(f)
        and args.dataset in os.path.basename(f)
        and 'Normal' not in os.path.basename(f))
    if not candidates:
        raise SystemExit(f'no stegotext files for {args.model} / {args.dataset} in {args.folder}')

    results = {}
    for candidate_file in candidates:
        name = os.path.basename(candidate_file)
        print(f'\n=== {name} ===')
        logging.warning(f'Evaluating the file: {name}')

        df = asw.read_stega_tsv(candidate_file)
        candidate_data = list(df['Text'].astype(str))
        references = reference_all[:len(candidate_data)]
        if len(references) != len(candidate_data):
            print(f'  ! {len(candidate_data)} stegotexts but only {len(references)} '
                  f'covertexts; comparing the first {len(references)}')
            candidate_data = candidate_data[:len(references)]

        scores = evaluate_group(candidate_data, references, reference_ppl, args)
        # Efficiency columns come straight out of the embedding run.
        scores['Capacity'] = float(df['BPT'].mean()) if 'BPT' in df else float('nan')
        scores['Entropy'] = float(df['Entropy'].mean())
        scores['Time'] = float(df['Time'].mean())
        scores['Samples'] = len(candidate_data)

        results[name] = scores
        for key, value in scores.items():
            print(f'  {key:11s} {value:.4f}' if isinstance(value, float) else
                  f'  {key:11s} {value}')
            logging.warning(f'{key}: {value}')

    asw.ensure_dir(args.output)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({'reference_ppl': reference_ppl, 'results': results}, f, indent=2)

    print(f'\n{"file":60s}{"dPPL":>9}{"BLEU":>8}{"ROUGE-L":>9}{"BERTScore":>11}'
          f'{"Capacity":>10}{"Time":>8}')
    for name, s in results.items():
        print(f'{name[:59]:60s}{s["dPPL"]:9.3f}{s["BLEU"]:8.3f}{s["ROUGE-L"]:9.3f}'
              f'{s["BERTScore"]:11.3f}{s["Capacity"]:10.3f}{s["Time"]:8.2f}')
    print(f'\nwritten to {args.output}')
