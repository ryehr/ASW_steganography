"""Average single-step KL divergence between full-context and windowed inference.

Produces Table 1 (and Table 6 for Llama): for each candidate context window, how
far its next-token distribution drifts from what the model would have said with
the whole history.  Appendix A ties this quantity to imperceptibility, so it is
the cheapest way to compare bridge contexts before running any steganography.

The measured quantity is

    D_KL(p_full || p_window)      (Equation 5)

which is the forward direction: it penalises a window that starves a token the
full context considered plausible.

One reference text is sampled per prompt from the full-context distribution, and
every strategy is scored against that same text, so the comparison is paired.

Example:
    python 2.Generation_window.py --context_window 10 --index_end 500
    python 2.Generation_window.py --context_window 10 \
        --strategies Baseline,Hard_0,Rand_5,Rand_10,Hard_missing,Hard_removed,Hard_1,Hard_2
"""

import argparse
import csv
import gc
import json
import os
import time

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import DynamicCache

import asw


def build_windows(model, tokenizer, args):
    """One ASWWindow per strategy, with every bridge materialised up front."""
    extra = json.loads(args.bridge_texts) if args.bridge_texts else {}
    windows = {}
    for name in args.strategies.split(','):
        name = name.strip()
        soft_embedding = None
        if name == 'Soft_0':
            generator = torch.Generator(device='cpu').manual_seed(args.seed)
            soft_embedding = torch.randn(
                args.default_soft_length, model.config.hidden_size,
                generator=generator).to(model.device, model.dtype)
        elif name.startswith('Soft'):
            path = args.soft_forward_path if name == 'Soft_forward' else args.soft_reverse_path
            if not path:
                raise SystemExit(f'--soft_{name.split("_")[1]}_path is required for {name}')
            soft_embedding = asw.load_soft_bridge(path, model)
        windows[name] = asw.ASWWindow(
            model, tokenizer, name, args.context_window,
            soft_embedding=soft_embedding, use_cache=not args.no_cache,
            rand_seed=args.seed, bridge_text=extra.get(name))
    return windows


def measure(model, tokenizer, windows, prompt_ids, args):
    """Sample a text under full context and score every window against it."""
    start_time = time.time()
    for window in windows.values():
        window.reset(prompt_ids)

    divergences = {name: 0.0 for name in windows}
    context = prompt_ids

    # The reference pass is plain autoregressive sampling, so it reuses its own
    # KV cache and stays linear in the sequence length.
    cache = DynamicCache()
    with torch.no_grad():
        model(context[:, :-1], past_key_values=cache, use_cache=True)

    steps = 0
    for step in range(args.token_max):
        with torch.no_grad():
            logits_full = model(context[:, -1:], past_key_values=cache,
                                use_cache=True).logits[0, -1, :]
        probs_full = F.softmax(logits_full.double(), dim=0)

        for name, window in windows.items():
            probs_window = F.softmax(window.logits(context).double(), dim=0)
            # torch's kl_div(input, target) is D_KL(target || exp(input)), so
            # the full-context distribution is the target and the window
            # supplies the log-probabilities.
            divergences[name] += F.kl_div(
                probs_window.log(), probs_full, reduction='sum').item()

        steps += 1
        next_token_id = torch.multinomial(probs_full.float(), num_samples=1)
        context = torch.cat([context, next_token_id.unsqueeze(0)], dim=1)
        if next_token_id.item() == tokenizer.eos_token_id:
            break

        if args.verbose and steps % 50 == 0:
            print(f'  step {steps}: ' +
                  ', '.join(f'{k}={v / steps:.3f}' for k, v in divergences.items()))

    for name in divergences:
        divergences[name] /= steps

    text = tokenizer.decode(context[0][-steps:], skip_special_tokens=True)
    return text, divergences, steps, time.time() - start_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--language_model', default='Qwen/Qwen2.5-7B-Instruct', type=str)
    parser.add_argument('--dataset', default='instinwild_en', type=str)
    parser.add_argument('--token_max', default=512, type=int)
    parser.add_argument('--context_window', default=10, type=int)
    parser.add_argument('--strategies', type=str,
                        default='Baseline,Hard_0,Rand_5,Rand_10,'
                                'Hard_missing,Hard_removed,Hard_1,Hard_2',
                        help='comma-separated; see asw.ALL_STRATEGIES plus Rand_<k> '
                             'and Text_<label> paired with --bridge_texts')
    parser.add_argument('--bridge_texts', default=None, type=str,
                        help='JSON mapping strategy name to bridge string, e.g. '
                             '\'{"Text_a": "[...]\\n"}\'')
    parser.add_argument('--soft_forward_path', default=None, type=str)
    parser.add_argument('--soft_reverse_path', default=None, type=str)
    parser.add_argument('--default_soft_length', default=8, type=int)
    parser.add_argument('--lora', default=0, type=int)
    parser.add_argument('--lora_path', default=None, type=str)
    parser.add_argument('--dtype', default='float32', choices=list(asw.DTYPES))
    parser.add_argument('--no_cache', action='store_true')
    parser.add_argument('--index_start', default=0, type=int)
    parser.add_argument('--index_end', default=-1, type=int, help='-1 means all')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--output', default=None, type=str,
                        help='defaults to 2.data/<model>_window_<w>_lora_<lora>.tsv')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)

    model, tokenizer = asw.load_model(
        args.language_model, dtype=args.dtype,
        lora_path=args.lora_path if args.lora else None)
    windows = build_windows(model, tokenizer, args)
    names = list(windows)

    stem = args.language_model.rsplit('/', 1)[-1]
    output_file = asw.ensure_dir(args.output or
        f'2.data/{stem}_window_{args.context_window}_lora_{args.lora}.tsv')
    header = ['Idx'] + names + ['Token_num', 'Time', 'Context', 'text']
    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f, delimiter='\t').writerow(header)
    else:
        # Runs append, so the header on disk has to match the strategy set of
        # this run for the columns to line up.
        with open(output_file, newline='', encoding='utf-8') as f:
            existing = next(csv.reader(f, delimiter='\t'))
        if existing != header:
            raise SystemExit(
                f'{output_file} was written with a different strategy set\n'
                f'  on disk: {", ".join(existing[1:-4])}\n'
                f'  now    : {", ".join(names)}\n'
                f'Pass --output to write somewhere else.')

    df = pd.read_csv(f'{args.dataset}.tsv', sep='\t', encoding='utf-8')
    df_test = df[df['part'] == 'test'].reset_index(drop=True)
    end = len(df_test) if args.index_end < 0 else min(args.index_end, len(df_test))

    for i in range(args.index_start, end):
        prompt = df_test['question'][i]
        idx = df_test['new_id'][i]
        print(f'[{i}] {prompt}')

        text = asw.chat_prompt(tokenizer, prompt)
        prompt_ids = tokenizer([text], return_tensors='pt').to(model.device)['input_ids']
        generated, divergences, token_num, duration = measure(
            model, tokenizer, windows, prompt_ids, args)
        print('  ' + ', '.join(f'{k}={divergences[k]:.3f}' for k in names))

        with open(output_file, 'a+', newline='', encoding='utf-8') as f:
            csv.writer(f, delimiter='\t').writerow(
                [idx] + [divergences[k] for k in names] +
                [token_num, duration, text, generated])
        gc.collect()
        torch.cuda.empty_cache()
