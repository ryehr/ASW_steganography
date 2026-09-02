"""Steganographic embedding with arithmetic coding inside the ASW framework.

Implements Algorithm 1 (soft bridge context) and Algorithm 3 (hard bridge
context); the two differ only in how the window is built, which asw.ASWWindow
takes care of.  The arithmetic coder follows Ziegler et al. (2019).

The companion decoder is 4.Extract_AC.py.  Both drive the same ASWWindow and the
same interval arithmetic, which is what makes the round trip lossless.

Example:
    python 3.Embed_AC.py --strategy Hard_1 --context_window 10
    python 3.Embed_AC.py --strategy Soft_forward --context_window 10 \
        --soft_bridge_path 1.soft_prompt/soft_length8_epoch9_reverse0_Qwen2.5-7B-Instruct_0.6792.pt
"""

import argparse
import csv
import gc
import json
import os
import random
import time

import pandas as pd
import torch
import torch.nn.functional as F

import asw


def encode_arithmetic(window, tokenizer, message, context, topk, args):
    """Embed ``message`` (a '0'/'1' string) into a text continuing ``context``.

    Returns the stegotext plus the statistics the evaluation scripts consume.
    """
    start_time = time.time()
    window.reset(context)

    precision = args.precision
    cur_interval = [0, 2 ** precision]  # bottom inclusive, top exclusive

    i = 0                      # bits of the message consumed so far
    total_num_for_stats = 0    # tokens emitted
    total_entropy_ptau = 0.0

    while total_num_for_stats < args.token_max:
        logits = window.logits(context)
        logits, indices = logits.sort(descending=True)
        logits = logits.double()
        probs_temp = F.softmax(logits / args.temp, dim=0)

        if i >= len(message):
            # Message exhausted: fall back to the most likely token.  With the
            # default 100k-bit payload this never fires before token_max.
            selection = 0
        else:
            cum_probs = asw.build_intervals(probs_temp, cur_interval, topk, precision)

            message_bits = message[i:i + precision]
            if len(message_bits) < precision:
                message_bits = message_bits + '0' * (precision - len(message_bits))
            message_idx = asw.bits2int(reversed(message_bits))

            selection = (cum_probs > message_idx).nonzero()[0].item()
            num_bits_encoded, _, cur_interval = asw.advance_interval(
                cum_probs, selection, cur_interval, precision)
            i += num_bits_encoded

            total_entropy_ptau += -torch.sum(
                probs_temp * torch.log2(probs_temp + 1e-10)).item()
            total_num_for_stats += 1

        prev = indices[selection].view(1)
        context = torch.cat((context, prev.unsqueeze(0)), dim=-1)
        if prev.item() == tokenizer.eos_token_id:
            break

    avg_entropy = total_entropy_ptau / total_num_for_stats
    bpt = i / total_num_for_stats
    stego_tokens = context[:, -total_num_for_stats:][0].tolist()
    stegotext = tokenizer.decode(context[:, -total_num_for_stats:].squeeze(0),
                                 skip_special_tokens=True)

    return {
        'Token_num': total_num_for_stats,
        'BPT': bpt,
        'Entropy': avg_entropy,
        'Utilization': bpt / avg_entropy,
        'Time': time.time() - start_time,
        'Text': stegotext,
        'Text_token': stego_tokens,
        'message': message[:i],
    }


# Settings that change the stegotexts themselves, so mixing two values of any of
# them inside one file makes the file meaningless.
SAMPLE_AFFECTING = ('language_model', 'dataset', 'token_max', 'top_k', 'temp',
                    'precision', 'context_window', 'strategy', 'default_soft_length',
                    'soft_bridge_path', 'bridge_text', 'lora', 'lora_path', 'dtype',
                    'no_cache', 'seed', 'message_bits')


def output_path(args):
    stem = args.language_model.rsplit('/', 1)[-1]
    if args.context_window <= 0:
        return f'3.Stega_data/AC_{stem}_full_{args.dataset}.tsv'
    name = (f'3.Stega_data/AC_{stem}_window_{args.context_window}'
            f'_strategy_{args.strategy}_lora_{args.lora}')
    if 'Soft' in args.strategy:
        name += f'_softlength_{args.default_soft_length}'
    return f'{name}_{args.dataset}.tsv'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--language_model', default='Qwen/Qwen2.5-7B-Instruct', type=str)
    parser.add_argument('--dataset', default='instinwild_en', type=str,
                        help="'instinwild_en', 'databricks-dolly' or 'supernatural'")
    parser.add_argument('--token_max', default=512, type=int)
    parser.add_argument('--top_k', default=-1, type=int)
    parser.add_argument('--temp', default=1.0, type=float)
    parser.add_argument('--precision', default=32, type=int, help='max 52')
    parser.add_argument('--context_window', default=10, type=int,
                        help='length w of the latest tokens; <=0 means full context')
    parser.add_argument('--strategy', default='Soft_forward', type=asw.strategy_arg)
    parser.add_argument('--bridge_text', default=None, type=str,
                        help='bridge string for a Text_<label> strategy')
    parser.add_argument('--default_soft_length', default=8, type=int)
    parser.add_argument('--soft_bridge_path', default=None, type=str,
                        help='trained soft bridge context (.pt) for the Soft_* strategies')
    parser.add_argument('--lora', default=0, type=int)
    parser.add_argument('--lora_path', default=None, type=str)
    parser.add_argument('--dtype', default='float32', choices=list(asw.DTYPES),
                        help='keep float32: extraction has to reproduce every logit')
    parser.add_argument('--no_cache', action='store_true',
                        help='disable the prefix KV cache (slower, same numbers)')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--index_start', default=0, type=int)
    parser.add_argument('--index_end', default=-1, type=int, help='-1 means all')
    parser.add_argument('--message_bits', default=100000, type=int)
    parser.add_argument('--overwrite', action='store_true',
                        help='replace an existing output file instead of appending')
    args = parser.parse_args()
    print(args)

    # With --context_window <= 0 the window is the whole history and no bridge is
    # used at all, so --strategy is inert and must not be validated.
    uses_bridge = args.context_window > 0
    if (uses_bridge and 'Soft' in args.strategy and args.strategy != 'Soft_0'
            and not args.soft_bridge_path):
        parser.error(f'--soft_bridge_path is required for --strategy {args.strategy}')

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, tokenizer = asw.load_model(
        args.language_model, dtype=args.dtype,
        lora_path=args.lora_path if args.lora else None)

    # The bridge is drawn or loaded once per run.  It has to stay fixed for the
    # whole text: the decoder rebuilds the same window at every step, so a bridge
    # that changed between steps could not be followed.
    soft_embedding = None
    if not uses_bridge:
        pass
    elif args.strategy == 'Soft_0':
        generator = torch.Generator(device='cpu').manual_seed(args.seed)
        soft_embedding = torch.randn(args.default_soft_length, model.config.hidden_size,
                                     generator=generator).to(model.device, model.dtype)
    elif 'Soft' in args.strategy:
        soft_embedding = asw.load_soft_bridge(args.soft_bridge_path, model,
                                              args.default_soft_length)

    window = asw.ASWWindow(model, tokenizer, args.strategy, args.context_window,
                           soft_embedding=soft_embedding, use_cache=not args.no_cache,
                           rand_seed=args.seed, bridge_text=args.bridge_text)

    # Pin down *which* bridge this run used, not just where it was loaded from.
    bridge_sha = asw.bridge_fingerprint(soft_embedding) if soft_embedding is not None else None

    topk = len(tokenizer.get_vocab()) if args.top_k < 0 else args.top_k

    file_name = asw.ensure_dir(output_path(args))
    config_path = file_name.replace('.tsv', '.config.json')
    header = ['Idx', 'Token_num', 'BPT', 'Entropy', 'Utilization', 'Time',
              'Context', 'Text', 'Context_token', 'Text_token', 'message']

    # Runs append, which is what makes sharding by --index_start/--index_end work.
    # Comparing against the recorded config keeps one file from ending up with
    # stegotexts generated under two different settings.
    if os.path.exists(file_name) and os.path.exists(config_path) and not args.overwrite:
        with open(config_path, encoding='utf-8') as f:
            previous = json.load(f)
        changed = [k for k in SAMPLE_AFFECTING
                   if k in previous and previous[k] != getattr(args, k)]
        if changed:
            raise SystemExit(
                f'{file_name} already holds stegotexts generated with different '
                f'settings ({", ".join(changed)}).\n'
                f'Pass --overwrite to replace it, or delete the file.')

    if args.overwrite or not os.path.exists(file_name):
        with open(file_name, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f, delimiter='\t').writerow(header)

    # Extraction has to rebuild the exact same distributions, so record how this
    # run was configured next to the stegotexts instead of relying on the reader
    # to pass matching flags by hand.
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump({**vars(args), 'soft_bridge_sha256': bridge_sha}, f, indent=2)

    df = pd.read_csv(f'{args.dataset}.tsv', sep='\t', encoding='utf-8')
    df_test = df[df['part'] == 'test'].reset_index(drop=True)
    end = len(df_test) if args.index_end < 0 else min(args.index_end, len(df_test))

    for i in range(args.index_start, end):
        prompt = df_test['question'][i]
        idx = df_test['new_id'][i]
        print(f'[{i}] {prompt}')

        text = asw.chat_prompt(tokenizer, prompt)
        model_inputs = tokenizer([text], return_tensors='pt').to(model.device)['input_ids']

        secret_bits = format(random.getrandbits(args.message_bits),
                             f'0{args.message_bits}b')
        result = encode_arithmetic(window, tokenizer, secret_bits, model_inputs, topk, args)

        with open(file_name, 'a+', newline='', encoding='utf-8') as f:
            csv.writer(f, delimiter='\t').writerow([
                idx, result['Token_num'], result['BPT'], result['Entropy'],
                result['Utilization'], result['Time'], text, result['Text'],
                model_inputs[0].tolist(), result['Text_token'], result['message'],
            ])
        gc.collect()
        torch.cuda.empty_cache()
