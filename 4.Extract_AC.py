"""Steganographic extraction with arithmetic coding inside the ASW framework.

Implements Algorithm 2 (soft bridge context) and Algorithm 4 (hard bridge
context): Bob's side of the channel.

Extraction inverts 3.Embed_AC.py step by step.  At every position the decoder
rebuilds the same context window, sorts the same distribution, slices the same
arithmetic-coding interval, and reads off the bits that the encoder's choice of
token must have consumed: the message index always lies between the two interval
bounds, so it shares their common prefix.

The run configuration is read from the .config.json that 3.Embed_AC.py writes, so
both sides use the same window, bridge and coder settings.

Examples:
    # lossless round trip
    python 4.Extract_AC.py --stego_file 3.Stega_data/AC_Qwen2.5-7B-Instruct_window_10_strategy_Hard_1_lora_0_instinwild_en.tsv

    # robustness under an active attack (Section 7.3)
    python 4.Extract_AC.py --stego_file ... --attack substitute --attack_num 1
"""

import argparse
import ast
import csv
import gc
import json
import os
import random

import pandas as pd
import torch
import torch.nn.functional as F

import asw


def decode_arithmetic(window, tokenizer, stego_tokens, context, topk, args):
    """Recover the embedded bits from ``stego_tokens``.

    ``context`` is the prompt; ``stego_tokens`` are the tokens Bob received.
    Returns the bit string plus the position at which decoding stopped, if the
    stegotext left the coded token set (which is what an attack looks like from
    the decoder's point of view).
    """
    window.reset(context)

    precision = args.precision
    cur_interval = [0, 2 ** precision]
    bits = []
    truncated_at = None

    for step, token_id in enumerate(stego_tokens):
        logits = window.logits(context)
        logits, indices = logits.sort(descending=True)
        logits = logits.double()
        probs_temp = F.softmax(logits / args.temp, dim=0)

        cum_probs = asw.build_intervals(probs_temp, cur_interval, topk, precision)

        # Where does the received token sit in the sorted distribution?
        match = (indices == token_id).nonzero()
        if match.shape[0] == 0:
            truncated_at = step
            break
        selection = match[0].item()

        if selection >= len(cum_probs):
            # Outside the interval the encoder could have used, so this token
            # did not come from this distribution.
            truncated_at = step
            break

        _, new_bits, cur_interval = asw.advance_interval(
            cum_probs, selection, cur_interval, precision)
        bits.extend(new_bits)

        context = torch.cat(
            (context, torch.tensor([[token_id]], device=context.device)), dim=-1)
        if token_id == tokenizer.eos_token_id:
            break

    return ''.join(str(b) for b in bits), truncated_at


def apply_attack(tokens, kind, num, vocab_size, rng, protected=()):
    """Simulate an active attack (Backes and Cachin, 2005) on a stegotext.

    ``substitute``/``delete``/``insert`` correspond to Tables 3, 7 and 8.  The
    eos token is left alone so that the attack models channel noise rather than
    truncation of the message.
    """
    tokens = list(tokens)
    if num <= 0:
        return tokens, []

    candidates = [i for i in range(len(tokens)) if tokens[i] not in protected]
    if not candidates:
        return tokens, []
    positions = sorted(rng.sample(candidates, min(num, len(candidates))))

    if kind == 'substitute':
        for p in positions:
            replacement = tokens[p]
            while replacement == tokens[p]:
                replacement = rng.randrange(vocab_size)
            tokens[p] = replacement
    elif kind == 'delete':
        for p in reversed(positions):
            del tokens[p]
    elif kind == 'insert':
        for p in reversed(positions):
            tokens.insert(p, rng.randrange(vocab_size))
    else:
        raise ValueError(f'unknown attack {kind!r}')
    return tokens, positions


def bit_accuracy(reference, recovered):
    """Fraction of the embedded payload that came back correctly.

    Scored over the full payload, so bits the decoder never produced count as
    errors rather than being left out of the average.
    """
    if not reference:
        return 0.0
    matched = sum(a == b for a, b in zip(reference, recovered))
    return matched / len(reference)


def correct_prefix(reference, recovered):
    """Length of the leading run of bits that came back correctly."""
    n = 0
    for a, b in zip(reference, recovered):
        if a != b:
            break
        n += 1
    return n


def load_config(stego_file, args):
    """Pull the embedding-time settings off disk so both sides agree."""
    config_path = stego_file.replace('.tsv', '.config.json')
    if not os.path.exists(config_path):
        print(f'! {config_path} not found; using the command-line flags. '
              f'They have to match the embedding run exactly.')
        return args
    with open(config_path, encoding='utf-8') as f:
        config = json.load(f)
    # Anything that changes the distribution has to come from the embedding run.
    for key in ('language_model', 'strategy', 'context_window', 'default_soft_length',
                'soft_bridge_path', 'bridge_text', 'lora', 'lora_path', 'precision',
                'temp', 'top_k', 'dtype', 'no_cache', 'seed', 'soft_bridge_sha256'):
        if key in config:
            setattr(args, key, config[key])
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stego_file', required=True, type=str,
                        help='TSV produced by 3.Embed_AC.py')
    parser.add_argument('--source', default='token', choices=('token', 'text'),
                        help="'token' replays Alice's exact tokens; 'text' retokenizes "
                             'the stegotext, which is what Bob really receives and which '
                             'also exposes the tokenization mismatch of Appendix H')
    parser.add_argument('--attack', default='none',
                        choices=('none', 'substitute', 'delete', 'insert'))
    parser.add_argument('--attack_num', default=0, type=int, help='m, the number of tokens attacked')
    parser.add_argument('--num_samples', default=-1, type=int, help='-1 means all')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--no_cache', action='store_true')
    # Overridden by the .config.json when one is present.
    parser.add_argument('--language_model', default='Qwen/Qwen2.5-7B-Instruct', type=str)
    parser.add_argument('--strategy', default='Hard_1', type=asw.strategy_arg)
    parser.add_argument('--context_window', default=10, type=int)
    parser.add_argument('--bridge_text', default=None, type=str,
                        help='bridge string for a Text_<label> strategy')
    parser.add_argument('--default_soft_length', default=8, type=int)
    parser.add_argument('--soft_bridge_path', default=None, type=str)
    parser.add_argument('--lora', default=0, type=int)
    parser.add_argument('--lora_path', default=None, type=str)
    parser.add_argument('--precision', default=32, type=int)
    parser.add_argument('--temp', default=1.0, type=float)
    parser.add_argument('--top_k', default=-1, type=int)
    parser.add_argument('--dtype', default='float32', choices=list(asw.DTYPES))
    args = parser.parse_args()

    args = load_config(args.stego_file, args)
    print(args)

    rng = random.Random(args.seed)
    model, tokenizer = asw.load_model(
        args.language_model, dtype=args.dtype,
        lora_path=args.lora_path if args.lora else None)

    soft_embedding = None
    if args.strategy == 'Soft_0':
        generator = torch.Generator(device='cpu').manual_seed(args.seed)
        soft_embedding = torch.randn(args.default_soft_length, model.config.hidden_size,
                                     generator=generator).to(model.device, model.dtype)
    elif 'Soft' in args.strategy:
        soft_embedding = asw.load_soft_bridge(args.soft_bridge_path, model,
                                              args.default_soft_length)

    # Confirm the loaded bridge is the one the stegotexts were produced with;
    # a different bridge gives a different distribution at every step.
    expected_sha = getattr(args, 'soft_bridge_sha256', None)
    if soft_embedding is not None and expected_sha:
        actual_sha = asw.bridge_fingerprint(soft_embedding)
        if actual_sha != expected_sha:
            raise SystemExit(
                f'{args.soft_bridge_path} no longer holds the bridge these stegotexts '
                f'were made with (embedded {expected_sha}, loaded {actual_sha}).\n'
                f'Point --soft_bridge_path at the checkpoint from the embedding run; '
                f'a training job that kept writing to the same file will have '
                f'overwritten it.')

    window = asw.ASWWindow(model, tokenizer, args.strategy, args.context_window,
                           soft_embedding=soft_embedding, use_cache=not args.no_cache,
                           rand_seed=args.seed, bridge_text=args.bridge_text)

    vocab_size = len(tokenizer.get_vocab())
    topk = vocab_size if args.top_k < 0 else args.top_k

    df = asw.read_stega_tsv(args.stego_file)
    # An empty field comes back from pandas as NaN rather than ''.
    df['message'] = df['message'].fillna('')
    if args.num_samples > 0:
        df = df.iloc[:args.num_samples]

    out_name = (args.stego_file.replace('3.Stega_data/', '4.Extract_data/')
                .replace('.tsv', f'_extract_{args.source}_{args.attack}{args.attack_num}.tsv'))
    asw.ensure_dir(out_name)
    with open(out_name, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f, delimiter='\t').writerow(
            ['Idx', 'Bits_embedded', 'Bits_recovered', 'Bit_accuracy',
             'Correct_prefix', 'Prefix_ratio', 'Exact', 'Truncated_at',
             'Attacked_positions'])

    accuracies, prefix_ratios, exact_hits = [], [], 0
    for row in range(len(df)):
        idx = df['Idx'][row]
        reference = df['message'][row]
        context = torch.tensor([ast.literal_eval(df['Context_token'][row])],
                               device=model.device)

        if args.source == 'token':
            stego_tokens = ast.literal_eval(df['Text_token'][row])
        else:
            stego_tokens = tokenizer(df['Text'][row], add_special_tokens=False)['input_ids']

        stego_tokens, attacked = apply_attack(
            stego_tokens, args.attack, args.attack_num, vocab_size, rng,
            protected=(tokenizer.eos_token_id,))

        recovered, truncated_at = decode_arithmetic(
            window, tokenizer, stego_tokens, context, topk, args)

        accuracy = bit_accuracy(reference, recovered)
        prefix = correct_prefix(reference, recovered)
        prefix_ratio = prefix / len(reference) if reference else 0.0
        exact = recovered == reference
        exact_hits += exact
        accuracies.append(accuracy)
        prefix_ratios.append(prefix_ratio)

        with open(out_name, 'a+', newline='', encoding='utf-8') as f:
            csv.writer(f, delimiter='\t').writerow(
                [idx, len(reference), len(recovered), accuracy, prefix,
                 prefix_ratio, int(exact), truncated_at, attacked])

        print(f'[{row}] idx={idx} bits={len(reference)}->{len(recovered)} '
              f'acc={accuracy:.4f} prefix={prefix}/{len(reference)} exact={exact}')
        gc.collect()
        torch.cuda.empty_cache()

    print()
    print(f'samples                 : {len(accuracies)}')
    print(f'average bit accuracy    : {sum(accuracies) / len(accuracies):.4f}')
    print(f'average correct prefix  : {sum(prefix_ratios) / len(prefix_ratios):.4f}')
    print(f'exact extraction rate   : {exact_hits / len(accuracies):.4f}')
    print(f'written to              : {out_name}')
