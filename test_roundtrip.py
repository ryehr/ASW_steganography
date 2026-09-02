"""Self-check: does a message survive the embed -> transmit -> extract round trip?

Extraction is lossless only while embedding and extraction agree on every logit,
which depends on the dtype, the bridge, the tokenization and the interval
arithmetic all lining up.  The text-quality metrics do not cover any of that, so
this script checks it directly before a long experiment is launched.

It runs the whole pipeline in memory, with no files and no dependency on a
trained bridge, and also confirms that an attack shows up as lost bits.

Example:
    python test_roundtrip.py --language_model Qwen/Qwen2.5-7B-Instruct
    python test_roundtrip.py --language_model Qwen/Qwen2.5-3B-Instruct --num_samples 2
"""

import argparse
import random
import sys
import types

import torch

import asw

# 3.Embed_AC.py and 4.Extract_AC.py are not importable by name (a module name
# cannot start with a digit), so load them the long way round.
import importlib.util


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def coder_args(args, **overrides):
    """The subset of flags the coder functions read."""
    base = dict(precision=args.precision, temp=1.0, token_max=args.token_max)
    base.update(overrides)
    return types.SimpleNamespace(**base)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language_model', default='Qwen/Qwen2.5-7B-Instruct', type=str)
    parser.add_argument('--strategies', default='Hard_1,Hard_2,Hard_0,Baseline,Soft_0',
                        type=str)
    parser.add_argument('--context_window', default=10, type=int)
    parser.add_argument('--token_max', default=48, type=int)
    parser.add_argument('--precision', default=32, type=int)
    parser.add_argument('--soft_length', default=8, type=int)
    parser.add_argument('--num_samples', default=3, type=int)
    parser.add_argument('--dtype', default='float32', choices=list(asw.DTYPES))
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()

    embed_mod = load_script('3.Embed_AC.py', 'embed_ac')
    extract_mod = load_script('4.Extract_AC.py', 'extract_ac')

    model, tokenizer = asw.load_model(args.language_model, dtype=args.dtype)
    vocab_size = len(tokenizer.get_vocab())
    topk = vocab_size

    import pandas as pd
    questions = list(pd.read_csv('instinwild_en.tsv', sep='\t')
                     .query("part == 'test'")['question'])[:args.num_samples]

    failures = []
    for strategy in args.strategies.split(','):
        strategy = strategy.strip()
        soft_embedding = None
        if strategy == 'Soft_0':
            generator = torch.Generator(device='cpu').manual_seed(args.seed)
            soft_embedding = torch.randn(args.soft_length, model.config.hidden_size,
                                         generator=generator).to(model.device, model.dtype)

        window = asw.ASWWindow(model, tokenizer, strategy, args.context_window,
                               soft_embedding=soft_embedding, rand_seed=args.seed)

        print(f'\n=== {strategy} (w={args.context_window}) ===')
        for i, question in enumerate(questions):
            text = asw.chat_prompt(tokenizer, question)
            prompt_ids = tokenizer([text], return_tensors='pt').to(model.device)['input_ids']

            rng = random.Random(args.seed + i)
            message = ''.join(str(rng.getrandbits(1)) for _ in range(2000))

            result = embed_mod.encode_arithmetic(
                window, tokenizer, message, prompt_ids, topk, coder_args(args))
            embedded = result['message']

            recovered, truncated = extract_mod.decode_arithmetic(
                window, tokenizer, result['Text_token'], prompt_ids, topk, coder_args(args))

            ok = recovered == embedded
            print(f'  [{i}] clean : {len(embedded):4d} bits, '
                  f'{"OK" if ok else "MISMATCH"}'
                  + ('' if ok else f' (recovered {len(recovered)}, '
                                   f'prefix {extract_mod.correct_prefix(embedded, recovered)})'))
            if not ok:
                failures.append(f'{strategy}[{i}] clean round trip')

            # A one-token substitution must cost bits, which confirms the
            # decoder is reading the window it is configured for.
            attacked, _ = extract_mod.apply_attack(
                result['Text_token'], 'substitute', 1, vocab_size, rng,
                protected=(tokenizer.eos_token_id,))
            recovered_attacked, _ = extract_mod.decode_arithmetic(
                window, tokenizer, attacked, prompt_ids, topk, coder_args(args))
            accuracy = extract_mod.bit_accuracy(embedded, recovered_attacked)
            prefix = extract_mod.correct_prefix(embedded, recovered_attacked)
            print(f'      attack: bit accuracy {accuracy:.3f}, '
                  f'correct prefix {prefix}/{len(embedded)}')
            if recovered_attacked == embedded:
                failures.append(f'{strategy}[{i}] attack had no effect')

    print()
    if failures:
        print(f'FAILED ({len(failures)}):')
        for failure in failures:
            print(f'  - {failure}')
        raise SystemExit(1)
    print('All round trips lossless, and every attack cost bits.')


if __name__ == '__main__':
    main()
