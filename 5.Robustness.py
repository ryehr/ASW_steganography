"""Robustness of ASW under active attacks (Section 7.3, Tables 3, 7 and 8).

Reports the *ratio of positions with unaffected inference*: the share of
generation steps whose context window an attacker did not touch.  Appendix F
argues for this proxy because it is independent of the encoding scheme -- every
LM-based method depends on the conditional distribution being stable at each
step, whichever way it turns that distribution into bits.

Whether a position survives depends only on which tokens its window covers, so
the whole measurement is combinatorial and needs no GPU.  Two paths are
provided and cross-checked against each other:

  * ``--mode analytic``  Proposition 1 in closed form (substitution only).
  * ``--mode empirical`` Monte-Carlo over the attacked positions, for all three
    attacks.  Deletion and insertion follow the paper's convention that the
    deleted or newly inserted positions count as affected.

Sequence lengths come from a real stegotext file when one is given.

Examples:
    python 5.Robustness.py --stego_file 3.Stega_data/AC_Qwen2.5-7B-Instruct_window_10_strategy_Hard_1_lora_0_instinwild_en.tsv
    python 5.Robustness.py --lengths 512 --attack delete
"""

import argparse
import random
from math import comb

import pandas as pd

import asw


# --------------------------------------------------------------------------- #
# Dependency sets
# --------------------------------------------------------------------------- #

def dependency_width(method, w, winstega_entropy_window):
    """How many preceding generated tokens a step's inference reads.

    ASW reads exactly the ``w`` latest tokens: the prompt and the bridge context
    are also in the window, but the prompt is held by both parties and the bridge
    is independent of the excluded segment, so neither is reachable by an
    attacker (Section 3).

    The truncated-window baseline additionally reads the span its entropy
    threshold is computed over, so its effective dependency is ``w`` plus
    ``winstega_entropy_window``.
    """
    if method == 'full':
        return None  # every preceding token
    if method == 'asw':
        return w
    if method == 'winstega':
        return w + winstega_entropy_window
    raise ValueError(f'unknown method {method!r}')


# --------------------------------------------------------------------------- #
# Closed form (Proposition 1)
# --------------------------------------------------------------------------- #

def unaffected_ratio_analytic(T, w, m):
    """Proposition 1: expected share of positions whose inference survives.

    E[influenced] = T - 1 - (1/C(T,m)) * sum_{i=1..min(w,T-1)} C(T-i, m)
                            - (1/C(T,m)) * max(0, T-1-w) * C(T-w, m)
    and this returns (T - E[influenced]) / T.
    """
    if m <= 0:
        return 1.0
    if m >= T:
        return 1.0 / T  # only the first position, which reads nothing, survives

    total = comb(T, m)
    if w is None:  # full context: position i survives iff nothing before it moved
        unaffected = sum(comb(T - i, m) for i in range(0, T) if T - i >= m) / total
        return unaffected / T

    unaffected = 1.0  # position 0 reads no generated token
    unaffected += sum(comb(T - i, m) for i in range(1, min(w, T - 1) + 1)) / total
    if T - 1 - w > 0:
        unaffected += max(0, T - 1 - w) * comb(T - w, m) / total
    return unaffected / T


# --------------------------------------------------------------------------- #
# Monte Carlo
# --------------------------------------------------------------------------- #

def unaffected_ratio_empirical(T, w, m, attack, rng, trials):
    """Same quantity, sampled, and defined for deletion and insertion too."""
    if m <= 0:
        return 1.0
    total = 0.0
    for _ in range(trials):
        attacked = set(rng.sample(range(T), min(m, T)))
        length = T + m if attack == 'insert' else T
        unaffected = 0
        for i in range(length):
            if attack in ('delete', 'insert') and i in attacked:
                continue  # the removed/added position is affected by convention
            low = 0 if w is None else max(0, i - w)
            if not any(low <= a < i for a in attacked):
                unaffected += 1
        total += unaffected / length
    return total / trials


# --------------------------------------------------------------------------- #

def sequence_lengths(args):
    if args.stego_file:
        df = asw.read_stega_tsv(args.stego_file)
        return [int(t) for t in df['Token_num']]
    return [args.lengths] * args.num_samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stego_file', default=None, type=str,
                        help='take stegotext lengths from this 3.Embed_AC.py output')
    parser.add_argument('--lengths', default=512, type=int,
                        help='fixed sequence length, when no --stego_file is given')
    parser.add_argument('--num_samples', default=500, type=int)
    parser.add_argument('--windows', default='10,30,50', type=str)
    parser.add_argument('--modifications', default='1,2,3', type=str)
    parser.add_argument('--attack', default='substitute',
                        choices=('substitute', 'delete', 'insert'))
    parser.add_argument('--mode', default='analytic', choices=('analytic', 'empirical'))
    parser.add_argument('--trials', default=200, type=int, help='Monte-Carlo trials per sequence')
    parser.add_argument('--winstega_entropy_window', default=None, type=int,
                        help='extra tokens WinStega reads for its entropy threshold; '
                             'defaults to w, i.e. an effective dependency of 2w')
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()

    if args.attack != 'substitute' and args.mode == 'analytic':
        parser.error('--mode analytic covers substitution only; use --mode empirical')

    rng = random.Random(args.seed)
    lengths = sequence_lengths(args)
    windows = [int(x) for x in args.windows.split(',')]
    mods = [int(x) for x in args.modifications.split(',')]

    print(f'attack     : {args.attack}')
    print(f'mode       : {args.mode}')
    print(f'sequences  : {len(lengths)} (mean length {sum(lengths) / len(lengths):.1f})')
    print()

    header = 'Method'.ljust(24) + ''.join(f'm={m}'.rjust(10) for m in mods)
    print(header)
    print('-' * len(header))

    def evaluate(method, w):
        width = dependency_width(method, w, args.winstega_entropy_window
                                 if args.winstega_entropy_window is not None else w)
        row = []
        for m in mods:
            if args.mode == 'analytic':
                values = [unaffected_ratio_analytic(T, width, m) for T in lengths]
            else:
                values = [unaffected_ratio_empirical(T, width, m, args.attack, rng, args.trials)
                          for T in lengths]
            row.append(sum(values) / len(values))
        return row

    row = evaluate('full', None)
    print('Full context'.ljust(24) + ''.join(f'{v:9.2%}' for v in row))

    for w in windows:
        for method, label in (('winstega', 'WinStega'), ('asw', 'ASW')):
            row = evaluate(method, w)
            name = f'w={w:<3d} {label}'
            print(name.ljust(24) + ''.join(f'{v:9.2%}' for v in row))

    if args.mode == 'empirical' and args.attack == 'substitute':
        # Both paths describe the same quantity for substitution, so they are
        # cross-checked against each other.
        T = lengths[0]
        for w in windows:
            for m in mods:
                a = unaffected_ratio_analytic(T, w, m)
                e = unaffected_ratio_empirical(T, w, m, 'substitute', rng, args.trials)
                assert abs(a - e) < 0.01, f'w={w} m={m}: analytic {a:.4f} vs empirical {e:.4f}'
        print('\nProposition 1 agrees with the Monte-Carlo estimate.')
