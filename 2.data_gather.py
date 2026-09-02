"""Summarise the per-sample KL divergences from 2.Generation_window.py into Table 1.

Reads whichever strategy columns the run actually produced, so adding a bridge
context to 2.Generation_window.py does not mean editing a hard-coded list here.

Example:
    python 2.data_gather.py --model Qwen2.5-7B-Instruct --window_size 10
    python 2.data_gather.py --window_size 10,20,30,40,50   # one column per w
"""

import argparse
import os

import pandas as pd

RESERVED = {'Idx', 'Token_num', 'Time', 'Context', 'text', 'stegotext'}


def summarise(path, split_at):
    df = pd.read_csv(path, sep='\t')
    strategies = [c for c in df.columns if c not in RESERVED]
    if split_at and split_at < len(df):
        groups = {'In-dataset': df.iloc[:split_at], 'Out-of-dataset': df.iloc[split_at:]}
    else:
        groups = {'All': df}
    return strategies, groups


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen2.5-7B-Instruct', type=str)
    parser.add_argument('--window_size', default='10', type=str,
                        help='comma-separated; one column per value')
    parser.add_argument('--lora', default=0, type=int)
    parser.add_argument('--folder', default='2.data', type=str)
    parser.add_argument('--split_at', default=0, type=int,
                        help='rows before this index count as in-dataset; 0 disables')
    parser.add_argument('--statistic', default='mean', choices=('mean', 'median'))
    args = parser.parse_args()

    windows = [int(w) for w in args.window_size.split(',')]
    columns, table = {}, {}

    for w in windows:
        path = os.path.join(args.folder, f'{args.model}_window_{w}_lora_{args.lora}.tsv')
        if not os.path.exists(path):
            print(f'! {path} not found, skipping')
            continue
        strategies, groups = summarise(path, args.split_at)
        for group_name, group in groups.items():
            for strategy in strategies:
                value = (group[strategy].mean() if args.statistic == 'mean'
                         else group[strategy].median())
                table.setdefault((group_name, strategy), {})[w] = value
            columns[(group_name, w)] = len(group)

    if not table:
        raise SystemExit('nothing to summarise')

    present = [w for w in windows if any(w in row for row in table.values())]
    for group_name in dict.fromkeys(key[0] for key in table):
        sizes = ', '.join(f'w={w}: {columns[(group_name, w)]}' for w in present)
        print(f'\n{group_name} ({args.statistic}; {sizes})')
        header = 'Strategy'.ljust(18) + ''.join(f'w={w}'.rjust(10) for w in present)
        print(header)
        print('-' * len(header))
        for (grp, strategy), values in table.items():
            if grp != group_name:
                continue
            print(strategy.ljust(18) +
                  ''.join(f'{values.get(w, float("nan")):10.3f}' for w in present))
