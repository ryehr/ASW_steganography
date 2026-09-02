"""Steganalysis: how well can Eve tell stegotext from covertext (Appendix C).

Fine-tunes a BERT binary discriminator on pairs of (reference text, stegotext)
and reports its test accuracy.  This is the "Steganalysis ACC" column of Table 2
and Figures 5-6, where lower means the stegotext is harder to spot.

Setup follows Appendix C: 500 pairs per experimental group, split 6:2:2 into
train/validation/test, AdamW at 1e-5, batch size 128, 5 epochs.  The model with
the best validation accuracy is the one evaluated, so a run that overfits in the
last epochs does not get reported as a stronger attacker than it is.

Example:
    python 6.Steganalysis.py \
        --stego_file 3.Stega_data/AC_Qwen2.5-7B-Instruct_window_10_strategy_Hard_1_lora_0_instinwild_en.tsv \
        --reference_file 3.Stega_data/Normal_Qwen2.5-7B-Instruct_instinwild_en.tsv
"""

import argparse
import copy
import glob
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import asw


class PairDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]


def make_collate(tokenizer, max_length):
    def collate(batch):
        texts, labels = zip(*batch)
        encoded = tokenizer(list(texts), return_tensors='pt', padding=True,
                            truncation=True, max_length=max_length)
        encoded['labels'] = torch.tensor(labels)
        return encoded
    return collate


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop('labels')
        preds = model(**batch).logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return correct / total


def run_group(stego_file, reference_file, args, device):
    """Train and evaluate one discriminator; returns its test accuracy."""
    stego = [t for t in asw.read_stega_tsv(stego_file)['Text'].astype(str)]
    reference = [t for t in asw.read_stega_tsv(reference_file)['Text'].astype(str)]

    n = min(len(stego), len(reference), args.num_pairs)
    if n < 10:
        raise ValueError(f'{stego_file}: only {n} usable pairs')
    stego, reference = stego[:n], reference[:n]

    # Split by pair, so a stegotext and its reference never straddle the split
    # and let the discriminator memorise the prompt instead of the style.
    order = list(range(n))
    random.Random(args.seed).shuffle(order)
    n_train, n_val = int(n * 0.6), int(n * 0.2)
    splits = {
        'train': order[:n_train],
        'validation': order[n_train:n_train + n_val],
        'test': order[n_train + n_val:],
    }

    tokenizer = AutoTokenizer.from_pretrained(args.discriminator)
    loaders = {}
    for name, index in splits.items():
        texts = [stego[i] for i in index] + [reference[i] for i in index]
        labels = [1] * len(index) + [0] * len(index)
        loaders[name] = DataLoader(
            PairDataset(texts, labels, tokenizer), batch_size=args.batch_size,
            shuffle=(name == 'train'), collate_fn=make_collate(tokenizer, args.max_length))

    model = AutoModelForSequenceClassification.from_pretrained(
        args.discriminator, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val, best_state = -1.0, None
    for epoch in range(args.epochs):
        model.train()
        for batch in loaders['train']:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        val = accuracy(model, loaders['validation'], device)
        print(f'  epoch {epoch + 1}/{args.epochs}  validation acc {val:.4f}')
        if val > best_val:
            best_val = val
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return accuracy(model, loaders['test'], device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stego_file', default=None, type=str,
                        help='one stegotext TSV; omit to sweep --stego_glob')
    parser.add_argument('--stego_glob', default=None, type=str,
                        help='evaluate every stegotext file matching this pattern')
    parser.add_argument('--reference_file', required=True, type=str,
                        help='covertext TSV from 3.Generation_normal.py')
    parser.add_argument('--discriminator', default='bert-base-uncased', type=str)
    parser.add_argument('--num_pairs', default=500, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--max_length', default=256, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--output', default='6.Steganalysis/results.json', type=str)
    args = parser.parse_args()

    if not args.stego_file and not args.stego_glob:
        parser.error('pass either --stego_file or --stego_glob')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    targets = [args.stego_file] if args.stego_file else sorted(glob.glob(args.stego_glob))
    targets = [t for t in targets if 'Normal' not in os.path.basename(t)]

    results = {}
    for target in targets:
        print(f'\n=== {os.path.basename(target)} ===')
        acc = run_group(target, args.reference_file, args, device)
        results[os.path.basename(target)] = acc
        print(f'  steganalysis accuracy: {acc:.4f}')

    asw.ensure_dir(args.output)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print('\nSteganalysis accuracy (lower is more imperceptible)')
    for name, acc in sorted(results.items(), key=lambda kv: kv[1]):
        print(f'  {acc:.4f}  {name}')
    print(f'\nwritten to {args.output}')
