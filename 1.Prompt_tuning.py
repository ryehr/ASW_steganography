"""Train the soft bridge context by self-distillation (Section 5).

The language model is frozen throughout.  The only trainable tensor is the soft
bridge context that sits between the anchored prompt and the latest tokens; it
learns to make the ASW window's next-token distribution match what the model
would have predicted from the full history:

    teacher  = f_LM(s_full)                        (full context)
    student  = f_LM_emb([E_prompt; theta_bridge; E_latest])
    loss     = D_KL(teacher || student)            forward,  Equation 9
             = D_KL(student || teacher)            reverse,  Equation 10

The defaults are the paper's: random initialisation, forward KL, lr 1e-3,
10 epochs, l_bridge=8, w=10.  --initialization selects an alternative starting
point for the bridge, --loss both interpolates the two divergences, and
--window_batch trades memory for speed; none of them changes the default run.

Example:
    python 1.Prompt_tuning.py --soft_prompt_length 8 --window_size 10 --KL_reverse 0
    python 1.Prompt_tuning.py --initialization 2 --loss both --window_batch 8
"""

import argparse
import gc
import os
import random

import pandas as pd
import torch
from accelerate import Accelerator
from torch.nn.functional import kl_div, log_softmax, softmax
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import asw


class QADataset(Dataset):
    def __init__(self, qa_pairs):
        self.qa_pairs = qa_pairs

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        question, answer = self.qa_pairs[idx]
        return {'question': question, 'answer': answer}


def make_collate(tokenizer, q_token_max, a_token_max):
    def collate(batch):
        q = tokenizer([b['question'] for b in batch], return_tensors='pt',
                      truncation=True, padding=True, max_length=q_token_max)
        a = tokenizer([b['answer'] for b in batch], return_tensors='pt',
                      truncation=True, padding=True, max_length=a_token_max)
        return {'q_input_ids': q['input_ids'], 'q_attention_mask': q['attention_mask'],
                'a_input_ids': a['input_ids'], 'a_attention_mask': a['attention_mask']}
    return collate


def init_soft_prompt(model, tokenizer, args):
    """Build the initial soft bridge context.

    ``0`` is the paper's ``torch.randn``, whose rows have norm ~sqrt(hidden_size).
    ``1`` starts from the embeddings of a hard bridge string, and ``2`` from
    embeddings sampled out of the vocabulary, following Lester et al. (2021);
    both of those start at the scale of real token embeddings.
    """
    embed = model.get_input_embeddings().weight
    hidden = model.config.hidden_size

    if args.initialization == 0:
        init = torch.randn(args.soft_prompt_length, hidden)
    elif args.initialization == 1:
        # Repeat the string so a bridge longer than it still fills up, then cut
        # to the requested length.
        text = (args.init_bridge_text or asw.HARD_BRIDGES['Hard_1']) * 100
        ids = tokenizer(text, add_special_tokens=False, return_tensors='pt',
                        max_length=args.soft_prompt_length, truncation=True)['input_ids']
        init = embed[ids.squeeze(0)].detach().cpu().clone()
    elif args.initialization == 2:
        generator = torch.Generator().manual_seed(args.seed)
        # Draw from the 5000 most frequent ids, which are ordinary word pieces
        # rather than the byte-fallback and unused tail of the vocabulary.
        ids = torch.randint(5000, (args.soft_prompt_length,), generator=generator)
        init = embed[ids].detach().cpu().clone()
    else:
        raise ValueError(f'unknown --initialization {args.initialization}')

    return torch.nn.Parameter(init.float())


def teacher_logits_at(model, input_ids, attention_mask, positions):
    """Full-context logits, but only at the positions the loss actually reads.

    The LM head is ~152k wide, so projecting the whole answer through it costs
    far more than the loss uses.  Taking the hidden states and projecting only
    the wanted positions gives the same numbers at a fraction of the cost.
    """
    with torch.no_grad():
        hidden = model.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        return model.lm_head(hidden[:, positions, :])


def distillation_loss(student_logits, teacher_logits, args):
    """Single-step KL between the two distributions (Equations 9 and 10)."""
    if args.loss == 'both':
        forward = kl_div(log_softmax(student_logits, dim=-1),
                         softmax(teacher_logits, dim=-1), reduction='batchmean')
        reverse = kl_div(log_softmax(teacher_logits, dim=-1),
                         softmax(student_logits, dim=-1), reduction='batchmean')
        return args.loss_alpha * forward + (1 - args.loss_alpha) * reverse
    if args.KL_reverse == 0:
        return kl_div(log_softmax(student_logits, dim=-1),
                      softmax(teacher_logits, dim=-1), reduction='batchmean')
    return kl_div(log_softmax(teacher_logits, dim=-1),
                  softmax(student_logits, dim=-1), reduction='batchmean')


def window_positions(a_attention_mask, initial_index, args, batch_size):
    """Start offsets of the sliding windows scored in one answer."""
    length = a_attention_mask.shape[1]
    return [s for s in range(initial_index, length - args.window_size, args.step)
            if a_attention_mask[:, s].sum() >= 0.8 * batch_size]


def run_batch(model, batch, soft_prompt, args, device, train):
    """Score every window position of one batch; returns (loss_sum, n_positions).

    All the windows of a sample have the same shape, so they are stacked into the
    batch dimension and scored in a single forward pass.
    """
    q_ids = batch['q_input_ids'].to(device)
    q_mask = batch['q_attention_mask'].to(device)
    a_ids = batch['a_input_ids'].to(device)
    a_mask = batch['a_attention_mask'].to(device)
    batch_size, q_len = q_ids.shape

    starts = window_positions(a_mask, args.initial_index, args, batch_size)
    if not starts:
        return 0.0, 0

    teacher_index = [q_len + s + args.window_size - 1 for s in starts]
    teacher = teacher_logits_at(
        model, torch.cat([q_ids, a_ids], dim=1),
        torch.cat([q_mask, a_mask], dim=1), teacher_index).detach()

    embed = model.get_input_embeddings()
    q_embeddings = embed(q_ids)
    soft_len = soft_prompt.shape[0]
    total_loss, scored = 0.0, 0

    for chunk_start in range(0, len(starts), args.window_batch):
        chunk = starts[chunk_start:chunk_start + args.window_batch]
        n = len(chunk)

        windows = torch.stack([a_ids[:, s:s + args.window_size] for s in chunk], dim=1)
        window_mask = torch.stack([a_mask[:, s:s + args.window_size] for s in chunk], dim=1)
        windows = windows.reshape(batch_size * n, args.window_size)
        window_mask = window_mask.reshape(batch_size * n, args.window_size)

        q_rep = q_embeddings.repeat_interleave(n, dim=0)
        q_mask_rep = q_mask.repeat_interleave(n, dim=0)
        soft_rep = soft_prompt.unsqueeze(0).expand(batch_size * n, -1, -1).to(device)
        soft_mask = torch.ones((batch_size * n, soft_len), device=device)

        student_embeddings = torch.cat([q_rep, soft_rep, embed(windows)], dim=1)
        student_mask = torch.cat([q_mask_rep, soft_mask, window_mask], dim=1)

        # logits_to_keep=1: the loss reads only the final position, so the rest
        # need not go through the LM head.
        student = model(inputs_embeds=student_embeddings, attention_mask=student_mask,
                        logits_to_keep=1).logits[:, -1, :]
        target = teacher[:, chunk_start:chunk_start + n, :].reshape(batch_size * n, -1)

        loss = distillation_loss(student, target, args)
        if train:
            loss.backward()
        total_loss += loss.item() * n
        scored += n

    args.initial_index = (args.initial_index + len(starts)) % args.step
    return total_loss, scored


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--language_model', default='Qwen/Qwen2.5-7B-Instruct', type=str)
    parser.add_argument('--a_token_max', default=512, type=int)
    parser.add_argument('--q_token_max', default=64, type=int)
    parser.add_argument('--window_size', default=10, type=int, help='w, the latest tokens')
    parser.add_argument('--initialization', default=0, type=int,
                        help='0: randn (paper), 1: hard bridge tokens, '
                             '2: sampled vocabulary embeddings (correctly scaled)')
    parser.add_argument('--init_bridge_text', default=None, type=str,
                        help="bridge string for --initialization 1; defaults to "
                             "the paper's '[CONTEXT TRUNCATED]\n'")
    parser.add_argument('--soft_prompt_length', default=8, type=int, help='l_bridge')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--KL_reverse', default=0, type=int, help='0: forward KL, 1: reverse KL')
    parser.add_argument('--loss', default='kl', choices=('kl', 'both'),
                        help="'both' interpolates forward and reverse KL")
    parser.add_argument('--loss_alpha', default=0.5, type=float,
                        help='weight on forward KL when --loss both')
    parser.add_argument('--step', default=50, type=int, help='stride between sampled windows')
    parser.add_argument('--window_batch', default=8, type=int,
                        help='window positions scored per forward pass')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--output_dir', default='1.soft_prompt', type=str)
    parser.add_argument('--max_train_samples', default=-1, type=int, help='-1 means all')
    parser.add_argument('--max_val_samples', default=-1, type=int, help='-1 means all')
    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    accelerator = Accelerator()
    device = accelerator.device

    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.language_model, torch_dtype='auto').to(torch.float32)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    soft_prompt = init_soft_prompt(model, tokenizer, args)
    print(f'soft bridge init: norm/row {soft_prompt.norm(dim=1).mean():.3f} '
          f'(real token embeddings average '
          f'{model.get_input_embeddings().weight.norm(dim=1).mean():.3f})')

    optimizer = torch.optim.AdamW([soft_prompt], lr=args.lr)

    df = pd.read_csv('instinwild_en.tsv', sep='\t', encoding='utf-8')
    qa_pairs = {}
    for part in ('train', 'validation'):
        subset = df[df['part'] == part]
        qa_pairs[part] = [
            (asw.chat_prompt(tokenizer, q), a)
            for q, a in zip(subset['question'], subset['answer'])
        ]

    if args.max_train_samples > 0:
        qa_pairs['train'] = qa_pairs['train'][:args.max_train_samples]
    if args.max_val_samples > 0:
        qa_pairs['validation'] = qa_pairs['validation'][:args.max_val_samples]

    collate = make_collate(tokenizer, args.q_token_max, args.a_token_max)
    train_loader = DataLoader(QADataset(qa_pairs['train']), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate)
    val_loader = DataLoader(QADataset(qa_pairs['validation']), batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate)

    model, train_loader, soft_prompt, optimizer = accelerator.prepare(
        model, train_loader, soft_prompt, optimizer)

    os.makedirs(args.output_dir, exist_ok=True)
    stem = args.language_model.split('/')[-1]
    args.initial_index = random.randint(0, args.step - 1)
    best_val = float('inf')

    for epoch in range(args.epochs):
        total_loss = total_steps = 0
        progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}') \
            if accelerator.is_main_process else train_loader

        for batch in progress:
            optimizer.zero_grad()
            loss_sum, scored = run_batch(model, batch, soft_prompt, args, device, train=True)
            if scored == 0:
                continue
            # One step per sample, over all of its window positions at once.
            optimizer.step()
            total_loss += loss_sum
            total_steps += scored
            if accelerator.is_main_process:
                progress.set_postfix({'KL Loss': total_loss / total_steps})

        gc.collect()
        torch.cuda.empty_cache()

        if accelerator.is_main_process:
            print('Running validation...')
        val_loss = val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                loss_sum, scored = run_batch(model, batch, soft_prompt, args, device, train=False)
                val_loss += loss_sum
                val_steps += scored
        val_avg = val_loss / max(val_steps, 1)

        if accelerator.is_main_process:
            print(f'Epoch {epoch + 1} Validation KL Loss: {val_avg:.4f}')
            tensor = soft_prompt.detach().cpu()
            torch.save(tensor, os.path.join(
                args.output_dir,
                f'soft_length{args.soft_prompt_length}_epoch{epoch + 1}'
                f'_reverse{args.KL_reverse}_{stem}_{val_avg:.4f}.pt'))
            if val_avg < best_val:
                best_val = val_avg
                # Stable name so the embedding scripts do not have to be told
                # which epoch happened to win.
                torch.save(tensor, os.path.join(
                    args.output_dir,
                    f'best_soft_length{args.soft_prompt_length}'
                    f'_reverse{args.KL_reverse}_{stem}.pt'))
                print(f'  new best ({best_val:.4f})')
