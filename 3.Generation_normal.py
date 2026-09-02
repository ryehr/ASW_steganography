"""Generate the covertexts that every quality metric is measured against.

These are ordinary multinomial samples from the full-context model, i.e. what the
channel looks like when nothing is embedded.  Section 7.2.1 uses them both as the
BLEU/ROUGE/BERTScore references and as the negative class for steganalysis.

Example:
    python 3.Generation_normal.py --dataset instinwild_en
"""

import argparse
import csv
import gc
import os
import time

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import DynamicCache

import asw


def generate(model, tokenizer, context, args):
    """Sample a continuation, reusing the KV cache across steps.

    The cache makes generation linear in the sequence length and leaves the
    distribution at every step unchanged.
    """
    start_time = time.time()
    prompt_len = context.shape[1]

    cache = DynamicCache()
    with torch.no_grad():
        model(context[:, :-1], past_key_values=cache, use_cache=True)

    total_entropy = 0.0
    steps = 0
    while steps < args.token_max:
        with torch.no_grad():
            logits = model(context[:, -1:], past_key_values=cache,
                           use_cache=True).logits[0, -1, :].double()
        probs = F.softmax(logits / args.temp, dim=0)

        total_entropy += -torch.sum(probs * torch.log2(probs + 1e-10)).item()
        steps += 1

        next_token_id = torch.multinomial(probs.float(), num_samples=1)
        context = torch.cat((context, next_token_id.unsqueeze(0)), dim=1)
        if next_token_id.item() == tokenizer.eos_token_id:
            break

    generated = context[:, prompt_len:]
    return {
        'Token_num': steps,
        'Entropy': total_entropy / steps,
        'Time': time.time() - start_time,
        'Text': tokenizer.decode(generated.squeeze(0), skip_special_tokens=True),
        'Text_token': generated[0].tolist(),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='instinwild_en', type=str,
                        help="'instinwild_en', 'databricks-dolly' or 'supernatural'")
    parser.add_argument('--language_model', default='Qwen/Qwen2.5-7B-Instruct', type=str)
    parser.add_argument('--token_max', default=512, type=int)
    parser.add_argument('--temp', default=1.0, type=float)
    parser.add_argument('--dtype', default='float32', choices=list(asw.DTYPES))
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--index_start', default=0, type=int)
    parser.add_argument('--index_end', default=-1, type=int, help='-1 means all')
    parser.add_argument('--overwrite', action='store_true',
                        help='replace an existing output file instead of appending')
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    model, tokenizer = asw.load_model(args.language_model, dtype=args.dtype)

    stem = args.language_model.rsplit('/', 1)[-1]
    file_name = asw.ensure_dir(f'3.Stega_data/Normal_{stem}_{args.dataset}.tsv')
    header = ['Idx', 'Token_num', 'Entropy', 'Time', 'Context', 'Text',
              'Context_token', 'Text_token']
    # Appending is what lets --index_start/--index_end shard a run across GPUs;
    # --overwrite is for restarting one cleanly.
    if args.overwrite or not os.path.exists(file_name):
        with open(file_name, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f, delimiter='\t').writerow(header)

    df = pd.read_csv(f'{args.dataset}.tsv', sep='\t', encoding='utf-8')
    df_test = df[df['part'] == 'test'].reset_index(drop=True)
    end = len(df_test) if args.index_end < 0 else min(args.index_end, len(df_test))

    for i in range(args.index_start, end):
        prompt = df_test['question'][i]
        idx = df_test['new_id'][i]
        print(f'[{i}] {prompt}')

        text = asw.chat_prompt(tokenizer, prompt)
        model_inputs = tokenizer([text], return_tensors='pt').to(model.device)['input_ids']
        result = generate(model, tokenizer, model_inputs, args)

        with open(file_name, 'a+', newline='', encoding='utf-8') as f:
            csv.writer(f, delimiter='\t').writerow([
                idx, result['Token_num'], result['Entropy'], result['Time'],
                text, result['Text'], model_inputs[0].tolist(), result['Text_token'],
            ])
        gc.collect()
        torch.cuda.empty_cache()
