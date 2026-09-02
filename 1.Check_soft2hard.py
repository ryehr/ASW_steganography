"""Read a trained soft bridge context back as text.

Projects each soft token onto its nearest vocabulary embedding by cosine
similarity, which gives a rough sense of what the tuned bridge ended up
"saying".  Diagnostic only: the bridge lives in the continuous embedding space
and is not equivalent to the tokens printed here.

Example:
    python 1.Check_soft2hard.py \
        --soft_bridge_path 1.soft_prompt/best_soft_length8_reverse0_Qwen2.5-7B-Instruct.pt
    python 1.Check_soft2hard.py --hard_bridge '[CONTEXT TRUNCATED]\n'   # sanity check
"""

import argparse

import torch
import torch.nn.functional as F

import asw

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--language_model', default='Qwen/Qwen2.5-7B-Instruct', type=str)
    parser.add_argument('--soft_bridge_path', default=None, type=str)
    parser.add_argument('--hard_bridge', default=None, type=str,
                        help='project this string instead; the round trip should '
                             'return the string itself')
    parser.add_argument('--top_k', default=5, type=int,
                        help='nearest neighbours to list per soft token')
    args = parser.parse_args()

    if not args.soft_bridge_path and not args.hard_bridge:
        parser.error('pass --soft_bridge_path or --hard_bridge')

    model, tokenizer = asw.load_model(args.language_model, trim_vocab=False)
    embedding_matrix = model.get_input_embeddings().weight

    if args.hard_bridge:
        ids = tokenizer(args.hard_bridge, add_special_tokens=False,
                        return_tensors='pt')['input_ids'].to(model.device)
        soft_bridge = model.get_input_embeddings()(ids).squeeze(0)
    else:
        soft_bridge = asw.load_soft_bridge(args.soft_bridge_path, model)

    print(f'bridge length {soft_bridge.shape[0]}, '
          f'mean row norm {soft_bridge.norm(dim=1).mean():.3f} '
          f'(real token embeddings average '
          f'{embedding_matrix.norm(dim=1).mean():.3f})')

    similarity = torch.matmul(F.normalize(soft_bridge.float(), dim=1),
                              F.normalize(embedding_matrix.float(), dim=1).T)
    scores, ids = similarity.topk(args.top_k, dim=1)

    print('\nnearest vocabulary tokens per position:')
    for position in range(soft_bridge.shape[0]):
        neighbours = ', '.join(
            f'{tokenizer.decode([i])!r} ({s:.3f})'
            for i, s in zip(ids[position].tolist(), scores[position].tolist()))
        print(f'  {position}: {neighbours}')

    print('\npseudo-text (top-1 per position):')
    print(tokenizer.decode(ids[:, 0].tolist(), clean_up_tokenization_spaces=True))
