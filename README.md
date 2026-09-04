# Anchored Sliding Window (ASW)

Code for **"Anchored Sliding Window: Toward Robust and Imperceptible Linguistic
Steganography"** (ACL 2026).

LM-based linguistic steganography usually assumes the stegotext arrives
unmodified, so a single altered token disrupts every subsequent extraction.
Truncating the context window addresses that but costs text quality. ASW keeps
the window short while preserving quality by anchoring the prompt and a **bridge
context** in front of the latest `w` tokens:

```
context window  =  prompt  ||  bridge context  ||  latest w tokens
```

The bridge context is either a hand-written string (`[CONTEXT TRUNCATED]\n`) or a
short sequence of tunable embeddings trained by self-distillation to close the
gap against full-context inference.

---

## Install

```bash
pip install -r requirements.txt
```

Tested with Python 3.12, PyTorch 2.13 and transformers 5.14 on RTX 6000 Ada
(48 GB). The 7B experiments run in float32 and need roughly 32 GB of VRAM.

## Data

The three evaluation sets are checked in, so nothing needs downloading:

| file | source | rows |
| --- | --- | --- |
| `instinwild_en.tsv` | InstructionWild | 12,500 train / 1,000 val / 500 test |
| `databricks-dolly.tsv` | databricks-dolly-15k, `open_qa` | 500 test |
| `supernatural.tsv` | Super-NaturalInstructions | 500 test |

Rebuild them with `0.Prepare_dataset.py` and `0.Prepare_other_evaluation.py`.

---

## Pipeline

```
0.Prepare_dataset.py            build instinwild_en.tsv
0.Prepare_other_evaluation.py   build the two out-of-domain sets
        |
1.Prompt_tuning.py              train the soft bridge context  (Section 5)
1.Prompt_tuning_LoRA.py         same, with a LoRA adapter      (Appendix G)
1.Check_soft2hard.py            nearest vocabulary tokens to a trained bridge
        |
2.Generation_window.py          single-step KL per strategy    (Table 1, Table 6)
2.data_gather.py                summarise those into a table
        |
3.Generation_normal.py          covertexts, the metric reference
3.Embed_AC.py                   Alice: embed                   (Algorithms 1, 3)
4.Extract_AC.py                 Bob: extract                   (Algorithms 2, 4)
        |
3.Stega_evaluation.py           dPPL / BLEU / ROUGE-L / BERTScore / capacity  (Table 2)
5.Robustness.py                 unaffected-inference ratio     (Tables 3, 7, 8)
6.Steganalysis.py               BERT discriminator accuracy    (Appendix C)
```

`asw.py` holds the shared pieces: model loading, the context window itself, and
the arithmetic-coding interval arithmetic. Embedding and extraction both go
through it, which is what guarantees they see identical distributions — a single
differing logit costs the whole message.

### Strategies

Passed as `--strategy` (embedding) or `--strategies` (KL measurement):

| name | window |
| --- | --- |
| `Baseline` | latest `w` tokens only — the truncated window of Figure 2b |
| `Hard_0` | prompt + latest tokens, no bridge |
| `Hard_1` | prompt + `[CONTEXT TRUNCATED]\n` + latest tokens |
| `Hard_2` | prompt + `...\n` + latest tokens |
| `Hard_missing`, `Hard_removed` | the other two strings from Table 1 |
| `Rand_<k>` | prompt + `k` random tokens + latest tokens (Table 1 control) |
| `Text_<label>` | prompt + a bridge string you supply + latest tokens |
| `Soft_0` | prompt + untrained random bridge + latest tokens |
| `Soft_forward`, `Soft_reverse` | prompt + trained soft bridge + latest tokens |

`--context_window 0` selects robustness-unaware full-context generation
(Figure 2a).

---

## Running the experiments

### Table 1 — average single-step KL divergence

```bash
for w in 10 20 30 40 50; do
  python 2.Generation_window.py --context_window $w --index_end 500
done
python 2.data_gather.py --window_size 10,20,30,40,50
```

`--strategies` selects which context windows to compare in one pass; every
strategy is scored against the same sampled text, so the comparison is paired.

### Table 2 — text quality, imperceptibility, efficiency

Train the two soft bridge contexts:

```bash
python 1.Prompt_tuning.py --soft_prompt_length 8 --window_size 10 --KL_reverse 0
python 1.Prompt_tuning.py --soft_prompt_length 8 --window_size 10 --KL_reverse 1
```

Generate the covertexts the metrics are measured against, then one stegotext set
per strategy:

```bash
python 3.Generation_normal.py --dataset instinwild_en

python 3.Embed_AC.py --context_window 0 --strategy Hard_0       # full context
python 3.Embed_AC.py --strategy Baseline --context_window 10
python 3.Embed_AC.py --strategy Hard_0   --context_window 10
python 3.Embed_AC.py --strategy Hard_1   --context_window 10
python 3.Embed_AC.py --strategy Hard_2   --context_window 10
python 3.Embed_AC.py --strategy Soft_0   --context_window 10
python 3.Embed_AC.py --strategy Soft_forward --context_window 10 \
    --soft_bridge_path 1.soft_prompt/best_soft_length8_reverse0_Qwen2.5-7B-Instruct.pt
python 3.Embed_AC.py --strategy Soft_reverse --context_window 10 \
    --soft_bridge_path 1.soft_prompt/best_soft_length8_reverse1_Qwen2.5-7B-Instruct.pt
```

Score them:

```bash
python 3.Stega_evaluation.py --model Qwen2.5-7B-Instruct --dataset instinwild_en
python 6.Steganalysis.py \
    --stego_glob '3.Stega_data/AC_Qwen2.5-7B-Instruct_*instinwild_en.tsv' \
    --reference_file 3.Stega_data/Normal_Qwen2.5-7B-Instruct_instinwild_en.tsv
```

### Tables 3, 7, 8 — robustness

```bash
python 5.Robustness.py --stego_file 3.Stega_data/AC_..._Hard_1_..._instinwild_en.tsv
python 5.Robustness.py --stego_file ... --attack delete --mode empirical
python 5.Robustness.py --stego_file ... --attack insert --mode empirical
```

`--mode analytic` evaluates Proposition 1 in closed form (substitution only);
`--mode empirical` samples the attacked positions and covers all three attacks.
Both take the sequence lengths from a real stegotext file. ASW's dependency width
is `w`; the truncated-window baseline additionally reads the tokens its entropy
threshold is computed over, which `--winstega_entropy_window` sets (default `w`,
for an effective width of `2w`).

### Figures 5 and 6 — hyperparameter variants

```bash
for l in 1 4 8 16 32; do
  python 1.Prompt_tuning.py --soft_prompt_length $l --window_size 10
done
for w in 10 30 50; do
  python 3.Embed_AC.py --strategy Hard_1 --context_window $w
done
```

### Appendix C — steganalysis

`6.Steganalysis.py` fine-tunes a BERT discriminator on 500 (covertext, stegotext)
pairs per group, split 6:2:2, AdamW at 1e-5, batch size 128, 5 epochs. Pass
`--stego_file` for one group or `--stego_glob` to sweep several.

### Appendix D — model and dataset variants

```bash
python 2.Generation_window.py --language_model Qwen/Qwen2.5-14B-Instruct --context_window 10
python 2.Generation_window.py --language_model meta-llama/Llama-3.1-8B-Instruct --context_window 10
python 3.Embed_AC.py --dataset databricks-dolly --strategy Hard_1 --context_window 10
python 3.Embed_AC.py --dataset supernatural     --strategy Hard_1 --context_window 10
```

### Appendix G — LoRA case study

```bash
python 1.Prompt_tuning_LoRA.py --soft_prompt_length 8 --window_size 10
python 3.Embed_AC.py --strategy Soft_forward --lora 1 \
    --lora_path 1.lora_checkpoints/best_soft_length_8_reverse_0_Qwen2.5-7B-Instruct \
    --soft_bridge_path 1.lora_checkpoints/best_lora_soft_soft_length_8_reverse_0_Qwen2.5-7B-Instruct.pt
```

### A quick end-to-end run

`run_small_scale.sh` runs the whole pipeline — covertexts, every strategy,
extraction, and all three evaluations — at a reduced sample count, which is
useful for checking a change before committing a full run to it:

```bash
SAMPLES=50 TOKEN_MAX=256 ./run_small_scale.sh
```

---

## Extraction

`4.Extract_AC.py` is Bob's side. It reads the `.config.json` that the embedding
run writes, so the two sides cannot silently disagree about the window, the
bridge or the coder.

```bash
# clean channel
python 4.Extract_AC.py --stego_file 3.Stega_data/AC_..._Hard_1_..._instinwild_en.tsv

# under an active attack
python 4.Extract_AC.py --stego_file ... --attack substitute --attack_num 1
python 4.Extract_AC.py --stego_file ... --attack delete     --attack_num 2
```

Reported per file:

- **bit accuracy** — matching bits over the embedded payload
- **correct prefix** — how far into the message extraction stayed exact
- **exact extraction rate** — fraction of stegotexts recovered bit for bit

`--source token` replays Alice's exact tokens; `--source text` re-tokenizes the
stegotext, which is what Bob really receives and which also exercises the
detokenize/retokenize pipeline discussed in Appendix H.

`test_roundtrip.py` checks the same thing in memory, with no files and no trained
bridge required:

```bash
python test_roundtrip.py --num_samples 3
```

---

## Options

### Custom bridge strings

Any bridge string can be used without editing the code, through a `Text_<label>`
strategy:

```bash
python 2.Generation_window.py --strategies Hard_1,Text_a,Text_b \
    --bridge_texts '{"Text_a": "[...]\n", "Text_b": "[omitted]\n"}'

python 3.Embed_AC.py --strategy Text_a --context_window 10 --bridge_text '[...]
'
```

`4.Extract_AC.py` picks the string up from the run's `.config.json`, so the
decoder needs no extra flag.

### Soft bridge initialisation

`1.Prompt_tuning.py --initialization` selects how the tunable bridge starts:

| value | initial bridge |
| --- | --- |
| `0` | `torch.randn(l_bridge, hidden_size)` (default) |
| `1` | embeddings of a hard bridge string, set with `--init_bridge_text` |
| `2` | embeddings sampled from the vocabulary |

### Training

- `--loss both` interpolates forward and reverse KL, weighted by `--loss_alpha`.
- `--window_batch` sets how many window positions are scored per forward pass.
- `--max_train_samples` / `--max_val_samples` cap the data for a quick run.

Each epoch is written to `1.soft_prompt/`, and the best validation loss is also
saved under a stable `best_*.pt` name so the embedding scripts need not be told
which epoch won.

### Output files

Runs append, so `--index_start` / `--index_end` can shard one output across
GPUs. `--overwrite` restarts a file instead. `3.Embed_AC.py` compares the current
settings against the config recorded beside an existing file and stops rather
than mixing two configurations into one file.

---

## Notes

**Numerical agreement.** Extraction has to reproduce every logit the encoder saw,
so embedding and extraction must use the same dtype, the same device and the same
cache setting. `--dtype float32` is the default for that reason; `bfloat16` is
fine for KL measurement but will drop bits in a round trip.

**KV caching.** The prompt and the bridge are constant while one text is
generated, so their keys and values are computed once per sample and only the `w`
latest tokens are re-run; full-context generation extends its cache one token at
a time instead of re-reading the whole prefix. Caching is on by default,
`asw.verify_cache_equivalence` asserts it reproduces the plain forward pass, and
`--no_cache` turns it off.

**Vocabulary trimming.** `asw.load_model` shrinks the output embedding to the
tokenizer's real vocabulary (Qwen2.5: 151,936 → 151,665). The padding slots are
never trained, and leaving them in lets the coder spend secret bits on tokens
that cannot be tokenized back — the "unreachable tokens" of Appendix H.

**Soft bridge fingerprints.** `3.Embed_AC.py` records a hash of the bridge tensor
alongside the stegotexts and `4.Extract_AC.py` checks it, so a checkpoint that
changed between the two runs is reported rather than silently producing the wrong
bits. Point `--soft_bridge_path` at the checkpoint the stegotexts were made with;
a training job still writing to a shared `best_*.pt` will move underneath it.

---

## Citation

```bibtex
@inproceedings{yan-etal-2026-anchored,
    title = "Anchored Sliding Window: Toward Robust and Imperceptible Linguistic Steganography",
    author = "Yan, Ruiyi  and
      Meng, Shiao  and
      Murawaki, Yugo",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.44/",
    doi = "10.18653/v1/2026.acl-long.44",
    pages = "993--1012",
    ISBN = "979-8-89176-390-6"
}
```

## Ethics

Steganography protects private communication but can also hide malicious traffic.
This code is released for academic research and was evaluated on public datasets
only; use it lawfully.
