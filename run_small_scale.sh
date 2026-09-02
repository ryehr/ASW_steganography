#!/usr/bin/env bash
# Small-scale end-to-end reproduction of Table 2 and its companions.
#
# Runs the whole pipeline -- covertexts, every embedding strategy, a round-trip
# extraction check, the quality metrics, steganalysis and the robustness table --
# at a sample count that finishes in about an hour on one GPU instead of the ~15
# hours the paper's 500 samples take.  The absolute numbers therefore carry more
# noise than the paper's; the ordering between strategies is what this is for.
#
# Usage:
#   ./run_small_scale.sh                          # 50 samples, 256 tokens
#   SAMPLES=100 TOKEN_MAX=512 ./run_small_scale.sh
#   SOFT_BRIDGE=1.soft_prompt/best_...pt ./run_small_scale.sh
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
STEM=${MODEL##*/}
DATASET=${DATASET:-instinwild_en}
SAMPLES=${SAMPLES:-50}
TOKEN_MAX=${TOKEN_MAX:-256}
WINDOW=${WINDOW:-10}
PYTHON=${PYTHON:-python}

# Trained soft bridge context; leave empty to skip the Soft_forward row.
SOFT_BRIDGE=${SOFT_BRIDGE:-}
# The bridge string that came out of the 2.Generation_window.py sweep.
BEST_BRIDGE=${BEST_BRIDGE:-'[The earlier part of this answer has been omitted. Continue seamlessly.]
'}

common="--language_model $MODEL --dataset $DATASET --token_max $TOKEN_MAX --index_end $SAMPLES"
embed_common="$common --overwrite"

covertexts="3.Stega_data/Normal_${STEM}_${DATASET}.tsv"
if [ "${REUSE_COVERTEXTS:-0}" = "1" ] && [ -f "$covertexts" ]; then
    echo "=== covertexts: reusing $covertexts ==="
else
    echo "=== covertexts (the metric reference) ==="
    $PYTHON 3.Generation_normal.py $common --overwrite
fi

echo
echo "=== embedding ==="
run_embed () {
    echo "--- $* ---"
    $PYTHON 3.Embed_AC.py $embed_common "$@"
}

run_embed --context_window 0 --strategy Hard_0                # Figure 2a, full context
run_embed --context_window "$WINDOW" --strategy Baseline      # Figure 2b, WinStega-style
run_embed --context_window "$WINDOW" --strategy Hard_0
run_embed --context_window "$WINDOW" --strategy Hard_1
run_embed --context_window "$WINDOW" --strategy Hard_2
run_embed --context_window "$WINDOW" --strategy Soft_0
run_embed --context_window "$WINDOW" --strategy Text_best --bridge_text "$BEST_BRIDGE"

if [ -n "$SOFT_BRIDGE" ]; then
    run_embed --context_window "$WINDOW" --strategy Soft_forward --soft_bridge_path "$SOFT_BRIDGE"
fi

echo
echo "=== round-trip extraction (must be lossless) ==="
for f in 3.Stega_data/AC_${STEM}_*${DATASET}.tsv; do
    echo "--- $(basename "$f") ---"
    $PYTHON 4.Extract_AC.py --stego_file "$f" | tail -4
done

echo
echo "=== extraction under a one-token substitution ==="
for f in 3.Stega_data/AC_${STEM}_*${DATASET}.tsv; do
    echo "--- $(basename "$f") ---"
    $PYTHON 4.Extract_AC.py --stego_file "$f" --attack substitute --attack_num 1 | tail -4
done

echo
echo "=== Table 2: text quality, capacity, runtime ==="
$PYTHON 3.Stega_evaluation.py --model "$STEM" --dataset "$DATASET"

echo
echo "=== Table 2: steganalysis accuracy ==="
$PYTHON 6.Steganalysis.py \
    --stego_glob "3.Stega_data/AC_${STEM}_*${DATASET}.tsv" \
    --reference_file "3.Stega_data/Normal_${STEM}_${DATASET}.tsv"

echo
echo "=== Table 3: unaffected-inference ratio ==="
$PYTHON 5.Robustness.py \
    --stego_file "3.Stega_data/AC_${STEM}_window_${WINDOW}_strategy_Hard_1_lora_0_${DATASET}.tsv"

echo
echo "done"
