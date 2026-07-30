#!/usr/bin/env bash
# Downstream classification eval (paper protocol: spatial 1280m + cap300 linear probe).
#
# CHECKPOINT may be a zoo alias, a pretrained/ folder, or a weight file:
#   CHECKPOINT=ted MODEL=TED bash scripts/eval_downstream.sh
#   CHECKPOINT=msm-hls-12b768 bash scripts/eval_downstream.sh
#   CHECKPOINT=pretrained/ntp-hls-12b768/pytorch_model.bin MODEL=NTP ...
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

CHECKPOINT="${CHECKPOINT:-ted}"
MODEL="${MODEL:-}"
MODEL_ID="${MODEL_ID:-}"
EVAL_MODE="${EVAL_MODE:-linear}"  # linear | knn
DATASETS="${DATASETS:-all}"
DOWNSTREAM_DATA_ROOT="${DOWNSTREAM_DATA_ROOT:-${REPO_ROOT}/dataset/downstream/classification}"
MAX_TRAIN_PER_CLASS="${MAX_TRAIN_PER_CLASS:-300}"
SPATIAL_BLOCK_M="${SPATIAL_BLOCK_M:-1280}"
# Tag CSV by checkpoint token (alias or stem)
CKPT_TAG="$(basename "${CHECKPOINT}" | sed 's/\.[^.]*$//')"
OUTPUT_CSV="${OUTPUT_CSV:-${REPO_ROOT}/results/downstream_${CKPT_TAG}_${EVAL_MODE}_cap${MAX_TRAIN_PER_CLASS}.csv}"
DEVICE="${DEVICE:-cuda:0}"

extra=()
if [ -n "$MODEL" ]; then
  extra+=(--model "$MODEL")
fi
if [ -n "$MODEL_ID" ]; then
  extra+=(--model_id "$MODEL_ID")
fi

python eval_downstream.py \
  --checkpoint "$CHECKPOINT" \
  --eval_mode "$EVAL_MODE" \
  --datasets "$DATASETS" \
  --downstream_data_root "$DOWNSTREAM_DATA_ROOT" \
  --max_train_per_class "$MAX_TRAIN_PER_CLASS" \
  --spatial_block_m "$SPATIAL_BLOCK_M" \
  --output_csv "$OUTPUT_CSV" \
  --device "$DEVICE" \
  "${extra[@]}"
