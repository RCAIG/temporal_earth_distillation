#!/usr/bin/env bash
# NTP | 12b384 | data ratio 0.2 | 50 epochs
# Train-only entrypoint for the shareable package.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

model_id="${MODEL_ID:-NTP-12b384d6h-data02-noImp-noLonLat-noMissingMaskEmbed-vs05-ep50-ablation}"
log_id="${LOG_ID:-$model_id}"
model="NTP"
root_path="${ROOT_PATH:-${REPO_ROOT}/dataset/}"

train_data_ratio="${TRAIN_DATA_RATIO:-0.2}"
valid_sample_threshold="${VALID_SAMPLE_THRESHOLD:-0.05}"
mask_rate=0
seq_len=732
sampling_stride=732
enc_in=7
c_out=7
num_workers="${NUM_WORKERS:-16}"
train_epochs="${TRAIN_EPOCHS:-50}"
batch_size="${BATCH_SIZE:-256}"
patience=60
learning_rate="${LEARNING_RATE:-1e-5}"
min_lr="${MIN_LR:-1e-8}"

e_layers=12
d_model=384
n_heads=6
d_ff=1536

devices="${DEVICES:-0,1,2,3,4,5,6,7}"
nproc_per_node="${NPROC_PER_NODE:-8}"
use_multi_gpu="${USE_MULTI_GPU:-1}"

if [ -z "${MASTER_PORT:-}" ]; then
  master_port=$(( (RANDOM % 10000) + 20000 ))
else
  master_port="${MASTER_PORT}"
fi

patch_len=3
stride=3
curriculum_strategy=none
mixed_batch_groups=1

warmup_epochs="${WARMUP_EPOCHS:-10}"
weight_decay="${WEIGHT_DECAY:-0.00005}"
weight_decay_end="${WEIGHT_DECAY_END:-0.00005}"
scaling_rule="${SCALING_RULE:-sqrt_wrt_1024}"
drop_path="${DROP_PATH:-0.15}"
geo_dropout_p="${GEO_DROPOUT_P:-0.0}"
missing_mask_embed_dropout="${MISSING_MASK_EMBED_DROPOUT:-0.0}"
use_missing_mask_embed="${USE_MISSING_MASK_EMBED:-0}"
use_lon_lat_embed="${USE_LON_LAT_EMBED:-0}"
lon_lat_n_fourier_freqs="${LON_LAT_N_FOURIER_FREQS:-4}"

export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES="$devices"

mkdir -p logs

echo "[NTP 12b384 data02 ep50] model_id=$model_id ratio=$train_data_ratio epochs=$train_epochs filter=$valid_sample_threshold use_missing_mask_embed=$use_missing_mask_embed"

_common_args=(
  --model_id "$model_id"
  --model "$model"
  --freq rs
  --root_path "$root_path"
  --enc_in "$enc_in"
  --c_out "$c_out"
  --mask_rate "$mask_rate"
  --train_data_ratio "$train_data_ratio"
  --valid_sample_threshold "$valid_sample_threshold"
  --seq_len "$seq_len"
  --sampling_stride "$sampling_stride"
  --patch_len "$patch_len"
  --stride "$stride"
  --train_epochs "$train_epochs"
  --batch_size "$batch_size"
  --patience "$patience"
  --num_workers "$num_workers"
  --learning_rate "$learning_rate"
  --d_model "$d_model"
  --n_heads "$n_heads"
  --e_layers "$e_layers"
  --d_ff "$d_ff"
  --scheduler_version cosine
  --warmup_epochs "$warmup_epochs"
  --min_lr "$min_lr"
  --weight_decay "$weight_decay"
  --weight_decay_end "$weight_decay_end"
  --scaling_rule "$scaling_rule"
  --drop_path "$drop_path"
  --geo_dropout_p "$geo_dropout_p"
  --missing_mask_embed_dropout "$missing_mask_embed_dropout"
  --use_missing_mask_embed "$use_missing_mask_embed"
  --use_lon_lat_embed "$use_lon_lat_embed"
  --lon_lat_n_fourier_freqs "$lon_lat_n_fourier_freqs"
  --curriculum_strategy "$curriculum_strategy"
  --mixed_batch_groups "$mixed_batch_groups"
  --ddp_find_unused_parameters 1
  --use_amp 1
  --no_attn_checkpoint
)

if [ "$use_multi_gpu" = "1" ] && [ "$nproc_per_node" -gt 1 ]; then
  torchrun --nproc_per_node="$nproc_per_node" --master_port="$master_port" train.py \
    "${_common_args[@]}" \
    --use_multi_gpu "$use_multi_gpu" \
    --devices "$devices" \
    > "logs/${log_id}.log" 2>&1
else
  export CUDA_VISIBLE_DEVICES="${devices%%,*}"
  python train.py \
    "${_common_args[@]}" \
    --use_multi_gpu 0 \
    --gpu 0 \
    > "logs/${log_id}.log" 2>&1
fi

echo "Done. Log: logs/${log_id}.log"
