#!/usr/bin/env bash
# TED | 12b768 | evidence-gap recipe | 80 epochs
# Train-only entrypoint for the shareable package.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT" || {
  echo "Error: cannot cd to REPO_ROOT=$REPO_ROOT"
  exit 1
}

model_id="${MODEL_ID:-TED-12b768d12h-data10-noImp-wd5e5-proto8192-teacherTemp07-studentTemp01-mom992-koleo010decay-noLonLat-noMissingMaskEmbed-evidenceGapRatioPosGate-alpha01-v25-indepFS-shortsFullSeq-multiT-ratio025to075-crop4rand2-mixedBatch-remote-cXattnB}"
log_id="${LOG_ID:-$model_id}"
model="TED"
root_path="${ROOT_PATH:-${REPO_ROOT}/dataset/}"

train_data_ratio="${TRAIN_DATA_RATIO:-1.0}"
use_pretrained_imputator=0
mask_rate=0
seq_len="${SEQ_LEN:-732}"
sampling_stride="${SAMPLING_STRIDE:-732}"
seq_window_align="${SEQ_WINDOW_ALIGN:-start}"
enc_in=7
c_out=7
num_workers="${NUM_WORKERS:-16}"
train_epochs="${TRAIN_EPOCHS:-80}"
max_train_steps="${MAX_TRAIN_STEPS:-0}"
batch_size="${BATCH_SIZE:-256}"
patience=60
learning_rate="${LEARNING_RATE:-1e-5}"
min_lr="${MIN_LR:-1e-8}"

e_layers=12
d_model=768
n_heads=12
d_ff=3072

devices="${DEVICES:-0,1,2,3,4,5,6,7}"
nproc_per_node="${NPROC_PER_NODE:-8}"
use_multi_gpu="${USE_MULTI_GPU:-1}"
n_storage_tokens="${N_STORAGE_TOKENS:-4}"
n_cls_tokens="${N_CLS_TOKENS:-1}"

ibot_head_n_prototypes="${IBOT_HEAD_N_PROTOTYPES:-8192}"
ibot_head_hidden_dim="${IBOT_HEAD_HIDDEN_DIM:-1536}"
ibot_head_bottleneck_dim="${IBOT_HEAD_BOTTLENECK_DIM:-256}"
ibot_head_nlayers="${IBOT_HEAD_NLAYERS:-3}"
dino_head_n_prototypes="${DINO_HEAD_N_PROTOTYPES:-8192}"
dino_head_hidden_dim="${DINO_HEAD_HIDDEN_DIM:-1536}"
dino_head_bottleneck_dim="${DINO_HEAD_BOTTLENECK_DIM:-256}"
dino_head_nlayers="${DINO_HEAD_NLAYERS:-3}"

mask_rate_v1="${MASK_RATE_V1:-0.3}"
mask_rate_v2="${MASK_RATE_V2:-0.6}"
mask_sample_probability="${MASK_SAMPLE_PROBABILITY:-0.5}"
valid_sample_threshold="${VALID_SAMPLE_THRESHOLD:-0.05}"

if [ -z "${MASTER_PORT:-}" ]; then
  master_port=$(( (RANDOM % 10000) + 20000 ))
else
  master_port="${MASTER_PORT}"
fi

patch_len=3
stride=3

lambda_fft_align="${LAMBDA_FFT_ALIGN:-1.0}"
fft_align_epoch_start="${FFT_ALIGN_EPOCH_START:-300}"
fft_align_epoch_end="${FFT_ALIGN_EPOCH_END:--1}"
fft_align_lambda_active="${FFT_ALIGN_LAMBDA_ACTIVE:-$lambda_fft_align}"
fft_align_lambda_inactive="${FFT_ALIGN_LAMBDA_INACTIVE:-0.0}"
fft_align_min_valid_ratio="${FFT_ALIGN_MIN_VALID_RATIO:-0.0}"
lambda_cls_proto="${LAMBDA_CLS_PROTO:-1.0}"
lambda_patch_proto="${LAMBDA_PATCH_PROTO:-1.0}"
lambda_koleo="${LAMBDA_KOLEO:-0.1}"
lambda_temporal="${LAMBDA_TEMPORAL:-0}"
lambda_cls_cons="${LAMBDA_CLS_CONS:-0}"

block_mask_ratio="${BLOCK_MASK_RATIO:-0.8}"
imputed_patch_weight="${IMPUTED_PATCH_WEIGHT:-1.0}"

warmup_epochs="${WARMUP_EPOCHS:-10}"
weight_decay="${WEIGHT_DECAY:-0.00005}"
weight_decay_end="${WEIGHT_DECAY_END:-0.00005}"
freeze_last_layer_epochs="${FREEZE_LAST_LAYER_EPOCHS:-5}"
drop_path="${DROP_PATH:-0.15}"

geo_dropout_p="${GEO_DROPOUT_P:-0.0}"
lon_lat_n_fourier_freqs="${LON_LAT_N_FOURIER_FREQS:-4}"
missing_mask_embed_dropout="${MISSING_MASK_EMBED_DROPOUT:-0.0}"
use_lon_lat_embed="${USE_LON_LAT_EMBED:-0}"
use_missing_mask_embed="${USE_MISSING_MASK_EMBED:-0}"

teacher_temp="${TEACHER_TEMP:-0.07}"
warmup_teacher_temp="${WARMUP_TEACHER_TEMP:-0.04}"
warmup_teacher_temp_epochs="${WARMUP_TEACHER_TEMP_EPOCHS:-10}"
student_temp="${STUDENT_TEMP:-0.1}"
momentum_teacher="${MOMENTUM_TEACHER:-0.992}"
final_momentum_teacher="${FINAL_MOMENTUM_TEACHER:-1.0}"
schedule_trunc_extra=0.0
scaling_rule="${SCALING_RULE:-sqrt_wrt_1024}"
curriculum_strategy="${CURRICULUM_STRATEGY:-mixed_batch}"
mixed_batch_groups="${MIXED_BATCH_GROUPS:-2}"
mixed_batch_weight_by_valid_samples="${MIXED_BATCH_WEIGHT_BY_VALID_SAMPLES:-1}"

evidence_gap_distill="${EVIDENCE_GAP_DISTILL:-1}"
evidence_gap_condition="${EVIDENCE_GAP_CONDITION:-1}"
evidence_gap_condition_readout="${EVIDENCE_GAP_CONDITION_READOUT:-cond_xattn_bottleneck}"
evidence_gap_condition_alpha="${EVIDENCE_GAP_CONDITION_ALPHA:-0.1}"
evidence_gap_condition_view_embed_dim="${EVIDENCE_GAP_CONDITION_VIEW_EMBED_DIM:-8}"
evidence_gap_condition_scalar_embed_dim="${EVIDENCE_GAP_CONDITION_SCALAR_EMBED_DIM:-8}"
evidence_gap_condition_scalar_n_freqs="${EVIDENCE_GAP_CONDITION_SCALAR_N_FREQS:-4}"
evidence_gap_condition_hidden_dim="${EVIDENCE_GAP_CONDITION_HIDDEN_DIM:-0}"
evidence_gap_version="${EVIDENCE_GAP_VERSION:-v2.5}"
evidence_gap_teacher_lengths="${EVIDENCE_GAP_TEACHER_LENGTHS:-61,122,183,244,366,488,732}"
evidence_gap_student_ratio_min="${EVIDENCE_GAP_STUDENT_RATIO_MIN:-0.25}"
evidence_gap_student_ratio_max="${EVIDENCE_GAP_STUDENT_RATIO_MAX:-0.75}"
evidence_gap_student_aug="${EVIDENCE_GAP_STUDENT_AUG:-strong}"
evidence_gap_n_short_crop="${EVIDENCE_GAP_N_SHORT_CROP:-4}"
evidence_gap_n_short_random="${EVIDENCE_GAP_N_SHORT_RANDOM:-2}"
evidence_gap_short_outside_teacher="${EVIDENCE_GAP_SHORT_OUTSIDE_TEACHER:-0}"
evidence_gap_independent_fullseq="${EVIDENCE_GAP_INDEPENDENT_FULLSEQ:-1}"
evidence_gap_same_short_multi_teacher_prob="${EVIDENCE_GAP_SAME_SHORT_MULTI_TEACHER_PROB:-0.0}"
evidence_gap_same_short_multi_teacher_count="${EVIDENCE_GAP_SAME_SHORT_MULTI_TEACHER_COUNT:-1}"
evidence_gap_same_short_anchor_from_crop_only="${EVIDENCE_GAP_SAME_SHORT_ANCHOR_FROM_CROP_ONLY:-1}"
evidence_gap_dual_teacher_cross="${EVIDENCE_GAP_DUAL_TEACHER_CROSS:-0}"
evidence_gap_dual_teacher_short_per_side="${EVIDENCE_GAP_DUAL_TEACHER_SHORT_PER_SIDE:-4}"
evidence_gap_dual_teacher_patch_cross="${EVIDENCE_GAP_DUAL_TEACHER_PATCH_CROSS:-0}"

resume_checkpoint="${RESUME_CHECKPOINT:-}"

compile_backbone="${COMPILE_BACKBONE:-0}"
ddp_find_unused_parameters="${DDP_FIND_UNUSED_PARAMETERS:-1}"
train_step_tensor_item_interval="${TRAIN_STEP_TENSOR_ITEM_INTERVAL:-25}"
train_step_console_log_enable="${TRAIN_STEP_CONSOLE_LOG_ENABLE:-0}"

export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES="$devices"

mkdir -p logs

extra_args=()
if [ "$compile_backbone" = "1" ]; then
  extra_args+=(--compile_backbone)
fi
if [ "$max_train_steps" != "0" ]; then
  extra_args+=(--max_train_steps "$max_train_steps")
fi
if [ -n "$resume_checkpoint" ]; then
  extra_args+=(--resume_checkpoint "$resume_checkpoint")
fi

echo "[TED 12b768 data10 cXattnB indepFS rr e80] REPO_ROOT=$REPO_ROOT"
echo "  model_id=$model_id"
echo "  log_id=$log_id"
echo "  arch=12b768d12h train_data_ratio=$train_data_ratio train_epochs=$train_epochs batch_size=$batch_size/GPU"
echo "  seq_len=$seq_len sampling_stride=$sampling_stride seq_window_align=$seq_window_align"
echo "  valid_sample_threshold=$valid_sample_threshold lambda_koleo=$lambda_koleo"
echo "  evidence_gap_condition=$evidence_gap_condition version=$evidence_gap_version readout=$evidence_gap_condition_readout"
echo "  teacher_lengths=$evidence_gap_teacher_lengths short_ratio=${evidence_gap_student_ratio_min}-${evidence_gap_student_ratio_max}"
echo "  crop/random=$evidence_gap_n_short_crop/$evidence_gap_n_short_random independent_fullseq=$evidence_gap_independent_fullseq"
echo "  dual_teacher_cross=$evidence_gap_dual_teacher_cross curriculum=$curriculum_strategy"

torchrun --nproc_per_node="$nproc_per_node" --master_port="$master_port" train.py \
  --model_id "$model_id" \
  --model "$model" \
  --root_path "$root_path" \
  --enc_in "$enc_in" \
  --c_out "$c_out" \
  --mask_rate "$mask_rate" \
  --mask_rate_v1 "$mask_rate_v1" \
  --mask_rate_v2 "$mask_rate_v2" \
  --mask_sample_probability "$mask_sample_probability" \
  --train_data_ratio "$train_data_ratio" \
  --use_pretrained_imputator "$use_pretrained_imputator" \
  --seq_len "$seq_len" \
  --sampling_stride "$sampling_stride" \
  --seq_window_align "$seq_window_align" \
  --patch_len "$patch_len" \
  --stride "$stride" \
  --train_epochs "$train_epochs" \
  --batch_size "$batch_size" \
  --patience "$patience" \
  --num_workers "$num_workers" \
  --learning_rate "$learning_rate" \
  --d_model "$d_model" \
  --n_heads "$n_heads" \
  --e_layers "$e_layers" \
  --d_ff "$d_ff" \
  --n_storage_tokens "$n_storage_tokens" \
  --n_cls_tokens "$n_cls_tokens" \
  --ibot_head_n_prototypes "$ibot_head_n_prototypes" \
  --ibot_head_hidden_dim "$ibot_head_hidden_dim" \
  --ibot_head_bottleneck_dim "$ibot_head_bottleneck_dim" \
  --ibot_head_nlayers "$ibot_head_nlayers" \
  --dino_head_n_prototypes "$dino_head_n_prototypes" \
  --dino_head_hidden_dim "$dino_head_hidden_dim" \
  --dino_head_bottleneck_dim "$dino_head_bottleneck_dim" \
  --dino_head_nlayers "$dino_head_nlayers" \
  --evidence_gap_distill "$evidence_gap_distill" \
  --evidence_gap_condition "$evidence_gap_condition" \
  --evidence_gap_condition_readout "$evidence_gap_condition_readout" \
  --evidence_gap_condition_alpha "$evidence_gap_condition_alpha" \
  --evidence_gap_condition_view_embed_dim "$evidence_gap_condition_view_embed_dim" \
  --evidence_gap_condition_scalar_embed_dim "$evidence_gap_condition_scalar_embed_dim" \
  --evidence_gap_condition_scalar_n_freqs "$evidence_gap_condition_scalar_n_freqs" \
  --evidence_gap_condition_hidden_dim "$evidence_gap_condition_hidden_dim" \
  --evidence_gap_version "$evidence_gap_version" \
  --evidence_gap_teacher_lengths "$evidence_gap_teacher_lengths" \
  --evidence_gap_student_ratio_min "$evidence_gap_student_ratio_min" \
  --evidence_gap_student_ratio_max "$evidence_gap_student_ratio_max" \
  --evidence_gap_student_aug "$evidence_gap_student_aug" \
  --evidence_gap_n_short_crop "$evidence_gap_n_short_crop" \
  --evidence_gap_n_short_random "$evidence_gap_n_short_random" \
  --evidence_gap_short_outside_teacher "$evidence_gap_short_outside_teacher" \
  --evidence_gap_independent_fullseq "$evidence_gap_independent_fullseq" \
  --evidence_gap_same_short_multi_teacher_prob "$evidence_gap_same_short_multi_teacher_prob" \
  --evidence_gap_same_short_multi_teacher_count "$evidence_gap_same_short_multi_teacher_count" \
  --evidence_gap_same_short_anchor_from_crop_only "$evidence_gap_same_short_anchor_from_crop_only" \
  --evidence_gap_dual_teacher_cross "$evidence_gap_dual_teacher_cross" \
  --evidence_gap_dual_teacher_short_per_side "$evidence_gap_dual_teacher_short_per_side" \
  --evidence_gap_dual_teacher_patch_cross "$evidence_gap_dual_teacher_patch_cross" \
  --lambda_cls_proto "$lambda_cls_proto" \
  --lambda_patch_proto "$lambda_patch_proto" \
  --lambda_koleo "$lambda_koleo" \
  --lambda_fft_align "$lambda_fft_align" \
  --fft_align_epoch_start "$fft_align_epoch_start" \
  --fft_align_epoch_end "$fft_align_epoch_end" \
  --fft_align_lambda_active "$fft_align_lambda_active" \
  --fft_align_lambda_inactive "$fft_align_lambda_inactive" \
  --fft_align_min_valid_ratio "$fft_align_min_valid_ratio" \
  --lambda_temporal "$lambda_temporal" \
  --lambda_cls_cons "$lambda_cls_cons" \
  --block_mask_ratio "$block_mask_ratio" \
  --imputed_patch_weight "$imputed_patch_weight" \
  --valid_sample_threshold "$valid_sample_threshold" \
  --use_multi_gpu "$use_multi_gpu" \
  --devices "$devices" \
  --scheduler_version cosine \
  --warmup_epochs "$warmup_epochs" \
  --min_lr "$min_lr" \
  --weight_decay "$weight_decay" \
  --weight_decay_end "$weight_decay_end" \
  --freeze_last_layer_epochs "$freeze_last_layer_epochs" \
  --teacher_temp "$teacher_temp" \
  --warmup_teacher_temp "$warmup_teacher_temp" \
  --warmup_teacher_temp_epochs "$warmup_teacher_temp_epochs" \
  --student_temp "$student_temp" \
  --momentum_teacher "$momentum_teacher" \
  --final_momentum_teacher "$final_momentum_teacher" \
  --schedule_trunc_extra "$schedule_trunc_extra" \
  --scaling_rule "$scaling_rule" \
  --mixed_batch_groups "$mixed_batch_groups" \
  --mixed_batch_weight_by_valid_samples "$mixed_batch_weight_by_valid_samples" \
  --curriculum_strategy "$curriculum_strategy" \
  --drop_path "$drop_path" \
  --geo_dropout_p "$geo_dropout_p" \
  --lon_lat_n_fourier_freqs "$lon_lat_n_fourier_freqs" \
  --missing_mask_embed_dropout "$missing_mask_embed_dropout" \
  --use_lon_lat_embed "$use_lon_lat_embed" \
  --use_missing_mask_embed "$use_missing_mask_embed" \
  --train_step_log_enable 1 \
  --train_step_tensor_item_interval "$train_step_tensor_item_interval" \
  --train_step_console_log_enable "$train_step_console_log_enable" \
  --progress_log_interval 10 \
  --ddp_find_unused_parameters "$ddp_find_unused_parameters" \
  --use_amp 1 \
  "${extra_args[@]}" \
  > "logs/${log_id}.log" 2>&1
