import argparse
import os
import torch
import torch.distributed as dist  # required import
from engine.ssl_trainer import SSLTrainer
from utils.tools import clamp_experiment_setting_for_checkpoint
import random
import numpy as np

# fix random seed
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# GPU performance settings
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

parser = argparse.ArgumentParser(description='Temporal Earth Distillation (TED)')

# basic config

parser.add_argument('--model_id', type=str, default='TED', help='model id')
parser.add_argument('--model', type=str, default='TED',
                    help='model name')
parser.add_argument('--pretrain_model', type=str, default=None, help='use pretrain model')

# data loader (HLS pretraining only)
parser.add_argument(
    '--root_path',
    type=str,
    default='./dataset/',
    help='directory containing train.nc / train.h5 (and optional val.nc)',
)
parser.add_argument(
    '--freq',
    type=str,
    default='rs',
    help="time-feature mode for HLS stamps (only 'rs' / remote-sensing is supported)",
)
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument(
    '--resume_checkpoint',
    type=str,
    default=None,
    help='resume training state path (file or directory). Supports full train-state resume including optimizer/scaler/RNG.',
)

# train data subsampling (for SSLTrainer / train.py training ablations)
parser.add_argument(
    '--train_data_ratio',
    type=float,
    default=1.0,
    help='ratio of training data to use (0 < ratio <= 1). Only affects flag=train in data provider.',
)

parser.add_argument('--seq_len', type=int, default=732, help='input sequence length (HLS window)')
parser.add_argument(
    '--seq_window_align',
    type=str,
    default='start',
    choices=['start', 'end', 'random_year'],
    help=(
        "HLS pretrain window align: 'start' (s_begin=w_idx*stride), 'end' (last seq_len steps), "
        "or 'random_year' (1 sample/loc; Uniform year block k*seq_len:(k+1)*seq_len)."
    ),
)

# model define

parser.add_argument('--enc_in', type=int, default=7, help='encoder input size (HLS bands)')

parser.add_argument('--c_out', type=int, default=7, help='output size (HLS bands)')
parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--e_layers', type=int, default=6, help='num of encoder layers')

parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')

parser.add_argument('--factor', type=int, default=1, help='attn factor')

parser.add_argument('--dropout', type=float, default=0.0, help='dropout (MHA/FFN); TED backbone default 0')
parser.add_argument(
    '--drop_path', '--drop_depth',
    type=float,
    default=0.1,
    help='stochastic depth prob (per-layer per-sample skip residual; same as --drop_depth; default 0.1 stable HLS; use 0.2 for stronger reg)',
)
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', default=True, action='store_true', help='whether to output attention in encoder')
parser.add_argument(
    '--curriculum_strategy',
    type=str,
    default='mixed_batch',
    help=(
        "curriculum / batch: "
        "'mixed_batch' splits each GPU's batch into micro-batches, each forward draws its own random "
        "sequence length and (TED) local-view patch ratio; backward uses ssl_num_valid_samples "
        "weighting when --mixed_batch_weight_by_valid_samples 1; grad accumulation then one optimizer step. "
        "'none'/'fast'/... single forward per GPU per step (faster; one random length per GPU per step)."
    ),
)
parser.add_argument('--curriculum_length_jitter', type=int, default=61,
                    help='mixed_batch start jitter radius (steps); ±jitter after year-aligned length; 0 off')
parser.add_argument('--curriculum_jitter_probability', type=float, default=0.5,
                    help='start jitter probability (0-1), default 0.5')
parser.add_argument(
    '--disable_view_augmentation',
    action='store_true',
    help='disable amp scale/shift/additive noise (and student global channel mask) on teacher/student views',
)
parser.add_argument('--global_shift_steps', type=int, default=0,
                    help='offset steps of second global view vs anchor window; exclusive with global_shift_ratio.'
                         'if steps/ratio both 0, on trigger sample another independent same-length window')
parser.add_argument('--global_shift_jitter_steps', type=int, default=0,
                    help='jitter radius on base shift steps (±); 0 disables step jitter (use jitter_ratio)')
parser.add_argument('--global_shift_ratio', type=float, default=0.0,
                    help='base offset of second global view as fraction of seq len; >0 overrides global_shift_steps')
parser.add_argument('--global_shift_jitter_ratio', type=float, default=0.0,
                    help='shift jitter as fraction of seq len; 0 uses only global_shift_jitter_steps')
parser.add_argument('--global_shift_probability', type=float, default=1.0,
                    help='prob (0-1) second global uses different window each forward; 0 shares anchor')
parser.add_argument('--global_shift_min_overlap_ratio', type=float, default=0.6,
                    help='min overlap ratio of two global windows (vs window len); default 0.6 same semantic segment')
parser.add_argument(
    '--global_shift_mode',
    type=str,
    default='base_jitter',
    choices=['uniform', 'base_jitter'],
    help='uniform: sample around base offset; base_jitter: base plus jitter. Only when ratio/steps>0',
)
parser.add_argument('--mixed_batch_groups', type=int, default=2,
                    help='curriculum_strategy=mixed_batch sub-batches per GPU, default 2 (speed vs variable length)')
parser.add_argument(
    '--mixed_batch_local_patch_divisors',
    type=str,
    default='8,6,4',
    help='mixed_batch local view patch fraction: local_token_num=max(1,N_patches//d),'
         '~1/d of full-window patches; comma-separated, same length as mixed_batch_local_patch_divisor_probs',
)
parser.add_argument(
    '--mixed_batch_local_patch_divisor_probs',
    type=str,
    default='0.5,0.25,0.25',
    help='Sampling probability of the same length as divisors (automatic normalize); only effective in curriculum_strategy=mixed_batch and train when',
)
parser.add_argument(
    '--mixed_batch_weight_by_valid_samples',
    type=int,
    default=1,
    help='1: mixed_batch backward weights by count above valid_sample_threshold,'
         'Then normalize according to the total number of valid samples before the optimizer step (align the samplemean gradient of "full batch forward");'
         '0: legacy equal 1/G sub-batch weighting',
)
parser.add_argument('--imputator_mode', type=str, default='full',
                    help="imputator usage policy: "
                         "'full' (teacher always imputator, student half), "
                         "'recon_only' (Teacher/Student all use raw,Only in reconstruction target use imputator), "
                         "'mixed_teacher' (teacher/student mix imputator/raw; patch filter kept)")
parser.add_argument(
    '--imputator_segment_stride',
    type=int,
    default=244,
    help='sliding-window stride for long sequences when using a pretrained imputator (e.g. 244 ≈ two years); window length follows the imputator (default 366)',
)
parser.add_argument(
    '--use_lon_lat_embed',
    type=int,
    default=1,
    help='1: multi-freq sin/cos lon/lat embed (WGS84 deg); 0: off for no geo coords',
)
parser.add_argument(
    '--lon_lat_n_fourier_freqs',
    type=int,
    default=4,
    help='Fourier freq count for lon/lat; embed dim=4*count then Linear to d_model (4 enough for climate zones)',
)
parser.add_argument(
    '--geo_dropout_p',
    type=float,
    default=0.5,
    help='train-time lon/lat embed dropout (CFG-style); 0=always; default 0.5 reduces geo shortcut',
)
parser.add_argument(
    '--missing_mask_embed_dropout',
    type=float,
    default=0.5,
    help='train-time whole-sample missing_mask embed dropout; 0=off; default 0.5 reduces missing shortcut',
)
parser.add_argument(
    '--use_missing_mask_embed',
    type=int,
    default=1,
    help='1: add missing-mask linear embedding to patches; 0: disable (weights remain load-compatible)',
)
parser.add_argument('--n_storage_tokens', type=int, default=0, help='number of storage tokens')
parser.add_argument(
    '--n_cls_tokens',
    type=int,
    default=1,
    help='CLS token count; >1 uses sigmoid-gated fusion before the sequence-state head',
)
parser.add_argument('--evidence_gap_distill', type=int, default=0, help='1: use single-teacher evidence-gap CLS distillation (TED); 0: off (default for MSM/NTP; TED scripts set this to 1)')
parser.add_argument('--evidence_gap_teacher_lengths', type=str, default='61,122,183,244,366,488,732', help='comma-separated teacher window lengths for evidence-gap distillation')
parser.add_argument('--evidence_gap_student_ratio_min', type=float, default=0.1, help='minimum student/teacher patch-token ratio in evidence-gap distillation')
parser.add_argument('--evidence_gap_student_ratio_max', type=float, default=0.9, help='maximum student/teacher patch-token ratio in evidence-gap distillation')
parser.add_argument('--evidence_gap_cls_bins', type=str, default='5,10,21,41,82', help='comma-separated token-gap upper bounds; N bounds create N+1 CLS groups')
parser.add_argument('--evidence_gap_condition', type=int, default=0, help='1: relation-conditioned CLS readout for short evidence-gap views (TED); 0: off by default')
parser.add_argument('--evidence_gap_condition_alpha', type=float, default=0.1, help='residual scale for relation-conditioned evidence-gap readout')
parser.add_argument('--evidence_gap_condition_view_embed_dim', type=int, default=8, help='view-type embedding dimension for relation condition')
parser.add_argument('--evidence_gap_condition_scalar_embed_dim', type=int, default=8, help='embedding dimension for each scalar relation condition')
parser.add_argument('--evidence_gap_condition_scalar_n_freqs', type=int, default=4, help='number of Fourier frequencies for scalar relation condition embeddings')
parser.add_argument('--evidence_gap_condition_hidden_dim', type=int, default=0, help='hidden dimension of relation condition adapter/MLP; 0 uses a small default')
parser.add_argument(
    '--evidence_gap_condition_readout',
    type=str,
    default='adapter',
    choices=[
        'adapter', 'direction', 'gate', 'film', 'cond_mlp', 'cond_sum_mlp', 'cond_film_mlp',
        'cond_res_mlp', 'cond_res_film_mlp', 'cond_gate_bottleneck', 'cond_mul_bottleneck',
        'cond_xattn_bottleneck', 'cond_blend_mlp',
    ],
    help='short-view condition readout; cond_* bottleneck variants reuse dino_head.mlp base',
)
parser.add_argument(
    '--evidence_gap_condition_drop_p',
    type=float,
    default=0.0,
    help='per-sample prob to skip condition readout on short views and use raw z->dino_head (train only); 0 disables',
)
parser.add_argument('--evidence_gap_student_aug', type=str, default='strong', choices=['none', 'weak', 'weak_local', 'strong'], help='student augmentation mode for evidence-gap distillation; strong is noise + channel masking')
parser.add_argument('--evidence_gap_n_short_crop', type=int, default=4, help='number of short contiguous crop students in evidence-gap distillation')
parser.add_argument('--evidence_gap_n_short_random', type=int, default=2, help='number of short random-token students in evidence-gap distillation')
parser.add_argument('--evidence_gap_short_outside_teacher', type=int, default=0, help='1: sample crop/random shorts from timeline regions outside the teacher window (global stays inside teacher)')
parser.add_argument('--evidence_gap_independent_fullseq', type=int, default=0, help='1: crop/random independently random-sample on full timeline; global stays in teacher window')
parser.add_argument('--evidence_gap_same_short_multi_teacher_prob', type=float, default=0.25, help='train-time probability to add same-short multi-teacher anchor rows per step')
parser.add_argument('--evidence_gap_same_short_multi_teacher_count', type=int, default=1, help='number of alternate teacher windows paired with a reused crop short z per anchor step')
parser.add_argument('--evidence_gap_same_short_anchor_from_crop_only', type=int, default=1, help='1: anchor same-short multi-teacher only from crop short views; 0: allow any short row (crop only implemented for now)')
parser.add_argument('--evidence_gap_dual_teacher_cross', type=int, default=0, help='1: sample two same-length teacher windows and form cross-window sequence-state pairs')
parser.add_argument('--evidence_gap_dual_teacher_short_per_side', type=int, default=4, help='legacy fallback: crop-only shorts per teacher window when dual_teacher_cross=1 and n_short_crop=n_short_random=0')
parser.add_argument('--evidence_gap_dual_teacher_patch_cross', type=int, default=1, help='1: patch-state loss uses cross-window teacher targets; 0: same-window patch pairing while sequence state may stay cross-window')
parser.add_argument(
    '--evidence_gap_drop_global_cls',
    type=int,
    default=0,
    help='1: exclude global student rows from evidence-to-context sequence-state CE; global forward still runs for patch-state supervision if lambda_patch_proto>0',
)
parser.add_argument(
    '--ibot_target_mode',
    type=str,
    default='dinov3',
    choices=['dinov3', 'jepa_block'],
    help='evidence-gap global patch mask sampler; dinov3=legacy; jepa_block=contiguous target blocks (utils.jepa_masking)',
)
parser.add_argument(
    '--ibot_context_target_attn',
    type=str,
    default='full',
    choices=['full', 'disjoint'],
    help='student attention over masked global patches; full=legacy bidirectional; disjoint=context/prefix cannot attend target slots',
)
parser.add_argument(
    '--ibot_jepa_target_ratio_min',
    type=float,
    default=0.25,
    help='jepa_block mode: min fraction of patches in contiguous target block(s)',
)
parser.add_argument(
    '--ibot_jepa_target_ratio_max',
    type=float,
    default=0.5,
    help='jepa_block mode: max fraction of patches in contiguous target block(s)',
)
parser.add_argument(
    '--ibot_jepa_n_blocks',
    type=int,
    default=1,
    help='jepa_block mode: number of contiguous target blocks per masked sample',
)
parser.add_argument('--evidence_gap_version', type=str, default='v2', choices=['v2', 'v2.5', 'v3', 'v4'], help='condition builder: v2=ratio+position in teacher window; v2.5=ratio+timeline offset; v4=teacher scale + timeline offset')
parser.add_argument('--dino_head_n_prototypes', type=int, default=256, help='number of categorical states in the sequence-state head')
parser.add_argument('--dino_head_hidden_dim', type=int, default=128, help='hidden dimension in the sequence-state head')
parser.add_argument('--dino_head_bottleneck_dim', type=int, default=64, help='bottleneck dimension in the sequence-state head')
parser.add_argument('--dino_head_nlayers', type=int, default=3, help='number of layers in the sequence-state head')
parser.add_argument('--ibot_head_n_prototypes', type=int, default=256, help='number of categorical states in the patch-state head')
parser.add_argument('--ibot_head_hidden_dim', type=int, default=128, help='hidden dimension in the patch-state head')
parser.add_argument('--ibot_head_bottleneck_dim', type=int, default=64, help='bottleneck dimension in the patch-state head')
parser.add_argument('--ibot_head_nlayers', type=int, default=3, help='number of layers in the patch-state head')
parser.add_argument('--teacher_temp', type=float, default=0.07, help='teacher temperature (final value)')
parser.add_argument('--warmup_teacher_temp', type=float, default=0.04, help='warmup teacher temperature (initial value)')
parser.add_argument('--warmup_teacher_temp_epochs', type=int, default=30, help='warmup epochs for teacher temperature')
parser.add_argument('--student_temp', type=float, default=0.1, help='student temperature (fixed)')
parser.add_argument(
    '--cls_global_loss_mode',
    type=str,
    default='dino',
    choices=['dino', 'overlap_compat'],
    help='global CLS loss: dino = cross-view CE with diagonal ignored; '
    'overlap_compat = self-view CE plus overlap-weighted cross-view CE for shifted temporal globals',
)
parser.add_argument(
    '--cls_global_compat_self_weight',
    type=float,
    default=1.0,
    help='overlap_compat: weight for same-window student/teacher global CE',
)
parser.add_argument(
    '--cls_global_compat_cross_floor',
    type=float,
    default=0.25,
    help='overlap_compat: minimum cross-window CE weight at the configured minimum overlap',
)
parser.add_argument(
    '--cls_global_compat_min_overlap',
    type=float,
    default=-1.0,
    help='overlap_compat: overlap mapped to the cross-weight floor; <0 uses global_shift_min_overlap_ratio',
)
parser.add_argument(
    '--cls_local_loss_mode',
    type=str,
    default='per_view',
    choices=['per_view', 'bag', 'crop_set'],
    help='local evidence vs teacher context: '
    'per_view = sequence-state CE over local evidence and teacher context windows; '
    'bag = logmeanexp bag over crop evidence windows vs mean teacher state (see cls_local_bag_loss); '
    'crop_set = set-posterior pool over crop evidence windows + optional per-crop anchor (see crop_view_loss). '
    'bag/crop_set auto-split crop vs random locals when cls_data provides counts.',
)
parser.add_argument(
    '--cls_local_crop_gamma',
    type=float,
    default=2.0,
    help='crop_set mode: generalized-mean sharpness for pooling crop views into set_prob',
)
parser.add_argument(
    '--cls_local_crop_lambda_set',
    type=float,
    default=0.5,
    help='crop_set mode: weight for set-level posterior loss vs teacher globals',
)
parser.add_argument(
    '--cls_local_crop_lambda_ind',
    type=float,
    default=1.0,
    help='crop_set mode: weight for per-crop anchor loss vs teacher globals',
)
parser.add_argument(
    '--cls_local_cross_teacher_beta',
    type=float,
    default=0.0,
    help='optional weak local-to-non-parent teacher CE weight; 0 keeps parent-only local supervision',
)
parser.add_argument(
    '--cls_local_cross_teacher_normalize',
    type=int,
    default=1,
    help='1 normalizes parent + weak cross local CE by its summed weights to preserve local loss scale',
)
parser.add_argument(
    '--lambda_cls_local_contrib',
    type=float,
    default=0.0,
    help='bag mode only: weight for weak per-local contribution regularizer (0 = disabled)',
)
parser.add_argument(
    '--cls_local_contrib_margin',
    type=float,
    default=0.0,
    help='bag mode: margin in relu(margin - contrib)^2 when lambda_cls_local_contrib > 0',
)
parser.add_argument('--momentum_teacher', type=float, default=0.992, help='EMA momentum (initial value)')
parser.add_argument('--final_momentum_teacher', type=float, default=1.0, help='EMA momentum (final value)')
parser.add_argument('--empty_cache_interval', type=int, default=0, help='every N steps call torch.cuda.empty_cache() (0=disabled, default; set e.g. 300 if fighting fragmentation/OOM)')
parser.add_argument('--weight_decay_end', type=float, default=None, help='final weight decay (default: weight_decay * 10)')
parser.add_argument('--min_lr', type=float, default=None, help='minimum learning rate (default: lr * 1e-6)')
parser.add_argument('--freeze_last_layer_epochs', type=int, default=1, help='epochs to freeze last layer')
parser.add_argument('--schedule_trunc_extra', type=float, default=0.0, help='schedule truncation extra')
parser.add_argument(
    '--scheduler_version',
    type=str,
    default='cosine',
    choices=['cosine', 'dinov3_v2'],
    help='LR/WD/momentum/teacher_temp: cosine=CosineScheduler (default); dinov3_v2=linear_warmup_cosine_decay',
)

parser.add_argument(
    '--sched_lr_peak',
    type=float,
    default=None,
    help='dinov3_v2 only: LR cosine peak (default None=scaled learning_rate)',
)
parser.add_argument(
    '--sched_lr_end',
    type=float,
    default=None,
    help='dinov3_v2 only: LR end (default None=scaled min_lr); same as peak for flat top',
)
parser.add_argument(
    '--sched_lr_warmup_epochs',
    type=int,
    default=None,
    help='dinov3_v2 only: LR linear warmup epochs (default None=warmup_epochs)',
)
parser.add_argument(
    '--sched_lr_cosine_epochs',
    type=int,
    default=None,
    help='dinov3_v2 only: LR cosine segment epochs (default None=fill rest)',
)
parser.add_argument(
    '--sched_wd_warmup_epochs',
    type=int,
    default=0,
    help='dinov3_v2 only: weight_decay linear warmup epochs',
)
parser.add_argument(
    '--sched_wd_cosine_epochs',
    type=int,
    default=None,
    help='dinov3_v2 only: WD cosine segment epochs (default None)',
)
parser.add_argument(
    '--sched_momentum_warmup_epochs',
    type=int,
    default=0,
    help='dinov3_v2 only: EMA momentum warmup epochs',
)
parser.add_argument(
    '--sched_momentum_cosine_epochs',
    type=int,
    default=None,
    help='dinov3_v2 only: momentum cosine segment epochs (default None)',
)
parser.add_argument(
    '--sched_teacher_temp_end',
    type=float,
    default=None,
    help='dinov3_v2 only: teacher temperature end (default=teacher_temp)',
)
parser.add_argument(
    '--sched_teacher_temp_cosine_epochs',
    type=int,
    default=None,
    help='dinov3_v2 only: teacher_temp cosine segment epochs (default None)',
)

# MSM patch masking

parser.add_argument('--patch_len', type=int, default=31, help='patch length')
parser.add_argument('--stride', type=int, default=31, help='stride')

parser.add_argument('--mask_rate', type=float, default=0.8, help='mask ratio (for input-level masking)')
parser.add_argument('--mask_rate_v1', type=float, default=0.3, help='min patch mask ratio for TED patch-state supervision')
parser.add_argument(
    '--patchtst_style_masking',
    type=int,
    default=1,
    help='MSM: 1=uniform random patch masking; 0=contiguous block-biased masking',
)
parser.add_argument(
    '--patchtst_mask_ratio',
    type=float,
    default=0.4,
    help='MSM: fraction of patches masked when patchtst_style_masking=1',
)
parser.add_argument('--mask_rate_v2', type=float, default=0.6, help='max patch mask ratio for TED patch-state supervision')
parser.add_argument(
    '--mask_sample_probability',
    type=float,
    default=0.5,
    help='fraction of global student rows that use patch-state masking; set 1.0 to mask all rows',
)
parser.add_argument(
    '--local_view_patch_divisor',
    type=int,
    default=8,
    help='TED local views (crop & random patch): '
         'local_token_num = max(1, num_patches_in_teacher_win // divisor) ≈ 1/divisor of patches. '
         'Must be >= 1 (default 8 matches previous hard-coded behavior). '
         'Independent of n_local_student (number of local views).',
)
parser.add_argument(
    '--ted_modular_n_local_student',
    type=int,
    default=8,
    help='TED only: number of student local views (temporal crop + random-token views). '
         'Default 8 matches previous behavior.',
)
parser.add_argument(
    '--ted_modular_n_local_random_views',
    type=int,
    default=2,
    help='TED only: how many of the local views use random patch-token sampling; '
         'the rest use temporal crop. Default 2 => 6 crop + 2 random. Set 0 for all crop (no random local views).',
)

parser.add_argument(
    '--sampling_stride',
    type=int,
    default=732,
    help='stride (in time steps) between consecutive HLS windows along the timeline',
)

# optional pretrained imputator (used inside TED when enabled)

parser.add_argument('--use_pretrained_imputator', type=int, default=0, help='whether to use pretrained imputator (0/1)')
parser.add_argument('--pretrained_imputator_path', type=str, default='./checkpoints/models/Transformer_Imputator.pth', help='path to pretrained imputator')
parser.add_argument('--imp_d_model', type=int, default=256, help='imputator d_model (used when loading pretrained imputator inside TED)')
parser.add_argument('--imp_n_heads', type=int, default=8, help='imputator n_heads')
parser.add_argument('--imp_e_layers', type=int, default=6, help='imputator e_layers')
parser.add_argument('--imp_d_ff', type=int, default=1024, help='imputator d_ff')
parser.add_argument(
    '--imp_n_storage_tokens',
    type=int,
    default=2,
    help='Imputator storage tokens (default 2, cluster training); -1 follows n_storage_tokens else 2. Must match ckpt when loading to TED. Unrelated to backbone --n_storage_tokens.',
)
parser.add_argument(
    '--imp_rec_loss',
    type=str,
    default='mse',
    choices=['mse', 'huber', 'mae'],
    help='Imputator (Transformer) reconstruction term: mse | huber | mae',
)
parser.add_argument('--imp_huber_delta', type=float, default=1.0, help='Huber delta when imp_rec_loss=huber')
parser.add_argument('--imp_rec_alpha', type=float, default=1.0, help='weight for reconstruction term in Imputator cal_rec_loss')
parser.add_argument('--imp_smooth_beta', type=float, default=0.5, help='weight for smooth_loss term in Imputator cal_rec_loss')
parser.add_argument(
    '--imp_smooth_mode',
    type=str,
    default='dy2',
    choices=['dy1', 'dy2'],
    help='Imputator smooth_loss: dy1=first-diff MSE, dy2=second-diff (default)',
)
parser.add_argument(
    '--imp_mask_min_p',
    type=float,
    default=0.4,
    help='Imputator train apply_mask random rate lower bound [min_p, mask_rate] (use_random_p=True); was 0.25',
)
parser.add_argument(
    '--imp_trim_topk_per_seq',
    type=int,
    default=0,
    help='per sample ignore topK largest recon errors; 0=off (try 3-10)',
)
parser.add_argument(
    '--imp_trim_min_keep',
    type=int,
    default=8,
    help='min supervised points after topK ignore to avoid empty supervision',
)

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument(
    '--no_attn_checkpoint',
    action='store_true',
    help='disable gradient checkpoint inside AttentionBlock (faster step time, higher VRAM)',
)
parser.add_argument(
    '--compile_backbone',
    action='store_true',
    help='torch.compile Backbone+teacher in TED (PyTorch 2.x; dynamic seq; first steps slow; measure on your GPU)',
)

parser.add_argument('--train_epochs', type=int, default=400, help='train epochs')
parser.add_argument(
    '--max_train_steps',
    type=int,
    default=0,
    help='>0: stop after this many successful optimizer steps (smoke/debug); 0=disabled',
)
parser.add_argument('--warmup_epochs', type=int, default=1, help='warmup epochs')
parser.add_argument(
    '--batch_size',
    type=int,
    default=256,
    help='per-process train batch size; global batch is this x world_size under DDP',
)
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='base learning rate (will be scaled by batch size)')
parser.add_argument(
    '--max_grad_norm',
    type=float,
    default=1.0,
    help='grad clip threshold; default 1.0',
)
parser.add_argument('--scaling_rule', type=str, default='sqrt_wrt_1024', help='LR scaling rule: sqrt_wrt_1024, linear_wrt_256, or none')

parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate')

parser.add_argument('--use_amp', type=int, default=0, help='use automatic mixed precision training (0/1)')
parser.add_argument(
    '--amp_dtype',
    type=str,
    default='auto',
    choices=['auto', 'float16', 'bfloat16'],
    help='With --use_amp: autocast dtype; auto uses bfloat16 when CUDA supports it (e.g. Hopper/Ampere), else float16.',
)

parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')

# loss weight params

parser.add_argument('--lambda_fft_align', type=float, default=0, help=(
    'Peak frequency-domain patch-alignment coefficient when fft_align_warmup_epochs>0 (value at end of warmup); '
    'otherwise base lambda (optionally gated by fft_align_epoch_*).'
))
parser.add_argument('--fft_align_epoch_start', type=int, default=-1,
                    help='enable fft align only from this 1-based epoch (inclusive); <=0 disables epoch gating')
parser.add_argument('--fft_align_epoch_end', type=int, default=-1,
                    help='enable fft align until this 1-based epoch (inclusive); '
                    'if fft_align_epoch_start>0 and this is <=0, run through final epoch (train_epochs); '
                    'if both start and end <=0, no epoch gating (use lambda_fft_align for all epochs)')
parser.add_argument('--fft_align_lambda_active', type=float, default=None,
                    help='fft align lambda inside [fft_align_epoch_start, fft_align_epoch_end]; None=use lambda_fft_align')
parser.add_argument('--fft_align_lambda_inactive', type=float, default=0.0,
                    help='fft align lambda outside gated epoch range (default 0.0)')
parser.add_argument(
    '--fft_align_warmup_epochs',
    type=int,
    default=0,
    help='If >0: ramp lambda_fft_align from fft_align_lambda_start to lambda_fft_align (peak) over '
    'this many 1-based epochs, then decay linearly to fft_align_lambda_end by train_epochs; '
    'ignores fft_align_epoch_start/end. If 0: use legacy epoch gate or constant lambda_fft_align.',
)
parser.add_argument(
    '--fft_align_lambda_start',
    type=float,
    default=0.0,
    help='frequency-domain patch-alignment coefficient at epoch 1 when fft_align_warmup_epochs>0.',
)
parser.add_argument(
    '--fft_align_lambda_end',
    type=float,
    default=0.0,
    help='frequency-domain patch-alignment coefficient at final epoch after warmup when fft_align_warmup_epochs>0.',
)
parser.add_argument(
    '--fft_align_all_patches',
    action='store_true',
    help='Use all patches for frequency-domain alignment over aligned student/teacher windows. '
    'Default (flag off): only masked patches, with the same support as patch-state supervision.',
)
parser.add_argument(
    '--fft_align_gram_mse_f_ref_bins',
    type=int,
    default=0,
    help='frequency-domain alignment rescale: 0=full-length F_ref from seq_len/patch/stride, loss * min(1,(F_act/F_ref)^2),'
    'decouple mean vs F for comparable micro-steps; full length scale=1; <0 disables',
)
parser.add_argument(
    '--fft_align_freq_keep_ratio',
    type=float,
    default=1.0,
    help='ratio-based low-frequency keep for frequency-domain patch alignment, applied on active bins F_act after DC removal (0,1]; 1.0 keeps all',
)
parser.add_argument(
    '--fft_align_freq_min_bins',
    type=int,
    default=1,
    help='minimum kept frequency bins for frequency-domain patch alignment after ratio truncation',
)
parser.add_argument(
    '--fft_align_freq_max_bins',
    type=int,
    default=0,
    help='maximum kept frequency bins for frequency-domain patch alignment after ratio truncation; <=0 means no upper bound',
)
parser.add_argument('--lambda_cls_proto', type=float, default=1, help='lambda cls proto')
parser.add_argument('--lambda_patch_proto', type=float, default=0.2, help='lambda patch proto')
parser.add_argument('--lambda_koleo', type=float, default=0.05, help='lambda koleo')
parser.add_argument('--lambda_temporal', type=float, default=0.05, help='lambda temporal')

# Frequency-domain alignment / masking / imputator related additional controls
parser.add_argument(
    '--fft_align_min_valid_ratio',
    type=float,
    default=0.0,
    help='only compute frequency-domain patch alignment when per-sample raw valid ratio '
    '(~missing_mask_orig) >= this; <=0 disables gating (use all valid_sample rows)',
)
parser.add_argument(
    '--valid_sample_threshold',
    type=float,
    default=0.0,
    help='TED / MSM / NTP: per-sample raw valid ratio '
    '(~missing_mask_orig.mean) must be >= this to join SSL loss; 0 keeps all samples',
)
# Option 2: frequency-domain alignment frequency-bin cutoff ratio

# Option 3: block-biased patch masking ratio (rest random)
parser.add_argument(
    '--block_mask_ratio',
    type=float,
    default=0.8,
    help='ratio of block-masked patches in block-biased masking (default 0.8)',
)
# Option 4: down-weight patch loss at imputator-filled positions
parser.add_argument(
    '--imputed_patch_weight',
    type=float,
    default=1.0,
    help='relative weight for patches that contain imputed (originally missing) positions in patch loss (1.0 = no reweighting)',
)
# Patch-state supervision: include all-missing patches in the loss when requested
parser.add_argument(
    '--ibot_patch_reliable_mode',
    type=str,
    default='filter',
    choices=['filter', 'keep_all'],
    help='filter: only patch states with at least one original valid observation participate in patch-state supervision; keep_all: no filter',
)
parser.add_argument(
    '--lambda_cls_cons',
    type=float,
    default=0,
    help='weight for raw-imputed CLS consistency loss (same sequence, align CLS from raw view and imputed view); 0 to disable',
)

# supervised classification split / early stopping (LCMAP & GlanCE)

# unified K for downstream KNN probes (all knn_probe_* in ExpProbe)

# Simulate multiple clouds: before encoding in the probe stage, randommask has a valid timestep with a certain ratio (set the entire step to NaN)

# training throughput: defaults favor wall-clock; set back to 1 for dense logs/checkpoints

parser.add_argument(
    '--train_step_log_enable',
    type=int,
    default=1,
    help='1: write per-step training metrics to JSONL file asynchronously (rank0 only). 0: disable.',
)
parser.add_argument(
    '--train_step_log_file',
    type=str,
    default='auto',
    help='per-step JSONL path; "auto" writes to logs/<model_id>_train_step_metrics.jsonl (rank0 only).',
)
parser.add_argument(
    '--train_step_console_log_enable',
    type=int,
    default=0,
    help='1: also print per-step detailed losses to stdout; 0: keep per-step logs only in JSONL.',
)
parser.add_argument(
    '--train_step_tensor_item_interval',
    type=int,
    default=25,
    help='convert Tensor log_vars to Python scalars every N steps (1=every step). Larger N reduces .item()/sync overhead; default 25.',
)
parser.add_argument(
    '--progress_log_interval',
    type=int,
    default=200,
    help='print progress every N batches (LR / speed / ETA); suggest 100-500',
)
parser.add_argument(
    '--checkpoint_epoch_save_interval',
    type=int,
    default=1,
    help=(
        'rank0 writes checkpoint_epoch_*.pth every N epochs (default 1); '
        'best checkpoint.pth on train loss unchanged. Large runs: 5+ to save disk.'
    ),
)
parser.add_argument(
    '--train_state_save_interval',
    type=int,
    default=5,
    help=(
        'every N epochs write train_state per rank; '
        'always at end of first and last epoch. 1=every epoch full resume state.'
    ),
)
parser.add_argument(
    '--ddp_find_unused_parameters',
    type=int,
    default=0,
    help='DDP: 1=find_unused_parameters (slower); 0=off (faster). Set 1 if backward errors.',
)

# GPU
# [change] use int for bool flags (more robust)
parser.add_argument('--use_gpu', type=int, default=1, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', type=int, default=1, help='use multiple gpus')
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

def main() -> None:
    args = parser.parse_args()

    # convert to bool
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.use_multi_gpu = True if torch.cuda.is_available() and args.use_multi_gpu else False
    args.use_pretrained_imputator = bool(args.use_pretrained_imputator)
    args.save_tsne_embeddings = bool(getattr(args, 'save_tsne_embeddings', 0))

    # --- DDP initialization ---
    if args.use_multi_gpu:
        # torchrun sets RANK and LOCAL_RANK automatically
        if "RANK" in os.environ and "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(local_rank)
            args.device = torch.device("cuda", local_rank)
            args.local_rank = local_rank
        else:
            print("Warning: use_multi_gpu is True but RANK/LOCAL_RANK not set. Falling back to single GPU mode.")
            print("To use multi-GPU training, please use: torchrun --nproc_per_node=N train.py ...")
            args.use_multi_gpu = False
            args.local_rank = 0
            if args.use_gpu and torch.cuda.is_available():
                device_id = args.gpu
                torch.cuda.set_device(device_id)
                args.device = torch.device(f"cuda:{device_id}")
            else:
                args.device = torch.device("cpu")
    else:
        args.local_rank = 0
        if args.use_gpu and torch.cuda.is_available():
            device_id = args.gpu
            torch.cuda.set_device(device_id)
            args.device = torch.device(f"cuda:{device_id}")
        else:
            args.device = torch.device("cpu")

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.local_rank

    if args.local_rank == 0:
        print('cuda.is_available:', torch.cuda.is_available())
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print('Args in experiment:')
        print(args)

    allowed_models = {
        'TED', 'MSM', 'NTP', 'Imputator',
        'TED_modular', 'Patch_Masked', 'Patch_NTP_TED', 'Transformer',
    }
    if args.model not in allowed_models:
        raise ValueError(
            f"Unsupported model={args.model!r}. Allowed: {sorted(allowed_models)}"
        )

    # Keep a stable experiment folder name (HLS-only package).
    args.data = 'HLS'
    args.embed = 'timeF'
    setting = '{}_{}_{}_sl{}_dm{}_nh{}_el{}_df{}_0'.format(
        args.model_id,
        args.model,
        args.data,
        args.seq_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_ff,
    )

    _setting_raw = setting
    setting = clamp_experiment_setting_for_checkpoint(setting)
    if setting != _setting_raw:
        args.checkpoint_setting_full = _setting_raw
    else:
        args.checkpoint_setting_full = None

    exp = SSLTrainer(args)

    try:
        if args.local_rank == 0:
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        torch.cuda.empty_cache()
    finally:
        if args.use_multi_gpu and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
