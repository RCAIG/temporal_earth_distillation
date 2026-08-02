"""Primary TED implementation: temporal backbone plus sequence-state and patch-state heads."""
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.backbone import Backbone
from utils.tools import (
    patchify,
    unpatchify,
    get_student_input,
    get_teacher_input,
    generate_local_view_crop,
    generate_local_view_random_sample,
    random_patch_masking_dinov3_style,
    imputator_sliding_window_overlap,
)


class ConditionCrossAttnDelta(nn.Module):
    """Single-token cross-attention: z queries condition embedding, projects to bottleneck delta."""

    def __init__(self, d_model, cond_embed_dim, bottleneck_dim, n_heads=4):
        super().__init__()
        n_heads = max(1, int(n_heads))
        while d_model % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(cond_embed_dim, 2 * d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.out_proj = nn.Linear(d_model, bottleneck_dim)
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, z, cond):
        q = self.q_proj(z).unsqueeze(1)
        kv = self.kv_proj(cond)
        k = kv[..., : kv.shape[-1] // 2].unsqueeze(1)
        v = kv[..., kv.shape[-1] // 2 :].unsqueeze(1)
        out, _ = self.attn(q, k, v)
        return self.out_proj(out.squeeze(1))


class ConditionFilmMLP(nn.Module):
    """Dedicated short-view projector: z MLP with FiLM modulation from embedded condition."""

    def __init__(self, in_dim, cond_embed_dim, bottleneck_dim, hidden_dim, nlayers):
        super().__init__()
        nlayers = max(int(nlayers), 1)
        self.nlayers = nlayers
        if nlayers == 1:
            self.layers = nn.ModuleList([nn.Linear(in_dim, bottleneck_dim)])
            self.film = nn.ModuleList()
            return
        dims = [in_dim] + [hidden_dim] * (nlayers - 1) + [bottleneck_dim]
        layers = nn.ModuleList()
        films = nn.ModuleList()
        for i in range(nlayers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < nlayers - 1:
                films.append(nn.Linear(cond_embed_dim, 2 * dims[i + 1]))
        self.layers = layers
        self.film = films

    def forward(self, x, cond):
        h = x
        for i, linear in enumerate(self.layers):
            h = linear(h)
            if i < len(self.film):
                h = F.gelu(h)
                gamma, beta = self.film[i](cond).chunk(2, dim=-1)
                h = h * (1.0 + gamma) + beta
        return h

    def zero_init_output(self):
        last = self.layers[-1]
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.constant_(last.bias, 0)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # 1. Student & Teacher Backbones
        self.backbone = Backbone(configs) # Student
        self.teacher = Backbone(configs)  # Teacher (EMA)
        # Student backbone uses norm_student only; frozen norm_teacher kept for interface avoids DDP unused params.
        for p in self.backbone.norm_teacher.parameters():
            p.requires_grad = False
        
        # init Teacher as frozen copy of Student
        self.teacher.load_state_dict(self.backbone.state_dict())
        self._copy_student_to_teacher()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.ema_params_lists = None
            
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.c_out = configs.c_out
        # Consistent with Backbone patchify/unfold logic:
        # N = ceil((T - patch_len) / stride) + 1 (when T > patch_len)
        self.num_patches = math.ceil((configs.seq_len - configs.patch_len + configs.stride) / configs.stride)
        self.local_view_patch_divisor = max(1, int(getattr(configs, "local_view_patch_divisor", 8)))
        self.num_local_patches = max(1, self.num_patches // self.local_view_patch_divisor)
        # Student local views: time crop vs in-patch random token mix (default 6 crop + 2 random, legacy hard-coded)
        self.ted_modular_n_local_student = max(0, int(getattr(configs, "ted_modular_n_local_student", 8)))
        _n_rand = int(getattr(configs, "ted_modular_n_local_random_views", 2))
        if self.ted_modular_n_local_student > 0:
            _n_rand = min(max(0, _n_rand), self.ted_modular_n_local_student)
        else:
            _n_rand = 0
        self.ted_modular_n_local_random_views = _n_rand
        # 2. Loss is computed by the external TED criterion; Model only performs forward passes
        
        # 3. Loss weights for sequence-state, patch-state and auxiliary regularization terms
        # keep base values for epoch scheduling during training
        # Backbone has no decoder: no pixel or frequency-domain reconstruction loss
        self.lambda_recon = 0.0
        self.lambda_fft_align = getattr(configs, 'lambda_fft_align', 0.05)  # frequency-domain patch alignment (unrelated to reconstruction)
        self.lambda_cls_proto = configs.lambda_cls_proto if hasattr(configs, 'lambda_cls_proto') else 1.0
        self.lambda_patch_proto = configs.lambda_patch_proto if hasattr(configs, 'lambda_patch_proto') else 1.0  # key fix: increased from 0.2 to 1.0
        self.lambda_koleo = configs.lambda_koleo if hasattr(configs, 'lambda_koleo') else 0.05
        self.lambda_temporal = configs.lambda_temporal if hasattr(configs, 'lambda_temporal') else 0.2  # reduced from 0.3 to 0.2 (balance)
        # Block-biased patch masking fraction for patch-state supervision
        self.block_mask_ratio = getattr(configs, 'block_mask_ratio', 0.8)
        # Only a subset of global rows receives patch masks; unmasked rows are excluded from patch-state loss
        self.mask_sample_probability = getattr(configs, 'mask_sample_probability', 0.5)
        # down-weight patch loss at imputator-filled positions (1.0 = no down-weight)
        self.imputed_patch_weight = getattr(configs, 'imputed_patch_weight', 1.0)
        # Raw-imputed CLS consistency: align CLS of raw/imputed views (0 disables)
        self.lambda_cls_cons = getattr(configs, 'lambda_cls_cons', 0.05)
        # Frequency-domain patch alignment runs only when original valid-observation ratio is high enough. Alignment is computed on each global-view pair
        # on samples passing threshold; do not filter rows by patch mask density.
        self.fft_align_min_valid_ratio = float(
            getattr(configs, "fft_align_min_valid_ratio", 0.0)
        )
        # Frequency-domain patch alignment defaults to masked patches, sharing support with patch-state loss; --fft_align_all_patches uses full windows
        self.fft_align_all_patches = bool(
            getattr(configs, "fft_align_all_patches", False)
        )

        # Teacher temperature for balanced categorical assignment
        self.teacher_temp = getattr(configs, 'teacher_temp', 0.07)
        
        # 4. training state tracking
        self.step_counter = 0
        self.log_interval = 10  # fallback step log interval (currently unused)
        self.last_log_time = time.time()
        self.log_interval_seconds = 300  # at least 5 minutes (300s) between logs
        self._current_epoch = 0
        
        # 5. validation thresholds
        self.valid_patch_threshold = 0.5  # threshold for valid patches
        # full-sequence original valid-observation ratio must exceed this for distillation and auxiliary regularization
        self.valid_sample_threshold = float(
            getattr(configs, "valid_sample_threshold", 0.0)
        )
        self.ibot_patch_reliable_mode = getattr(
            configs, 'ibot_patch_reliable_mode', 'filter'
        )

        # 6. curriculum learning config
        # 'fast': first 10% 1-3 years only, then 1-6 years full range (recommended, cosine anneal)
        # 'balanced': 0-15%: 1-3year, 15-35%: 1-4year, 35-55%: 1-5year, 55-100%: 1-6year
        # 'conservative': 0-20%: 1-3year, 20-40%: 1-4year, 40-60%: 1-5year, 60-100%: 1-6year
        # 'none': no curriculum; 1-6 years full range directly
        self.curriculum_strategy = getattr(configs, 'curriculum_strategy', 'fast')
        self._curriculum_length_jitter = int(getattr(configs, 'curriculum_length_jitter', 61))
        self.curriculumJitterProbability = float(getattr(configs, 'curriculum_jitter_probability', 0.5))
        self.disable_view_augmentation = bool(
            getattr(configs, 'disable_view_augmentation', False)
        )

        # mixed_batch: each sub-forward samples local patch ratio (divisor d -> ~1/d of teacher-window patches)
        divs_s = str(getattr(configs, "mixed_batch_local_patch_divisors", "8,6,4"))
        prb_s = str(getattr(configs, "mixed_batch_local_patch_divisor_probs", "0.5,0.25,0.25"))
        divs_list = [max(1, int(x.strip())) for x in divs_s.split(",") if x.strip()]
        prbs_list = [float(x.strip()) for x in prb_s.split(",") if x.strip()]
        if len(divs_list) != len(prbs_list) or not divs_list:
            raise ValueError(
                "mixed_batch_local_patch_divisors and mixed_batch_local_patch_divisor_probs "
                f"must be same-length non-empty lists; got divisors={divs_list!r} probs={prbs_list!r}"
            )
        _pt = torch.tensor(prbs_list, dtype=torch.double)
        _pt = _pt / _pt.sum().clamp(min=1e-12)
        self._mixed_batch_local_divisors_list = divs_list
        self.register_buffer("_mixed_batch_local_div_probs_t", _pt, persistent=False)
        
        # 7. Imputator usestrategy
        # 'full': Teacher uses all imputator, Student uses half imputator and half raw.
        # The missing_mask on the Teacher side is set to all 0, all patches are considered reliable, and reconstructionuse perfect target
        # 'recon_only' : Teacher / Student viewall use no imputator (all see raw),
        # missing_mask / patch filter maintains originallogic, but the reconstruction target still uses the perfect target generated by the imputator
        # 'mixed_teacher': Student is still 50% imputator view, Teacher use [imp, raw] true mixedview,
        # no longer use 0.5 mask (only 0/1) under and mixed, reconstruction target use perfect target
        # 'woMask': Teacher uses all imputators, Student uses half imputator and half raw.
        # and all viewpass in backbone's missing_mask treated as all 0 (do not use missing mask embedding)
        # 'wMask': Teacher all use imputator, Student all use no imputator (all use raw),
        # Teacher / Student both keeporiginal missing_mask (based on physical missing)
        self.imputator_mode = getattr(configs, 'imputator_mode', 'full')
        # Ultra-long sequence imputator slidingwindowstride (default 244≈two years, overlaps with pred_len=366 by about 122 points)
        self.imputator_segment_stride = getattr(configs, 'imputator_segment_stride', 244)
        # lon/latembedding CFG: trainingwhen uses probabilitydrop conditions to facilitate the inference stage without lon/lat; cooperate with geo_keep backbone
        self.geo_dropout_p = float(getattr(configs, "geo_dropout_p", 0.5))
        self.global_shift_steps = int(getattr(configs, 'global_shift_steps', 0))
        self.global_shift_jitter_steps = int(getattr(configs, 'global_shift_jitter_steps', 0))
        self.global_shift_probability = float(getattr(configs, 'global_shift_probability', 1.0))
        self.global_shift_min_overlap_ratio = float(
            getattr(configs, "global_shift_min_overlap_ratio", 0.6)
        )
        # With train.py argparse defaultconsistent (no parameter is passed whendisable temporal shift)
        self.global_shift_ratio = float(getattr(configs, "global_shift_ratio", 0.0))
        self.global_shift_jitter_ratio = float(
            getattr(configs, "global_shift_jitter_ratio", 0.0)
        )
        self.global_shift_mode = str(
            getattr(configs, "global_shift_mode", "base_jitter")
        ).lower()

        # Optional: only compile Backbone (student/teacher), do not compile the whole picture _forward (too many Python branches)
        self.evidence_gap_distill = bool(int(getattr(configs, "evidence_gap_distill", 0)))
        self.evidence_gap_teacher_lengths = self._parse_int_list(
            getattr(configs, "evidence_gap_teacher_lengths", "61,122,183,244,366,488,732"),
            default=[61, 122, 183, 244, 366, 488, 732],
        )
        self.evidence_gap_student_ratio_min = float(
            getattr(configs, "evidence_gap_student_ratio_min", 0.1)
        )
        self.evidence_gap_student_ratio_max = float(
            getattr(configs, "evidence_gap_student_ratio_max", 0.9)
        )
        self.evidence_gap_cls_bins = self._parse_int_list(
            getattr(configs, "evidence_gap_cls_bins", "5,10,21,41,82"),
            default=[5, 10, 21, 41, 82],
        )
        self.evidence_gap_student_aug = str(
            getattr(configs, "evidence_gap_student_aug", "strong")
        )
        self.evidence_gap_n_short_crop = max(
            0, int(getattr(configs, "evidence_gap_n_short_crop", 4))
        )
        self.evidence_gap_n_short_random = max(
            0, int(getattr(configs, "evidence_gap_n_short_random", 2))
        )
        self.evidence_gap_short_outside_teacher = bool(
            int(getattr(configs, "evidence_gap_short_outside_teacher", 0))
        )
        self.evidence_gap_independent_fullseq = bool(
            int(getattr(configs, "evidence_gap_independent_fullseq", 0))
        )
        self.evidence_gap_same_short_multi_teacher_prob = float(
            getattr(configs, "evidence_gap_same_short_multi_teacher_prob", 0.25)
        )
        self.evidence_gap_same_short_multi_teacher_count = max(
            0, int(getattr(configs, "evidence_gap_same_short_multi_teacher_count", 1))
        )
        self.evidence_gap_same_short_anchor_from_crop_only = bool(
            int(getattr(configs, "evidence_gap_same_short_anchor_from_crop_only", 1))
        )
        self.evidence_gap_dual_teacher_cross = bool(
            int(getattr(configs, "evidence_gap_dual_teacher_cross", 0))
        )
        self.evidence_gap_dual_teacher_short_per_side = max(
            0,
            int(getattr(configs, "evidence_gap_dual_teacher_short_per_side", 4)),
        )
        self.evidence_gap_dual_teacher_patch_cross = bool(
            int(getattr(configs, "evidence_gap_dual_teacher_patch_cross", 1))
        )
        self.evidence_gap_drop_global_cls = bool(
            int(getattr(configs, "evidence_gap_drop_global_cls", 0))
        )
        # Latent-prediction CE ablation knobs; defaults preserve the paper release path.
        self.ibot_target_mode = str(
            getattr(configs, "ibot_target_mode", "dinov3")
        ).lower()
        self.ibot_context_target_attn = str(
            getattr(configs, "ibot_context_target_attn", "full")
        ).lower()
        self.ibot_jepa_target_ratio_min = float(
            getattr(configs, "ibot_jepa_target_ratio_min", 0.25)
        )
        self.ibot_jepa_target_ratio_max = float(
            getattr(configs, "ibot_jepa_target_ratio_max", 0.5)
        )
        self.ibot_jepa_n_blocks = max(
            1, int(getattr(configs, "ibot_jepa_n_blocks", 1))
        )
        self.evidence_gap_version = str(
            getattr(configs, "evidence_gap_version", "v2")
        ).lower()
        self.evidence_gap_n_bins = len(self.evidence_gap_cls_bins) + 1
        self.evidence_gap_condition = self.evidence_gap_distill and bool(
            int(getattr(configs, "evidence_gap_condition", 1))
        )
        self.evidence_gap_condition_alpha = float(
            getattr(configs, "evidence_gap_condition_alpha", 0.1)
        )
        self.evidence_gap_condition_drop_p = float(
            getattr(configs, "evidence_gap_condition_drop_p", 0.0)
        )
        requested_readout = str(
            getattr(configs, "evidence_gap_condition_readout", "adapter")
        ).strip().lower()
        if requested_readout not in {
            "adapter",
            "direction",
            "gate",
            "film",
            "cond_mlp",
            "cond_sum_mlp",
            "cond_film_mlp",
            "cond_res_mlp",
            "cond_res_film_mlp",
            "cond_gate_bottleneck",
            "cond_mul_bottleneck",
            "cond_xattn_bottleneck",
            "cond_blend_mlp",
        }:
            raise ValueError(
                "evidence_gap_condition_readout must be 'adapter', 'direction', "
                "'gate', 'cond_mlp', 'cond_sum_mlp', 'cond_film_mlp', "
                "'cond_res_mlp', 'cond_res_film_mlp', 'cond_gate_bottleneck', "
                "'cond_mul_bottleneck', 'cond_xattn_bottleneck', 'cond_blend_mlp', "
                "or legacy 'film'; "
                f"got {requested_readout!r}"
            )
        self.evidence_gap_condition_readout = requested_readout
        self.evidence_gap_condition_view_embed_dim = int(
            getattr(configs, "evidence_gap_condition_view_embed_dim", 8)
        )
        self.evidence_gap_condition_scalar_embed_dim = int(
            getattr(configs, "evidence_gap_condition_scalar_embed_dim", 8)
        )
        self.evidence_gap_condition_scalar_n_freqs = max(
            1, int(getattr(configs, "evidence_gap_condition_scalar_n_freqs", 4))
        )
        cond_hidden = int(getattr(configs, "evidence_gap_condition_hidden_dim", 0))
        if cond_hidden <= 0:
            cond_hidden = 64
        d_model = self.backbone.d_model
        if self.evidence_gap_condition:
            self.evidence_gap_condition_view_embed = nn.Embedding(
                3, self.evidence_gap_condition_view_embed_dim
            )
            if self.evidence_gap_condition_readout == "film":
                cond_in_dim = 2 + self.evidence_gap_condition_view_embed_dim
                self.evidence_gap_condition_mlp = nn.Sequential(
                    nn.Linear(cond_in_dim, cond_hidden),
                    nn.GELU(),
                    nn.Linear(cond_hidden, 2 * d_model),
                )
                nn.init.zeros_(self.evidence_gap_condition_mlp[-1].weight)
                nn.init.zeros_(self.evidence_gap_condition_mlp[-1].bias)
                self.evidence_gap_condition_ratio_embed = None
                self.evidence_gap_condition_position_embed = None
                self.evidence_gap_condition_norm = None
                self.evidence_gap_condition_adapter = None
                self.evidence_gap_condition_direction = None
                self.evidence_gap_condition_norm_no_affine = None
                self.evidence_gap_condition_z_to_basis = None
                self.evidence_gap_condition_cond_to_gate = None
                self.evidence_gap_condition_basis_to_z = None
                self.evidence_gap_cond_mlp = None
                self.evidence_gap_cond_z_mlp = None
                self.evidence_gap_cond_c_mlp = None
                self.evidence_gap_cond_film_mlp = None
            elif self.evidence_gap_condition_readout == "gate":
                scalar_feature_dim = 1 + 2 * self.evidence_gap_condition_scalar_n_freqs
                scalar_embed_dim = max(1, self.evidence_gap_condition_scalar_embed_dim)
                cond_embed_dim = 2 * scalar_embed_dim + self.evidence_gap_condition_view_embed_dim
                gate_rank = cond_hidden
                self.register_buffer(
                    "evidence_gap_condition_fourier_freqs",
                    torch.pow(
                        torch.tensor(2.0),
                        torch.arange(self.evidence_gap_condition_scalar_n_freqs).float(),
                    ),
                    persistent=False,
                )
                self.evidence_gap_condition_ratio_embed = nn.Sequential(
                    nn.Linear(scalar_feature_dim, scalar_embed_dim),
                    nn.GELU(),
                    nn.Linear(scalar_embed_dim, scalar_embed_dim),
                )
                self.evidence_gap_condition_position_embed = nn.Sequential(
                    nn.Linear(scalar_feature_dim, scalar_embed_dim),
                    nn.GELU(),
                    nn.Linear(scalar_embed_dim, scalar_embed_dim),
                )
                self.evidence_gap_condition_norm = None
                self.evidence_gap_condition_adapter = None
                self.evidence_gap_condition_direction = None
                self.evidence_gap_condition_norm_no_affine = nn.LayerNorm(
                    d_model, elementwise_affine=False
                )
                self.evidence_gap_condition_z_to_basis = nn.Linear(
                    d_model, gate_rank, bias=False
                )
                nn.init.normal_(self.evidence_gap_condition_z_to_basis.weight, std=0.02)
                self.evidence_gap_condition_cond_to_gate = nn.Sequential(
                    nn.Linear(cond_embed_dim, cond_hidden),
                    nn.GELU(),
                    nn.Linear(cond_hidden, gate_rank, bias=False),
                )
                nn.init.normal_(self.evidence_gap_condition_cond_to_gate[-1].weight, std=0.02)
                self.evidence_gap_condition_basis_to_z = nn.Linear(
                    gate_rank, d_model, bias=False
                )
                nn.init.normal_(self.evidence_gap_condition_basis_to_z.weight, std=0.005)
                self.evidence_gap_condition_mlp = None
                self.evidence_gap_cond_mlp = None
                self.evidence_gap_cond_z_mlp = None
                self.evidence_gap_cond_c_mlp = None
                self.evidence_gap_cond_film_mlp = None
            elif self.evidence_gap_condition_readout == "cond_mlp":
                scalar_feature_dim = 1 + 2 * self.evidence_gap_condition_scalar_n_freqs
                scalar_embed_dim = max(1, self.evidence_gap_condition_scalar_embed_dim)
                cond_embed_dim = 2 * scalar_embed_dim + self.evidence_gap_condition_view_embed_dim
                bottleneck_dim = int(
                    getattr(configs, "dino_head_bottleneck_dim", 256)
                )
                cond_mlp_hidden = int(getattr(configs, "dino_head_hidden_dim", 1536))
                cond_mlp_nlayers = int(getattr(configs, "dino_head_nlayers", 3))
                self.register_buffer(
                    "evidence_gap_condition_fourier_freqs",
                    torch.pow(
                        torch.tensor(2.0),
                        torch.arange(self.evidence_gap_condition_scalar_n_freqs).float(),
                    ),
                    persistent=False,
                )
                self.evidence_gap_condition_ratio_embed = nn.Sequential(
                    nn.Linear(scalar_feature_dim, scalar_embed_dim),
                    nn.GELU(),
                    nn.Linear(scalar_embed_dim, scalar_embed_dim),
                )
                self.evidence_gap_condition_position_embed = nn.Sequential(
                    nn.Linear(scalar_feature_dim, scalar_embed_dim),
                    nn.GELU(),
                    nn.Linear(scalar_embed_dim, scalar_embed_dim),
                )
                self.evidence_gap_condition_norm = None
                self.evidence_gap_condition_adapter = None
                self.evidence_gap_condition_direction = None
                self.evidence_gap_condition_norm_no_affine = None
                self.evidence_gap_condition_z_to_basis = None
                self.evidence_gap_condition_cond_to_gate = None
                self.evidence_gap_condition_basis_to_z = None
                self.evidence_gap_condition_mlp = None
                self.evidence_gap_cond_mlp = self._build_condition_cond_mlp(
                    d_model + cond_embed_dim,
                    bottleneck_dim,
                    cond_mlp_hidden,
                    cond_mlp_nlayers,
                )
                self.evidence_gap_cond_mlp.apply(self._init_condition_cond_mlp_weights)
                self.evidence_gap_cond_z_mlp = None
                self.evidence_gap_cond_c_mlp = None
                self.evidence_gap_cond_film_mlp = None
            elif self.evidence_gap_condition_readout == "cond_sum_mlp":
                cond_embed_dim = self._init_v25_condition_scalar_embed(configs)
                bottleneck_dim, cond_mlp_hidden, cond_mlp_nlayers = (
                    self._condition_predictor_dims(configs)
                )
                self._clear_z_condition_readout_modules()
                self.evidence_gap_cond_mlp = None
                self.evidence_gap_cond_film_mlp = None
                self.evidence_gap_cond_z_mlp = self._build_condition_cond_mlp(
                    d_model, bottleneck_dim, cond_mlp_hidden, cond_mlp_nlayers
                )
                self.evidence_gap_cond_c_mlp = self._build_condition_cond_mlp(
                    cond_embed_dim, bottleneck_dim, cond_mlp_hidden, cond_mlp_nlayers
                )
                self.evidence_gap_cond_z_mlp.apply(self._init_condition_cond_mlp_weights)
                self.evidence_gap_cond_c_mlp.apply(self._init_condition_cond_mlp_weights)
            elif self.evidence_gap_condition_readout == "cond_film_mlp":
                cond_embed_dim = self._init_v25_condition_scalar_embed(configs)
                bottleneck_dim, cond_mlp_hidden, cond_mlp_nlayers = (
                    self._condition_predictor_dims(configs)
                )
                self._clear_z_condition_readout_modules()
                self.evidence_gap_cond_mlp = None
                self.evidence_gap_cond_z_mlp = None
                self.evidence_gap_cond_c_mlp = None
                self.evidence_gap_cond_film_mlp = ConditionFilmMLP(
                    d_model,
                    cond_embed_dim,
                    bottleneck_dim,
                    cond_mlp_hidden,
                    cond_mlp_nlayers,
                )
                self.evidence_gap_cond_film_mlp.apply(self._init_condition_cond_mlp_weights)
            elif self.evidence_gap_condition_readout in {
                "cond_res_mlp",
                "cond_res_film_mlp",
                "cond_gate_bottleneck",
                "cond_mul_bottleneck",
                "cond_xattn_bottleneck",
                "cond_blend_mlp",
            }:
                cond_embed_dim = self._init_v25_condition_scalar_embed(configs)
                bottleneck_dim, cond_mlp_hidden, cond_mlp_nlayers = (
                    self._condition_predictor_dims(configs)
                )
                self._clear_z_condition_readout_modules()
                self._clear_bottleneck_condition_modules()
                self.evidence_gap_cond_mlp = None
                self.evidence_gap_cond_z_mlp = None
                self.evidence_gap_cond_c_mlp = None
                self.evidence_gap_cond_film_mlp = None
                readout = self.evidence_gap_condition_readout
                if readout == "cond_res_mlp":
                    self.evidence_gap_cond_res_mlp = self._build_condition_cond_mlp(
                        d_model + cond_embed_dim,
                        bottleneck_dim,
                        cond_mlp_hidden,
                        cond_mlp_nlayers,
                    )
                    self.evidence_gap_cond_res_mlp.apply(
                        self._init_condition_cond_mlp_weights
                    )
                    self._zero_init_last_linear(self.evidence_gap_cond_res_mlp)
                elif readout == "cond_res_film_mlp":
                    self.evidence_gap_cond_res_film_mlp = ConditionFilmMLP(
                        d_model,
                        cond_embed_dim,
                        bottleneck_dim,
                        cond_mlp_hidden,
                        cond_mlp_nlayers,
                    )
                    self.evidence_gap_cond_res_film_mlp.apply(
                        self._init_condition_cond_mlp_weights
                    )
                    self.evidence_gap_cond_res_film_mlp.zero_init_output()
                elif readout == "cond_gate_bottleneck":
                    gate_rank = cond_hidden
                    self.evidence_gap_cond_b_norm_no_affine = nn.LayerNorm(
                        bottleneck_dim, elementwise_affine=False
                    )
                    self.evidence_gap_cond_b_to_basis = nn.Linear(
                        bottleneck_dim, gate_rank, bias=False
                    )
                    nn.init.normal_(self.evidence_gap_cond_b_to_basis.weight, std=0.02)
                    self.evidence_gap_cond_b_cond_to_gate = nn.Sequential(
                        nn.Linear(cond_embed_dim, cond_hidden),
                        nn.GELU(),
                        nn.Linear(cond_hidden, gate_rank, bias=False),
                    )
                    nn.init.normal_(
                        self.evidence_gap_cond_b_cond_to_gate[-1].weight, std=0.02
                    )
                    self.evidence_gap_cond_b_basis_to_b = nn.Linear(
                        gate_rank, bottleneck_dim, bias=False
                    )
                    nn.init.normal_(self.evidence_gap_cond_b_basis_to_b.weight, std=0.005)
                elif readout == "cond_mul_bottleneck":
                    self.evidence_gap_cond_mul_mlp = self._build_condition_cond_mlp(
                        cond_embed_dim,
                        bottleneck_dim,
                        cond_mlp_hidden,
                        max(cond_mlp_nlayers, 2),
                    )
                    self.evidence_gap_cond_mul_mlp.apply(
                        self._init_condition_cond_mlp_weights
                    )
                    self._zero_init_last_linear(self.evidence_gap_cond_mul_mlp)
                elif readout == "cond_xattn_bottleneck":
                    self.evidence_gap_cond_xattn = ConditionCrossAttnDelta(
                        d_model,
                        cond_embed_dim,
                        bottleneck_dim,
                        n_heads=4,
                    )
                elif readout == "cond_blend_mlp":
                    self.evidence_gap_cond_blend_mlp = self._build_condition_cond_mlp(
                        d_model + cond_embed_dim,
                        bottleneck_dim,
                        cond_mlp_hidden,
                        cond_mlp_nlayers,
                    )
                    self.evidence_gap_cond_blend_mlp.apply(
                        self._init_condition_cond_mlp_weights
                    )
                    self.evidence_gap_cond_blend_gate = nn.Sequential(
                        nn.Linear(cond_embed_dim, cond_hidden),
                        nn.GELU(),
                        nn.Linear(cond_hidden, 1),
                    )
                    nn.init.zeros_(self.evidence_gap_cond_blend_gate[-1].weight)
                    nn.init.constant_(self.evidence_gap_cond_blend_gate[-1].bias, -4.0)
            else:
                scalar_feature_dim = 1 + 2 * self.evidence_gap_condition_scalar_n_freqs
                scalar_embed_dim = max(1, self.evidence_gap_condition_scalar_embed_dim)
                cond_embed_dim = 2 * scalar_embed_dim + self.evidence_gap_condition_view_embed_dim
                self.register_buffer(
                    "evidence_gap_condition_fourier_freqs",
                    torch.pow(
                        torch.tensor(2.0),
                        torch.arange(self.evidence_gap_condition_scalar_n_freqs).float(),
                    ),
                    persistent=False,
                )
                self.evidence_gap_condition_ratio_embed = nn.Sequential(
                    nn.Linear(scalar_feature_dim, scalar_embed_dim),
                    nn.GELU(),
                    nn.Linear(scalar_embed_dim, scalar_embed_dim),
                )
                self.evidence_gap_condition_position_embed = nn.Sequential(
                    nn.Linear(scalar_feature_dim, scalar_embed_dim),
                    nn.GELU(),
                    nn.Linear(scalar_embed_dim, scalar_embed_dim),
                )
                self.evidence_gap_condition_norm = nn.LayerNorm(d_model)
                condition_delta = nn.Sequential(
                    nn.Linear(d_model + cond_embed_dim, cond_hidden),
                    nn.GELU(),
                    nn.Linear(cond_hidden, d_model),
                )
                nn.init.zeros_(condition_delta[-1].weight)
                nn.init.zeros_(condition_delta[-1].bias)
                if self.evidence_gap_condition_readout == "direction":
                    self.evidence_gap_condition_adapter = None
                    self.evidence_gap_condition_direction = condition_delta
                else:
                    self.evidence_gap_condition_adapter = condition_delta
                    self.evidence_gap_condition_direction = None
                self.evidence_gap_condition_norm_no_affine = None
                self.evidence_gap_condition_z_to_basis = None
                self.evidence_gap_condition_cond_to_gate = None
                self.evidence_gap_condition_basis_to_z = None
                self.evidence_gap_condition_mlp = None
                self.evidence_gap_cond_mlp = None
                self.evidence_gap_cond_z_mlp = None
                self.evidence_gap_cond_c_mlp = None
                self.evidence_gap_cond_film_mlp = None
        else:
            self.evidence_gap_condition_view_embed = None
            self.evidence_gap_condition_mlp = None
            self.evidence_gap_condition_ratio_embed = None
            self.evidence_gap_condition_position_embed = None
            self.evidence_gap_condition_norm = None
            self.evidence_gap_condition_adapter = None
            self.evidence_gap_condition_direction = None
            self.evidence_gap_condition_norm_no_affine = None
            self.evidence_gap_condition_z_to_basis = None
            self.evidence_gap_condition_cond_to_gate = None
            self.evidence_gap_condition_basis_to_z = None
            self.evidence_gap_cond_mlp = None
            self.evidence_gap_cond_z_mlp = None
            self.evidence_gap_cond_c_mlp = None
            self.evidence_gap_cond_film_mlp = None
        if (
            self.evidence_gap_distill
            and not self.evidence_gap_condition
            and self.backbone.n_cls_tokens != 1
            and self.backbone.n_cls_tokens < self.evidence_gap_n_bins
        ):
            raise ValueError(
                "evidence_gap_distill requires n_cls_tokens >= "
                f"{self.evidence_gap_n_bins}, got {self.backbone.n_cls_tokens}. "
                "Use --n_cls_tokens 6 for gap-bin CLS routing, --n_cls_tokens 1 with "
                "--evidence_gap_condition 0 for raw CLS ablation, or keep "
                "--evidence_gap_condition 1 for relation-conditioned routing."
            )

        self._maybe_compile_backbones(configs)

    def _maybe_compile_backbones(self, configs):
        if not getattr(configs, 'compile_backbone', False):
            return
        # defaultdisable compile (more stable, avoids occasional problems under dynamic shape); it is only enabled when the environment variable is explicitly set
        # usage：export TED_ENABLE_COMPILE_BACKBONE=1
        enable_compile = os.environ.get("TED_ENABLE_COMPILE_BACKBONE", "0").strip().lower()
        if enable_compile not in {"1", "true", "yes", "on"}:
            rk = getattr(configs, 'local_rank', 0)
            if rk == 0 or rk is None:
                print("[TED] compile_backbone requested but disabled (set TED_ENABLE_COMPILE_BACKBONE=1 to enable).")
            return
        try:
            # Curriculum causes T and patch number N to change at each step, must be dynamic; fullgraph=False reduces control flow compilation failures.
            self.backbone = torch.compile(self.backbone, dynamic=True, fullgraph=False)
            self.teacher = torch.compile(self.teacher, dynamic=True, fullgraph=False)
            rk = getattr(configs, 'local_rank', 0)
            if rk == 0 or rk is None:
                print('[TED] torch.compile(backbone + teacher) enabled (dynamic=True)')
        except Exception as e:
            rk = getattr(configs, 'local_rank', 0)
            if rk == 0 or rk is None:
                print(f'[TED] torch.compile skipped: {e}')

    @staticmethod
    def _teacher_target_name(student_name):
        if "norm_teacher" in student_name:
            return None
        if "norm_student" in student_name:
            return student_name.replace("norm_student", "norm_teacher")
        return student_name

    @torch.no_grad()
    def _copy_student_to_teacher(self):
        teacher_params = dict(self.teacher.named_parameters())
        for name_s, param_s in self.backbone.named_parameters():
            target_name = self._teacher_target_name(name_s)
            if target_name is None:
                continue
            param_t = teacher_params.get(target_name)
            if param_t is not None:
                param_t.copy_(param_s)

    def _sample_missing_mask_embed_keep(self, batch_size, device):
        """Per-sample keep mask for missing mask embed dropout; shared by teacher/student."""
        bb = self.backbone
        if not self.training:
            return None
        if not getattr(bb, "use_missing_mask_embed", False):
            return None
        drop_p = float(getattr(bb, "missing_mask_embed_dropout", 0.0))
        if drop_p <= 0.0:
            return None
        return torch.rand(batch_size, 1, 1, device=device, dtype=torch.float32) >= drop_p

    @staticmethod
    def _repeat_missing_mask_embed_keep(keep, repeat_factor):
        if keep is None or int(repeat_factor) <= 1:
            return keep
        return keep.repeat(int(repeat_factor), 1, 1)

    def _build_ema_param_lists(self):
        student_param_list = []
        teacher_param_list = []
        teacher_params = dict(self.teacher.named_parameters())

        for name_s, param_s in self.backbone.named_parameters():
            target_name = self._teacher_target_name(name_s)
            if target_name is None:
                continue

            param_t = teacher_params.get(target_name)
            if param_t is None:
                continue
            if param_t.shape != param_s.shape:
                raise ValueError(
                    f"EMA shape mismatch: student {name_s} {tuple(param_s.shape)} "
                    f"-> teacher {target_name} {tuple(param_t.shape)}"
                )

            student_param_list.append(param_s)
            teacher_param_list.append(param_t)

        return student_param_list, teacher_param_list

    def _update_teacher(self, m=None):
        """EMA teacher update for TED, with normalization-layer mapping."""
        if m is None:
            m = 0.992

        if self.ema_params_lists is None:
            self.ema_params_lists = self._build_ema_param_lists()
        student_param_list, teacher_param_list = self.ema_params_lists

        with torch.no_grad():
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    @staticmethod
    def _parse_int_list(value, default):
        if value is None:
            return list(default)
        if isinstance(value, (list, tuple)):
            vals = [int(v) for v in value]
        else:
            vals = [int(v.strip()) for v in str(value).split(",") if v.strip()]
        return vals if len(vals) > 0 else list(default)

    def _num_patches_for_len(self, length):
        length = int(length)
        if length <= self.patch_len:
            return 1
        return int(math.ceil((length - self.patch_len) / self.stride) + 1)

    def _egap_log_view_lengths(self, tag: str, **fields) -> None:
        max_logs = int(os.environ.get("EGAP_LENGTH_DEBUG_MAX", "0"))
        if max_logs <= 0:
            return
        count = int(getattr(self, "_egap_log_count", 0))
        if count >= max_logs:
            return
        rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        if rank != 0:
            return
        self._egap_log_count = count + 1
        step = int(getattr(self, "_current_iteration", count))
        parts = [f"[EGAP-LEN #{count + 1} iter={step} {tag}]"]
        for key, value in fields.items():
            parts.append(f"{key}={value}")
        print(" ".join(parts), flush=True)

    def _gap_cls_id(self, gap_tokens):
        gap_tokens = int(gap_tokens)
        for idx, upper in enumerate(self.evidence_gap_cls_bins):
            if gap_tokens <= int(upper):
                return idx
        return len(self.evidence_gap_cls_bins)

    def _student_context_target_attn(self):
        """Only enable disjoint attn when JEPA-CE mode explicitly requests it."""
        if str(getattr(self, "ibot_context_target_attn", "full")).lower() == "disjoint":
            return "disjoint"
        return "full"

    def _make_ibot_collated_masks(
        self, B, num_tokens, device, mask_rate_v1, mask_rate_v2
    ):
        """
        Patch-mask factory for evidence-gap global students.
        Default: block-biased patch sampler. JEPA-CE: contiguous target blocks
        via utils.jepa_masking (imported only when selected).
        """
        if mask_rate_v2 is None:
            mask_ratio_tuple = (mask_rate_v1, mask_rate_v1)
        else:
            mask_ratio_tuple = (
                min(mask_rate_v1, mask_rate_v2),
                max(mask_rate_v1, mask_rate_v2),
            )
        mode = str(getattr(self, "ibot_target_mode", "dinov3")).lower()
        if mode == "jepa_block":
            from utils.jepa_masking import random_patch_masking_jepa_blocks

            rmin = float(getattr(self, "ibot_jepa_target_ratio_min", 0.25))
            rmax = float(getattr(self, "ibot_jepa_target_ratio_max", 0.5))
            return random_patch_masking_jepa_blocks(
                B,
                (rmin, rmax),
                float(self.mask_sample_probability),
                int(num_tokens),
                device,
                n_blocks=int(getattr(self, "ibot_jepa_n_blocks", 1)),
                min_context_patches=1,
            )
        return random_patch_masking_dinov3_style(
            B,
            mask_ratio_tuple,
            float(self.mask_sample_probability),
            int(num_tokens),
            device,
            block_ratio=self.block_mask_ratio,
        )

    def _evidence_gap_teacher_token_range(self):
        vals = []
        for length in self.evidence_gap_teacher_lengths:
            try:
                vals.append(max(1, self._num_patches_for_len(int(length))))
            except Exception:
                continue
        if not vals:
            vals = [max(1, int(self.num_patches))]
        return float(min(vals)), float(max(vals))

    def _evidence_gap_teacher_length_range(self):
        vals = [int(v) for v in self.evidence_gap_teacher_lengths if int(v) > 0]
        if not vals:
            vals = [max(1, int(getattr(self, "num_patches", 1)))]
        return float(min(vals)), float(max(vals))

    @staticmethod
    def _log_normalize_tensor(value, min_value, max_value):
        min_value = max(float(min_value), 1e-6)
        max_value = max(float(max_value), min_value + 1e-6)
        denom = math.log(max_value) - math.log(min_value)
        value = value.float().clamp(min=min_value, max=max_value)
        out = 2.0 * (torch.log(value) - math.log(min_value)) / max(denom, 1e-6) - 1.0
        return out.clamp(-1.0, 1.0)

    @staticmethod
    def _build_condition_cond_mlp(in_dim, bottleneck_dim, hidden_dim, nlayers):
        nlayers = max(int(nlayers), 1)
        if nlayers == 1:
            return nn.Linear(in_dim, bottleneck_dim)
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(nlayers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def _init_condition_cond_mlp_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _condition_predictor_dims(self, configs):
        bottleneck_dim = int(getattr(configs, "dino_head_bottleneck_dim", 256))
        hidden_dim = int(getattr(configs, "dino_head_hidden_dim", 1536))
        nlayers = int(getattr(configs, "dino_head_nlayers", 3))
        return bottleneck_dim, hidden_dim, nlayers

    def _init_v25_condition_scalar_embed(self, configs):
        scalar_feature_dim = 1 + 2 * self.evidence_gap_condition_scalar_n_freqs
        scalar_embed_dim = max(1, self.evidence_gap_condition_scalar_embed_dim)
        cond_embed_dim = 2 * scalar_embed_dim + self.evidence_gap_condition_view_embed_dim
        self.register_buffer(
            "evidence_gap_condition_fourier_freqs",
            torch.pow(
                torch.tensor(2.0),
                torch.arange(self.evidence_gap_condition_scalar_n_freqs).float(),
            ),
            persistent=False,
        )
        self.evidence_gap_condition_ratio_embed = nn.Sequential(
            nn.Linear(scalar_feature_dim, scalar_embed_dim),
            nn.GELU(),
            nn.Linear(scalar_embed_dim, scalar_embed_dim),
        )
        self.evidence_gap_condition_position_embed = nn.Sequential(
            nn.Linear(scalar_feature_dim, scalar_embed_dim),
            nn.GELU(),
            nn.Linear(scalar_embed_dim, scalar_embed_dim),
        )
        return cond_embed_dim

    def _clear_z_condition_readout_modules(self):
        self.evidence_gap_condition_norm = None
        self.evidence_gap_condition_adapter = None
        self.evidence_gap_condition_direction = None
        self.evidence_gap_condition_norm_no_affine = None
        self.evidence_gap_condition_z_to_basis = None
        self.evidence_gap_condition_cond_to_gate = None
        self.evidence_gap_condition_basis_to_z = None
        self.evidence_gap_condition_mlp = None

    def _clear_bottleneck_condition_modules(self):
        self.evidence_gap_cond_res_mlp = None
        self.evidence_gap_cond_res_film_mlp = None
        self.evidence_gap_cond_b_norm_no_affine = None
        self.evidence_gap_cond_b_to_basis = None
        self.evidence_gap_cond_b_cond_to_gate = None
        self.evidence_gap_cond_b_basis_to_b = None
        self.evidence_gap_cond_mul_mlp = None
        self.evidence_gap_cond_xattn = None
        self.evidence_gap_cond_blend_mlp = None
        self.evidence_gap_cond_blend_gate = None

    @staticmethod
    def _zero_init_last_linear(module):
        if isinstance(module, nn.Sequential):
            last = module[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                if last.bias is not None:
                    nn.init.constant_(last.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _dino_head_mlp_bottleneck(self, z):
        return self.backbone.dino_head.mlp(z)

    def _dino_head_bottleneck_logits(self, bottleneck):
        eps = 1e-6 if bottleneck.dtype == torch.float16 else 1e-12
        normed = F.normalize(bottleneck, dim=-1, p=2, eps=eps)
        return self.backbone.dino_head.last_layer(normed)

    def _build_evidence_gap_condition_num(
        self,
        teacher_tokens,
        student_tokens,
        start_frac,
        end_frac,
    ):
        ratio_min = max(float(self.evidence_gap_student_ratio_min), 1e-6)
        ratio_max = max(float(self.evidence_gap_student_ratio_max), ratio_min + 1e-6)

        start_frac = start_frac.float().clamp(0.0, 1.0)
        end_frac = end_frac.float().clamp(0.0, 1.0)
        ratio_value = torch.full_like(
            start_frac,
            float(max(1, int(student_tokens))) / float(max(1, int(teacher_tokens))),
        )

        ratio_scaled = self._log_normalize_tensor(
            ratio_value, ratio_min, ratio_max
        )
        center_frac = (0.5 * (start_frac + end_frac)).clamp(0.0, 1.0)
        relative_position = (2.0 * center_frac - 1.0).clamp(-1.0, 1.0)
        return torch.stack([ratio_scaled, relative_position], dim=-1)

    def _build_condition_for_short_in_teacher(
        self,
        short_abs_starts,
        short_len,
        short_tokens,
        teacher_start,
        teacher_len,
        timeline_len=None,
    ):
        teacher_tokens = self._num_patches_for_len(int(teacher_len))
        short_tokens = int(short_tokens)
        if timeline_len is not None or self._is_v25_condition_version():
            tl = int(timeline_len) if timeline_len is not None else int(teacher_len)
            return self._build_single_teacher_short_condition(
                short_abs_starts,
                int(short_len),
                int(short_tokens),
                int(teacher_start),
                int(teacher_len),
                tl,
            )
        short_offset = short_abs_starts.float() - float(teacher_start)
        short_start_token = short_offset / max(float(self.stride), 1.0)
        start_frac = short_start_token / max(float(teacher_tokens), 1.0)
        end_frac = (
            short_start_token + float(short_tokens)
        ) / max(float(teacher_tokens), 1.0)
        return self._build_evidence_gap_condition_num(
            teacher_tokens,
            short_tokens,
            start_frac,
            end_frac,
        )

    def _build_v4_condition_num(
        self,
        student_abs_starts,
        student_len,
        teacher_start,
        teacher_len,
        timeline_len,
    ):
        """v4 scaleOffset: teacher scale + timeline offset (center_t - center_s)/(T-1)."""
        device = student_abs_starts.device
        batch_size = int(student_abs_starts.reshape(-1).numel())
        teacher_len = int(teacher_len)
        teacher_start = int(teacher_start)
        student_len = int(student_len)
        timeline_len = int(timeline_len)

        teacher_min, teacher_max = self._evidence_gap_teacher_length_range()
        scale_value = torch.full(
            (batch_size,),
            float(teacher_len),
            device=device,
            dtype=torch.float32,
        )
        scale_scaled = self._log_normalize_tensor(scale_value, teacher_min, teacher_max)

        center_t = float(teacher_start) + 0.5 * float(teacher_len)
        s_start = student_abs_starts.float().reshape(-1)
        if int(s_start.numel()) == 1:
            s_start = s_start.expand(batch_size)
        elif int(s_start.numel()) != batch_size:
            raise ValueError(
                f"student_abs_starts must be scalar or [B={batch_size}], "
                f"got shape {tuple(student_abs_starts.shape)}"
            )
        center_s = s_start + 0.5 * float(student_len)
        denom = max(float(timeline_len) - 1.0, 1.0)
        offset = ((float(center_t) - center_s) / denom).clamp(-1.0, 1.0)
        return torch.stack([scale_scaled, offset], dim=-1)

    def _is_v25_condition_version(self):
        return str(self.evidence_gap_version) in ("v2.5", "v25")

    def _timeline_offset_from_centers(
        self,
        student_abs_starts,
        student_len,
        teacher_start,
        teacher_len,
        timeline_len,
        batch_size,
    ):
        center_t = float(teacher_start) + 0.5 * float(teacher_len)
        s_start = student_abs_starts.float().reshape(-1)
        if int(s_start.numel()) == 1:
            s_start = s_start.expand(batch_size)
        elif int(s_start.numel()) != batch_size:
            raise ValueError(
                f"student_abs_starts must be scalar or [B={batch_size}], "
                f"got shape {tuple(student_abs_starts.shape)}"
            )
        center_s = s_start + 0.5 * float(student_len)
        denom = max(float(timeline_len) - 1.0, 1.0)
        return ((center_t - center_s) / denom).clamp(-1.0, 1.0)

    def _build_v25_condition_num(
        self,
        short_abs_starts,
        short_len,
        short_tokens,
        teacher_start,
        teacher_len,
        teacher_tokens,
        timeline_len,
    ):
        """v2.5 ratioTimelineOffset: v2 ratio + v4 timeline offset (shorts only)."""
        device = short_abs_starts.device
        batch_size = int(short_abs_starts.reshape(-1).numel())
        ratio_min = max(float(self.evidence_gap_student_ratio_min), 1e-6)
        ratio_max = max(float(self.evidence_gap_student_ratio_max), ratio_min + 1e-6)
        ratio_value = torch.full(
            (batch_size,),
            float(max(1, int(short_tokens))) / float(max(1, int(teacher_tokens))),
            device=device,
            dtype=torch.float32,
        )
        ratio_scaled = self._log_normalize_tensor(ratio_value, ratio_min, ratio_max)
        offset = self._timeline_offset_from_centers(
            short_abs_starts,
            int(short_len),
            int(teacher_start),
            int(teacher_len),
            int(timeline_len),
            batch_size,
        )
        return torch.stack([ratio_scaled, offset], dim=-1)

    def _build_single_teacher_short_condition(
        self,
        short_abs_starts,
        short_len,
        short_tokens,
        teacher_start,
        teacher_len,
        timeline_len,
        start_frac=None,
        end_frac=None,
    ):
        """Condition for crop/random short views only (not global student)."""
        if self.evidence_gap_version == "v4":
            return self._build_v4_condition_num(
                short_abs_starts,
                int(short_len),
                int(teacher_start),
                int(teacher_len),
                int(timeline_len),
            )
        if self._is_v25_condition_version():
            teacher_tokens = self._num_patches_for_len(int(teacher_len))
            tl = int(timeline_len) if timeline_len is not None else int(teacher_len)
            return self._build_v25_condition_num(
                short_abs_starts,
                int(short_len),
                int(short_tokens),
                int(teacher_start),
                int(teacher_len),
                int(teacher_tokens),
                tl,
            )
        teacher_tokens = self._num_patches_for_len(int(teacher_len))
        return self._build_evidence_gap_condition_num(
            teacher_tokens,
            int(short_tokens),
            start_frac,
            end_frac,
        )

    def _build_dual_teacher_cross_condition(
        self,
        student_abs_starts,
        student_len,
        short_tokens,
        teacher_start,
        teacher_len,
        timeline_len,
    ):
        if self.evidence_gap_version == "v4":
            return self._build_v4_condition_num(
                student_abs_starts,
                int(student_len),
                int(teacher_start),
                int(teacher_len),
                int(timeline_len),
            )
        if self._is_v25_condition_version():
            return self._build_single_teacher_short_condition(
                student_abs_starts,
                int(student_len),
                int(short_tokens),
                int(teacher_start),
                int(teacher_len),
                int(timeline_len),
            )
        return self._build_condition_for_short_in_teacher(
            student_abs_starts,
            int(student_len),
            int(short_tokens),
            int(teacher_start),
            int(teacher_len),
        )

    def _sample_alt_teacher_containing_short(
        self,
        T,
        short_abs_starts,
        short_len,
        used_teachers,
        device,
    ):
        short_len = int(short_len)
        T = int(T)
        short_abs_starts = short_abs_starts.long()
        min_abs = int(short_abs_starts.min().item())
        max_end = int((short_abs_starts + short_len).max().item())

        teacher_choices = [
            int(v) for v in self.evidence_gap_teacher_lengths if 0 < int(v) <= T
        ]
        if len(teacher_choices) == 0:
            teacher_choices = [T]

        candidates = []
        for teacher_len in teacher_choices:
            teacher_len = int(teacher_len)
            if teacher_len < short_len:
                continue
            min_start = max(0, max_end - teacher_len)
            max_start = min(min_abs, T - teacher_len)
            if min_start > max_start:
                continue
            start = int(
                torch.randint(
                    min_start,
                    max_start + 1,
                    (1,),
                    device=device,
                ).item()
            )
            spec = (start, teacher_len)
            if spec in used_teachers:
                continue
            candidates.append(spec)

        if len(candidates) == 0:
            return None

        choice_idx = int(
            torch.randint(len(candidates), (1,), device=device).item()
        )
        return candidates[choice_idx]

    def _sample_dual_teacher_starts(self, T, teacher_len, device):
        T = int(T)
        teacher_len = int(teacher_len)
        if teacher_len >= T:
            return 0, 0
        max_start = T - teacher_len
        start_a = int(torch.randint(max_start + 1, (1,), device=device).item())
        if max_start <= 0:
            return start_a, start_a
        for _ in range(12):
            start_b = int(torch.randint(max_start + 1, (1,), device=device).item())
            if start_b != start_a:
                return start_a, start_b
        start_b = (start_a + max(1, max_start // 2)) % (max_start + 1)
        if start_b == start_a:
            start_b = (start_a + 1) % (max_start + 1)
        return start_a, start_b

    def _patch_time_span(self, patch_index):
        p_start = int(patch_index) * int(self.stride)
        p_end = p_start + int(self.patch_len)
        return p_start, p_end

    def _eligible_crop_abs_starts_outside_teacher(
        self, timeline_len, short_len, teacher_start, teacher_len
    ):
        timeline_len = int(timeline_len)
        short_len = int(short_len)
        teacher_start = int(teacher_start)
        teacher_end = int(teacher_start) + int(teacher_len)
        if timeline_len < short_len:
            return []
        eligible = []
        for start in range(0, timeline_len - short_len + 1):
            end = start + short_len
            if end <= teacher_start or start >= teacher_end:
                eligible.append(start)
        return eligible

    def _sample_crop_abs_starts_outside_teacher(
        self,
        n_rows,
        timeline_len,
        short_len,
        teacher_start,
        teacher_len,
        device,
    ):
        eligible = self._eligible_crop_abs_starts_outside_teacher(
            timeline_len, short_len, teacher_start, teacher_len
        )
        if len(eligible) == 0:
            return None
        pick = torch.randint(
            0, len(eligible), (int(n_rows),), device=device
        )
        return torch.tensor(
            [eligible[int(i)] for i in pick.tolist()],
            device=device,
            dtype=torch.long,
        )

    def _eligible_patch_indices_outside_teacher(
        self, timeline_len, teacher_start, teacher_len, device
    ):
        n_patches = self._num_patches_for_len(int(timeline_len))
        teacher_start = int(teacher_start)
        teacher_end = int(teacher_start) + int(teacher_len)
        eligible = []
        for patch_idx in range(int(n_patches)):
            p_start, p_end = self._patch_time_span(patch_idx)
            if p_end <= teacher_start or p_start >= teacher_end:
                eligible.append(int(patch_idx))
        if len(eligible) == 0:
            return None
        return torch.tensor(eligible, device=device, dtype=torch.long)

    def _sample_patch_indices_outside_teacher(
        self,
        n_rows,
        short_tokens,
        timeline_len,
        teacher_start,
        teacher_len,
        device,
    ):
        pool = self._eligible_patch_indices_outside_teacher(
            timeline_len, teacher_start, teacher_len, device
        )
        if pool is None:
            return None
        short_tokens = int(short_tokens)
        if int(pool.numel()) < short_tokens:
            pick = pool[
                torch.randint(0, int(pool.numel()), (n_rows, short_tokens), device=device)
            ]
        else:
            pick = torch.stack(
                [
                    pool[torch.randperm(int(pool.numel()), device=device)[:short_tokens]]
                    for _ in range(int(n_rows))
                ],
                dim=0,
            )
        return torch.sort(pick, dim=1).values

    def _evidence_gap_condition_raw_vector(self, condition_num, view_type, device, dtype):
        cond_flat = condition_num.reshape(-1, condition_num.shape[-1]).to(
            device=device, dtype=dtype
        )
        view_flat = view_type.reshape(-1).to(device=device, dtype=torch.long)
        # view_type: -1=global, 0=crop, 1=random -> embedding indices 0,1,2
        view_emb = self.evidence_gap_condition_view_embed(view_flat + 1).to(dtype=dtype)
        return torch.cat([cond_flat, view_emb], dim=-1)

    def _scalar_condition_fourier(self, value):
        value = value.clamp(-1.0, 1.0)
        freqs = self.evidence_gap_condition_fourier_freqs.to(
            device=value.device, dtype=value.dtype
        )
        angles = math.pi * value * freqs.view(1, -1)
        return torch.cat([value, torch.sin(angles), torch.cos(angles)], dim=-1)

    def _evidence_gap_condition_embedding(self, condition_num, view_type, device, dtype):
        cond_flat = condition_num.reshape(-1, condition_num.shape[-1]).to(
            device=device, dtype=dtype
        )
        ratio = cond_flat[:, 0:1]
        position = cond_flat[:, 1:2]
        ratio_emb = self.evidence_gap_condition_ratio_embed(
            self._scalar_condition_fourier(ratio)
        )
        position_emb = self.evidence_gap_condition_position_embed(
            self._scalar_condition_fourier(position)
        )
        view_flat = view_type.reshape(-1).to(device=device, dtype=torch.long)
        view_emb = self.evidence_gap_condition_view_embed(view_flat + 1).to(dtype=dtype)
        return torch.cat([ratio_emb, position_emb, view_emb], dim=-1)

    def _film_evidence_gap_logits(self, z_cls, condition_num, view_type):
        original_shape = z_cls.shape[:-1]
        z_flat = z_cls.reshape(-1, z_cls.shape[-1])
        n_flat = int(z_flat.shape[0])
        drop_p = float(self.evidence_gap_condition_drop_p)
        if self.training and drop_p > 0.0:
            use_raw = torch.rand(n_flat, device=z_flat.device) < drop_p
        else:
            use_raw = torch.zeros(n_flat, dtype=torch.bool, device=z_flat.device)

        h = z_flat
        cond_mask = ~use_raw
        if cond_mask.any():
            cond = self._evidence_gap_condition_raw_vector(
                condition_num, view_type, z_flat.device, z_flat.dtype
            )
            gamma_beta = self.evidence_gap_condition_mlp(cond[cond_mask])
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            alpha = float(self.evidence_gap_condition_alpha)
            h = z_flat.clone()
            if use_raw.any():
                h[use_raw] = z_flat[use_raw]
            h[cond_mask] = (1.0 + alpha * torch.tanh(gamma)) * z_flat[cond_mask] + alpha * beta

        logits = self.backbone.dino_head(h)
        return logits.view(*original_shape, logits.shape[-1])

    def _adapter_evidence_gap_logits(self, z_cls, condition_num, view_type):
        original_shape = z_cls.shape[:-1]
        z_flat = z_cls.reshape(-1, z_cls.shape[-1])
        n_flat = int(z_flat.shape[0])
        drop_p = float(self.evidence_gap_condition_drop_p)
        if self.training and drop_p > 0.0:
            use_raw = torch.rand(n_flat, device=z_flat.device) < drop_p
        else:
            use_raw = torch.zeros(n_flat, dtype=torch.bool, device=z_flat.device)

        h = z_flat
        cond_mask = ~use_raw
        if cond_mask.any():
            cond = self._evidence_gap_condition_embedding(
                condition_num, view_type, z_flat.device, z_flat.dtype
            )
            z_norm = self.evidence_gap_condition_norm(z_flat)
            adapter_in = torch.cat([z_norm[cond_mask], cond[cond_mask]], dim=-1)
            delta = self.evidence_gap_condition_adapter(adapter_in)
            alpha = float(self.evidence_gap_condition_alpha)
            h = z_flat.clone()
            if use_raw.any():
                h[use_raw] = z_flat[use_raw]
            h[cond_mask] = z_flat[cond_mask] + alpha * delta

        logits = self.backbone.dino_head(h)
        return logits.view(*original_shape, logits.shape[-1])

    def _direction_evidence_gap_logits(self, z_cls, condition_num, view_type):
        original_shape = z_cls.shape[:-1]
        z_flat = z_cls.reshape(-1, z_cls.shape[-1])
        n_flat = int(z_flat.shape[0])
        self._last_evidence_gap_direction_stats = {}
        drop_p = float(self.evidence_gap_condition_drop_p)
        if self.training and drop_p > 0.0:
            use_raw = torch.rand(n_flat, device=z_flat.device) < drop_p
        else:
            use_raw = torch.zeros(n_flat, dtype=torch.bool, device=z_flat.device)

        h = z_flat
        cond_mask = ~use_raw
        if cond_mask.any():
            cond = self._evidence_gap_condition_embedding(
                condition_num, view_type, z_flat.device, z_flat.dtype
            )
            z_normed = self.evidence_gap_condition_norm(z_flat)
            direction_in = torch.cat([z_normed[cond_mask], cond[cond_mask]], dim=-1)
            direction = self.evidence_gap_condition_direction(direction_in)

            # Treat condition as a local direction around z, not as a free
            # condition-specific representation. Remove radial motion and cap
            # the step size before applying the usual alpha scale.
            z_base = z_flat[cond_mask]
            z_unit = F.normalize(z_base.detach(), dim=-1, eps=1e-6)
            direction = direction - (direction * z_unit).sum(dim=-1, keepdim=True) * z_unit
            raw_direction_norm = direction.norm(dim=-1, keepdim=True)
            direction_norm = direction.norm(dim=-1, keepdim=True)
            max_norm = z_base.detach().norm(dim=-1, keepdim=True).clamp_min(1.0)
            direction = direction * torch.clamp(max_norm / direction_norm.clamp_min(1e-6), max=1.0)

            alpha = float(self.evidence_gap_condition_alpha)
            h = z_flat.clone()
            if use_raw.any():
                h[use_raw] = z_flat[use_raw]
            h[cond_mask] = z_base + alpha * direction
            with torch.no_grad():
                step_norm = (alpha * direction).norm(dim=-1)
                z_norm = z_base.detach().norm(dim=-1).clamp_min(1e-6)
                self._last_evidence_gap_direction_stats = {
                    "condition_direction_raw_norm_mean": float(
                        raw_direction_norm.detach().mean().item()
                    ),
                    "condition_direction_step_norm_mean": float(
                        step_norm.detach().mean().item()
                    ),
                    "condition_direction_step_to_z_norm_mean": float(
                        (step_norm / z_norm).detach().mean().item()
                    ),
                }

        logits = self.backbone.dino_head(h)
        return logits.view(*original_shape, logits.shape[-1])

    def _gate_evidence_gap_logits(self, z_cls, condition_num, view_type):
        """Condition gates a z-derived subspace; c cannot answer when z=0."""
        original_shape = z_cls.shape[:-1]
        z_flat = z_cls.reshape(-1, z_cls.shape[-1])
        n_flat = int(z_flat.shape[0])
        self._last_evidence_gap_gate_stats = {}
        drop_p = float(self.evidence_gap_condition_drop_p)
        if self.training and drop_p > 0.0:
            use_raw = torch.rand(n_flat, device=z_flat.device) < drop_p
        else:
            use_raw = torch.zeros(n_flat, dtype=torch.bool, device=z_flat.device)

        h = z_flat
        cond_mask = ~use_raw
        if cond_mask.any():
            cond = self._evidence_gap_condition_embedding(
                condition_num, view_type, z_flat.device, z_flat.dtype
            )
            alpha = float(self.evidence_gap_condition_alpha)
            z_base = z_flat[cond_mask]
            zn = self.evidence_gap_condition_norm_no_affine(z_base)
            u = self.evidence_gap_condition_z_to_basis(zn)
            g_raw = self.evidence_gap_condition_cond_to_gate(cond[cond_mask])
            g = torch.tanh(g_raw)
            u_cond = u * g
            delta = self.evidence_gap_condition_basis_to_z(u_cond)

            h = z_flat.clone()
            if use_raw.any():
                h[use_raw] = z_flat[use_raw]
            h[cond_mask] = z_base + alpha * delta
            with torch.no_grad():
                delta_norm = delta.detach().norm(dim=-1)
                z_norm = z_base.detach().norm(dim=-1).clamp_min(1e-6)
                self._last_evidence_gap_gate_stats = {
                    "condition_gate_mean": float(g.detach().mean().item()),
                    "condition_gate_abs_mean": float(g.detach().abs().mean().item()),
                    "condition_gate_raw_std": float(g_raw.detach().std().item()),
                    "condition_gate_delta_norm_mean": float(delta_norm.mean().item()),
                    "condition_gate_delta_to_z_norm_mean": float(
                        (delta_norm / z_norm).mean().item()
                    ),
                }

        logits = self.backbone.dino_head(h)
        return logits.view(*original_shape, logits.shape[-1])

    def _cond_mlp_evidence_gap_logits(self, z_cls, condition_num, view_type):
        """Short views: concat(z, embed(condition)) -> conditioner MLP -> shared categorical-state layer."""
        original_shape = z_cls.shape[:-1]
        z_flat = z_cls.reshape(-1, z_cls.shape[-1])
        n_flat = int(z_flat.shape[0])
        out_dim = int(self.backbone.dino_head.last_layer.out_features)
        self._last_evidence_gap_cond_mlp_stats = {}
        drop_p = float(self.evidence_gap_condition_drop_p)
        if self.training and drop_p > 0.0:
            use_raw = torch.rand(n_flat, device=z_flat.device) < drop_p
        else:
            use_raw = torch.zeros(n_flat, dtype=torch.bool, device=z_flat.device)

        logits = torch.empty(
            (n_flat, out_dim), device=z_flat.device, dtype=z_flat.dtype
        )
        if use_raw.any():
            logits[use_raw] = self.backbone.dino_head(z_flat[use_raw]).to(
                dtype=z_flat.dtype
            )
        cond_mask = ~use_raw
        if cond_mask.any():
            cond = self._evidence_gap_condition_embedding(
                condition_num, view_type, z_flat.device, z_flat.dtype
            )
            z_cond = z_flat[cond_mask]
            cond_sel = cond[cond_mask]
            bottleneck = self.evidence_gap_cond_mlp(
                torch.cat([z_cond, cond_sel], dim=-1)
            )
            logits[cond_mask] = self._dino_head_bottleneck_logits(bottleneck).to(
                dtype=z_flat.dtype
            )
            with torch.no_grad():
                self._last_evidence_gap_cond_mlp_stats = {
                    "condition_cond_mlp_bottleneck_norm_mean": float(
                        bottleneck.detach().norm(dim=-1).mean().item()
                    ),
                    "condition_cond_mlp_z_norm_mean": float(
                        z_cond.detach().norm(dim=-1).mean().item()
                    ),
                }

        return logits.view(*original_shape, out_dim)

    def _cond_sum_mlp_evidence_gap_logits(self, z_cls, condition_num, view_type):
        """Short views: MLP_z(z) + MLP_c(embed(c)) -> shared categorical-state layer."""
        original_shape = z_cls.shape[:-1]
        z_flat = z_cls.reshape(-1, z_cls.shape[-1])
        n_flat = int(z_flat.shape[0])
        out_dim = int(self.backbone.dino_head.last_layer.out_features)
        self._last_evidence_gap_cond_sum_stats = {}
        drop_p = float(self.evidence_gap_condition_drop_p)
        if self.training and drop_p > 0.0:
            use_raw = torch.rand(n_flat, device=z_flat.device) < drop_p
        else:
            use_raw = torch.zeros(n_flat, dtype=torch.bool, device=z_flat.device)

        logits = torch.empty(
            (n_flat, out_dim), device=z_flat.device, dtype=z_flat.dtype
        )
        if use_raw.any():
            logits[use_raw] = self.backbone.dino_head(z_flat[use_raw]).to(
                dtype=z_flat.dtype
            )
        cond_mask = ~use_raw
        if cond_mask.any():
            cond = self._evidence_gap_condition_embedding(
                condition_num, view_type, z_flat.device, z_flat.dtype
            )
            z_cond = z_flat[cond_mask]
            cond_sel = cond[cond_mask]
            bottleneck = self.evidence_gap_cond_z_mlp(z_cond) + self.evidence_gap_cond_c_mlp(
                cond_sel
            )
            logits[cond_mask] = self._dino_head_bottleneck_logits(bottleneck).to(
                dtype=z_flat.dtype
            )
            with torch.no_grad():
                self._last_evidence_gap_cond_sum_stats = {
                    "condition_cond_sum_bottleneck_norm_mean": float(
                        bottleneck.detach().norm(dim=-1).mean().item()
                    ),
                    "condition_cond_sum_z_branch_norm_mean": float(
                        self.evidence_gap_cond_z_mlp(z_cond)
                        .detach()
                        .norm(dim=-1)
                        .mean()
                        .item()
                    ),
                    "condition_cond_sum_c_branch_norm_mean": float(
                        self.evidence_gap_cond_c_mlp(cond_sel)
                        .detach()
                        .norm(dim=-1)
                        .mean()
                        .item()
                    ),
                }

        return logits.view(*original_shape, out_dim)

    def _cond_film_mlp_evidence_gap_logits(self, z_cls, condition_num, view_type):
        """Short views: FiLM-MLP(z, embed(c)) -> shared categorical-state layer."""
        original_shape = z_cls.shape[:-1]
        z_flat = z_cls.reshape(-1, z_cls.shape[-1])
        n_flat = int(z_flat.shape[0])
        out_dim = int(self.backbone.dino_head.last_layer.out_features)
        self._last_evidence_gap_cond_film_stats = {}
        drop_p = float(self.evidence_gap_condition_drop_p)
        if self.training and drop_p > 0.0:
            use_raw = torch.rand(n_flat, device=z_flat.device) < drop_p
        else:
            use_raw = torch.zeros(n_flat, dtype=torch.bool, device=z_flat.device)

        logits = torch.empty(
            (n_flat, out_dim), device=z_flat.device, dtype=z_flat.dtype
        )
        if use_raw.any():
            logits[use_raw] = self.backbone.dino_head(z_flat[use_raw]).to(
                dtype=z_flat.dtype
            )
        cond_mask = ~use_raw
        if cond_mask.any():
            cond = self._evidence_gap_condition_embedding(
                condition_num, view_type, z_flat.device, z_flat.dtype
            )
            z_cond = z_flat[cond_mask]
            cond_sel = cond[cond_mask]
            bottleneck = self.evidence_gap_cond_film_mlp(z_cond, cond_sel)
            logits[cond_mask] = self._dino_head_bottleneck_logits(bottleneck).to(
                dtype=z_flat.dtype
            )
            with torch.no_grad():
                self._last_evidence_gap_cond_film_stats = {
                    "condition_cond_film_bottleneck_norm_mean": float(
                        bottleneck.detach().norm(dim=-1).mean().item()
                    ),
                }

        return logits.view(*original_shape, out_dim)

    def _short_view_bottleneck_condition_logits(
        self,
        z_cls,
        condition_num,
        view_type,
        *,
        stats_attr,
        apply_bottleneck_fn,
    ):
        """Shared short-view path: b_base=mlp(z), b=fn(b_base,z,c), shared last_layer."""
        original_shape = z_cls.shape[:-1]
        z_flat = z_cls.reshape(-1, z_cls.shape[-1])
        n_flat = int(z_flat.shape[0])
        out_dim = int(self.backbone.dino_head.last_layer.out_features)
        setattr(self, stats_attr, {})
        drop_p = float(self.evidence_gap_condition_drop_p)
        if self.training and drop_p > 0.0:
            use_raw = torch.rand(n_flat, device=z_flat.device) < drop_p
        else:
            use_raw = torch.zeros(n_flat, dtype=torch.bool, device=z_flat.device)

        logits = torch.empty(
            (n_flat, out_dim), device=z_flat.device, dtype=z_flat.dtype
        )
        if use_raw.any():
            logits[use_raw] = self.backbone.dino_head(z_flat[use_raw]).to(
                dtype=z_flat.dtype
            )
        cond_mask = ~use_raw
        if cond_mask.any():
            cond = self._evidence_gap_condition_embedding(
                condition_num, view_type, z_flat.device, z_flat.dtype
            )
            z_cond = z_flat[cond_mask]
            cond_sel = cond[cond_mask]
            b_base = self._dino_head_mlp_bottleneck(z_cond)
            b, stats = apply_bottleneck_fn(b_base, z_cond, cond_sel)
            logits[cond_mask] = self._dino_head_bottleneck_logits(b).to(
                dtype=z_flat.dtype
            )
            if stats:
                with torch.no_grad():
                    setattr(self, stats_attr, stats)
        return logits.view(*original_shape, out_dim)

    def _cond_res_mlp_evidence_gap_logits(self, z_cls, condition_num, view_type):
        alpha = float(self.evidence_gap_condition_alpha)

        def apply_fn(b_base, z_cond, cond_sel):
            delta = self.evidence_gap_cond_res_mlp(
                torch.cat([z_cond, cond_sel], dim=-1)
            )
            b = b_base + alpha * delta
            stats = {
                "condition_cond_res_bottleneck_norm_mean": float(
                    b.detach().norm(dim=-1).mean().item()
                ),
                "condition_cond_res_delta_norm_mean": float(
                    delta.detach().norm(dim=-1).mean().item()
                ),
            }
            return b, stats

        return self._short_view_bottleneck_condition_logits(
            z_cls,
            condition_num,
            view_type,
            stats_attr="_last_evidence_gap_cond_res_stats",
            apply_bottleneck_fn=apply_fn,
        )

    def _cond_res_film_mlp_evidence_gap_logits(self, z_cls, condition_num, view_type):
        alpha = float(self.evidence_gap_condition_alpha)

        def apply_fn(b_base, z_cond, cond_sel):
            delta = self.evidence_gap_cond_res_film_mlp(z_cond, cond_sel)
            b = b_base + alpha * delta
            stats = {
                "condition_cond_res_film_delta_norm_mean": float(
                    delta.detach().norm(dim=-1).mean().item()
                ),
            }
            return b, stats

        return self._short_view_bottleneck_condition_logits(
            z_cls,
            condition_num,
            view_type,
            stats_attr="_last_evidence_gap_cond_res_film_stats",
            apply_bottleneck_fn=apply_fn,
        )

    def _cond_gate_bottleneck_evidence_gap_logits(self, z_cls, condition_num, view_type):
        alpha = float(self.evidence_gap_condition_alpha)

        def apply_fn(b_base, _z_cond, cond_sel):
            bn = self.evidence_gap_cond_b_norm_no_affine(b_base)
            u = self.evidence_gap_cond_b_to_basis(bn)
            g = torch.tanh(self.evidence_gap_cond_b_cond_to_gate(cond_sel))
            delta = self.evidence_gap_cond_b_basis_to_b(u * g)
            b = b_base + alpha * delta
            stats = {
                "condition_cond_gate_b_abs_mean": float(g.detach().abs().mean().item()),
                "condition_cond_gate_b_delta_norm_mean": float(
                    delta.detach().norm(dim=-1).mean().item()
                ),
            }
            return b, stats

        return self._short_view_bottleneck_condition_logits(
            z_cls,
            condition_num,
            view_type,
            stats_attr="_last_evidence_gap_cond_gate_b_stats",
            apply_bottleneck_fn=apply_fn,
        )

    def _cond_mul_bottleneck_evidence_gap_logits(self, z_cls, condition_num, view_type):
        def apply_fn(b_base, _z_cond, cond_sel):
            scale = 1.0 + torch.tanh(self.evidence_gap_cond_mul_mlp(cond_sel))
            b = b_base * scale
            stats = {
                "condition_cond_mul_scale_mean": float(scale.detach().mean().item()),
            }
            return b, stats

        return self._short_view_bottleneck_condition_logits(
            z_cls,
            condition_num,
            view_type,
            stats_attr="_last_evidence_gap_cond_mul_b_stats",
            apply_bottleneck_fn=apply_fn,
        )

    def _cond_xattn_bottleneck_evidence_gap_logits(self, z_cls, condition_num, view_type):
        alpha = float(self.evidence_gap_condition_alpha)

        def apply_fn(b_base, z_cond, cond_sel):
            delta = self.evidence_gap_cond_xattn(z_cond, cond_sel)
            b = b_base + alpha * delta
            stats = {
                "condition_cond_xattn_delta_norm_mean": float(
                    delta.detach().norm(dim=-1).mean().item()
                ),
            }
            return b, stats

        return self._short_view_bottleneck_condition_logits(
            z_cls,
            condition_num,
            view_type,
            stats_attr="_last_evidence_gap_cond_xattn_b_stats",
            apply_bottleneck_fn=apply_fn,
        )

    def _cond_blend_mlp_evidence_gap_logits(self, z_cls, condition_num, view_type):
        def apply_fn(b_base, z_cond, cond_sel):
            s = torch.sigmoid(self.evidence_gap_cond_blend_gate(cond_sel))
            b_alt = self.evidence_gap_cond_blend_mlp(
                torch.cat([z_cond, cond_sel], dim=-1)
            )
            b = (1.0 - s) * b_base + s * b_alt
            stats = {
                "condition_cond_blend_s_mean": float(s.detach().mean().item()),
            }
            return b, stats

        return self._short_view_bottleneck_condition_logits(
            z_cls,
            condition_num,
            view_type,
            stats_attr="_last_evidence_gap_cond_blend_stats",
            apply_bottleneck_fn=apply_fn,
        )

    def _conditioned_evidence_gap_logits(self, z_cls, condition_num, view_type):
        if not self.evidence_gap_condition:
            return self.backbone.dino_head(z_cls)
        # Global student (and other unconditioned rows) pass None condition.
        if condition_num is None or view_type is None:
            return self.backbone.dino_head(z_cls)
        if self.evidence_gap_condition_readout == "film":
            return self._film_evidence_gap_logits(z_cls, condition_num, view_type)
        if self.evidence_gap_condition_readout == "direction":
            return self._direction_evidence_gap_logits(z_cls, condition_num, view_type)
        if self.evidence_gap_condition_readout == "gate":
            return self._gate_evidence_gap_logits(z_cls, condition_num, view_type)
        if self.evidence_gap_condition_readout == "cond_mlp":
            return self._cond_mlp_evidence_gap_logits(z_cls, condition_num, view_type)
        if self.evidence_gap_condition_readout == "cond_sum_mlp":
            return self._cond_sum_mlp_evidence_gap_logits(z_cls, condition_num, view_type)
        if self.evidence_gap_condition_readout == "cond_film_mlp":
            return self._cond_film_mlp_evidence_gap_logits(z_cls, condition_num, view_type)
        if self.evidence_gap_condition_readout == "cond_res_mlp":
            return self._cond_res_mlp_evidence_gap_logits(z_cls, condition_num, view_type)
        if self.evidence_gap_condition_readout == "cond_res_film_mlp":
            return self._cond_res_film_mlp_evidence_gap_logits(
                z_cls, condition_num, view_type
            )
        if self.evidence_gap_condition_readout == "cond_gate_bottleneck":
            return self._cond_gate_bottleneck_evidence_gap_logits(
                z_cls, condition_num, view_type
            )
        if self.evidence_gap_condition_readout == "cond_mul_bottleneck":
            return self._cond_mul_bottleneck_evidence_gap_logits(
                z_cls, condition_num, view_type
            )
        if self.evidence_gap_condition_readout == "cond_xattn_bottleneck":
            return self._cond_xattn_bottleneck_evidence_gap_logits(
                z_cls, condition_num, view_type
            )
        if self.evidence_gap_condition_readout == "cond_blend_mlp":
            return self._cond_blend_mlp_evidence_gap_logits(z_cls, condition_num, view_type)
        return self._adapter_evidence_gap_logits(z_cls, condition_num, view_type)

    def _forward_evidence_gap(
        self,
        x_enc,
        time_mark,
        mask_rate_v1,
        mask_rate_v2=None,
        mode="train",
        imputator=None,
        lon_lat=None,
    ):
        B, T, C = x_enc.shape
        device = x_enc.device
        if lon_lat is not None:
            if lon_lat.shape[0] != B or lon_lat.shape[1] != T or lon_lat.shape[-1] != 2:
                raise ValueError(
                    f"lon_lat expected [B,T,2], got {tuple(lon_lat.shape)} for x_enc [{B},{T},...]"
                )

        missing_mask_orig = torch.isnan(x_enc).any(dim=-1)
        missing_mask = missing_mask_orig.clone()
        x_clean_filled = x_enc.nan_to_num(0.0)
        x_target_perfect = x_clean_filled
        imputator_mode = getattr(self, "imputator_mode", "full")
        if imputator is not None:
            with torch.no_grad():
                imp_device = next(imputator.parameters()).device
                imputed_out = imputator_sliding_window_overlap(
                    x_enc,
                    time_mark,
                    missing_mask_orig,
                    imputator,
                    window_len=getattr(imputator, "pred_len", 366),
                    stride=getattr(self, "imputator_segment_stride", 244),
                    device=device,
                    imp_device=imp_device,
                )
            mask_expanded = missing_mask_orig.unsqueeze(-1).float()
            x_target_perfect = x_clean_filled * (1 - mask_expanded) + imputed_out * mask_expanded
            if imputator_mode in ["full", "woMask", "mixed_teacher"]:
                missing_mask = torch.zeros_like(missing_mask_orig)

        teacher_choices = [
            int(v) for v in self.evidence_gap_teacher_lengths if 0 < int(v) <= int(T)
        ]
        if len(teacher_choices) == 0:
            teacher_choices = [int(T)]
        choice_idx = int(torch.randint(len(teacher_choices), (1,), device=device).item())
        teacher_len = int(max(1, min(T, teacher_choices[choice_idx])))
        teacher_start = 0
        if teacher_len < T:
            teacher_start = int(torch.randint(T - teacher_len + 1, (1,), device=device).item())

        teacher_tokens = self._num_patches_for_len(teacher_len)
        r_min = max(0.0, min(1.0, self.evidence_gap_student_ratio_min))
        r_max = max(r_min, min(1.0, self.evidence_gap_student_ratio_max))
        if teacher_tokens <= 1:
            student_tokens = 1
        else:
            min_student_tokens = max(1, int(math.ceil(r_min * teacher_tokens)))
            max_student_tokens = max(
                min_student_tokens,
                min(teacher_tokens - 1, int(math.floor(r_max * teacher_tokens))),
            )
            student_tokens = int(
                torch.randint(
                    min_student_tokens,
                    max_student_tokens + 1,
                    (1,),
                    device=device,
                ).item()
            )

        if teacher_len <= self.patch_len:
            student_len = teacher_len
            student_patch_offset = 0
        else:
            student_len = min(
                teacher_len,
                self.patch_len + (student_tokens - 1) * self.stride,
            )
            student_tokens = self._num_patches_for_len(student_len)
            max_offset_by_tokens = max(0, teacher_tokens - student_tokens)
            max_offset_by_time = max(0, (teacher_len - student_len) // max(1, self.stride))
            max_patch_offset = min(max_offset_by_tokens, max_offset_by_time)
            student_patch_offset = int(
                torch.randint(max_patch_offset + 1, (1,), device=device).item()
            ) if max_patch_offset > 0 else 0

        student_start = teacher_start + student_patch_offset * self.stride
        gap_tokens = max(0, teacher_tokens - student_tokens)
        gap_cls_id = self._gap_cls_id(gap_tokens)

        def _slice_win(tensor, start, length):
            return tensor[:, start : start + length] if tensor is not None else None

        use_imputed_teacher = imputator is not None and imputator_mode in ["full", "woMask"]
        teacher_base_full = x_target_perfect if use_imputed_teacher else x_clean_filled
        teacher_mask_full = missing_mask if use_imputed_teacher else missing_mask_orig
        student_base_full = x_target_perfect if (imputator is not None and imputator_mode == "woMask") else x_clean_filled
        student_mask_full = missing_mask if (imputator is not None and imputator_mode == "woMask") else missing_mask_orig

        x_teacher_base = _slice_win(teacher_base_full, teacher_start, teacher_len)
        x_student_base = _slice_win(student_base_full, student_start, student_len)
        missing_mask_teacher = _slice_win(teacher_mask_full, teacher_start, teacher_len).float()
        missing_mask_student = _slice_win(student_mask_full, student_start, student_len).float()
        missing_mask_teacher_orig = _slice_win(missing_mask_orig, teacher_start, teacher_len)
        missing_mask_student_orig = _slice_win(missing_mask_orig, student_start, student_len)
        time_mark_teacher = _slice_win(time_mark, teacher_start, teacher_len)
        time_mark_student = _slice_win(time_mark, student_start, student_len)
        lon_lat_teacher = _slice_win(lon_lat, teacher_start, teacher_len)
        lon_lat_student = _slice_win(lon_lat, student_start, student_len)

        geo_keep = None
        if lon_lat is not None and getattr(self.backbone, "lon_lat_proj", None) is not None:
            if self.training and mode == "train" and self.geo_dropout_p > 0:
                geo_keep = (
                    torch.rand(B, 1, 1, device=device, dtype=torch.float32) >= self.geo_dropout_p
                ).to(dtype=x_enc.dtype)
            else:
                geo_keep = torch.ones(B, 1, 1, device=device, dtype=x_enc.dtype)

        x_teacher = get_teacher_input(x_teacher_base, "none", is_train=self.training)
        student_aug = "none" if self.disable_view_augmentation else self.evidence_gap_student_aug
        x_student = get_student_input(x_student_base, student_aug, is_train=self.training)

        if mask_rate_v2 is None:
            mask_ratio_tuple = (mask_rate_v1, mask_rate_v1)
        else:
            mask_ratio_tuple = (min(mask_rate_v1, mask_rate_v2), max(mask_rate_v1, mask_rate_v2))
        collated_masks, mask_indices, _ = self._make_ibot_collated_masks(
            B, student_tokens, device, mask_rate_v1, mask_rate_v2
        )

        teacher_was_training = self.teacher.training
        self.teacher.eval()
        teacher_temp = getattr(self, "_current_teacher_temp", getattr(self, "teacher_temp", 0.07))
        mm_keep = self._sample_missing_mask_embed_keep(B, device)
        with torch.no_grad():
            out_teacher = self.teacher(
                x_teacher,
                missing_mask_teacher,
                time_mark_teacher,
                mask_map=None,
                is_student=False,
                lon_lat=lon_lat_teacher,
                geo_keep=geo_keep,
                missing_mask_embed_keep=mm_keep,
            )
            t_logits_bank = out_teacher["logits_cls_bank"].detach()
            t_logits_global_raw = out_teacher["logits_global"].detach()
            t_z_patch = out_teacher["z_patch_enc"].detach()
        self.teacher.train(teacher_was_training)

        out_student = self.backbone(
            x_student,
            missing_mask_student,
            time_mark_student,
            mask_map=collated_masks,
            is_student=True,
            lon_lat=lon_lat_student,
            geo_keep=geo_keep,
            missing_mask_embed_keep=mm_keep,
            context_target_attn=self._student_context_target_attn(),
        )
        s_logits_bank = out_student["logits_cls_bank"]
        s_z_cls_bank = out_student["z_cls_bank"]
        s_z_patch = out_student["z_patch_enc"]

        t_logits_gap = t_logits_bank[:, gap_cls_id, :]
        s_logits_gap = s_logits_bank[:, gap_cls_id, :]
        s_z_gap = s_z_cls_bank[:, gap_cls_id, :]

        valid_ratio_per_sample = torch.minimum(
            (~missing_mask_teacher_orig).float().mean(dim=-1),
            (~missing_mask_student_orig).float().mean(dim=-1),
        )
        valid_sample_indices = torch.where(
            valid_ratio_per_sample >= float(self.valid_sample_threshold)
        )[0]

        def _filter0(tensor):
            if tensor is None:
                return None
            if valid_sample_indices.numel() == 0:
                return tensor[:0]
            return tensor.index_select(0, valid_sample_indices)

        s_logits_global_valid = _filter0(s_logits_gap).unsqueeze(0)
        t_logits_global_valid = _filter0(t_logits_gap).unsqueeze(0)
        s_z_gap_valid = _filter0(s_z_gap)
        s_z_patch_valid = _filter0(s_z_patch)

        fft_masked_s_pre = None
        fft_masked_t_pre = None
        fft_masked_row_ids = None
        fft_masked_patch_idx = None
        s_patch_masked = None
        t_patch_masked = None
        masks_weight_valid = None
        collated_masks_valid = None
        mask_indices_valid = None
        ibot_denom_rows = None
        all_valid_tokens = None
        s_patch_tokens_all = None
        t_patch_tokens_all = None

        t_z_patch_aligned = t_z_patch[
            :,
            student_patch_offset : student_patch_offset + student_tokens,
            :,
        ]
        if valid_sample_indices.numel() > 0:
            collated_masks_valid = collated_masks.unsqueeze(0)[:, valid_sample_indices, :]
            valid_mask_flat = collated_masks_valid.flatten()
            mask_indices_valid = valid_mask_flat.nonzero(as_tuple=False).squeeze(1)

            orig_valid_student = (~missing_mask_student_orig).float().unsqueeze(-1)
            patch_valid_ratio = patchify(
                orig_valid_student, self.patch_len, self.stride
            ).mean(dim=-1)[:, :student_tokens]
            if imputator is not None and imputator_mode == "full":
                is_patch_reliable = torch.ones_like(patch_valid_ratio, dtype=torch.bool)
            elif getattr(self, "ibot_patch_reliable_mode", "filter") == "keep_all":
                is_patch_reliable = torch.ones_like(patch_valid_ratio, dtype=torch.bool)
            else:
                is_patch_reliable = patch_valid_ratio > 0
            is_patch_reliable_valid = is_patch_reliable.index_select(
                0, valid_sample_indices
            )

            if mask_indices_valid.numel() > 0:
                reliable_flat = is_patch_reliable_valid.unsqueeze(0).flatten()
                reliable_flags = reliable_flat[mask_indices_valid]
                mask_indices_valid = mask_indices_valid[reliable_flags]

            if mask_indices_valid.numel() > 0:
                B_valid = int(valid_sample_indices.numel())
                N_patch = int(student_tokens)
                s_patch_flat = s_z_patch_valid[:, :N_patch, :].reshape(B_valid * N_patch, -1)
                t_patch_valid = t_z_patch_aligned.index_select(0, valid_sample_indices)
                t_patch_flat = t_patch_valid[:, :N_patch, :].reshape(B_valid * N_patch, -1)
                s_pre = torch.index_select(s_patch_flat, dim=0, index=mask_indices_valid)
                t_pre = torch.index_select(t_patch_flat, dim=0, index=mask_indices_valid)
                fft_masked_s_pre = s_pre
                fft_masked_t_pre = t_pre.detach()
                fft_masked_row_ids = mask_indices_valid // N_patch
                fft_masked_patch_idx = mask_indices_valid % N_patch
                s_patch_masked = self.backbone.ibot_head(s_pre)
                with torch.no_grad():
                    t_patch_masked = self.teacher.ibot_head(t_pre).detach()

                valid_tokens_per_row = (
                    (is_patch_reliable_valid[:, :N_patch] > 0)
                    .sum(dim=-1)
                    .float()
                    .clamp(min=1.0)
                )
                retained_per_row = torch.bincount(
                    fft_masked_row_ids,
                    minlength=B_valid,
                ).to(device=device, dtype=valid_tokens_per_row.dtype).clamp(min=1.0)
                masks_weight_valid = (
                    valid_tokens_per_row[fft_masked_row_ids]
                    / retained_per_row[fft_masked_row_ids]
                )
                active_rows = torch.unique(fft_masked_row_ids)
                all_valid_tokens = valid_tokens_per_row[active_rows].sum().clamp(min=1.0)
                ibot_denom_rows = float(max(1, int(active_rows.numel())))

            if self.fft_align_all_patches and s_z_patch_valid.shape[0] > 0:
                t_patch_valid_all = t_z_patch_aligned.index_select(0, valid_sample_indices)
                s_patch_tokens_all = s_z_patch_valid[:, :student_tokens, :]
                t_patch_tokens_all = t_patch_valid_all[:, :student_tokens, :].detach()

        _ssl_nv = int(valid_sample_indices.numel())
        return {
            "z_global": s_z_gap,
            "valid_sample_indices": valid_sample_indices,
            "recon_data": {"rec_seq_valid": None, "target_seq_valid": None},
            "spectral_data": {
                "s_patch_tokens_all": s_patch_tokens_all,
                "t_patch_tokens_all": t_patch_tokens_all,
                "fft_masked_s_pre": fft_masked_s_pre,
                "fft_masked_t_pre": fft_masked_t_pre,
                "fft_masked_row_ids": fft_masked_row_ids,
                "fft_masked_patch_idx": fft_masked_patch_idx,
            },
            "cls_data": {
                "cls_loss_mode": "evidence_gap",
                "s_logits_global_valid": s_logits_global_valid,
                "s_logits_local_valid": None,
                "t_logits_global_valid": t_logits_global_valid,
                "dino_global_scale": 1.0,
                "dino_local_scale": 0.0,
                "gap_cls_id": int(gap_cls_id),
                "gap_tokens": int(gap_tokens),
                "teacher_tokens": int(teacher_tokens),
                "student_tokens": int(student_tokens),
                "teacher_len": int(teacher_len),
                "student_len": int(student_len),
            },
            "patch_data": {
                "s_patch_masked": s_patch_masked,
                "t_patch_masked": t_patch_masked,
                "masks_weight_global_valid": masks_weight_valid,
                "collated_masks_global_valid": collated_masks_valid,
                "mask_indices_list_global_valid": mask_indices_valid,
                "ibot_denom_rows": ibot_denom_rows,
                "all_valid_tokens": all_valid_tokens,
                "ibot_debug": None,
            },
            "koleo_data": {"s_z_cls_flat": s_z_gap_valid if _ssl_nv > 0 else None},
            "temporal_data": {
                "s_z_patch_enc_valid": s_z_patch_valid if _ssl_nv > 0 else None
            },
            "cls_consistency_data": None,
            "lambda_weights": {
                "lambda_recon": 0.0,
                "lambda_fft_align": self.lambda_fft_align,
                "lambda_cls_proto": self.lambda_cls_proto,
                "lambda_patch_proto": self.lambda_patch_proto,
                "lambda_koleo": self.lambda_koleo,
                "lambda_temporal": self.lambda_temporal,
                "lambda_cls_cons": 0.0,
            },
            "teacher_temp": teacher_temp,
            "ssl_num_valid_samples": _ssl_nv,
            "ssl_subgroup_batch_size": int(B),
        }

    def _forward_evidence_gap_v2(
        self,
        x_enc,
        time_mark,
        mask_rate_v1,
        mask_rate_v2=None,
        mode="train",
        imputator=None,
        lon_lat=None,
    ):
        B, T, C = x_enc.shape
        device = x_enc.device
        if lon_lat is not None:
            if lon_lat.shape[0] != B or lon_lat.shape[1] != T or lon_lat.shape[-1] != 2:
                raise ValueError(
                    f"lon_lat expected [B,T,2], got {tuple(lon_lat.shape)} for x_enc [{B},{T},...]"
                )

        missing_mask_orig = torch.isnan(x_enc).any(dim=-1)
        missing_mask = missing_mask_orig.clone()
        x_clean_filled = x_enc.nan_to_num(0.0)
        x_target_perfect = x_clean_filled
        imputator_mode = getattr(self, "imputator_mode", "full")
        if imputator is not None:
            with torch.no_grad():
                imp_device = next(imputator.parameters()).device
                imputed_out = imputator_sliding_window_overlap(
                    x_enc,
                    time_mark,
                    missing_mask_orig,
                    imputator,
                    window_len=getattr(imputator, "pred_len", 366),
                    stride=getattr(self, "imputator_segment_stride", 244),
                    device=device,
                    imp_device=imp_device,
                )
            mask_expanded = missing_mask_orig.unsqueeze(-1).float()
            x_target_perfect = x_clean_filled * (1 - mask_expanded) + imputed_out * mask_expanded
            if imputator_mode in ["full", "woMask", "mixed_teacher"]:
                missing_mask = torch.zeros_like(missing_mask_orig)

        teacher_choices = [
            int(v) for v in self.evidence_gap_teacher_lengths if 0 < int(v) <= int(T)
        ]
        if len(teacher_choices) == 0:
            teacher_choices = [int(T)]
        choice_idx = int(torch.randint(len(teacher_choices), (1,), device=device).item())
        teacher_len = int(max(1, min(T, teacher_choices[choice_idx])))
        teacher_start = 0
        if teacher_len < T:
            teacher_start = int(torch.randint(T - teacher_len + 1, (1,), device=device).item())

        teacher_tokens = self._num_patches_for_len(teacher_len)
        r_min = max(0.0, min(1.0, self.evidence_gap_student_ratio_min))
        r_max = max(r_min, min(1.0, self.evidence_gap_student_ratio_max))
        if teacher_tokens <= 1:
            short_tokens = 1
            short_len = teacher_len
        else:
            min_short_tokens = max(1, int(math.ceil(r_min * teacher_tokens)))
            max_short_tokens = max(
                min_short_tokens,
                min(teacher_tokens - 1, int(math.floor(r_max * teacher_tokens))),
            )
            short_tokens = int(
                torch.randint(
                    min_short_tokens,
                    max_short_tokens + 1,
                    (1,),
                    device=device,
                ).item()
            )
            short_len = min(
                teacher_len,
                self.patch_len + (short_tokens - 1) * self.stride,
            )
            short_tokens = self._num_patches_for_len(short_len)

        self._egap_log_view_lengths(
            "v2",
            path="evidence_gap_v2",
            input_T=int(T),
            teacher_choices=teacher_choices,
            teacher_len=int(teacher_len),
            teacher_start=int(teacher_start),
            teacher_tokens=int(teacher_tokens),
            short_len=int(short_len),
            short_tokens=int(short_tokens),
            ratio_min=float(r_min),
            ratio_max=float(r_max),
            global_student_len=int(teacher_len),
            n_short_crop=int(self.evidence_gap_n_short_crop),
            n_short_random=int(self.evidence_gap_n_short_random),
            dual_cross=0,
        )

        global_gap_tokens = 0
        short_gap_tokens = max(0, teacher_tokens - short_tokens)
        global_gap_cls_id = self._gap_cls_id(global_gap_tokens)
        short_gap_cls_id = self._gap_cls_id(short_gap_tokens)

        def _slice_win(tensor, start, length):
            return tensor[:, start : start + length] if tensor is not None else None

        use_imputed_teacher = imputator is not None and imputator_mode in ["full", "woMask"]
        teacher_base_full = x_target_perfect if use_imputed_teacher else x_clean_filled
        teacher_mask_full = missing_mask if use_imputed_teacher else missing_mask_orig
        student_base_full = (
            x_target_perfect
            if (imputator is not None and imputator_mode == "woMask")
            else x_clean_filled
        )
        student_mask_full = (
            missing_mask
            if (imputator is not None and imputator_mode == "woMask")
            else missing_mask_orig
        )

        independent_fullseq = bool(self.evidence_gap_independent_fullseq)
        x_teacher_base = _slice_win(teacher_base_full, teacher_start, teacher_len)
        # Global student always shares teacher window; only crop/random may sample elsewhere.
        global_student_start = int(teacher_start)
        x_student_global_base = _slice_win(
            student_base_full, teacher_start, teacher_len
        )
        x_student_parent = x_student_global_base
        missing_mask_student_global = _slice_win(
            student_mask_full, teacher_start, teacher_len
        ).float()
        missing_mask_student_parent = missing_mask_student_global
        time_mark_student_global = _slice_win(time_mark, teacher_start, teacher_len)
        lon_lat_student_global = _slice_win(lon_lat, teacher_start, teacher_len)
        missing_mask_teacher = _slice_win(teacher_mask_full, teacher_start, teacher_len).float()
        missing_mask_teacher_orig = _slice_win(missing_mask_orig, teacher_start, teacher_len)
        time_mark_teacher = _slice_win(time_mark, teacher_start, teacher_len)
        lon_lat_teacher = _slice_win(lon_lat, teacher_start, teacher_len)

        geo_keep = None
        if lon_lat is not None and getattr(self.backbone, "lon_lat_proj", None) is not None:
            if self.training and mode == "train" and self.geo_dropout_p > 0:
                geo_keep = (
                    torch.rand(B, 1, 1, device=device, dtype=torch.float32) >= self.geo_dropout_p
                ).to(dtype=x_enc.dtype)
            else:
                geo_keep = torch.ones(B, 1, 1, device=device, dtype=x_enc.dtype)

        no_view_aug = bool(getattr(self, "disable_view_augmentation", False))
        global_aug = "none" if no_view_aug else self.evidence_gap_student_aug
        short_aug = "none" if no_view_aug else "weak_local"

        x_teacher = get_teacher_input(x_teacher_base, "none", is_train=self.training)
        x_student_global = get_student_input(
            x_student_global_base, global_aug, is_train=self.training
        )

        if mask_rate_v2 is None:
            mask_ratio_tuple = (mask_rate_v1, mask_rate_v1)
        else:
            mask_ratio_tuple = (min(mask_rate_v1, mask_rate_v2), max(mask_rate_v1, mask_rate_v2))
        collated_masks, _, _ = self._make_ibot_collated_masks(
            B, teacher_tokens, device, mask_rate_v1, mask_rate_v2
        )

        teacher_was_training = self.teacher.training
        self.teacher.eval()
        teacher_temp = getattr(self, "_current_teacher_temp", getattr(self, "teacher_temp", 0.07))
        mm_keep = self._sample_missing_mask_embed_keep(B, device)
        with torch.no_grad():
            out_teacher = self.teacher(
                x_teacher,
                missing_mask_teacher,
                time_mark_teacher,
                mask_map=None,
                is_student=False,
                lon_lat=lon_lat_teacher,
                geo_keep=geo_keep,
                missing_mask_embed_keep=mm_keep,
            )
            t_logits_bank = out_teacher["logits_cls_bank"].detach()
            t_logits_global_raw = out_teacher["logits_global"].detach()
            t_z_patch = out_teacher["z_patch_enc"].detach()
        self.teacher.train(teacher_was_training)

        out_global = self.backbone(
            x_student_global,
            missing_mask_student_global,
            time_mark_student_global,
            mask_map=collated_masks,
            is_student=True,
            lon_lat=lon_lat_student_global,
            geo_keep=geo_keep,
            missing_mask_embed_keep=mm_keep,
            context_target_attn=self._student_context_target_attn(),
        )
        s_logits_global_bank = out_global["logits_cls_bank"]
        s_z_global_bank = out_global["z_cls_bank"]
        s_z_patch_global = out_global["z_patch_enc"]

        n_short_crop = int(self.evidence_gap_n_short_crop)
        n_short_random = int(self.evidence_gap_n_short_random)
        short_outside_teacher = bool(
            self.evidence_gap_short_outside_teacher
        ) and not independent_fullseq
        short_outside_fallback = 0
        short_batches = []
        short_time_batches = []
        short_mask_batches = []
        short_lon_lat_batches = []
        short_condition_num_batches = []
        short_view_type_batches = []
        crop_starts_all = None

        if n_short_crop > 0:
            n_crop_rows = n_short_crop * B
            batch_idx = torch.arange(n_crop_rows, device=device).unsqueeze(1)
            row_idx = batch_idx.expand(-1, int(short_len))
            crop_on_fullseq = independent_fullseq
            crop_inside_teacher = (not short_outside_teacher) and (not crop_on_fullseq)
            if crop_on_fullseq:
                x_crop_parent = student_base_full.repeat(n_short_crop, 1, 1)
                x_crop, crop_starts = generate_local_view_crop(
                    x_crop_parent, short_len, device
                )
                crop_starts_all = crop_starts.view(n_short_crop, B)
                time_idx = crop_starts.unsqueeze(1) + torch.arange(
                    short_len, device=device
                ).unsqueeze(0)
                parent_mask_rep = student_mask_full.float().repeat(n_short_crop, 1)
                abs_starts_crop = crop_starts_all.long()
                parent_time_src = time_mark
                parent_ll_src = lon_lat
            elif short_outside_teacher:
                abs_starts_flat = self._sample_crop_abs_starts_outside_teacher(
                    n_crop_rows,
                    int(T),
                    int(short_len),
                    int(teacher_start),
                    int(teacher_len),
                    device,
                )
                if abs_starts_flat is None:
                    short_outside_fallback += 1
                    x_crop_parent = x_student_parent.repeat(n_short_crop, 1, 1)
                    x_crop, crop_starts = generate_local_view_crop(
                        x_crop_parent, short_len, device
                    )
                    crop_starts_all = crop_starts.view(n_short_crop, B)
                    time_idx = crop_starts.unsqueeze(1) + torch.arange(
                        short_len, device=device
                    ).unsqueeze(0)
                    parent_mask_rep = missing_mask_student_parent.repeat(n_short_crop, 1)
                    abs_starts_crop = int(teacher_start) + crop_starts_all.long()
                    parent_time_src = time_mark_teacher
                    parent_ll_src = lon_lat_teacher
                else:
                    crop_starts_all = abs_starts_flat.view(n_short_crop, B)
                    x_crop_parent = student_base_full.repeat(n_short_crop, 1, 1)
                    time_idx = abs_starts_flat.unsqueeze(1) + torch.arange(
                        short_len, device=device
                    ).unsqueeze(0)
                    parent_mask_rep = student_mask_full.float().repeat(n_short_crop, 1)
                    x_crop = x_crop_parent[row_idx, time_idx]
                    abs_starts_crop = crop_starts_all.long()
                    parent_time_src = time_mark
                    parent_ll_src = lon_lat
            else:
                x_crop_parent = x_student_parent.repeat(n_short_crop, 1, 1)
                x_crop, crop_starts = generate_local_view_crop(
                    x_crop_parent, short_len, device
                )
                crop_starts_all = crop_starts.view(n_short_crop, B)
                time_idx = crop_starts.unsqueeze(1) + torch.arange(
                    short_len, device=device
                ).unsqueeze(0)
                parent_mask_rep = missing_mask_student_parent.repeat(n_short_crop, 1)
                abs_starts_crop = int(teacher_start) + crop_starts_all.long()
                parent_time_src = time_mark_teacher
                parent_ll_src = lon_lat_teacher

            short_batches.append(
                get_student_input(x_crop, short_aug, is_train=self.training)
            )
            short_mask_batches.append(parent_mask_rep[row_idx, time_idx])
            crop_start_token = (
                abs_starts_crop.float() - float(teacher_start)
            ) / max(float(self.stride), 1.0)
            crop_start_frac = (
                crop_start_token / max(float(teacher_tokens), 1.0)
            ).view(n_short_crop, B)
            crop_end_frac = (
                (crop_start_token + float(short_tokens)) / max(float(teacher_tokens), 1.0)
            ).view(n_short_crop, B)
            for row in range(n_short_crop):
                short_condition_num_batches.append(
                    self._build_single_teacher_short_condition(
                        abs_starts_crop[row],
                        int(short_len),
                        int(short_tokens),
                        int(teacher_start),
                        int(teacher_len),
                        int(T),
                        start_frac=crop_start_frac[row],
                        end_frac=crop_end_frac[row],
                    ).unsqueeze(0)
                )
            short_view_type_batches.append(
                torch.full((n_short_crop, B), 0, device=device, dtype=torch.long)
            )
            if parent_time_src is not None:
                parent_time_rep = parent_time_src.repeat(n_short_crop, 1, 1)
                short_time_batches.append(parent_time_rep[row_idx, time_idx])
            if parent_ll_src is not None:
                parent_ll_rep = parent_ll_src.repeat(n_short_crop, 1, 1)
                short_lon_lat_batches.append(parent_ll_rep[row_idx, time_idx])

        if n_short_random > 0:
            n_rand_rows = n_short_random * B
            batch_idx = torch.arange(n_rand_rows, device=device).unsqueeze(1)
            rand_on_fullseq = independent_fullseq
            rand_inside_teacher = (not short_outside_teacher) and (not rand_on_fullseq)
            if rand_on_fullseq:
                x_rand_parent = student_base_full.repeat(n_short_random, 1, 1)
                x_rand_patches = patchify(x_rand_parent, self.patch_len, self.stride)
                x_rand_sampled, token_indices = generate_local_view_random_sample(
                    x_rand_patches, short_tokens, device
                )
                parent_mask_rep = student_mask_full.float().repeat(n_short_random, 1)
                parent_time_src = time_mark
                parent_ll_src = lon_lat
            elif short_outside_teacher:
                x_rand_parent = student_base_full.repeat(n_short_random, 1, 1)
                x_rand_patches = patchify(x_rand_parent, self.patch_len, self.stride)
                token_indices = self._sample_patch_indices_outside_teacher(
                    n_rand_rows,
                    int(short_tokens),
                    int(T),
                    int(teacher_start),
                    int(teacher_len),
                    device,
                )
                if token_indices is None:
                    short_outside_fallback += 1
                    rand_inside_teacher = True
                else:
                    x_rand_sampled = x_rand_patches[batch_idx, token_indices]
                    parent_mask_rep = student_mask_full.float().repeat(n_short_random, 1)
                    parent_time_src = time_mark
                    parent_ll_src = lon_lat

            if rand_inside_teacher:
                x_rand_parent = x_student_parent.repeat(n_short_random, 1, 1)
                x_rand_patches = patchify(x_rand_parent, self.patch_len, self.stride)
                x_rand_sampled, token_indices = generate_local_view_random_sample(
                    x_rand_patches, short_tokens, device
                )
                parent_mask_rep = missing_mask_student_parent.repeat(n_short_random, 1)
                parent_time_src = time_mark_teacher
                parent_ll_src = lon_lat_teacher
            x_rand = unpatchify(
                x_rand_sampled, short_len, self.patch_len, self.c_out
            )
            mask_patches = patchify(
                parent_mask_rep.unsqueeze(-1).float(), self.patch_len, self.stride
            )
            mask_sampled = mask_patches[batch_idx, token_indices]
            short_batches.append(get_student_input(x_rand, short_aug, is_train=self.training))
            short_mask_batches.append(
                unpatchify(mask_sampled, short_len, self.patch_len, 1).squeeze(-1)
            )
            rand_center_frac = (
                (token_indices.float().mean(dim=1) + 0.5)
                / max(
                    float(teacher_tokens if rand_inside_teacher else self._num_patches_for_len(int(T))),
                    1.0,
                )
            ).view(n_short_random, B)
            token_indices_view = token_indices.view(
                n_short_random, B, short_tokens
            )
            for row in range(n_short_random):
                min_tok = token_indices_view[row].min(dim=-1).values
                if rand_inside_teacher:
                    abs_starts_row = int(teacher_start) + (
                        min_tok.float() * float(self.stride)
                    ).long()
                else:
                    abs_starts_row = (min_tok.float() * float(self.stride)).long()
                short_condition_num_batches.append(
                    self._build_single_teacher_short_condition(
                        abs_starts_row,
                        int(short_len),
                        int(short_tokens),
                        int(teacher_start),
                        int(teacher_len),
                        int(T),
                        start_frac=rand_center_frac[row],
                        end_frac=rand_center_frac[row],
                    ).unsqueeze(0)
                )
            short_view_type_batches.append(
                torch.full((n_short_random, B), 1, device=device, dtype=torch.long)
            )
            if parent_time_src is not None:
                parent_time_rep = parent_time_src.repeat(n_short_random, 1, 1)
                time_patches = patchify(parent_time_rep, self.patch_len, self.stride)
                time_sampled = time_patches[batch_idx, token_indices]
                short_time_batches.append(
                    unpatchify(time_sampled, short_len, self.patch_len, 2)
                )
            if parent_ll_src is not None:
                parent_ll_rep = parent_ll_src.repeat(n_short_random, 1, 1)
                ll_patches = patchify(parent_ll_rep, self.patch_len, self.stride)
                ll_sampled = ll_patches[batch_idx, token_indices]
                short_lon_lat_batches.append(
                    unpatchify(ll_sampled, short_len, self.patch_len, 2)
                )

        n_short_actual = n_short_crop + n_short_random
        s_logits_short_gap = None
        s_z_short_gap = None
        short_condition_num = None
        short_view_type = None
        condition_log = {}
        if n_short_actual > 0 and len(short_batches) > 0:
            x_short = torch.cat(short_batches, dim=0)
            missing_short = torch.cat(short_mask_batches, dim=0).float()
            time_short = (
                torch.cat(short_time_batches, dim=0)
                if len(short_time_batches) > 0
                else None
            )
            lon_lat_short = (
                torch.cat(short_lon_lat_batches, dim=0)
                if len(short_lon_lat_batches) > 0
                else None
            )
            geo_keep_short = (
                geo_keep.repeat(n_short_actual, 1, 1) if geo_keep is not None else None
            )
            mm_keep_short = self._repeat_missing_mask_embed_keep(mm_keep, n_short_actual)
            out_short = self.backbone(
                x_short,
                missing_short,
                time_short,
                mask_map=None,
                is_student=True,
                lon_lat=lon_lat_short,
                geo_keep=geo_keep_short,
                missing_mask_embed_keep=mm_keep_short,
            )
            s_z_short_gap = out_short["z_cls"].view(n_short_actual, B, -1)
            if self.evidence_gap_condition:
                short_condition_num = torch.cat(short_condition_num_batches, dim=0)
                short_view_type = torch.cat(short_view_type_batches, dim=0)
                with torch.no_grad():
                    cond_float = short_condition_num.detach().float()
                    view_float = short_view_type.detach().float()
                    ratio_raw = float(short_tokens) / float(max(1, int(teacher_tokens)))
                    condition_log = {
                        "condition_ratio": float(ratio_raw),
                        "condition_ratio_scaled_mean": float(cond_float[..., 0].mean().item()),
                        "condition_ratio_scaled_min": float(cond_float[..., 0].min().item()),
                        "condition_ratio_scaled_max": float(cond_float[..., 0].max().item()),
                        "condition_relative_position_mean": float(cond_float[..., 1].mean().item()),
                        "condition_relative_position_min": float(cond_float[..., 1].min().item()),
                        "condition_relative_position_max": float(cond_float[..., 1].max().item()),
                        "condition_view_type_mean": float(view_float.mean().item()),
                        "condition_crop_views": int((short_view_type == 0).sum().item()),
                        "condition_random_views": int((short_view_type == 1).sum().item()),
                        "condition_short_view_types": short_view_type[:, 0].detach(),
                    }
                s_logits_short_gap = self._conditioned_evidence_gap_logits(
                    s_z_short_gap,
                    short_condition_num,
                    short_view_type,
                )
                if self.evidence_gap_condition_readout == "direction":
                    condition_log.update(
                        getattr(self, "_last_evidence_gap_direction_stats", {})
                    )
                elif self.evidence_gap_condition_readout == "gate":
                    condition_log.update(
                        getattr(self, "_last_evidence_gap_gate_stats", {})
                    )
                elif self.evidence_gap_condition_readout == "cond_mlp":
                    condition_log.update(
                        getattr(self, "_last_evidence_gap_cond_mlp_stats", {})
                    )
                elif self.evidence_gap_condition_readout == "cond_sum_mlp":
                    condition_log.update(
                        getattr(self, "_last_evidence_gap_cond_sum_stats", {})
                    )
                elif self.evidence_gap_condition_readout == "cond_film_mlp":
                    condition_log.update(
                        getattr(self, "_last_evidence_gap_cond_film_stats", {})
                    )
                elif self.evidence_gap_condition_readout == "cond_res_mlp":
                    condition_log.update(
                        getattr(self, "_last_evidence_gap_cond_res_stats", {})
                    )
                elif self.evidence_gap_condition_readout == "cond_res_film_mlp":
                    condition_log.update(
                        getattr(self, "_last_evidence_gap_cond_res_film_stats", {})
                    )
                elif self.evidence_gap_condition_readout == "cond_gate_bottleneck":
                    condition_log.update(
                        getattr(self, "_last_evidence_gap_cond_gate_b_stats", {})
                    )
                elif self.evidence_gap_condition_readout == "cond_mul_bottleneck":
                    condition_log.update(
                        getattr(self, "_last_evidence_gap_cond_mul_b_stats", {})
                    )
                elif self.evidence_gap_condition_readout == "cond_xattn_bottleneck":
                    condition_log.update(
                        getattr(self, "_last_evidence_gap_cond_xattn_b_stats", {})
                    )
                elif self.evidence_gap_condition_readout == "cond_blend_mlp":
                    condition_log.update(
                        getattr(self, "_last_evidence_gap_cond_blend_stats", {})
                    )
            elif self.backbone.n_cls_tokens == 1:
                s_logits_short_gap = self._conditioned_evidence_gap_logits(
                    s_z_short_gap,
                    None,
                    None,
                )
            else:
                s_logits_short_bank = out_short["logits_cls_bank"].view(
                    n_short_actual, B, self.backbone.n_cls_tokens, -1
                )
                s_z_short_bank = out_short["z_cls_bank"].view(
                    n_short_actual, B, self.backbone.n_cls_tokens, -1
                )
                s_logits_short_gap = s_logits_short_bank[:, :, short_gap_cls_id, :]
                s_z_short_gap = s_z_short_bank[:, :, short_gap_cls_id, :]
        else:
            n_short_actual = 0

        if independent_fullseq:
            condition_log["independent_fullseq"] = 1
        if short_outside_teacher:
            condition_log["short_outside_teacher"] = 1
            condition_log["short_outside_fallback"] = int(short_outside_fallback)

        n_anchor_rows = 0
        s_logits_anchor = None
        t_logits_extra = []
        anchor_teacher_row_indices = []

        can_anchor = (
            int(self.evidence_gap_same_short_multi_teacher_count) > 0
            and n_short_crop > 0
            and n_short_actual > 0
            and s_z_short_gap is not None
            and crop_starts_all is not None
            and (self.evidence_gap_condition or self.backbone.n_cls_tokens == 1)
        )
        if (
            can_anchor
            and self.training
            and mode == "train"
            and float(torch.rand((), device=device).item())
            < float(self.evidence_gap_same_short_multi_teacher_prob)
        ):
            anchor_row = int(
                torch.randint(n_short_crop, (1,), device=device).item()
            )
            z_anchor = s_z_short_gap[anchor_row]
            abs_starts = teacher_start + crop_starts_all[anchor_row].long()
            if short_view_type is not None:
                view_anchor = short_view_type[anchor_row]
            else:
                view_anchor = torch.zeros(B, device=device, dtype=torch.long)

            used_teachers = {(int(teacher_start), int(teacher_len))}
            anchor_logits_rows = []

            for _ in range(int(self.evidence_gap_same_short_multi_teacher_count)):
                alt_spec = self._sample_alt_teacher_containing_short(
                    T=int(T),
                    short_abs_starts=abs_starts,
                    short_len=int(short_len),
                    used_teachers=used_teachers,
                    device=device,
                )
                if alt_spec is None:
                    break
                alt_start, alt_len = alt_spec
                used_teachers.add((int(alt_start), int(alt_len)))

                x_teacher_alt = _slice_win(teacher_base_full, alt_start, alt_len)
                missing_mask_teacher_alt = _slice_win(
                    teacher_mask_full, alt_start, alt_len
                ).float()
                time_mark_teacher_alt = _slice_win(time_mark, alt_start, alt_len)
                lon_lat_teacher_alt = _slice_win(lon_lat, alt_start, alt_len)
                x_teacher_alt_in = get_teacher_input(
                    x_teacher_alt, "none", is_train=self.training
                )
                with torch.no_grad():
                    out_teacher_alt = self.teacher(
                        x_teacher_alt_in,
                        missing_mask_teacher_alt,
                        time_mark_teacher_alt,
                        mask_map=None,
                        is_student=False,
                        lon_lat=lon_lat_teacher_alt,
                        geo_keep=geo_keep,
                        missing_mask_embed_keep=mm_keep,
                    )
                    t_logits_alt = out_teacher_alt["logits_global"].detach()
                t_logits_extra.append(t_logits_alt)
                anchor_teacher_row_indices.append(len(t_logits_extra))

                c_alt = self._build_condition_for_short_in_teacher(
                    short_abs_starts=abs_starts,
                    short_len=int(short_len),
                    short_tokens=int(short_tokens),
                    teacher_start=int(alt_start),
                    teacher_len=int(alt_len),
                    timeline_len=int(T),
                )
                logits_alt = self._conditioned_evidence_gap_logits(
                    z_anchor.unsqueeze(0),
                    c_alt.unsqueeze(0),
                    view_anchor.unsqueeze(0),
                )
                anchor_logits_rows.append(logits_alt)

            if len(anchor_logits_rows) > 0:
                s_logits_anchor = torch.cat(anchor_logits_rows, dim=0)
                n_anchor_rows = int(s_logits_anchor.shape[0])
                condition_log["condition_anchor_views"] = n_anchor_rows

        if self.evidence_gap_condition or self.backbone.n_cls_tokens == 1:
            # Global student: no gate condition; only crop/random shorts use condition.
            s_logits_global_gap = self._conditioned_evidence_gap_logits(
                out_global["z_cls"].unsqueeze(0), None, None
            ).squeeze(0)
            s_z_global_gap = out_global["z_cls"]
        else:
            s_logits_global_gap = s_logits_global_bank[:, global_gap_cls_id, :]
            s_z_global_gap = s_z_global_bank[:, global_gap_cls_id, :]
        s_logits_rows = [s_logits_global_gap.unsqueeze(0)]
        s_z_rows = [s_z_global_gap.unsqueeze(0)]
        student_gap_cls_ids = [int(global_gap_cls_id)]
        if n_short_actual > 0:
            s_logits_rows.append(s_logits_short_gap)
            s_z_rows.append(s_z_short_gap)
            student_gap_cls_ids.extend([int(short_gap_cls_id)] * n_short_actual)
        if s_logits_anchor is not None and n_anchor_rows > 0:
            s_logits_rows.append(s_logits_anchor)
        s_logits_all = torch.cat(s_logits_rows, dim=0)
        s_z_all = torch.cat(s_z_rows, dim=0)

        if self.evidence_gap_condition or self.backbone.n_cls_tokens == 1:
            teacher_row_indices = [0]
            if n_short_actual > 0:
                teacher_row_indices.extend([0] * n_short_actual)
            teacher_row_indices.extend(anchor_teacher_row_indices)
            if len(t_logits_extra) > 0:
                t_logits_unique = torch.stack(
                    [t_logits_global_raw] + t_logits_extra, dim=0
                )
            else:
                t_logits_unique = t_logits_global_raw.unsqueeze(0)
        else:
            unique_cls_ids = []
            teacher_row_indices = []
            for cls_id in student_gap_cls_ids:
                if cls_id not in unique_cls_ids:
                    unique_cls_ids.append(int(cls_id))
                teacher_row_indices.append(unique_cls_ids.index(int(cls_id)))
            t_logits_unique = torch.stack(
                [t_logits_bank[:, cls_id, :] for cls_id in unique_cls_ids],
                dim=0,
            )

        valid_ratio_per_sample = (~missing_mask_teacher_orig).float().mean(dim=-1)
        valid_sample_indices = torch.where(
            valid_ratio_per_sample >= float(self.valid_sample_threshold)
        )[0]

        def _filter_rows(tensor):
            if tensor is None:
                return None
            if valid_sample_indices.numel() == 0:
                return tensor[:, :0]
            return tensor.index_select(1, valid_sample_indices)

        def _filter0(tensor):
            if tensor is None:
                return None
            if valid_sample_indices.numel() == 0:
                return tensor[:0]
            return tensor.index_select(0, valid_sample_indices)

        s_logits_global_valid = _filter_rows(s_logits_all)
        t_logits_global_valid = _filter_rows(t_logits_unique)
        s_z_all_valid = _filter_rows(s_z_all)
        s_z_global_valid = _filter0(s_z_global_gap)
        s_z_patch_global_valid = _filter0(s_z_patch_global)

        fft_masked_s_pre = None
        fft_masked_t_pre = None
        fft_masked_row_ids = None
        fft_masked_patch_idx = None
        s_patch_masked = None
        t_patch_masked = None
        masks_weight_valid = None
        collated_masks_valid = None
        mask_indices_valid = None
        ibot_denom_rows = None
        all_valid_tokens = None
        s_patch_tokens_all = None
        t_patch_tokens_all = None

        if valid_sample_indices.numel() > 0:
            collated_masks_valid = collated_masks.unsqueeze(0)[:, valid_sample_indices, :]
            valid_mask_flat = collated_masks_valid.flatten()
            mask_indices_valid = valid_mask_flat.nonzero(as_tuple=False).squeeze(1)

            orig_valid_global = (~missing_mask_teacher_orig).float().unsqueeze(-1)
            patch_valid_ratio = patchify(
                orig_valid_global, self.patch_len, self.stride
            ).mean(dim=-1)[:, :teacher_tokens]
            if imputator is not None and imputator_mode == "full":
                is_patch_reliable = torch.ones_like(patch_valid_ratio, dtype=torch.bool)
            elif getattr(self, "ibot_patch_reliable_mode", "filter") == "keep_all":
                is_patch_reliable = torch.ones_like(patch_valid_ratio, dtype=torch.bool)
            else:
                is_patch_reliable = patch_valid_ratio > 0
            is_patch_reliable_valid = is_patch_reliable.index_select(
                0, valid_sample_indices
            )

            if mask_indices_valid.numel() > 0:
                reliable_flat = is_patch_reliable_valid.unsqueeze(0).flatten()
                reliable_flags = reliable_flat[mask_indices_valid]
                mask_indices_valid = mask_indices_valid[reliable_flags]

            if mask_indices_valid.numel() > 0:
                B_valid = int(valid_sample_indices.numel())
                N_patch = int(teacher_tokens)
                s_patch_flat = s_z_patch_global_valid[:, :N_patch, :].reshape(
                    B_valid * N_patch, -1
                )
                t_patch_valid = t_z_patch.index_select(0, valid_sample_indices)
                t_patch_flat = t_patch_valid[:, :N_patch, :].reshape(
                    B_valid * N_patch, -1
                )
                s_pre = torch.index_select(s_patch_flat, dim=0, index=mask_indices_valid)
                t_pre = torch.index_select(t_patch_flat, dim=0, index=mask_indices_valid)
                fft_masked_s_pre = s_pre
                fft_masked_t_pre = t_pre.detach()
                fft_masked_row_ids = mask_indices_valid // N_patch
                fft_masked_patch_idx = mask_indices_valid % N_patch
                s_patch_masked = self.backbone.ibot_head(s_pre)
                with torch.no_grad():
                    t_patch_masked = self.teacher.ibot_head(t_pre).detach()

                valid_tokens_per_row = (
                    (is_patch_reliable_valid[:, :N_patch] > 0)
                    .sum(dim=-1)
                    .float()
                    .clamp(min=1.0)
                )
                retained_per_row = torch.bincount(
                    fft_masked_row_ids,
                    minlength=B_valid,
                ).to(device=device, dtype=valid_tokens_per_row.dtype).clamp(min=1.0)
                masks_weight_valid = (
                    valid_tokens_per_row[fft_masked_row_ids]
                    / retained_per_row[fft_masked_row_ids]
                )
                active_rows = torch.unique(fft_masked_row_ids)
                all_valid_tokens = valid_tokens_per_row[active_rows].sum().clamp(min=1.0)
                ibot_denom_rows = float(max(1, int(active_rows.numel())))

            if self.fft_align_all_patches and s_z_patch_global_valid.shape[0] > 0:
                t_patch_valid_all = t_z_patch.index_select(0, valid_sample_indices)
                s_patch_tokens_all = s_z_patch_global_valid[:, :teacher_tokens, :]
                t_patch_tokens_all = t_patch_valid_all[:, :teacher_tokens, :].detach()

        _ssl_nv = int(valid_sample_indices.numel())
        koleo_flat = None
        if _ssl_nv > 0 and s_z_global_valid is not None:
            # Feature-spread regularization: global student only (raw sequence token); short views excluded.
            koleo_flat = s_z_global_valid
        return {
            "z_global": s_z_global_gap,
            "valid_sample_indices": valid_sample_indices,
            "recon_data": {"rec_seq_valid": None, "target_seq_valid": None},
            "spectral_data": {
                "s_patch_tokens_all": s_patch_tokens_all,
                "t_patch_tokens_all": t_patch_tokens_all,
                "fft_masked_s_pre": fft_masked_s_pre,
                "fft_masked_t_pre": fft_masked_t_pre,
                "fft_masked_row_ids": fft_masked_row_ids,
                "fft_masked_patch_idx": fft_masked_patch_idx,
            },
            "cls_data": {
                "cls_loss_mode": "evidence_gap",
                "evidence_gap_pairwise": True,
                "evidence_gap_condition": bool(self.evidence_gap_condition),
                "evidence_gap_condition_readout": str(self.evidence_gap_condition_readout),
                "evidence_gap_condition_alpha": float(self.evidence_gap_condition_alpha),
                "evidence_gap_teacher_row_indices": teacher_row_indices,
                "evidence_gap_n_short_original": int(n_short_actual),
                "evidence_gap_n_short_anchor": int(n_anchor_rows),
                "evidence_gap_n_global_cls_rows": 1,
                "evidence_gap_drop_global_cls": bool(self.evidence_gap_drop_global_cls),
                "s_logits_global_valid": s_logits_global_valid,
                "s_logits_local_valid": None,
                "t_logits_global_valid": t_logits_global_valid,
                "dino_global_scale": 1.0,
                "dino_local_scale": 0.0,
                "gap_cls_id": int(short_gap_cls_id),
                "gap_tokens": int(short_gap_tokens),
                "teacher_tokens": int(teacher_tokens),
                "student_tokens": int(short_tokens),
                "teacher_len": int(teacher_len),
                "student_len": int(short_len),
                **condition_log,
            },
            "patch_data": {
                "s_patch_masked": s_patch_masked,
                "t_patch_masked": t_patch_masked,
                "masks_weight_global_valid": masks_weight_valid,
                "collated_masks_global_valid": collated_masks_valid,
                "mask_indices_list_global_valid": mask_indices_valid,
                "ibot_denom_rows": ibot_denom_rows,
                "all_valid_tokens": all_valid_tokens,
                "ibot_debug": None,
            },
            "koleo_data": {
                "s_z_cls_flat": (
                    None
                    if self.evidence_gap_drop_global_cls
                    else koleo_flat
                )
            },
            "temporal_data": {
                "s_z_patch_enc_valid": s_z_patch_global_valid if _ssl_nv > 0 else None
            },
            "cls_consistency_data": None,
            "lambda_weights": {
                "lambda_recon": 0.0,
                "lambda_fft_align": self.lambda_fft_align,
                "lambda_cls_proto": self.lambda_cls_proto,
                "lambda_patch_proto": self.lambda_patch_proto,
                "lambda_koleo": self.lambda_koleo,
                "lambda_temporal": self.lambda_temporal,
                "lambda_cls_cons": 0.0,
            },
            "teacher_temp": teacher_temp,
            "ssl_num_valid_samples": _ssl_nv,
            "ssl_subgroup_batch_size": int(B),
        }

    def _forward_evidence_gap_dual_teacher_cross(
        self,
        x_enc,
        time_mark,
        mask_rate_v1,
        mask_rate_v2=None,
        mode="train",
        imputator=None,
        lon_lat=None,
    ):
        """
        Dual-teacher cross-window sequence-state pairing:
          - sample teacher_len, two distinct offsets (W_a, W_b)
          - 2 global students (one per window) cross-paired to opposite teachers
          - n_short_crop + n_short_random short views per window, cross-paired to opposite teachers
        """
        B, T, C = x_enc.shape
        device = x_enc.device
        if lon_lat is not None:
            if lon_lat.shape[0] != B or lon_lat.shape[1] != T or lon_lat.shape[-1] != 2:
                raise ValueError(
                    f"lon_lat expected [B,T,2], got {tuple(lon_lat.shape)} for x_enc [{B},{T},...]"
                )

        missing_mask_orig = torch.isnan(x_enc).any(dim=-1)
        missing_mask = missing_mask_orig.clone()
        x_clean_filled = x_enc.nan_to_num(0.0)
        x_target_perfect = x_clean_filled
        imputator_mode = getattr(self, "imputator_mode", "full")
        if imputator is not None:
            with torch.no_grad():
                imp_device = next(imputator.parameters()).device
                imputed_out = imputator_sliding_window_overlap(
                    x_enc,
                    time_mark,
                    missing_mask_orig,
                    imputator,
                    window_len=getattr(imputator, "pred_len", 366),
                    stride=getattr(self, "imputator_segment_stride", 244),
                    device=device,
                    imp_device=imp_device,
                )
            mask_expanded = missing_mask_orig.unsqueeze(-1).float()
            x_target_perfect = x_clean_filled * (1 - mask_expanded) + imputed_out * mask_expanded
            if imputator_mode in ["full", "woMask", "mixed_teacher"]:
                missing_mask = torch.zeros_like(missing_mask_orig)

        teacher_choices = [
            int(v) for v in self.evidence_gap_teacher_lengths if 0 < int(v) <= int(T)
        ]
        if len(teacher_choices) == 0:
            teacher_choices = [int(T)]
        choice_idx = int(torch.randint(len(teacher_choices), (1,), device=device).item())
        teacher_len = int(max(1, min(T, teacher_choices[choice_idx])))
        teacher_start_a, teacher_start_b = self._sample_dual_teacher_starts(
            T, teacher_len, device
        )

        teacher_tokens = self._num_patches_for_len(teacher_len)
        r_min = max(0.0, min(1.0, self.evidence_gap_student_ratio_min))
        r_max = max(r_min, min(1.0, self.evidence_gap_student_ratio_max))
        if teacher_tokens <= 1:
            short_tokens = 1
            short_len = teacher_len
        else:
            min_short_tokens = max(1, int(math.ceil(r_min * teacher_tokens)))
            max_short_tokens = max(
                min_short_tokens,
                min(teacher_tokens - 1, int(math.floor(r_max * teacher_tokens))),
            )
            short_tokens = int(
                torch.randint(
                    min_short_tokens,
                    max_short_tokens + 1,
                    (1,),
                    device=device,
                ).item()
            )
            short_len = min(
                teacher_len,
                self.patch_len + (short_tokens - 1) * self.stride,
            )
            short_tokens = self._num_patches_for_len(short_len)

        self._egap_log_view_lengths(
            "dualT2",
            path="dual_teacher_cross",
            input_T=int(T),
            teacher_choices=teacher_choices,
            teacher_len=int(teacher_len),
            teacher_start_a=int(teacher_start_a),
            teacher_start_b=int(teacher_start_b),
            teacher_tokens=int(teacher_tokens),
            short_len=int(short_len),
            short_tokens=int(short_tokens),
            ratio_min=float(r_min),
            ratio_max=float(r_max),
            n_short_crop=int(self.evidence_gap_n_short_crop),
            n_short_random=int(self.evidence_gap_n_short_random),
            n_short_per_side=int(self.evidence_gap_dual_teacher_short_per_side),
            dual_cross=1,
        )

        short_gap_tokens = max(0, teacher_tokens - short_tokens)
        short_gap_cls_id = self._gap_cls_id(short_gap_tokens)

        def _slice_win(tensor, start, length):
            return tensor[:, start : start + length] if tensor is not None else None

        use_imputed_teacher = imputator is not None and imputator_mode in ["full", "woMask"]
        teacher_base_full = x_target_perfect if use_imputed_teacher else x_clean_filled
        teacher_mask_full = missing_mask if use_imputed_teacher else missing_mask_orig
        student_base_full = (
            x_target_perfect
            if (imputator is not None and imputator_mode == "woMask")
            else x_clean_filled
        )
        student_mask_full = (
            missing_mask
            if (imputator is not None and imputator_mode == "woMask")
            else missing_mask_orig
        )

        def _window_pack(start):
            return {
                "start": int(start),
                "x_teacher": _slice_win(teacher_base_full, start, teacher_len),
                "x_student": _slice_win(student_base_full, start, teacher_len),
                "missing_teacher": _slice_win(teacher_mask_full, start, teacher_len).float(),
                "missing_student": _slice_win(student_mask_full, start, teacher_len).float(),
                "missing_teacher_orig": _slice_win(missing_mask_orig, start, teacher_len),
                "time_mark": _slice_win(time_mark, start, teacher_len),
                "lon_lat": _slice_win(lon_lat, start, teacher_len),
            }

        win_a = _window_pack(teacher_start_a)
        win_b = _window_pack(teacher_start_b)

        geo_keep = None
        if lon_lat is not None and getattr(self.backbone, "lon_lat_proj", None) is not None:
            if self.training and mode == "train" and self.geo_dropout_p > 0:
                geo_keep = (
                    torch.rand(B, 1, 1, device=device, dtype=torch.float32) >= self.geo_dropout_p
                ).to(dtype=x_enc.dtype)
            else:
                geo_keep = torch.ones(B, 1, 1, device=device, dtype=x_enc.dtype)

        no_view_aug = bool(getattr(self, "disable_view_augmentation", False))
        global_aug = "none" if no_view_aug else self.evidence_gap_student_aug
        short_aug = "none" if no_view_aug else "weak_local"

        x_teacher_in = get_teacher_input(
            torch.cat([win_a["x_teacher"], win_b["x_teacher"]], dim=0),
            "none",
            is_train=self.training,
        )
        missing_mask_teacher = torch.cat(
            [win_a["missing_teacher"], win_b["missing_teacher"]], dim=0
        )
        time_mark_teacher = (
            torch.cat([win_a["time_mark"], win_b["time_mark"]], dim=0)
            if win_a["time_mark"] is not None
            else None
        )
        lon_lat_teacher = (
            torch.cat([win_a["lon_lat"], win_b["lon_lat"]], dim=0)
            if win_a["lon_lat"] is not None
            else None
        )
        geo_keep_teacher = (
            torch.cat([geo_keep, geo_keep], dim=0) if geo_keep is not None else None
        )

        if mask_rate_v2 is None:
            mask_ratio_tuple = (mask_rate_v1, mask_rate_v1)
        else:
            mask_ratio_tuple = (min(mask_rate_v1, mask_rate_v2), max(mask_rate_v1, mask_rate_v2))
        collated_masks_a, _, _ = self._make_ibot_collated_masks(
            B, teacher_tokens, device, mask_rate_v1, mask_rate_v2
        )
        collated_masks_b, _, _ = self._make_ibot_collated_masks(
            B, teacher_tokens, device, mask_rate_v1, mask_rate_v2
        )

        teacher_was_training = self.teacher.training
        self.teacher.eval()
        teacher_temp = getattr(self, "_current_teacher_temp", getattr(self, "teacher_temp", 0.07))
        mm_keep = self._sample_missing_mask_embed_keep(B, device)
        mm_keep_teacher = self._repeat_missing_mask_embed_keep(mm_keep, 2)
        with torch.no_grad():
            out_teacher = self.teacher(
                x_teacher_in,
                missing_mask_teacher,
                time_mark_teacher,
                mask_map=None,
                is_student=False,
                lon_lat=lon_lat_teacher,
                geo_keep=geo_keep_teacher,
                missing_mask_embed_keep=mm_keep_teacher,
            )
            t_logits_global = out_teacher["logits_global"].view(2, B, -1).detach()
            t_z_patch = out_teacher["z_patch_enc"].view(2, B, teacher_tokens, -1).detach()
        self.teacher.train(teacher_was_training)

        def _student_global(win, masks):
            x_global = get_student_input(win["x_student"], global_aug, is_train=self.training)
            return self.backbone(
                x_global,
                win["missing_student"],
                win["time_mark"],
                mask_map=masks,
                is_student=True,
                lon_lat=win["lon_lat"],
                geo_keep=geo_keep,
                missing_mask_embed_keep=mm_keep,
                context_target_attn=self._student_context_target_attn(),
            )

        out_global_a = _student_global(win_a, collated_masks_a)
        out_global_b = _student_global(win_b, collated_masks_b)
        s_z_global_a = out_global_a["z_cls"]
        s_z_global_b = out_global_b["z_cls"]
        s_z_patch_a = out_global_a["z_patch_enc"]
        s_z_patch_b = out_global_b["z_patch_enc"]

        def _global_condition_for_cross(src_start, tgt_start):
            short_abs = torch.full((B,), int(src_start), device=device, dtype=torch.long)
            cond = self._build_dual_teacher_cross_condition(
                short_abs,
                int(teacher_len),
                int(teacher_tokens),
                int(tgt_start),
                int(teacher_len),
                int(T),
            )
            view = torch.full((B,), -1, device=device, dtype=torch.long)
            return cond, view

        def _logits_from_z(z_cls, condition_num, view_type):
            if self.evidence_gap_condition:
                return self._conditioned_evidence_gap_logits(
                    z_cls.unsqueeze(0),
                    condition_num.unsqueeze(0),
                    view_type.unsqueeze(0),
                ).squeeze(0)
            if self.backbone.n_cls_tokens == 1:
                return self._conditioned_evidence_gap_logits(
                    z_cls.unsqueeze(0), None, None
                ).squeeze(0)
            raise RuntimeError(
                "dual_teacher_cross without condition requires n_cls_tokens==1"
            )

        cond_a_for_b, view_global = _global_condition_for_cross(
            teacher_start_a, teacher_start_b
        )
        cond_b_for_a, _ = _global_condition_for_cross(teacher_start_b, teacher_start_a)
        s_logits_global_a = _logits_from_z(s_z_global_a, cond_a_for_b, view_global)
        s_logits_global_b = _logits_from_z(s_z_global_b, cond_b_for_a, view_global)

        n_short_crop_side = int(self.evidence_gap_n_short_crop)
        n_short_random_side = int(self.evidence_gap_n_short_random)
        if n_short_crop_side == 0 and n_short_random_side == 0:
            n_short_crop_side = int(self.evidence_gap_dual_teacher_short_per_side)
        n_short_per_side = n_short_crop_side + n_short_random_side
        short_logits_rows = []
        short_z_rows = []
        condition_log = {
            "dual_teacher_cross": True,
            "dual_teacher_start_a": int(teacher_start_a),
            "dual_teacher_start_b": int(teacher_start_b),
            "dual_teacher_len": int(teacher_len),
            "dual_teacher_short_crop_per_side": int(n_short_crop_side),
            "dual_teacher_short_random_per_side": int(n_short_random_side),
            "dual_teacher_short_per_side": int(n_short_per_side),
            "dual_teacher_patch_cross": bool(self.evidence_gap_dual_teacher_patch_cross),
        }

        def _short_views_for_window(win, tgt_start, teacher_row_for_loss_log):
            if n_short_per_side <= 0:
                return
            x_parent = win["x_student"]
            short_batches = []
            short_mask_batches = []
            short_time_batches = []
            short_lon_lat_batches = []
            cond_rows = []
            view_rows = []

            if n_short_crop_side > 0:
                x_crop_parent = x_parent.repeat(n_short_crop_side, 1, 1)
                x_crop, crop_starts = generate_local_view_crop(
                    x_crop_parent, short_len, device
                )
                crop_starts_view = crop_starts.view(n_short_crop_side, B)
                n_rows = n_short_crop_side * B
                batch_idx = torch.arange(n_rows, device=device).unsqueeze(1)
                time_idx = crop_starts.unsqueeze(1) + torch.arange(
                    short_len, device=device
                ).unsqueeze(0)
                parent_mask_rep = win["missing_student"].repeat(n_short_crop_side, 1)
                short_batches.append(
                    get_student_input(x_crop, short_aug, is_train=self.training)
                )
                short_mask_batches.append(parent_mask_rep[batch_idx, time_idx])
                if win["time_mark"] is not None:
                    parent_time_rep = win["time_mark"].repeat(n_short_crop_side, 1, 1)
                    short_time_batches.append(parent_time_rep[batch_idx, time_idx])
                if win["lon_lat"] is not None:
                    parent_ll_rep = win["lon_lat"].repeat(n_short_crop_side, 1, 1)
                    short_lon_lat_batches.append(parent_ll_rep[batch_idx, time_idx])
                abs_starts = int(win["start"]) + crop_starts_view.long()
                for row in range(n_short_crop_side):
                    c_row = self._build_dual_teacher_cross_condition(
                        abs_starts[row],
                        int(short_len),
                        int(short_tokens),
                        int(tgt_start),
                        int(teacher_len),
                        int(T),
                    )
                    cond_rows.append(c_row.unsqueeze(0))
                    view_rows.append(
                        torch.zeros(B, device=device, dtype=torch.long).unsqueeze(0)
                    )

            if n_short_random_side > 0:
                x_rand_parent = x_parent.repeat(n_short_random_side, 1, 1)
                x_rand_patches = patchify(x_rand_parent, self.patch_len, self.stride)
                x_rand_sampled, token_indices = generate_local_view_random_sample(
                    x_rand_patches, short_tokens, device
                )
                x_rand = unpatchify(
                    x_rand_sampled, short_len, self.patch_len, self.c_out
                )
                n_rand_rows = n_short_random_side * B
                batch_idx = torch.arange(n_rand_rows, device=device).unsqueeze(1)
                parent_mask_rep = win["missing_student"].repeat(n_short_random_side, 1)
                mask_patches = patchify(
                    parent_mask_rep.unsqueeze(-1).float(), self.patch_len, self.stride
                )
                mask_sampled = mask_patches[batch_idx, token_indices]
                short_batches.append(
                    get_student_input(x_rand, short_aug, is_train=self.training)
                )
                short_mask_batches.append(
                    unpatchify(mask_sampled, short_len, self.patch_len, 1).squeeze(-1)
                )
                if win["time_mark"] is not None:
                    parent_time_rep = win["time_mark"].repeat(n_short_random_side, 1, 1)
                    time_patches = patchify(parent_time_rep, self.patch_len, self.stride)
                    time_sampled = time_patches[batch_idx, token_indices]
                    short_time_batches.append(
                        unpatchify(time_sampled, short_len, self.patch_len, 2)
                    )
                if win["lon_lat"] is not None:
                    parent_ll_rep = win["lon_lat"].repeat(n_short_random_side, 1, 1)
                    ll_patches = patchify(parent_ll_rep, self.patch_len, self.stride)
                    ll_sampled = ll_patches[batch_idx, token_indices]
                    short_lon_lat_batches.append(
                        unpatchify(ll_sampled, short_len, self.patch_len, 2)
                    )
                token_indices_view = token_indices.view(
                    n_short_random_side, B, short_tokens
                )
                for row in range(n_short_random_side):
                    min_tok = token_indices_view[row].min(dim=-1).values
                    abs_starts_row = int(win["start"]) + (
                        min_tok.float() * float(self.stride)
                    ).long()
                    c_row = self._build_dual_teacher_cross_condition(
                        abs_starts_row,
                        int(short_len),
                        int(short_tokens),
                        int(tgt_start),
                        int(teacher_len),
                        int(T),
                    )
                    cond_rows.append(c_row.unsqueeze(0))
                    view_rows.append(
                        torch.ones(B, device=device, dtype=torch.long).unsqueeze(0)
                    )

            x_short = torch.cat(short_batches, dim=0)
            missing_short = torch.cat(short_mask_batches, dim=0).float()
            time_short = (
                torch.cat(short_time_batches, dim=0)
                if len(short_time_batches) > 0
                else None
            )
            lon_lat_short = (
                torch.cat(short_lon_lat_batches, dim=0)
                if len(short_lon_lat_batches) > 0
                else None
            )
            geo_keep_short = (
                geo_keep.repeat(n_short_per_side, 1, 1) if geo_keep is not None else None
            )
            mm_keep_short = self._repeat_missing_mask_embed_keep(mm_keep, n_short_per_side)
            out_short = self.backbone(
                x_short,
                missing_short,
                time_short,
                mask_map=None,
                is_student=True,
                lon_lat=lon_lat_short,
                geo_keep=geo_keep_short,
                missing_mask_embed_keep=mm_keep_short,
            )
            z_short = out_short["z_cls"].view(n_short_per_side, B, -1)
            cond_num = torch.cat(cond_rows, dim=0)
            view_type = torch.cat(view_rows, dim=0)
            if self.evidence_gap_condition:
                logits_short = self._conditioned_evidence_gap_logits(
                    z_short, cond_num, view_type
                )
            elif self.backbone.n_cls_tokens == 1:
                logits_short = self._conditioned_evidence_gap_logits(
                    z_short, None, None
                )
            else:
                raise RuntimeError(
                    "dual_teacher_cross without condition requires n_cls_tokens==1"
                )
            short_logits_rows.append(logits_short)
            short_z_rows.append(z_short)
            _ = teacher_row_for_loss_log

        _short_views_for_window(win_a, teacher_start_b, 1)
        _short_views_for_window(win_b, teacher_start_a, 0)

        s_logits_rows = [
            s_logits_global_a.unsqueeze(0),
            s_logits_global_b.unsqueeze(0),
        ]
        s_z_rows = [s_z_global_a.unsqueeze(0), s_z_global_b.unsqueeze(0)]
        n_short_a = 0
        n_short_b = 0
        if len(short_logits_rows) > 0:
            s_logits_rows.append(short_logits_rows[0])
            s_z_rows.append(short_z_rows[0])
            n_short_a = int(short_logits_rows[0].shape[0])
        if len(short_logits_rows) > 1:
            s_logits_rows.append(short_logits_rows[1])
            s_z_rows.append(short_z_rows[1])
            n_short_b = int(short_logits_rows[1].shape[0])

        s_logits_all = torch.cat(s_logits_rows, dim=0)
        s_z_all = torch.cat(s_z_rows, dim=0)
        t_logits_unique = t_logits_global
        teacher_row_indices = (
            [1, 0]
            + [1] * n_short_a
            + [0] * n_short_b
        )

        missing_mask_teacher_orig_a = win_a["missing_teacher_orig"]
        valid_ratio_per_sample = (~missing_mask_teacher_orig_a).float().mean(dim=-1)
        valid_sample_indices = torch.where(
            valid_ratio_per_sample >= float(self.valid_sample_threshold)
        )[0]

        def _filter_rows(tensor):
            if tensor is None:
                return None
            if valid_sample_indices.numel() == 0:
                return tensor[:, :0]
            return tensor.index_select(1, valid_sample_indices)

        def _filter0(tensor):
            if tensor is None:
                return None
            if valid_sample_indices.numel() == 0:
                return tensor[:0]
            return tensor.index_select(0, valid_sample_indices)

        s_logits_global_valid = _filter_rows(s_logits_all)
        t_logits_global_valid = _filter_rows(t_logits_unique)
        s_z_all_valid = _filter_rows(s_z_all)
        s_z_global_a_valid = _filter0(s_z_global_a)
        s_z_global_b_valid = _filter0(s_z_global_b)
        s_z_patch_a_valid = _filter0(s_z_patch_a)
        s_z_patch_b_valid = _filter0(s_z_patch_b)

        fft_masked_s_pre = None
        fft_masked_t_pre = None
        fft_masked_row_ids = None
        fft_masked_patch_idx = None
        s_patch_masked = None
        t_patch_masked = None
        masks_weight_valid = None
        collated_masks_valid = None
        mask_indices_valid = None
        ibot_denom_rows = None
        all_valid_tokens = None
        s_patch_tokens_all = None
        t_patch_tokens_all = None

        if valid_sample_indices.numel() > 0:
            collated_masks_valid = collated_masks_a.unsqueeze(0)[:, valid_sample_indices, :]
            orig_valid = (~missing_mask_teacher_orig_a).float().unsqueeze(-1)
            patch_valid_ratio = patchify(
                orig_valid, self.patch_len, self.stride
            ).mean(dim=-1)[:, :teacher_tokens]
            if imputator is not None and imputator_mode == "full":
                is_patch_reliable = torch.ones_like(patch_valid_ratio, dtype=torch.bool)
            elif getattr(self, "ibot_patch_reliable_mode", "filter") == "keep_all":
                is_patch_reliable = torch.ones_like(patch_valid_ratio, dtype=torch.bool)
            else:
                is_patch_reliable = patch_valid_ratio > 0
            is_patch_reliable_valid = is_patch_reliable.index_select(
                0, valid_sample_indices
            )

            if self.evidence_gap_dual_teacher_patch_cross:
                patch_pairs = [
                    (s_z_patch_a_valid, t_z_patch[1], collated_masks_a),
                    (s_z_patch_b_valid, t_z_patch[0], collated_masks_b),
                ]
            else:
                patch_pairs = [
                    (s_z_patch_a_valid, t_z_patch[0], collated_masks_a),
                    (s_z_patch_b_valid, t_z_patch[1], collated_masks_b),
                ]
            s_pre_list = []
            t_pre_list = []
            row_id_list = []
            patch_idx_list = []
            row_base = 0
            for s_patch_valid, t_patch_side, masks_side in patch_pairs:
                masks_v = masks_side.unsqueeze(0)[:, valid_sample_indices, :]
                mask_flat = masks_v.flatten()
                mask_idx = mask_flat.nonzero(as_tuple=False).squeeze(1)
                if mask_idx.numel() == 0:
                    row_base += int(valid_sample_indices.numel())
                    continue
                B_valid = int(valid_sample_indices.numel())
                N_patch = int(teacher_tokens)
                reliable_flat = is_patch_reliable_valid.unsqueeze(0).flatten()
                reliable_flags = reliable_flat[mask_idx]
                mask_idx = mask_idx[reliable_flags]
                if mask_idx.numel() == 0:
                    row_base += B_valid
                    continue
                s_flat = s_patch_valid[:, :N_patch, :].reshape(B_valid * N_patch, -1)
                t_flat = t_patch_side.index_select(0, valid_sample_indices)[:, :N_patch, :]
                t_flat = t_flat.reshape(B_valid * N_patch, -1)
                s_pre_list.append(torch.index_select(s_flat, dim=0, index=mask_idx))
                t_pre_list.append(
                    torch.index_select(t_flat, dim=0, index=mask_idx).detach()
                )
                row_id_list.append(mask_idx // N_patch + row_base)
                patch_idx_list.append(mask_idx % N_patch)
                row_base += B_valid

            if len(s_pre_list) > 0:
                s_pre = torch.cat(s_pre_list, dim=0)
                t_pre = torch.cat(t_pre_list, dim=0)
                fft_masked_row_ids = torch.cat(row_id_list, dim=0)
                fft_masked_patch_idx = torch.cat(patch_idx_list, dim=0)
                fft_masked_s_pre = s_pre
                fft_masked_t_pre = t_pre
                s_patch_masked = self.backbone.ibot_head(s_pre)
                with torch.no_grad():
                    t_patch_masked = self.teacher.ibot_head(t_pre).detach()
                B_valid = int(valid_sample_indices.numel())
                valid_tokens_per_row = (
                    (is_patch_reliable_valid[:, :teacher_tokens] > 0)
                    .sum(dim=-1)
                    .float()
                    .clamp(min=1.0)
                )
                retained_per_row = torch.bincount(
                    fft_masked_row_ids,
                    minlength=max(1, 2 * B_valid),
                ).to(device=device, dtype=valid_tokens_per_row.dtype).clamp(min=1.0)
                masks_weight_valid = (
                    valid_tokens_per_row[fft_masked_row_ids % B_valid]
                    / retained_per_row[fft_masked_row_ids].clamp(min=1.0)
                )
                active_rows = torch.unique(fft_masked_row_ids)
                all_valid_tokens = valid_tokens_per_row[:B_valid].sum().clamp(min=1.0)
                ibot_denom_rows = float(max(1, int(active_rows.numel())))

            if self.fft_align_all_patches and s_z_patch_a_valid.shape[0] > 0:
                s_patch_tokens_all = s_z_patch_a_valid[:, :teacher_tokens, :]
                t_patch_tokens_all = t_z_patch[1].index_select(
                    0, valid_sample_indices
                )[:, :teacher_tokens, :].detach()

        _ssl_nv = int(valid_sample_indices.numel())
        koleo_flat = None
        if _ssl_nv > 0:
            koleo_flat = torch.cat([s_z_global_a_valid, s_z_global_b_valid], dim=0)

        n_short_total = int(n_short_a + n_short_b)
        return {
            "z_global": s_z_global_a,
            "valid_sample_indices": valid_sample_indices,
            "recon_data": {"rec_seq_valid": None, "target_seq_valid": None},
            "spectral_data": {
                "s_patch_tokens_all": s_patch_tokens_all,
                "t_patch_tokens_all": t_patch_tokens_all,
                "fft_masked_s_pre": fft_masked_s_pre,
                "fft_masked_t_pre": fft_masked_t_pre,
                "fft_masked_row_ids": fft_masked_row_ids,
                "fft_masked_patch_idx": fft_masked_patch_idx,
            },
            "cls_data": {
                "cls_loss_mode": "evidence_gap",
                "evidence_gap_pairwise": True,
                "evidence_gap_condition": bool(self.evidence_gap_condition),
                "evidence_gap_condition_readout": str(self.evidence_gap_condition_readout),
                "evidence_gap_condition_alpha": float(self.evidence_gap_condition_alpha),
                "evidence_gap_teacher_row_indices": teacher_row_indices,
                "evidence_gap_n_short_original": int(n_short_total),
                "evidence_gap_n_short_anchor": 0,
                "evidence_gap_n_global_cls_rows": 2,
                "evidence_gap_drop_global_cls": bool(self.evidence_gap_drop_global_cls),
                "s_logits_global_valid": s_logits_global_valid,
                "s_logits_local_valid": None,
                "t_logits_global_valid": t_logits_global_valid,
                "dino_global_scale": 1.0,
                "dino_local_scale": 0.0,
                "gap_cls_id": int(short_gap_cls_id),
                "gap_tokens": int(short_gap_tokens),
                "teacher_tokens": int(teacher_tokens),
                "student_tokens": int(short_tokens),
                "teacher_len": int(teacher_len),
                "student_len": int(short_len),
                **condition_log,
            },
            "patch_data": {
                "s_patch_masked": s_patch_masked,
                "t_patch_masked": t_patch_masked,
                "masks_weight_global_valid": masks_weight_valid,
                "collated_masks_global_valid": collated_masks_valid,
                "mask_indices_list_global_valid": mask_indices_valid,
                "ibot_denom_rows": ibot_denom_rows,
                "all_valid_tokens": all_valid_tokens,
                "ibot_debug": None,
            },
            "koleo_data": {
                "s_z_cls_flat": (
                    None
                    if self.evidence_gap_drop_global_cls
                    else koleo_flat
                )
            },
            "temporal_data": {
                "s_z_patch_enc_valid": s_z_patch_a_valid if _ssl_nv > 0 else None
            },
            "cls_consistency_data": None,
            "lambda_weights": {
                "lambda_recon": 0.0,
                "lambda_fft_align": self.lambda_fft_align,
                "lambda_cls_proto": self.lambda_cls_proto,
                "lambda_patch_proto": self.lambda_patch_proto,
                "lambda_koleo": self.lambda_koleo,
                "lambda_temporal": self.lambda_temporal,
                "lambda_cls_cons": 0.0,
            },
            "teacher_temp": teacher_temp,
            "ssl_num_valid_samples": _ssl_nv,
            "ssl_subgroup_batch_size": int(B),
        }

    def _forward(self, x_enc, time_mark, mask_rate_v1, mask_rate_v2=None, mode='train', imputator=None, lon_lat=None):
        """
        Args:
            x_enc: input data
            time_mark: time mark
            mask_rate_v1:mask rate 1
            mask_rate_v2:mask rate 2
            mode: mode
            imputator:interpolator (optional)
            lon_lat: optional [B, T, 2], WGS84 degrees (lon,lat); synccrop with time_mark. Without then, no geo embedding is added.
        """
        B, T, C = x_enc.shape
        device = x_enc.device
        # Kept for downstream alignment code; shared teacher/student crops keep this at 0.
        global_shift_patch_offset = 0
        if lon_lat is not None:
            if lon_lat.shape[0] != B or lon_lat.shape[1] != T or lon_lat.shape[-1] != 2:
                raise ValueError(
                    f"lon_lat expected [B,T,2], got {tuple(lon_lat.shape)}，vs x_enc [{B},{T},…] mismatch"
                )
        
        # 1. Data Prep
        # [Key modification]: saveoriginal’s physical missing mask is used for raw student views that do not go through the imputator.
        missing_mask_orig = torch.isnan(x_enc).any(dim=-1)
        
        # Initialize logic missing_mask, default is equal to original mask
        # If the imputator is subsequently used, this mask will be set to all 0s.
        missing_mask = missing_mask_orig.clone()  # clone to ensure safety (may be modified later)
        
        # Do full sequence padding first, then crop later.
        x_clean_filled = x_enc.nan_to_num(0.0)
        
        # 2. Imputator Logic: pred_len (such as 366) is the window length; super long usestride imputator_segment_stride (default 244) slidingoverlap, multi-window prediction at the missing point takes the mean
        x_target_perfect = None
        imputator_mode = getattr(self, 'imputator_mode', 'full')
        if imputator is not None:
            with torch.no_grad():
                imp_device = next(imputator.parameters()).device
                max_imp_len = getattr(imputator, 'pred_len', 366)
                stride_imp = getattr(self, 'imputator_segment_stride', 244)
                imputed_out = imputator_sliding_window_overlap(
                    x_enc,
                    time_mark,
                    missing_mask_orig,
                    imputator,
                    window_len=max_imp_len,
                    stride=stride_imp,
                    device=device,
                    imp_device=imp_device,
                )
            
            # [Key modification]: Synthesize perfect target when use missing_mask_orig (indicate where need is completed)
            mask_expanded = missing_mask_orig.unsqueeze(-1).float()
            x_target_perfect = x_clean_filled * (1 - mask_expanded) + imputed_out * mask_expanded
            
            # 【mode1：full】
            # It is considered that the target generated by the imputator is reliable in all positions:
            # - missing_mask is considered fully valid in the downstream perspective
            # - Patch filterlogic no longer distinguishes between seriously missing patches.
            #
            # 【mode2：recon_only / wMask】
            # keep missing_mask = missing_mask_orig, so that logic such as patch filter is still based on original observation quality.
            #
            # 【mode3：mixed_teacher】
            # mixed no longer uses 0.5, only keeps 0/1:
            # - For imputator view, missing_mask is regarded as fully valid (0);
            # - For raw view, explicitly use missing_mask_orig (0/1) later.
            if imputator_mode in ['full', 'woMask', 'mixed_teacher']:
                missing_mask = torch.zeros_like(missing_mask_orig)
        else:
            # Without imputator, main missing_mask remains as missing_mask_orig
            # x_target_perfect = create_smooth_target(x_clean_filled, missing_mask_orig, kernel_size=5)
            x_target_perfect = x_clean_filled

        # 3. Dynamic/fixed sequencelength sampling
        # If curriculum_strategy == 'fixed', then useoriginallength T is completely used and no crop is done.
        curriculum_strategy = getattr(self, 'curriculum_strategy', 'fast')
        if curriculum_strategy == 'fixed':
            chosen_len = T
            start_idx = 0
            end_idx = T
        else:
            # Dynamic sequencelength sampling (1year/2year/3year/4year/5year/6year), crop after imputator processes done
            #
            # 【option1：curriculum learningstrategy（currentimplementation）】
            # Gradually increase lengthrange during the training process to improve training stability.
            # Advantages: simple and easy to implement; stable training, fast loss reduction
            # Disadvantage: need to know the total number of training steps
            #
            # [option2: Grouping within Batch (optional)]
            # Divide the batch into multiple groups, each group usedifferentlength, and then process them separately
            # Advantages: Each batch can see multiple lengths without knowing the number of training steps.
            # Disadvantages: complex implementation, need to handle variable lengthsequence or group forward
            # Implementation ideas:
            # - Divide the batch into 3-4 groups
            # - Choose the length independently for each group (1-3year, 2-4year, 3-5year, 4-6year)
            # - forward and loss weighted average respectively
            #
            # [option3: Each sampleindependentlength (optional)]
            # Choose length for each sampleindependent, usepadding to maximum length
            # Advantages: Most flexible
            # Disadvantages: padding wastes calculations and implementation is complex
            #
            # Assume T is 6yearlength, define 1-6year in equal parts
            base_year_len = max(self.patch_len, T // 6)
            
            # get currenttraining progress (iteration/epoch)
            # Can be controlled by the iteration parameter of self.step_counter or externalpass in
            current_iteration = getattr(self, '_current_iteration', 0)
            total_iterations = getattr(self, '_total_iterations', 100000)  # default100,000 steps
            
            # [Curriculum learning strategy after optimize]: Speed ​​up the progress and start training when the long sequence is at a higher learning rate.
            # Considering cosineannealing, the learning rate is higher in the early stage and lower in the later stage.
            # Therefore, the progress of curriculum learning is accelerated, so that the model can be exposed to long sequences at a higher learning rate.
            progress = current_iteration / max(total_iterations, 1)
            
            if curriculum_strategy == 'none':
                # do not usecurriculum learning，directly1-6yearfull range
                max_year = 6
            elif curriculum_strategy == 'fast':
                # Quick strategy: first 10% use only1-3year, then 1-6yearfull range (recommended, suited for cosineannealing)
                if progress < 0.1:
                    max_year = 3
                else:
                    max_year = 6
            elif curriculum_strategy == 'balanced':
                # Balanced strategy: 0-15%: 1-3year, 15-35%: 1-4year, 35-55%: 1-5year, 55-100%: 1-6year
                if progress < 0.15:
                    max_year = 3
                elif progress < 0.35:
                    max_year = 4
                elif progress < 0.55:
                    max_year = 5
                else:
                    max_year = 6
            elif curriculum_strategy == 'conservative':
                # Conservative strategy: 0-20%: 1-3year, 20-40%: 1-4year, 40-60%: 1-5year, 60-100%: 1-6year
                if progress < 0.2:
                    max_year = 3
                elif progress < 0.4:
                    max_year = 4
                elif progress < 0.6:
                    max_year = 5
                else:
                    max_year = 6
            elif curriculum_strategy == 'mixed_batch':
                # mixed_batch: SSLTrainer only does batch dimension splitting, when the interdimensional crop is done here
                # (The imputator has processed the complete long sequence in top, and crop here does not affect the imputation quality)
                max_year = 6
            else:
                # Other unknown strategies, defaultuse fast strategy
                if progress < 0.1:
                    max_year = 3
                else:
                    max_year = 6
            
            # Generate the length selection allowed in the current stage (1~6year)
            length_choices = sorted(set([
                base_year_len,               # 1year
                min(2 * base_year_len, T),   # 2year
                min(3 * base_year_len, T),   # 3year
            ]))
            if max_year >= 4:
                length_choices.append(min(4 * base_year_len, T))  # 4year
            if max_year >= 5:
                length_choices.append(min(5 * base_year_len, T))  # 5year
            if max_year >= 6:
                length_choices.append(T)  # 6year（fulllength）

            length_choices = torch.tensor(length_choices, device=device)
            chosen_len = int(length_choices[torch.randint(len(length_choices), (1,), device=device)])
            start_idx = 0
            if chosen_len < T:
                max_start = max(T - chosen_len, 0)
                start_idx = int(torch.randint(max_start + 1, (1,), device=device))
                # After choosing a year-aligned length, jitter the shared
                # teacher/student crop start. This adds temporal diversity
                # without forcing invariance between different time windows.
                start_jitter = int(getattr(self, '_curriculum_length_jitter', 61))
                jitter_prob = float(getattr(self, 'curriculumJitterProbability', 0.5))
                if start_jitter > 0 and torch.rand(1, device=device).item() < jitter_prob:
                    jitter = int(torch.randint(-start_jitter, start_jitter + 1, (1,), device=device))
                    start_idx = max(0, min(start_idx + jitter, max_start))
            end_idx = start_idx + chosen_len
        
        # === Anchor temporal crop + optional second global crop ===
        # View 0 remains the anchor window. View 1 can shift to another
        # same-length window so the two global views differ in temporal content,
        # matching multi-window self-distillation more closely.
        T_full = int(T)
        te_start = int(start_idx)
        st_start = int(start_idx)
        win_len = int(chosen_len)
        max_global_start = max(0, T_full - win_len)

        def _sample_second_global_start(base_start: int) -> int:
            if max_global_start <= 0:
                # Full window (win_len=T): Unable to shift, two globals are in the same class, consistent with probe fixed [0:T]
                return int(base_start)
            if not (self.training and mode == "train"):
                return int(base_start)

            prob = float(max(0.0, min(1.0, self.global_shift_probability)))
            if prob <= 0.0:
                return int(base_start)
            if prob < 1.0 and torch.rand(1, device=device).item() >= prob:
                return int(base_start)

            ratio = float(max(0.0, self.global_shift_ratio))
            steps = int(max(0, self.global_shift_steps))
            jitter_ratio = float(max(0.0, self.global_shift_jitter_ratio))
            jitter_steps = int(max(0, self.global_shift_jitter_steps))
            min_overlap = float(max(0.0, min(1.0, self.global_shift_min_overlap_ratio)))
            max_delta_from_overlap = int(math.floor((1.0 - min_overlap) * win_len))
            if max_delta_from_overlap <= 0:
                return int(base_start)

            base_delta = int(round(ratio * win_len)) if ratio > 0 else steps
            jitter = int(round(jitter_ratio * win_len)) if jitter_ratio > 0 else jitter_steps
            force_shift = prob >= 1.0

            candidate = int(base_start)
            if base_delta > 0:
                base_delta = min(base_delta, max_delta_from_overlap)
                if self.global_shift_mode == "uniform":
                    low = max(1, base_delta - max(0, jitter))
                    high = min(max_delta_from_overlap, max(low, base_delta + max(0, jitter)))
                    delta = int(torch.randint(low, high + 1, (1,), device=device).item())
                else:
                    delta = base_delta
                    if jitter > 0:
                        delta = min(
                            max_delta_from_overlap,
                            max(
                                1,
                                delta + int(
                                    torch.randint(-jitter, jitter + 1, (1,), device=device).item()
                                ),
                            ),
                        )
                sign = -1 if torch.rand(1, device=device).item() < 0.5 else 1
                candidate = int(max(0, min(base_start + sign * delta, max_global_start)))

            if candidate == int(base_start) and (force_shift or base_delta <= 0):
                low = max(0, int(base_start) - max_delta_from_overlap)
                high = min(max_global_start, int(base_start) + max_delta_from_overlap)
                for _ in range(8):
                    candidate = int(
                        torch.randint(low, high + 1, (1,), device=device).item()
                    )
                    if candidate != int(base_start):
                        break
                if candidate == int(base_start):
                    candidate = low if int(base_start) != low else high
            return int(candidate)

        global_view_starts = [int(start_idx), _sample_second_global_start(int(start_idx))]
        global_views_share_content = all(s == global_view_starts[0] for s in global_view_starts)
        g0_start, g1_start = int(global_view_starts[0]), int(global_view_starts[1])
        g0_end, g1_end = g0_start + win_len, g1_start + win_len
        global_view_overlap_steps = max(0, min(g0_end, g1_end) - max(g0_start, g1_start))
        global_view_overlap_ratio = float(global_view_overlap_steps) / float(max(win_len, 1))
        global_view_shift_ratio = abs(g1_start - g0_start) / float(max(win_len, 1))

        def _slice_win(tensor, s0, length):
            return tensor[:, s0 : s0 + length] if tensor is not None else None

        # The teacher / student windows must be taken from the full-length tensor that has not yet been pressed te_start crop**
        _tp, _cf = x_target_perfect, x_clean_filled
        _mm, _mmo = missing_mask, missing_mask_orig
        _tm, _ll = time_mark, lon_lat

        x_target_perfect_student = _slice_win(_tp, st_start, win_len)
        x_clean_filled_student = _slice_win(_cf, st_start, win_len)
        missing_mask_student = _slice_win(_mm, st_start, win_len)
        missing_mask_orig_student = _slice_win(_mmo, st_start, win_len)
        time_mark_student = _slice_win(_tm, st_start, win_len)
        lon_lat_student = _slice_win(_ll, st_start, win_len)

        x_target_perfect = _slice_win(_tp, te_start, win_len)
        x_clean_filled = _slice_win(_cf, te_start, win_len)
        missing_mask = _slice_win(_mm, te_start, win_len)
        missing_mask_orig = _slice_win(_mmo, te_start, win_len)
        time_mark = _slice_win(_tm, te_start, win_len)
        lon_lat = _slice_win(_ll, te_start, win_len)

        T = win_len
        # Each teacher/global-student pair uses the same crop window, so there
        # is no patch offset inside a paired view even when the two global views differ.
        global_shift_patch_offset = 0

        # CFG formula geo conditions: Teacher/Student, the same batch of samples use geo_keep, which is convenient for distillation alignment
        geo_keep = None
        if lon_lat is not None and getattr(self.backbone, "lon_lat_proj", None) is not None:
            if self.training and mode == "train" and self.geo_dropout_p > 0:
                geo_keep = (
                    torch.rand(B, 1, 1, device=device, dtype=torch.float32) >= self.geo_dropout_p
                ).to(dtype=x_enc.dtype)
            else:
                geo_keep = torch.ones(B, 1, 1, device=device, dtype=x_enc.dtype)

        num_patches_cur = math.ceil((T - self.patch_len + self.stride) / self.stride)
        _curr_strat = getattr(self, "curriculum_strategy", "fast")
        if (
            _curr_strat == "mixed_batch"
            and self.training
            and mode == "train"
        ):
            _p = self._mixed_batch_local_div_probs_t
            _idx = int(torch.multinomial(_p, num_samples=1, replacement=True).item())
            _d_cur = max(1, int(self._mixed_batch_local_divisors_list[_idx]))
        else:
            _d_cur = self.local_view_patch_divisor
        num_local_patches_cur = max(1, num_patches_cur // _d_cur)

        # Recalculate is_target_reliable and is_patch_reliable_for_ibot based on the length after crop
        # - In full mode: all patch targets are considered reliable (imputator provides complete imputation), no filter
        # - Other cases: ibot_patch_reliable_mode is filter (default) or keep_all
        # Same as when pre-calculating whether each patch includes imputator imputation position (used for subsequent patch loss down-weight)
        orig_valid = (~missing_mask_orig_student).float().unsqueeze(-1)
        orig_valid_map_flat = patchify(orig_valid, self.patch_len, self.stride)  # [B, N_cur, patch_len]
        patch_valid_ratio_orig = orig_valid_map_flat.mean(dim=-1)  # [B, N_cur], 1 means that the patch is all original observation
        has_imputed_patch = patch_valid_ratio_orig < 1.0  # True means that the patch includes at least 1 originalmissing point

        if imputator is not None and imputator_mode == 'full':
            is_target_reliable = torch.ones(B, num_patches_cur, device=device, dtype=torch.bool)
            is_patch_reliable_for_ibot = torch.ones(B, num_patches_cur, device=device, dtype=torch.bool)
        else:
            # Use the shared student/teacher crop mask.
            missing_mask_float = missing_mask_student.float()
            is_valid_pixel = 1.0 - missing_mask_float
            valid_map_flat = patchify(is_valid_pixel.unsqueeze(-1), self.patch_len, self.stride)
            is_target_reliable = valid_map_flat.mean(dim=-1) > self.valid_patch_threshold
            if getattr(self, 'ibot_patch_reliable_mode', 'filter') == 'keep_all':
                is_patch_reliable_for_ibot = torch.ones(
                    B, num_patches_cur, device=device, dtype=torch.bool
                )
            else:
                is_patch_reliable_for_ibot = patch_valid_ratio_orig > 0

        # 2. Build multi-view - optimizeversion: batch generation
        
        # usefull viewsconfig (training and validate remain consistent to ensure lossvalidity)
        n_teacher_views = 2
        n_global_student = 2
        n_local_student = self.ted_modular_n_local_student

        # Determine whether Student global view uses imputator based on imputator_mode
        # - full : [True, False] (half use imputator, half use raw)
        # - mixed_teacher: [True, False] (half use imputator, half use raw)
        # - recon_only   : [False, False]（all use raw）
        # - woMask: [True, False] (half use imputator, half use raw, but missing_mask is uniformly regarded as 0)
        # - wMask        : [False, False]（all use raw，keeporiginal missing_mask）
        if imputator is not None and imputator_mode in ['full', 'mixed_teacher', 'woMask']:
            x_student_global_use_imputator = [True, False]
        else:
            x_student_global_use_imputator = [False, False]
        
        # Teacher Views: weak augmentation views
        # - full: Teacher viewalluse imputator; Student half imp half raw
        # - recon_only   : Teacher viewalluse raw（x_clean_filled）
        # - mixed_teacher: Teacher viewuse [imp, raw] true mixed; Student half imp half raw
        teacher_sources = []
        if imputator is not None:
            if imputator_mode == 'full':
                teacher_sources = ['imp', 'imp']
            elif imputator_mode == 'recon_only':
                teacher_sources = ['raw', 'raw']
            elif imputator_mode == 'mixed_teacher':
                teacher_sources = ['imp', 'raw']
            else:  # 'woMask' / 'wMask' and other strings default use imp,imp
                teacher_sources = ['imp', 'imp']
        else:
            teacher_sources = ['raw', 'raw']

        x_teacher_bases = []
        teacher_masks_list = []
        teacher_time_list = []
        teacher_lon_lat_list = []
        x_global_bases = []
        global_masks_list = []
        global_time_list = []
        global_lon_lat_list = []
        local_parent_base_list = []
        local_parent_time_list = []
        local_parent_mask_list = []
        local_parent_lon_lat_list = []
        patch_valid_ratio_orig_global_list = []
        has_imputed_patch_global_list = []
        is_patch_reliable_for_ibot_global_list = []
        valid_ratio_per_global_view_list = []
        for view_idx, gv_start in enumerate(global_view_starts):
            gv_tp = _slice_win(_tp, gv_start, win_len)
            gv_cf = _slice_win(_cf, gv_start, win_len)
            gv_mm = _slice_win(_mm, gv_start, win_len)
            gv_mmo = _slice_win(_mmo, gv_start, win_len)
            gv_tm = _slice_win(_tm, gv_start, win_len)
            gv_ll = _slice_win(_ll, gv_start, win_len)

            teacher_src = teacher_sources[view_idx]
            if teacher_src == 'imp':
                x_teacher_bases.append(gv_tp)
                # It is considered that there is no missing
                teacher_masks_list.append(gv_mm)
            else:
                x_teacher_bases.append(gv_cf)
                teacher_masks_list.append(gv_mmo)
            teacher_time_list.append(gv_tm)
            teacher_lon_lat_list.append(gv_ll)

            use_imp = x_student_global_use_imputator[view_idx]
            x_global_bases.append(gv_tp if use_imp else gv_cf)
            global_masks_list.append(gv_mm if use_imp else gv_mmo)
            global_time_list.append(gv_tm)
            global_lon_lat_list.append(gv_ll)
            local_parent_base_list.append(gv_tp)
            local_parent_time_list.append(gv_tm)
            local_parent_mask_list.append(gv_mm)
            local_parent_lon_lat_list.append(gv_ll)

            gv_orig_valid = (~gv_mmo).float().unsqueeze(-1)
            gv_orig_valid_map_flat = patchify(gv_orig_valid, self.patch_len, self.stride)
            gv_patch_valid_ratio_orig = gv_orig_valid_map_flat.mean(dim=-1)
            patch_valid_ratio_orig_global_list.append(gv_patch_valid_ratio_orig)
            has_imputed_patch_global_list.append(gv_patch_valid_ratio_orig < 1.0)
            if imputator is not None and imputator_mode == 'full':
                gv_is_patch_reliable = torch.ones(
                    B, num_patches_cur, device=device, dtype=torch.bool
                )
            elif getattr(self, 'ibot_patch_reliable_mode', 'filter') == 'keep_all':
                gv_is_patch_reliable = torch.ones(
                    B, num_patches_cur, device=device, dtype=torch.bool
                )
            else:
                gv_is_patch_reliable = gv_patch_valid_ratio_orig > 0
            is_patch_reliable_for_ibot_global_list.append(gv_is_patch_reliable)
            valid_ratio_per_global_view_list.append((~gv_mmo).float().mean(dim=-1))

        x_teacher_base = torch.cat(x_teacher_bases, dim=0)  # [n_teacher*B, T, C]
        patch_valid_ratio_orig_global = torch.stack(patch_valid_ratio_orig_global_list, dim=0)
        has_imputed_patch_global = torch.stack(has_imputed_patch_global_list, dim=0)
        is_patch_reliable_for_ibot_global = torch.stack(
            is_patch_reliable_for_ibot_global_list, dim=0
        )
        valid_ratio_per_global_view = torch.stack(valid_ratio_per_global_view_list, dim=0)
        _no_view_aug = bool(getattr(self, 'disable_view_augmentation', False))
        _teacher_aug = 'none' if _no_view_aug else 'weak'
        _student_global_aug = 'none' if _no_view_aug else 'strong'
        _student_local_aug = 'none' if _no_view_aug else 'weak_local'
        x_teacher_views_batch = get_teacher_input(
            x_teacher_base, _teacher_aug, is_train=True
        )

        # Student Global Views: strong augmentation, with randompatch mask
        x_global_bases_batch = torch.cat(x_global_bases, dim=0)  # [n_global*B, T, C]
        x_student_global_views_batch = get_student_input(
            x_global_bases_batch, _student_global_aug, is_train=True
        )
        # optimize: avoid list derivation, directly use batch (if need is accessed separately, you can use view)
        # x_student_global_views = [x_student_global_views_batch[i*B:(i+1)*B] for i in range(n_global_student)] # No longer used, keep comment
        
        # 3. Teacher Forward (EMA) - Batch all teacher views (need done before generating local views)
        teacher_was_training = self.teacher.training
        self.teacher.eval()
        
        teacher_temp = getattr(self, '_current_teacher_temp', getattr(self, 'teacher_temp', 0.07))
        
        # Teacher use the corresponding missing_mask (for views based on imputator, mask can be all 0)
        missing_mask_teacher = torch.cat(teacher_masks_list, dim=0).float()  # [n_teacher*B, T]
        time_mark_teacher = (
            torch.cat(teacher_time_list, dim=0) if time_mark is not None else None
        )
        lon_lat_teacher = (
            torch.cat(teacher_lon_lat_list, dim=0) if lon_lat is not None else None
        )
        geo_keep_teacher = geo_keep.repeat(n_teacher_views, 1, 1) if geo_keep is not None else None
        mm_keep = self._sample_missing_mask_embed_keep(B, device)
        mm_keep_teacher = self._repeat_missing_mask_embed_keep(mm_keep, n_teacher_views)
        
        with torch.no_grad():
            out_teacher_batch = self.teacher(
                x_teacher_views_batch, missing_mask_teacher, time_mark_teacher,
                mask_map=None, is_student=False,
                lon_lat=lon_lat_teacher, geo_keep=geo_keep_teacher,
                missing_mask_embed_keep=mm_keep_teacher,
            )

            t_logits_global_raw = out_teacher_batch['logits_global'].view(n_teacher_views, B, -1).detach()
            # note: Teacher’s ibot_head is now not calculated in the backbone and only returns normalized features
            t_z_cls_batch = out_teacher_batch['z_cls'].view(n_teacher_views, B, -1).detach()
            t_z_patch_batch = out_teacher_batch['z_patch_enc'].view(n_teacher_views, B, -1, out_teacher_batch['z_patch_enc'].shape[-1]).detach()
        
        t_logits_global = t_logits_global_raw
        # t_logits_patch is no longer calculated here. ibot_head will be used later after selecting masked patches according to mask_indices_list.
        
        self.teacher.train(teacher_was_training)

        # 4. Student Local Views: always crop or random token from **teacher time window** (x_target_perfect / time_mark / missing_mask),
            # Then weak_local augmentation; local views stay inside the shared teacher/student crop.
        _n_rand_lv = self.ted_modular_n_local_random_views if n_local_student > 0 else 0
        _n_crop_lv = n_local_student - _n_rand_lv
        local_view_types = ['crop'] * _n_crop_lv + ['random'] * _n_rand_lv
        x_student_local_views = []
        local_cross_overlap_list = []
        local_time_mark_list = []  # Save the time_mark corresponding to each local view
        local_missing_mask_list = []  # Save the missing_mask corresponding to each local view
        local_lon_lat_list = []  # lon_lat sync with time_mark (optional)
        local_parent_id_list = []  # crop / random both record parent global id (0 or 1)
        
        if n_local_student > 0:
            local_token_num = num_local_patches_cur
            local_time_steps = local_token_num * self.patch_len

            def _overlap_ratio_1d(start_tensor, span_len, other_start, other_len):
                starts = start_tensor.to(device=device, dtype=torch.float32)
                ends = starts + float(span_len)
                other_s = float(other_start)
                other_e = other_s + float(other_len)
                overlap = (
                    torch.minimum(ends, starts.new_full((), other_e))
                    - torch.maximum(starts, starts.new_full((), other_s))
                ).clamp_min(0.0)
                return overlap / max(float(span_len), 1.0)
            
            # crop local: linked to parent global; random local: only sampled from the patch in the corresponding parent time window
            crop_indices = [i for i, t in enumerate(local_view_types) if t == 'crop']
            random_indices = [i for i, t in enumerate(local_view_types) if t == 'random']

            if len(crop_indices) > 0:
                n_crop0 = len(crop_indices) // 2
                n_crop1 = len(crop_indices) - n_crop0
                crop_parent_ids = ([0] * n_crop0) + ([1] * n_crop1)
                if len(crop_indices) == 1:
                    crop_parent_ids = [0]

                for parent_id in sorted(set(crop_parent_ids)):
                    parent_view_indices = [
                        crop_idx for crop_idx, pid in zip(crop_indices, crop_parent_ids) if pid == parent_id
                    ]
                    if len(parent_view_indices) == 0:
                        continue

                    x_parent_base = local_parent_base_list[parent_id].repeat(len(parent_view_indices), 1, 1)
                    x_crop_batch, crop_start_indices = generate_local_view_crop(
                        x_parent_base, local_time_steps, device
                    )

                    batch_indices = torch.arange(len(parent_view_indices) * B, device=device).unsqueeze(1)
                    time_indices = crop_start_indices.unsqueeze(1) + torch.arange(
                        local_time_steps, device=device
                    ).unsqueeze(0)

                    parent_time = local_parent_time_list[parent_id]
                    parent_lon_lat = local_parent_lon_lat_list[parent_id]
                    parent_mask = local_parent_mask_list[parent_id]

                    if parent_time is not None:
                        time_mark_crop = parent_time.repeat(len(parent_view_indices), 1, 1)[
                            batch_indices, time_indices
                        ]
                    else:
                        time_mark_crop = None

                    if parent_lon_lat is not None:
                        lon_lat_crop = parent_lon_lat.repeat(len(parent_view_indices), 1, 1)[
                            batch_indices, time_indices
                        ]
                    else:
                        lon_lat_crop = None

                    missing_mask_crop = parent_mask.repeat(len(parent_view_indices), 1)[
                        batch_indices, time_indices
                    ]

                    x_crop_aug = get_student_input(
                        x_crop_batch, _student_local_aug, is_train=True
                    )

                    for idx, crop_idx in enumerate(parent_view_indices):
                        other_parent_id = 1 - int(parent_id) if len(global_view_starts) > 1 else int(parent_id)
                        local_abs_starts = (
                            crop_start_indices[idx * B:(idx + 1) * B]
                            + int(global_view_starts[parent_id])
                        )
                        view_cross_overlap = _overlap_ratio_1d(
                            local_abs_starts,
                            local_time_steps,
                            int(global_view_starts[other_parent_id]),
                            win_len,
                        ).mean().detach()
                        x_student_local_views.append((crop_idx, x_crop_aug[idx * B:(idx + 1) * B]))
                        if time_mark_crop is not None:
                            local_time_mark_list.append(time_mark_crop[idx * B:(idx + 1) * B])
                        else:
                            local_time_mark_list.append(None)
                        if lon_lat_crop is not None:
                            local_lon_lat_list.append(lon_lat_crop[idx * B:(idx + 1) * B])
                        else:
                            local_lon_lat_list.append(None)
                        local_missing_mask_list.append(missing_mask_crop[idx * B:(idx + 1) * B])
                        local_parent_id_list.append(parent_id)
                        local_cross_overlap_list.append(view_cross_overlap)
            
            if len(random_indices) > 0:
                # Random local: Like crop, it is linked to a single parent global and only samples from the teacher time window.
                # Avoid calculating the loss of two teachers at the same time after union sampling, causing target ambiguity.
                n_rand0 = len(random_indices) // 2
                n_rand1 = len(random_indices) - n_rand0
                random_parent_ids = ([0] * n_rand0) + ([1] * n_rand1)
                if len(random_indices) == 1:
                    random_parent_ids = [0]

                for parent_id in sorted(set(random_parent_ids)):
                    parent_view_indices = [
                        rand_idx for rand_idx, pid in zip(random_indices, random_parent_ids) if pid == parent_id
                    ]
                    if len(parent_view_indices) == 0:
                        continue

                    parent_base = local_parent_base_list[parent_id]
                    parent_time = local_parent_time_list[parent_id]
                    parent_lon_lat = local_parent_lon_lat_list[parent_id]
                    parent_mask = local_parent_mask_list[parent_id]

                    x_random_base = parent_base.repeat(len(parent_view_indices), 1, 1)
                    x_random_patches = patchify(x_random_base, self.patch_len, self.stride)
                    x_random_patches_sampled, token_indices = generate_local_view_random_sample(
                        x_random_patches, local_token_num, device
                    )
                    x_random_local = unpatchify(
                        x_random_patches_sampled, local_time_steps, self.patch_len, self.c_out
                    )

                    n_parent_random_samples = len(parent_view_indices) * B
                    batch_indices = torch.arange(n_parent_random_samples, device=device).unsqueeze(1)

                    time_mark_random_base = (
                        parent_time.repeat(len(parent_view_indices), 1, 1)
                        if parent_time is not None
                        else None
                    )
                    lon_lat_random_base = (
                        parent_lon_lat.repeat(len(parent_view_indices), 1, 1)
                        if parent_lon_lat is not None
                        else None
                    )
                    missing_mask_random_base = parent_mask.repeat(len(parent_view_indices), 1)

                    if time_mark_random_base is not None:
                        tm_patches = patchify(time_mark_random_base, self.patch_len, self.stride)
                        tm_patches_sampled = tm_patches[batch_indices, token_indices]
                        time_mark_random = unpatchify(
                            tm_patches_sampled, local_time_steps, self.patch_len, 2
                        )
                    else:
                        time_mark_random = None

                    if lon_lat_random_base is not None:
                        ll_patches = patchify(lon_lat_random_base, self.patch_len, self.stride)
                        ll_patches_sampled = ll_patches[batch_indices, token_indices]
                        lon_lat_random = unpatchify(
                            ll_patches_sampled, local_time_steps, self.patch_len, 2
                        )
                    else:
                        lon_lat_random = None

                    mm_patches = patchify(
                        missing_mask_random_base.unsqueeze(-1).float(), self.patch_len, self.stride
                    )
                    mm_patches_sampled = mm_patches[batch_indices, token_indices]
                    missing_mask_random = unpatchify(
                        mm_patches_sampled, local_time_steps, self.patch_len, 1
                    ).squeeze(-1)

                    x_random_aug = get_student_input(
                        x_random_local, _student_local_aug, is_train=True
                    )

                    for idx, random_idx in enumerate(parent_view_indices):
                        other_parent_id = 1 - int(parent_id) if len(global_view_starts) > 1 else int(parent_id)
                        token_idx_view = token_indices[idx * B:(idx + 1) * B]
                        patch_abs_starts = (
                            token_idx_view.reshape(-1) * int(self.stride)
                            + int(global_view_starts[parent_id])
                        )
                        view_cross_overlap = _overlap_ratio_1d(
                            patch_abs_starts,
                            self.patch_len,
                            int(global_view_starts[other_parent_id]),
                            win_len,
                        ).mean().detach()
                        x_student_local_views.append((random_idx, x_random_aug[idx * B:(idx + 1) * B]))
                        if time_mark_random is not None:
                            local_time_mark_list.append(time_mark_random[idx * B:(idx + 1) * B])
                        else:
                            local_time_mark_list.append(None)
                        if lon_lat_random is not None:
                            local_lon_lat_list.append(lon_lat_random[idx * B:(idx + 1) * B])
                        else:
                            local_lon_lat_list.append(None)
                        local_missing_mask_list.append(missing_mask_random[idx * B:(idx + 1) * B])
                        local_parent_id_list.append(parent_id)
                        local_cross_overlap_list.append(view_cross_overlap)
            
            # sync sorting: sorted by crop_idx/random_idx
            sorted_pairs = sorted(
                zip(
                    x_student_local_views,
                    local_time_mark_list,
                    local_missing_mask_list,
                    local_lon_lat_list,
                    local_parent_id_list,
                    local_cross_overlap_list,
                ),
                key=lambda x: x[0][0],
            )
            x_student_local_views = [pair[0][1] for pair in sorted_pairs]  # Extract data partial
            local_time_mark_list = [pair[1] for pair in sorted_pairs]
            local_missing_mask_list = [pair[2] for pair in sorted_pairs]
            local_lon_lat_list = [pair[3] for pair in sorted_pairs]
            local_cross_overlap_list = [pair[5] for pair in sorted_pairs]
            local_crop_parent_ids = [
                int(pair[4]) for pair in sorted_pairs[: len(crop_indices)] if pair[4] is not None
            ]
            local_random_parent_ids = [
                int(pair[4]) for pair in sorted_pairs[len(crop_indices) :] if pair[4] is not None
            ]
            local_crop_cross_overlaps = (
                torch.stack(local_cross_overlap_list[: len(crop_indices)])
                if len(crop_indices) > 0
                else None
            )
            local_random_cross_overlaps = (
                torch.stack(local_cross_overlap_list[len(crop_indices) :])
                if len(random_indices) > 0
                else None
            )
        else:
            local_crop_parent_ids = []
            local_random_parent_ids = []
            local_crop_cross_overlaps = None
            local_random_cross_overlaps = None

        # 5. Student Forward (Backbone) - handle multiple student views
        
        # use block-biased patch masks with random patches
        # The mask ratio interval is controlled by mask_rate_v1 / mask_rate_v2, which provides an external interface instead of hard-coding.
        # If only one value is given (mask_rate_v2 is empty), then use a fixed mask ratio.
        if mask_rate_v2 is None:
            mask_ratio_tuple = (mask_rate_v1, mask_rate_v1)
        else:
            mask_ratio_tuple = (min(mask_rate_v1, mask_rate_v2), max(mask_rate_v1, mask_rate_v2))
        mask_sample_probability = float(self.mask_sample_probability)

        collated_masks_global, mask_indices_list_global, masks_weight_global = random_patch_masking_dinov3_style(
            B * n_global_student,
            mask_ratio_tuple,
            mask_sample_probability,
            num_patches_cur,
            device,
            block_ratio=self.block_mask_ratio,
        )
        collated_masks_global = collated_masks_global.view(n_global_student, B, num_patches_cur)
        
        # Global student views (with mask) - batch processing
        # [Key modification]: Dynamically build missing_mask_global
        missing_mask_global = torch.cat(global_masks_list, dim=0).float() # [n_global*B, T]
        # In woMask mode, do not use missing_mask embedding, uniformly treated as all 0
        if imputator_mode == 'woMask':
            missing_mask_global = torch.zeros_like(missing_mask_global)

        time_mark_global = (
            torch.cat(global_time_list, dim=0) if time_mark is not None else None
        )
        lon_lat_global = (
            torch.cat(global_lon_lat_list, dim=0) if lon_lat is not None else None
        )
        geo_keep_global = geo_keep.repeat(n_global_student, 1, 1) if geo_keep is not None else None
        mask_map_global_batch = collated_masks_global.view(n_global_student * B, num_patches_cur)
        mm_keep_global = self._repeat_missing_mask_embed_keep(mm_keep, n_global_student)
        
        # Forward all global student views at once
        out_student_global_batch = self.backbone(
            x_student_global_views_batch, missing_mask_global, time_mark_global,
            mask_map=mask_map_global_batch, is_student=True,
            lon_lat=lon_lat_global, geo_keep=geo_keep_global,
            missing_mask_embed_keep=mm_keep_global,
        )
        
        s_logits_global = out_student_global_batch['logits_global'].view(n_global_student, B, -1)
        s_z_cls_global_list = out_student_global_batch['z_cls'].view(n_global_student, B, -1)
        # note: ibot_head is now not calculated in backbone, only normalized features are returned
        s_z_patch_global = out_student_global_batch['z_patch_enc'].view(n_global_student, B, -1, out_student_global_batch['z_patch_enc'].shape[-1])
        
        # note: ibot_head's application of masked patches will be filtered according to valid_sample_indices later.
        # Here save pre-head features first
        
        s_z_patch_enc = s_z_patch_global[0]  # [B, N, D]
        s_z_global = s_z_cls_global_list[0]  # [B, D]

        # Local student views (without mask) - batch processing
        if n_local_student > 0 and len(x_student_local_views) > 0:
            x_student_local_batch = torch.cat(x_student_local_views, dim=0)
            B_local, T_local, C_local = x_student_local_views[0].shape
            
            # [key fix]: use time_mark and missing_mask corresponding to crop/random position
            # optimize: concat after filter None (if time_mark is None, there will be None in local_time_mark_list)
            if len(local_time_mark_list) > 0 and local_time_mark_list[0] is not None:
                time_mark_local = torch.cat([tm for tm in local_time_mark_list if tm is not None], dim=0)
            else:
                time_mark_local = None
            
            if len(local_lon_lat_list) > 0 and local_lon_lat_list[0] is not None:
                lon_lat_local = torch.cat([z for z in local_lon_lat_list if z is not None], dim=0)
            else:
                lon_lat_local = None
            
            # missing_mask: generated from x_target_perfect, missing_mask may not be all 0 (depends on whether there is an imputator)
            if len(local_missing_mask_list) > 0:
                missing_mask_local = torch.cat(local_missing_mask_list, dim=0)  # [n_local_student * B, T_local]
            else:
                missing_mask_local = torch.zeros(n_local_student * B, T_local, device=device, dtype=torch.bool)
            
            geo_keep_local = geo_keep.repeat(n_local_student, 1, 1) if geo_keep is not None else None
            mm_keep_local = self._repeat_missing_mask_embed_keep(mm_keep, n_local_student)
            out_student_local_batch = self.backbone(
                x_student_local_batch, missing_mask_local.float(), time_mark_local,
                mask_map=None, is_student=True,
                lon_lat=lon_lat_local, geo_keep=geo_keep_local,
                missing_mask_embed_keep=mm_keep_local,
            )
            s_logits_local = out_student_local_batch['logits_global'].view(n_local_student, B, -1)
        else:
            s_logits_local = None

        # 6. Loss Calculation Preparation
        
        # Auxiliary function: filter based on Valid Sample (based on original observation quality, not missing_mask after imputator)
        # Notes:
        # - Even if the missing value is padded using imputator, if there are very few original valid observation points, the pseudo target of the sample is still unreliable.
        # Should not participate in sequence-state or patch-state distillation.
        # - So here use missing_mask_orig to measure samplevalidobservationratio.
        valid_ratio_per_sample = valid_ratio_per_global_view.min(dim=0).values
        valid_sample_threshold = float(self.valid_sample_threshold)
        valid_sample_indices = torch.where(valid_ratio_per_sample >= valid_sample_threshold)[0]
        
        if len(valid_sample_indices) == 0:
            # No sample passes valid_sample_threshold: exclude cls/patch/koleo/temporal;
            # Frequency-domain alignment should also skip samples that do not pass the threshold (previously, when thr_ff<=0, it would mistakenly usefull batch).
            return {
                'z_global': s_z_global,
                'valid_sample_indices': valid_sample_indices,
                'recon_data': {'rec_seq_valid': None, 'target_seq_valid': None},
                'spectral_data': {
                    's_patch_tokens_all': None,
                    't_patch_tokens_all': None,
                    'fft_masked_s_pre': None,
                    'fft_masked_t_pre': None,
                    'fft_masked_row_ids': None,
                    'fft_masked_patch_idx': None,
                },
                'cls_data': {
                    's_logits_global_valid': None,
                    's_logits_local_valid': None,
                    't_logits_global_valid': None,
                    'dino_global_scale': None,
                    'dino_local_scale': None,
                    'cls_local_n_crop_views': None,
                    'cls_local_n_random_views': None,
                    'cls_local_crop_parent_ids': None,
                    'cls_local_random_parent_ids': None,
                    'cls_local_crop_cross_overlaps': local_crop_cross_overlaps,
                    'cls_local_random_cross_overlaps': local_random_cross_overlaps,
                    'cls_global_overlap_ratio': global_view_overlap_ratio,
                    'cls_global_shift_ratio': global_view_shift_ratio,
                },
                'patch_data': {'s_patch_masked': None, 't_patch_masked_centered': None, 'masks_weight_global_valid': None, 'collated_masks_global_valid': None, 'mask_indices_list_global_valid': None, 'ibot_denom_rows': None},
                'koleo_data': {'s_z_cls_flat': None},
                'cls_consistency_data': None,
                'temporal_data': {'s_z_patch_enc_valid': None},
                'lambda_weights': {
                    'lambda_recon': 0.0,
                    'lambda_fft_align': self.lambda_fft_align,
                    'lambda_cls_proto': self.lambda_cls_proto,
                    'lambda_patch_proto': self.lambda_patch_proto,
                    'lambda_koleo': self.lambda_koleo,
                    'lambda_temporal': self.lambda_temporal,
                    'lambda_cls_cons': self.lambda_cls_cons,
                },
                'teacher_temp': teacher_temp,
                'ssl_num_valid_samples': 0,
                'ssl_subgroup_batch_size': int(B),
            }

        def filter_batch(tensor, indices):
            """
Filter the data on the batch dimension according to the sample index to ensure that the empty set when returns an empty tensor of shapecompatible.
            """
            if tensor is None:
                return None

            # only when tensor dimension >=3 when batch is considered to be at dim1; 2D/1D is considered to be batch at dim0
            batch_dim = 1 if tensor.dim() >= 3 else 0
            max_idx = tensor.size(batch_dim)
            if max_idx == 0:
                return tensor

            valid_indices = indices[(indices >= 0) & (indices < max_idx)]
            if len(valid_indices) == 0:
                # Construct an empty tensor with the same dimension as the original tensor (batch dimension length is 0)
                shape = list(tensor.shape)
                shape[batch_dim] = 0
                return tensor.new_empty(shape)

            if batch_dim == 1:
                return tensor[:, valid_indices]
            else:
                return tensor[valid_indices]

        # B. Sequence-state loss data preparation
        max_batch_size = s_logits_global.shape[1] if len(s_logits_global.shape) >= 2 else s_logits_global.shape[0]
        valid_sample_indices_safe = valid_sample_indices[valid_sample_indices < max_batch_size]
        
        if len(valid_sample_indices_safe) == 0:
            s_logits_global_valid = s_logits_global[:, :0, :] if len(s_logits_global.shape) == 3 else s_logits_global[:0]
            s_logits_local_valid = s_logits_local[:, :0, :] if len(s_logits_local.shape) == 3 else s_logits_local[:0]
            t_logits_global_valid = t_logits_global[:, :0, :] if len(t_logits_global.shape) == 3 else t_logits_global[:0]
            dino_global_scale = None
            dino_local_scale = None
        else:
            s_logits_global_valid = filter_batch(s_logits_global, valid_sample_indices_safe)
            s_logits_local_valid = filter_batch(s_logits_local, valid_sample_indices_safe)
            t_logits_global_valid = filter_batch(t_logits_global, valid_sample_indices_safe)
            
            if len(valid_sample_indices_safe) > 0 and s_logits_global_valid.shape[1] > 0:
                n_global_crops = s_logits_global_valid.shape[0]
                n_local_crops = s_logits_local_valid.shape[0] if s_logits_local_valid is not None else 0
                if n_local_crops > 0:
                    dino_global_terms = n_global_crops * (n_global_crops - 1)
                    n_crop_valid = min(int(_n_crop_lv), int(n_local_crops))
                    n_rand_valid = max(0, int(n_local_crops) - n_crop_valid)
                    # crop / random locals are parent-aware, each align a single teacher global
                    dino_local_terms = n_crop_valid + n_rand_valid
                    dino_global_scale = dino_global_terms / (dino_global_terms + dino_local_terms)
                    dino_local_scale = dino_local_terms / (dino_global_terms + dino_local_terms)
                else:
                    dino_global_scale = 1.0
                    dino_local_scale = 0.0
            else:
                dino_global_scale = None
                dino_local_scale = None

        # C. Patch-state loss data preparation
        fft_masked_s_pre = None
        fft_masked_t_pre = None
        fft_masked_row_ids = None
        fft_masked_patch_idx = None
        # note: the patch-state head is used only for masked patches
        s_patch_masked = None
        t_patch_masked = None
        masks_weight_global_valid = None
        collated_masks_global_valid = None
        mask_indices_list_global_valid = None
        ibot_denom_rows = None
        all_valid_tokens_in_batch = None
        ibot_debug_info = None

        if len(mask_indices_list_global) > 0:
            collated_masks_global_valid = collated_masks_global[:, valid_sample_indices_safe, :]

            # align patch dimension length to avoid mismatch between mask and token numbers in dynamic length scenarios causing gather to go out of bounds
            n_patch_mask = int(collated_masks_global_valid.shape[2])
            n_patch_student = int(s_z_patch_global.shape[2])
            n_patch_teacher = int(t_z_patch_batch.shape[2])
            n_patch_offset = int(max(0, min(global_shift_patch_offset, n_patch_teacher)))
            n_patch_aligned = min(n_patch_mask, n_patch_student, max(0, n_patch_teacher - n_patch_offset))
            if n_patch_aligned <= 0:
                collated_masks_global_valid = None
                mask_indices_list_global_valid = None
            else:
                if n_patch_aligned != n_patch_mask:
                    collated_masks_global_valid = collated_masks_global_valid[:, :, :n_patch_aligned]

                # Recalculate the mask_indices of valid sample (based on filtered valid_sample_indices_safe)
                valid_mask_flatten = collated_masks_global_valid.flatten()  # [n_global_student * B_valid * N]
                mask_indices_list_global_valid = valid_mask_flatten.nonzero(as_tuple=False).squeeze(1)

            # [Patch reliability filter] filter: remove patch_valid_ratio_orig==0; keep_all: do not remove
            # In full mode is_patch_reliable_for_ibot is all True, no filter
            if (
                collated_masks_global_valid is not None
                and mask_indices_list_global_valid is not None
                and len(mask_indices_list_global_valid) > 0
            ):
                _ibot_dbg = os.environ.get("TED_IBOT_PATCH_DEBUG", "").strip().lower() in (
                    "1",
                    "true",
                    "yes",
                )
                if _ibot_dbg:
                    ibot_debug_info = {
                        "n_masked_pre_reliable_filter": int(mask_indices_list_global_valid.shape[0]),
                    }
                n_gs, B_valid, N_patch = collated_masks_global_valid.shape
                is_patch_reliable_valid = is_patch_reliable_for_ibot_global[:, valid_sample_indices_safe, :N_patch]
                view_idx = mask_indices_list_global_valid // (B_valid * N_patch)
                flat_rest = mask_indices_list_global_valid % (B_valid * N_patch)
                patch_sample_idx = flat_rest // N_patch
                patch_patch_idx = flat_rest % N_patch
                reliable_flags = is_patch_reliable_valid[view_idx, patch_sample_idx, patch_patch_idx]  # [n_masked]
                mask_indices_list_global_valid = mask_indices_list_global_valid[reliable_flags]
                if _ibot_dbg and ibot_debug_info is not None:
                    ibot_debug_info["n_masked_post_reliable_filter"] = int(
                        mask_indices_list_global_valid.shape[0]
                    )

            if (
                collated_masks_global_valid is not None
                and mask_indices_list_global_valid is not None
                and len(mask_indices_list_global_valid) > 0
            ):
                _, B_valid, N_patch = collated_masks_global_valid.shape

                # Student: Select the pre-head features of masked patches and then use ibot_head
                s_patch_flat_valid = s_z_patch_global[:, valid_sample_indices_safe, :N_patch, :].flatten(0, 1).flatten(0, 1)  # [n_global_student * B_valid * N_common, D]
                max_student_idx = int(s_patch_flat_valid.shape[0])

                # Security filter illegal index (theoretically it should not occur; thennotes boundary length appears and there is a mismatch)
                valid_idx_s = (mask_indices_list_global_valid >= 0) & (mask_indices_list_global_valid < max_student_idx)
                if not bool(valid_idx_s.all()):
                    if not hasattr(self, "_ibot_index_guard_warned_s"):
                        self._ibot_index_guard_warned_s = False
                    if not self._ibot_index_guard_warned_s:
                        dropped_s = int((~valid_idx_s).sum().item())
                        print(
                            f"[patch-state index guard] drop {dropped_s} invalid student mask indices; max={max_student_idx}",
                            flush=True,
                        )
                        self._ibot_index_guard_warned_s = True
                    mask_indices_list_global_valid = mask_indices_list_global_valid[valid_idx_s]

                if len(mask_indices_list_global_valid) > 0:
                    # Teacher: select pre-head features of masked patches before the patch-state head
                    t_patch_all_global = t_z_patch_batch.view(n_teacher_views, B, -1, t_z_patch_batch.shape[-1])  # [n_teacher, B, N, D]
                    t_patch_all_global_valid = t_patch_all_global[:, valid_sample_indices_safe, n_patch_offset:n_patch_offset + N_patch, :]  # [n_teacher, B_valid, N_common, D]
                    t_patch_flat_valid = t_patch_all_global_valid.flatten(0, 1).flatten(0, 1)  # [n_teacher * B_valid * N, D]
                    max_teacher_idx = int(t_patch_flat_valid.shape[0])

                    valid_idx_t = mask_indices_list_global_valid < max_teacher_idx
                    if not bool(valid_idx_t.all()):
                        if not hasattr(self, "_ibot_index_guard_warned_t"):
                            self._ibot_index_guard_warned_t = False
                        if not self._ibot_index_guard_warned_t:
                            dropped_t = int((~valid_idx_t).sum().item())
                            print(
                                f"[patch-state index guard] drop {dropped_t} invalid teacher mask indices; max={max_teacher_idx}",
                                flush=True,
                            )
                            self._ibot_index_guard_warned_t = True
                        mask_indices_list_global_valid = mask_indices_list_global_valid[valid_idx_t]

                    if len(mask_indices_list_global_valid) > 0:
                        s_masked_patches_pre_head = torch.index_select(s_patch_flat_valid, dim=0, index=mask_indices_list_global_valid)
                        t_masked_patches_pre_head = torch.index_select(t_patch_flat_valid, dim=0, index=mask_indices_list_global_valid)
                        if not self.fft_align_all_patches:
                            fft_masked_s_pre = s_masked_patches_pre_head
                            fft_masked_t_pre = t_masked_patches_pre_head.detach()
                            fft_masked_row_ids = mask_indices_list_global_valid // N_patch
                            fft_masked_patch_idx = mask_indices_list_global_valid % N_patch
                        s_patch_masked = self.backbone.ibot_head(s_masked_patches_pre_head)  # [n_masked_valid, K]

                        # Teacher use no_grad and detach
                        with torch.no_grad():
                            t_patch_masked = self.teacher.ibot_head(t_masked_patches_pre_head).detach()  # [n_masked_valid, K]

                        # Patch-state weights:
                        # The original token-level loss averages masked tokens within each row because all patches are valid.
                        # TED first drops unreliable masked patches, so use the retained masked count
                        # per row. This keeps row weight linear in the raw-valid patch count instead
                        # of multiplying it again by retained/original masked ratio.
                        n_gs, B_valid, N_patch = collated_masks_global_valid.shape
                        row_ids_all = mask_indices_list_global_valid // N_patch  # [n_masked_valid]

                        # Raw-valid patch count per (global_view, sample) row.
                        patch_valid_ratio_valid = patch_valid_ratio_orig_global[:, valid_sample_indices_safe, :N_patch]
                        valid_tokens_per_row = (
                            (patch_valid_ratio_valid > 0).sum(dim=-1).float().reshape(-1).clamp(min=1.0)
                        )  # [n_global_student * B_valid]

                        n_rows_total = n_gs * B_valid
                        retained_per_row = torch.bincount(
                            row_ids_all,
                            minlength=n_rows_total,
                        ).to(device=device, dtype=valid_tokens_per_row.dtype).clamp(min=1.0)
                        masks_weight_global_valid = (
                            valid_tokens_per_row[row_ids_all] / retained_per_row[row_ids_all]
                        )  # [n_masked_valid]

                        active_row_ids = torch.unique(row_ids_all)
                        all_valid_tokens_in_batch = (
                            valid_tokens_per_row[active_row_ids].sum().clamp(min=1.0)
                        )

                        # Option 4: Down-weight the patch that includes imputator imputation position (only takes effect when use imputator is enabled and weights < 1)
                        if imputator is not None and self.imputed_patch_weight != 1.0:
                            n_gs, B_valid, N_patch = collated_masks_global_valid.shape
                            has_imputed_patch_valid = has_imputed_patch_global[:, valid_sample_indices_safe, :N_patch]
                            view_idx = mask_indices_list_global_valid // (B_valid * N_patch)
                            flat_rest = mask_indices_list_global_valid % (B_valid * N_patch)
                            patch_sample_idx = flat_rest // N_patch
                            patch_patch_idx = flat_rest % N_patch
                            has_imputed_for_masked = has_imputed_patch_valid[
                                view_idx, patch_sample_idx, patch_patch_idx
                            ]  # [n_masked_valid]
                            # Multiply the included imputed patches by imputed_patch_weight, and keep the rest at 1
                            weight_factor = torch.ones_like(masks_weight_global_valid)
                            weight_factor = torch.where(
                                has_imputed_for_masked,
                                torch.full_like(weight_factor, self.imputed_patch_weight),
                                weight_factor,
                            )
                            masks_weight_global_valid = masks_weight_global_valid * weight_factor

                        n_masked = len(t_patch_masked)
                        if masks_weight_global_valid is not None and len(masks_weight_global_valid) != n_masked:
                            if len(masks_weight_global_valid) > n_masked:
                                masks_weight_global_valid = masks_weight_global_valid[:n_masked]
                            else:
                                padding = torch.ones(n_masked - len(masks_weight_global_valid), device=masks_weight_global_valid.device, dtype=masks_weight_global_valid.dtype)
                                masks_weight_global_valid = torch.cat([masks_weight_global_valid, padding])

                        _, _, N_patch_denom = collated_masks_global_valid.shape
                        row_ids = mask_indices_list_global_valid // N_patch_denom
                        ibot_denom_rows = float(max(1, int(torch.unique(row_ids).numel())))
                        if ibot_debug_info is not None:
                            ibot_debug_info["ibot_denom_rows"] = float(ibot_denom_rows)
                            ibot_debug_info["ibot_patch_reliable_mode"] = getattr(
                                self, "ibot_patch_reliable_mode", "filter"
                            )

        # D. Feature-spread regularization
        s_z_cls_global_valid = filter_batch(s_z_cls_global_list, valid_sample_indices_safe)
        s_z_cls_flat = None
        if s_z_cls_global_valid.shape[0] > 0 and s_z_cls_global_valid.shape[1] > 0:
            s_z_cls_flat = s_z_cls_global_valid.flatten(0, 1)

        # D'. Raw–Imputed CLS consistency: only when there is imputator and student, the same as when there are two ways of raw and imputed global view when provided
        cls_consistency_data = None
        if (
            imputator is not None
            and len(x_student_global_use_imputator) >= 2
            and sum(x_student_global_use_imputator) == 1
            and global_views_share_content
            and s_z_cls_global_valid.shape[0] >= 2
            and s_z_cls_global_valid.shape[1] > 0
        ):
            cls_consistency_data = {
                's_z_cls_per_view': s_z_cls_global_valid,
                'use_imputator_per_view': x_student_global_use_imputator,
            }

        # E. Temporal Loss
        s_z_patch_enc_valid = filter_batch(s_z_patch_enc, valid_sample_indices_safe)

        # F. Frequency-domain patch alignment: full windows only when fft_align_all_patches; otherwise use masked pre-head patch tokens
        s_patch_tokens_all = None
        t_patch_tokens_all = None
        if self.fft_align_all_patches:
            n_patch_student = int(s_z_patch_global.shape[2])
            n_patch_teacher = int(t_z_patch_batch.shape[2])
            n_po_fft = int(max(0, min(global_shift_patch_offset, max(0, n_patch_teacher - 1))))
            n_patch_align = min(n_patch_student, max(0, n_patch_teacher - n_po_fft))
            thr_ff = float(getattr(self, "fft_align_min_valid_ratio", 0.0))
            if n_patch_align <= 0:
                s_patch_tokens_all = None
                t_patch_tokens_all = None
            else:
                if n_patch_student != n_patch_teacher or n_po_fft > 0:
                    if not hasattr(self, "_fft_align_patch_warned"):
                        self._fft_align_patch_warned = False
                    if not self._fft_align_patch_warned:
                        print(
                            f"[frequency-domain patch align] student={n_patch_student}, teacher={n_patch_teacher}, "
                            f"teacher_offset={n_po_fft}, align={n_patch_align}",
                            flush=True,
                        )
                        self._fft_align_patch_warned = True

                n_pair = min(int(n_global_student), int(n_teacher_views))
                s_rows = []
                t_rows = []
                for g in range(n_pair):
                    if len(valid_sample_indices_safe) == 0:
                        continue
                    s_sub = s_z_patch_global[g, valid_sample_indices_safe, :n_patch_align, :]
                    t_sub = (
                        t_z_patch_batch[g, valid_sample_indices_safe, n_po_fft : n_po_fft + n_patch_align, :]
                        .detach()
                    )
                    if thr_ff <= 0:
                        keep = torch.ones(len(valid_sample_indices_safe), device=device, dtype=torch.bool)
                    else:
                        keep = valid_ratio_per_sample[valid_sample_indices_safe] >= thr_ff
                    if not bool(keep.any()):
                        continue
                    s_view = s_sub[keep]
                    t_view = t_sub[keep]

                    s_rows.append(s_view)
                    t_rows.append(t_view)

                if len(s_rows) == 0:
                    s_patch_tokens_all = None
                    t_patch_tokens_all = None
                else:
                    s_patch_tokens_all = torch.cat(s_rows, dim=0)
                    t_patch_tokens_all = torch.cat(t_rows, dim=0)

        _ssl_nv = int(valid_sample_indices_safe.numel())

        return {
            'z_global': s_z_global,
            'valid_sample_indices': valid_sample_indices_safe,
            'recon_data': {'rec_seq_valid': None, 'target_seq_valid': None},
            'spectral_data': {
                's_patch_tokens_all': s_patch_tokens_all,
                't_patch_tokens_all': t_patch_tokens_all,
                'fft_masked_s_pre': fft_masked_s_pre,
                'fft_masked_t_pre': fft_masked_t_pre,
                'fft_masked_row_ids': fft_masked_row_ids,
                'fft_masked_patch_idx': fft_masked_patch_idx,
            },
            'cls_data': {
                's_logits_global_valid': s_logits_global_valid,
                's_logits_local_valid': s_logits_local_valid,
                't_logits_global_valid': t_logits_global_valid,
                'dino_global_scale': dino_global_scale,
                'dino_local_scale': dino_local_scale,
                # local logits dim0: first crop (when window fragment), then random (sampling within patch)
                'cls_local_n_crop_views': int(_n_crop_lv),
                'cls_local_n_random_views': int(_n_rand_lv),
                'cls_local_crop_parent_ids': local_crop_parent_ids,
                'cls_local_random_parent_ids': local_random_parent_ids,
                'cls_local_crop_cross_overlaps': local_crop_cross_overlaps,
                'cls_local_random_cross_overlaps': local_random_cross_overlaps,
                'cls_global_overlap_ratio': global_view_overlap_ratio,
                'cls_global_shift_ratio': global_view_shift_ratio,
            },
            'patch_data': {
                's_patch_masked': s_patch_masked,  # [n_masked_valid, K] - should use ibot_head
                't_patch_masked': t_patch_masked,  # [n_masked_valid, K] - should use ibot_head
                # note: t_logits_patch_full is no longer needed because ibot_head should only be used for masked patches
                'masks_weight_global_valid': masks_weight_global_valid,
                'collated_masks_global_valid': collated_masks_global_valid,
                'mask_indices_list_global_valid': mask_indices_list_global_valid,
                'ibot_denom_rows': ibot_denom_rows,
                'all_valid_tokens': all_valid_tokens_in_batch,
                'ibot_debug': ibot_debug_info,
            },
            'koleo_data': {'s_z_cls_flat': s_z_cls_flat},
            'temporal_data': {'s_z_patch_enc_valid': s_z_patch_enc_valid},
            'cls_consistency_data': cls_consistency_data,
            'lambda_weights': {
                'lambda_recon': 0.0,
                'lambda_fft_align': self.lambda_fft_align,
                'lambda_cls_proto': self.lambda_cls_proto,
                'lambda_patch_proto': self.lambda_patch_proto,
                'lambda_koleo': self.lambda_koleo,
                'lambda_temporal': self.lambda_temporal,
                'lambda_cls_cons': self.lambda_cls_cons,
            },
            'teacher_temp': teacher_temp,
            'ssl_num_valid_samples': _ssl_nv,
            'ssl_subgroup_batch_size': int(B),
        }

    def forward(self, x_enc, time_mark=None, valid_mask=None, next_x_enc=None, mode='train', mask_rate_v1=0.3, mask_rate_v2=0.6, imputator=None, teacher_temp=None, iteration=None, total_iterations=None, current_epoch=None, lon_lat=None):
        # If the dynamic teacher_temp is passed in, save it
        if teacher_temp is not None:
            self._current_teacher_temp = teacher_temp
        
        # save iteration / epoch information is used in curriculum learning and other schedules
        if iteration is not None:
            self._current_iteration = iteration
        if total_iterations is not None:
            self._total_iterations = total_iterations
        if current_epoch is not None:
            self._current_epoch = int(current_epoch)
        
        if mode == 'train':
            if self.evidence_gap_distill:
                if self.evidence_gap_dual_teacher_cross:
                    return self._forward_evidence_gap_dual_teacher_cross(
                        x_enc,
                        time_mark,
                        mask_rate_v1,
                        mask_rate_v2,
                        mode='train',
                        imputator=imputator,
                        lon_lat=lon_lat,
                    )
                return self._forward_evidence_gap_v2(
                    x_enc,
                    time_mark,
                    mask_rate_v1,
                    mask_rate_v2,
                    mode='train',
                    imputator=imputator,
                    lon_lat=lon_lat,
                )
            return self._forward(x_enc, time_mark, mask_rate_v1, mask_rate_v2, mode='train', imputator=imputator, lon_lat=lon_lat)

        raise ValueError(
            f"TED.forward only supports mode='train', current mode='{mode}'."
            "inference/feature extraction please use encode(), please use visualize for visualization()。"
        )

    def encode(self, xEnc, timeMark=None, imputator=None, lon_lat=None):
        """
Coding method
        Args:
            xEnc: input data [B, T, C]
            timeMark: time mark [B, T, 2]（optional）
            imputator:interpolator (optional), if provided then use the paddingmissing value
            lon_lat: optional [B, T, 2], WGS84 degree; do not pass then and do not add geo embedding (consistent with behavior after training CFG drop)
        Returns:
            include cls_token, storage_tokens,Dictionary of patch_tokens
        """
        try:
            # Construct missing_mask / imputator_mode logic consistently with the training phase
            missing_mask_orig = torch.isnan(xEnc).any(dim=-1).float()
            imputator_mode = getattr(self, "imputator_mode", "full")
            
            if imputator is not None:
                if imputator_mode in ["full", "woMask", "mixed_teacher"]:
                    # full / woMask: During the training phase, the teacher view is considered to have no missing
                    missing_mask = torch.zeros_like(missing_mask_orig)
                else:  # recon_only and others: keep physical missing
                    missing_mask = missing_mask_orig
            else:
                # There is no imputator when, training phase missing_mask = missing_mask_orig
                missing_mask = missing_mask_orig

            x_in = xEnc.nan_to_num(0.0)
            if timeMark is None:
                timeMark = torch.zeros(xEnc.shape[0], xEnc.shape[1], 1, device=xEnc.device)
            
            # Uniformly changed to Teacher branch encoding; imputator performs segmentation imputation according to pred_len in backbone.encode, which is consistent with training _forward
            if imputator is not None:
                return self.teacher.encode(
                    x_in, missing_mask, timeMark, imputator=imputator, use_student_norm=False, lon_lat=lon_lat, geo_keep=None
                )
            else:
                return self.teacher.encode(
                    x_in, missing_mask, timeMark, imputator=None, use_student_norm=False, lon_lat=lon_lat, geo_keep=None
                )
        except Exception as e:
            print(f"Encode Error: {e}")
            return {'cls_token': torch.zeros(xEnc.shape[0], self.backbone.d_model, device=xEnc.device), 'patch_tokens': None}

    def visualize(self, xEnc, timeMark=None, imputator=None, lon_lat=None):
        """
Visualization method: return the attention weights of each layer, CLS token and the cos similarity of all tokens and other information
        
        Args:
            xEnc: input data [B, T, C]
            timeMark: time mark [B, T, 2]（optional）
            imputator:interpolator (optional), if provided then use the paddingmissing value
            lon_lat: optional [B, T, 2], WGS84 degrees
        
        Returns:
include a dictionary with the following information:
            - all_attns:List of attention weights for each layer
            - cls_cos_sim_all:Cos similarity between CLS token and all tokens[B, 1+R+N]
            - cls_cos_sim_patch:cos similarity between CLS token and patch tokens[B, N]
            - cls_token:CLS token characteristics[B, D]
            - patch_tokens:Patch tokens features[B, N, D]
            - storage_tokens:Storage tokens characteristics[B, R, D] or None
        """
        try:
            missing_mask_orig = torch.isnan(xEnc).any(dim=-1).float()
            imputator_mode = getattr(self, "imputator_mode", "full")

            if imputator is not None:
                if imputator_mode in ["full", "woMask", "mixed_teacher"]:
                    missing_mask = torch.zeros_like(missing_mask_orig)
                else:
                    missing_mask = missing_mask_orig
            else:
                missing_mask = missing_mask_orig

            x_in = xEnc.nan_to_num(0.0)
            if timeMark is None: 
                timeMark = torch.zeros(xEnc.shape[0], xEnc.shape[1], 2, device=xEnc.device)
            
            # use Teacher branch for visualization (more stable)
            # Adjust the visualize method of use teacher
            return self.teacher.visualize(
                x_in, missing_mask, timeMark, imputator=imputator, use_student_norm=False, lon_lat=lon_lat, geo_keep=None
            )
        except Exception as e:
            print(f"Visualize Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'all_attns': [],
                'cls_cos_sim_all': None,
                'cls_cos_sim_patch': None,
                'cls_token': torch.zeros(xEnc.shape[0], self.backbone.d_model, device=xEnc.device),
                'patch_tokens': None,
                'storage_tokens': None,
            }
