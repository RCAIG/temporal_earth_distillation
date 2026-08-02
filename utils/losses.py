"""TED loss terms for sequence-state and patch-state self-distillation."""

from __future__ import annotations

import math
import warnings

import torch
import torch as t
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

try:
    from dinov3.distributed import get_process_subgroup
except ImportError:
    def get_process_subgroup():
        return None


def mse_loss(pred, target, mask=None):
    """
    Compute masked Mean Squared Error (MSE) loss.

    Args:
        pred (torch.Tensor): Predicted values, shape [batch_size, seq_len, target_channels] or similar.
        target (torch.Tensor): Target values, same shape as pred.
        mask (torch.Tensor, optional): Mask tensor, shape [batch_size, seq_len] or broadcastable.
                                      1 for positions to include, 0 to exclude. If None, no masking.

    Returns:
        torch.Tensor: Scalar MSE loss, averaged over masked positions.
    """
    # ensure pred and target have the same shape
    assert pred.shape == target.shape, f"pred shape {pred.shape} != target shape {target.shape}"

    # compute element-wise squared differences
    mse = (pred - target) ** 2  # [batch_size, seq_len, target_channels]

    if mask is None:
        # without mask, mean over all positions
        loss = mse.mean()
    else:
        # ensure mask broadcasts with mse
        if mask.dim() == 2:  # [batch_size, seq_len] -> [batch_size, seq_len, 1]
            mask = mask.unsqueeze(-1)

        # apply mask and compute loss over masked region
        masked_mse = mse * mask  # [batch_size, seq_len, target_channels]
        sum_loss = masked_mse.sum()  # scalar total loss over masked region
        num_valid = mask.sum()  # scalar count of 1s in mask

        # avoid div-by-zero; empty mask still ties to pred so total_loss has grad on mixed_batch paths
        if num_valid > 0:
            loss = sum_loss / num_valid
        else:
            loss = (pred * 0).sum()

    return loss


def mae_loss(pred, target, mask=None):
    """Masked mean absolute error (L1), same masking convention as mse_loss."""
    assert pred.shape == target.shape, f"pred shape {pred.shape} != target shape {target.shape}"
    abs_diff = t.abs(pred - target)
    if mask is None:
        return abs_diff.mean()
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    masked = abs_diff * mask
    num_valid = mask.sum().clamp(min=1)
    return masked.sum() / num_valid


def huber_loss(pred, target, mask=None, delta=3.0):
    """
    Compute masked Huber loss.

    Args:
        pred (torch.Tensor): Predicted values, shape [batch_size, seq_len, target_channels] or similar.
        target (torch.Tensor): Target values, same shape as pred.
        mask (torch.Tensor, optional): Mask tensor, shape [batch_size, seq_len] or broadcastable.
                                      1 for positions to include, 0 to exclude. If None, no masking.
        delta (float, optional): Huber loss threshold. Defaults to 1.0.

    Returns:
        torch.Tensor: Scalar Huber loss, averaged over masked positions.
    """
    # Ensure pred and target have the same shape
    assert pred.shape == target.shape, f"pred shape {pred.shape} != target shape {target.shape}"

    # Compute element-wise absolute difference
    abs_diff = t.abs(pred - target)  # [batch_size, seq_len, target_channels]

    # Compute Huber loss
    quadratic = t.min(abs_diff, t.tensor(delta, device=pred.device, dtype=pred.dtype))
    linear = abs_diff - quadratic
    huber = quadratic ** 2 + delta * linear  # [batch_size, seq_len, target_channels]

    if mask is None:
        # Without mask, compute mean over all positions
        loss = huber.mean()
    else:
        # Ensure mask is broadcastable with huber
        if mask.dim() == 2:  # [batch_size, seq_len] -> [batch_size, seq_len, 1]
            mask = mask.unsqueeze(-1)

        # Apply mask and compute loss over masked positions
        masked_huber = huber * mask  # [batch_size, seq_len, target_channels]
        sum_loss = masked_huber.sum()  # Scalar, total loss over masked positions
        num_valid = mask.sum()  # Scalar, number of masked positions

        # Avoid division by zero
        if num_valid > 0:
            loss = sum_loss / num_valid
        else:
            loss = t.tensor(0.0, device=pred.device, dtype=pred.dtype)

    return loss


def smooth_loss(pred, mode='dy2'):
    """compute improved second-order difference loss emphasizing direction changes"""
    # first check whether input contains NaN/Inf
    if t.isnan(pred).any() or t.isinf(pred).any():
        warnings.warn("Found NaN/Inf in prediction tensor")
        return t.tensor(0.0, device=pred.device)
    # compute first-order differences
    dy = pred[:, 1:] - pred[:, :-1]  # dy[i] = pred[i+1] - pred[i], length N-1
    # compute second-order differences
    d2y = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]  # length N-2
    if mode == 'dy1':
        loss = t.mean(dy ** 2)
    else:
        loss = t.mean(d2y ** 2)

    return loss


class BalancedCategoricalAssignment(nn.Module):
    """
    Balanced categorical-assignment module for teacher state distributions.
    It supports sequence-state rows and masked patch-state rows under DDP, where
    the number of valid rows may differ across ranks.
    """
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, teacher_output, teacher_temp, n_masked_patches_tensor=None, n_iterations=3):
        """
        teacher_output: [N_local, K] 
                        Note: if this rank has no samples, pass a dummy tensor [1, K]
        teacher_temp: temperature coefficient
        n_masked_patches_tensor: optional number of masked patch-state rows;
                                 if None, infer the count from teacher_output.shape[0]
        """
        teacher_output = teacher_output.float()
        # Temperature-scaled teacher logits for balanced assignment; no clamp.
        Q = torch.exp(teacher_output / teacher_temp).t()  # Q shape: [K, N_local]
        
        K = Q.shape[0] # categorical states
        B_local = Q.shape[1] # Local Batch Size

        # --- Key fix: dynamically compute global batch size (B) ---
        # Supports two modes:
        # 1. patch-state path: use the explicit masked-row count when provided
        # 2. sequence-state path: infer row count from teacher_output.shape[0]
        if n_masked_patches_tensor is not None:
            # patch-state path: use passed tensor
            B = n_masked_patches_tensor.clone().float()
            if dist.is_initialized():
                dist.all_reduce(B, group=get_process_subgroup())
            B_total = B.item()
        else:
            # sequence-state path: dynamic computation
            # cannot assume B_global = B_local * world_size because valid sample counts differ per rank
            # even dummy tensors contribute B_local (usually 1), keeping the balanced-assignment denominator non-zero
            # slightly dilutes the global distribution (extra dummy samples) but avoids deadlock and keeps numerics stable.
            B_tensor = torch.tensor(B_local, device=teacher_output.device, dtype=torch.float32)
            if dist.is_initialized():
                dist.all_reduce(B_tensor, group=get_process_subgroup())
            B_total = B_tensor.item()
        
        # avoid division by zero（extreme edge case）
        if B_total == 0: B_total = 1.0

        # 2. normalize matrix sum
        # keep assignment numerics unclamped for consistency across state heads
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q, group=get_process_subgroup())
        Q /= sum_Q

        # 3. Balanced-assignment iterations
        # keep assignment numerics unclamped for consistency across state heads and extra NaN checks
        for _ in range(n_iterations):
            # row normalize: each categorical state has total weight 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows, group=get_process_subgroup())
            Q /= sum_of_rows
            Q /= K

            # column normalize: each sample total weight 1/B_total
            # local operation; no communication needed
            sum_of_cols = torch.sum(Q, dim=0, keepdim=True)
            Q /= sum_of_cols
            Q /= B_total

        Q *= B_total  # the columns must sum to 1 so that Q is an assignment
        return Q.t() # return [N_local, K]


class SequenceStateLoss(nn.Module):
    def __init__(self, out_dim, student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.sinkhorn = BalancedCategoricalAssignment()

    def forward(self, student_logits, teacher_probs, ignore_diagonal=False):
        """
        student_logits: [n_student_windows, B, K]
        teacher_probs:  [n_teacher_windows, B, K] after balanced assignment
        """
        student_crops, B, K = student_logits.shape
        teacher_crops, _, _ = teacher_probs.shape
        # Sequence-state CE uses log_softmax directly, with no extra clamp.
        student_logits = F.log_softmax(student_logits.float() / self.student_temp, dim=-1)
        
        if not ignore_diagonal:
            # standard cross-entropy
            loss = -torch.einsum("s b k, t b k -> ", student_logits, teacher_probs)
            return loss / (B * student_crops * teacher_crops)
        else:
            # ignore matched global-global rows when requested
            loss = -torch.einsum("s b k, t b k -> s t", student_logits, teacher_probs)
            min_st = min(student_crops, teacher_crops)
            loss = torch.diagonal_scatter(loss, loss.new_zeros(min_st))
            return loss.sum() / (B * student_crops * teacher_crops - B * min_st)

    def weighted_forward(self, student_logits, teacher_probs, pair_weights):
        """
        Weighted CE over student/teacher temporal-window pairs.
        pair_weights: [n_student_windows, n_teacher_windows], zeros disable a pair.
        """
        student_crops, B, _ = student_logits.shape
        teacher_crops, _, _ = teacher_probs.shape
        log_probs = F.log_softmax(student_logits.float() / self.student_temp, dim=-1)
        weights = pair_weights.to(device=log_probs.device, dtype=log_probs.dtype)
        if tuple(weights.shape) != (student_crops, teacher_crops):
            raise ValueError(
                f"pair_weights shape {tuple(weights.shape)} does not match "
                f"student/teacher crops {(student_crops, teacher_crops)}"
            )
        denom = weights.sum().clamp_min(1e-12)
        pair_loss = -torch.einsum("s b k, t b k -> s t", log_probs, teacher_probs)
        return (pair_loss * weights).sum() / (B * denom)


def cls_local_bag_loss(
    s_logits_local,
    t_probs,
    student_temp,
    lambda_contrib=0.0,
    contrib_margin=0.0,
):
    """
    Bag-style local sequence state: local temporal windows pool to one distribution against averaged teacher targets.

    s_logits_local: [L, B, K]
    t_probs:        [N_teacher_windows, B, K] after balanced assignment
    """
    L, B, K = s_logits_local.shape
    st = float(student_temp)
    log_p_local = F.log_softmax(s_logits_local.float() / st, dim=-1)
    log_p_bag = torch.logsumexp(log_p_local - math.log(L), dim=0)
    q_avg = t_probs.mean(dim=0).detach().float()
    loss_bag = -(q_avg * log_p_bag).sum(dim=-1).mean()

    if lambda_contrib <= 0:
        return loss_bag

    p_local = F.softmax(s_logits_local.float() / st, dim=-1)
    q_dot_p = (p_local * q_avg.unsqueeze(0)).sum(dim=-1)
    q_dot_q = (q_avg * q_avg).sum(dim=-1).unsqueeze(0)
    uniform = 1.0 / float(K)
    contrib = (q_dot_p - uniform) / (q_dot_q - uniform + 1e-6)
    loss_contrib = F.relu(float(contrib_margin) - contrib).pow(2).mean()
    return loss_bag + float(lambda_contrib) * loss_contrib


def crop_view_loss(
    s_logits_local,
    t_probs,
    student_temp=0.1,
    n_crop=6,
    gamma=2.0,
    lambda_set=0.5,
    lambda_ind=1.0,
):
    """
    Set-posterior local sequence state: pool temporal crops with generalized mean (gamma),
    then compute CE against each teacher context. Optional per-crop anchoring keeps
    individual evidence windows tied to teacher targets.

    s_logits_local: [L, B, K], first n_crop rows are temporal evidence crops
    t_probs:        [T, B, K] teacher context states
    """
    eps = 1e-6
    nc = int(n_crop)
    s_crop = s_logits_local[:nc]
    st = float(student_temp)
    log_p_crop = F.log_softmax(s_crop.float() / st, dim=-1)

    q = t_probs.detach().float()  # [T, B, K]
    gm = float(gamma)

    # Compute the generalized-mean pooled set posterior in log-domain for stability:
    # log_score_k = (1/gamma) * log(mean_i exp(gamma * log p_i(k)))
    log_score = torch.logsumexp(gm * log_p_crop - math.log(nc), dim=0) / gm
    log_set = log_score - torch.logsumexp(log_score, dim=-1, keepdim=True)

    # L_set = (1/T) sum_t CE(q_t, p_set): pool crops first, then CE per teacher, then mean.
    # Do NOT average teacher targets before CE — that would mix nonlinear set pooling with q̄.
    loss_set = torch.tensor(0.0, device=s_crop.device, dtype=q.dtype)
    n_teacher = q.shape[0]
    for t in range(n_teacher):
        q_t = q[t]  # [B, K]
        loss_set = loss_set + -(q_t * log_set).sum(dim=-1).mean()
    loss_set = loss_set / n_teacher

    # Per-crop anchor: same rule — CE(q_t, p_i) per (crop, teacher), then average.
    loss_ind = torch.tensor(0.0, device=s_crop.device, dtype=q.dtype)
    n_terms = nc * n_teacher
    for i in range(nc):
        log_p_i = log_p_crop[i]  # [B, K]
        for t in range(n_teacher):
            loss_ind = loss_ind + -(q[t] * log_p_i).sum(dim=-1).mean()
    loss_ind = loss_ind / n_terms

    ls = float(lambda_set)
    li = float(lambda_ind)
    denom = max(ls + li, eps)
    return (ls * loss_set + li * loss_ind) / denom


class PatchStateLoss(nn.Module):
    def __init__(self, patch_out_dim, student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.sinkhorn = BalancedCategoricalAssignment()

    def forward_fft_bins(self, student_logits_flat, teacher_soft_flat):
        """
        Frequency-bin state loss: same CE as forward_masked per row; each row is
        (view, sample, freq_bin), and teacher_soft_flat is a balanced assignment.
        """
        log_p = F.log_softmax(student_logits_flat.float() / self.student_temp, dim=-1)
        per_row = -(teacher_soft_flat.float() * log_p).sum(dim=-1)
        return per_row.mean()

    def forward_masked(
        self,
        student_patch_tokens_masked,
        teacher_patch_tokens_masked,
        student_masks_flat,
        n_masked_patches=None,
        masks_weight=None,
        ibot_denom_rows=None,
    ):
        t = teacher_patch_tokens_masked
        s = student_patch_tokens_masked
        
        # compute patch-state cross-entropy
        loss = torch.sum(t.float() * F.log_softmax(s.float() / self.student_temp, dim=-1), dim=-1)
        
        if masks_weight is None:
            if student_masks_flat is not None:
                # compute weights dynamically
                masks_weight = (
                    (1 / student_masks_flat.sum(-1).clamp(min=1.0))
                    .unsqueeze(-1)
                    .expand_as(student_masks_flat)[student_masks_flat]
                )
            else:
                masks_weight = 1.0
                
        if n_masked_patches is not None:
            loss = loss[:n_masked_patches]
            
        loss = loss * masks_weight

        # Denominator:
        # - If TED passes ibot_denom_rows: count (student window, batch) rows that still contribute >=1 masked patch after reliable-patch filtering,
        #   (global_student_view, batch) rows, avoiding numerator on reliable patches only while denominator uses unfiltered mask rows
        #   which keeps the patch-state scalar from being abnormally low.
        # - else: number of global rows with at least one masked patch.
        if ibot_denom_rows is not None:
            denom = max(float(ibot_denom_rows), 1.0)
        elif student_masks_flat is not None:
            n_rows_active = (student_masks_flat.sum(dim=-1) > 0).sum()
            denom = n_rows_active.float().clamp(min=1.0)
        else:
            denom = 1.0

        return -loss.sum() / denom


class FeatureSpreadLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
        self.pdist = nn.PairwiseDistance(2, eps=epsilon)

    def forward(self, x):
        if x.shape[0] < 2: return torch.tensor(0.0, device=x.device)
        with torch.autocast("cuda", enabled=False):
            x = F.normalize(x, eps=self.epsilon, p=2, dim=-1)
            # dot products to find nearest neighbors
            dots = torch.mm(x, x.t())
            dots.view(-1)[:: (x.shape[0] + 1)].fill_(-1) # exclude self
            _, indices = torch.max(dots, dim=1)
            distances = self.pdist(x, x[indices])
            loss = -torch.log(distances + self.epsilon).mean()
        return loss


class FrequencyDomainPatchAlignment(nn.Module):
    """
    Frequency-domain patch alignment (light constraint):
    1) rFFT along patch sequence dim;
    2) remove DC;
    3) log1p(|·|)；
    4) multiply normalized frequency coords;
    5) L2-normalize freq tokens, Gram MSE vs teacher.

    gram_mse_f_ref_bins：Multiply Gram MSE by min(1, (F_act/F_ref)^2) for mixed_batch variable length
    gradient scale related to mean over F^2 elements; F_ref is no-DC freq count at full seq_len.
    scale=1 at full length; shorter windows slightly down-weighted; no extra boost beyond F_ref. None=no scaling.
    """

    def __init__(
        self,
        alpha=1.0,
        eps=1e-6,
        gram_mse_f_ref_bins: int | None = None,
        freq_keep_ratio: float = 1.0,
        freq_min_bins: int = 1,
        freq_max_bins: int = 0,
    ):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.gram_mse_f_ref_bins = gram_mse_f_ref_bins
        self.freq_keep_ratio = float(freq_keep_ratio)
        self.freq_min_bins = int(freq_min_bins)
        self.freq_max_bins = int(freq_max_bins)

    def _build_freq_tokens(self, patches):
        """
        patches: [B, N, D]，N N is number of patches.
        returns [B, F_no_dc, D] or None if too short

        rFFT keeps patches dtype (e.g. FP16/BF16 under AMP), avoiding full [B,N,D].float();
        magnitude to FP32 before the frequency-domain Gram calculation.
        """
        x_fft = torch.fft.rfft(patches, dim=1, norm='ortho')
        x_mag = torch.log1p(torch.abs(x_fft).float())

        if x_mag.shape[1] <= 1:
            return None
        x_mag = x_mag[:, 1:, :]

        n_freq = int(x_mag.shape[1])
        if n_freq <= 1:
            freq_coord = torch.ones(
                (1, n_freq, 1),
                device=x_mag.device,
                dtype=torch.float32,
            )
        else:
            freq_coord = torch.linspace(
                0.0,
                1.0,
                steps=n_freq + 1,
                device=x_mag.device,
                dtype=torch.float32,
            )[1:].view(1, n_freq, 1)
        x_mag = x_mag * freq_coord
        x_mag = F.normalize(x_mag, p=2, dim=-1, eps=self.eps)
        return x_mag

    def _gram(self, x):
        return torch.matmul(x, x.transpose(-1, -2))

    def _resolve_keep_bins(self, f_align: int) -> int:
        if f_align <= 0:
            return 0
        ratio = float(self.freq_keep_ratio)
        if ratio <= 0:
            return 0
        if ratio >= 1.0:
            keep = int(f_align)
        else:
            keep = int(round(ratio * float(f_align)))
        keep = max(int(self.freq_min_bins), keep)
        if int(self.freq_max_bins) > 0:
            keep = min(keep, int(self.freq_max_bins))
        keep = min(keep, int(f_align))
        return max(0, keep)

    def forward(self, sPatches, tPatches):
        """
        sPatches / tPatches: [B, N, D]，teacher detached inside loss.
        """
        s_freq = self._build_freq_tokens(sPatches)
        t_freq = self._build_freq_tokens(tPatches)
        if s_freq is None or t_freq is None:
            return torch.tensor(0.0, device=sPatches.device)

        f_align = min(int(s_freq.shape[1]), int(t_freq.shape[1]))
        if f_align <= 0:
            return torch.tensor(0.0, device=sPatches.device)
        keep_bins = self._resolve_keep_bins(f_align)
        if keep_bins <= 0:
            return torch.tensor(0.0, device=sPatches.device)
        s_freq = s_freq[:, :keep_bins, :]
        t_freq = t_freq[:, :keep_bins, :].detach()

        s_gram = self._gram(s_freq)
        t_gram = self._gram(t_freq)
        mse = F.mse_loss(s_gram, t_gram)
        f_act = int(s_freq.shape[1])
        if self.gram_mse_f_ref_bins is not None and int(self.gram_mse_f_ref_bins) > 0:
            f_ref = float(max(1, int(self.gram_mse_f_ref_bins)))
            # counteract gradient scale drift from mean over F^2; scale=1 at full length.
            # no boost beyond F_ref (avoid favoring longer windows); shorter windows scale<1 slightly reduce gradient share.
            scale = (float(f_act) / f_ref) ** 2
            scale = min(1.0, scale)
            mse = mse * scale
        return self.alpha * mse


def fft_gram_align_masked_patch_rows(fft_mod, s_pre, t_pre, row_ids, patch_idx):
    """
    Frequency-domain alignment on masked patches ordered in time within each
    (student window x batch) row; skip rows with <2 masked patches.
    """
    if (
        s_pre is None
        or t_pre is None
        or row_ids is None
        or patch_idx is None
        or s_pre.shape[0] == 0
    ):
        z = torch.tensor(0.0)
        if s_pre is not None and torch.is_tensor(s_pre):
            z = z.to(device=s_pre.device, dtype=s_pre.dtype)
        return z

    device = s_pre.device
    acc = None
    n_terms = 0
    for r in torch.unique(row_ids):
        m = row_ids == r
        idx = torch.nonzero(m, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        ord_ = torch.argsort(patch_idx[idx])
        sel = idx[ord_]
        s_row = s_pre[sel].unsqueeze(0)
        t_row = t_pre[sel].unsqueeze(0)
        if s_row.shape[1] < 2:
            continue
        term = fft_mod(s_row, t_row)
        acc = term if acc is None else acc + term
        n_terms += 1
    if n_terms == 0 or acc is None:
        return torch.zeros((), device=device, dtype=s_pre.dtype)
    return acc / float(n_terms)


def temporal_neighbor_loss(z_patch):
    z1 = z_patch[:, :-1]
    z2 = z_patch[:, 1:]
    return 1.0 - F.cosine_similarity(z1, z2, dim=-1).mean()


class TEDCriterion(nn.Module):
    """
    TED self-distillation criterion.
    Combines sequence-state inference, patch-state supervision, feature-spread
    regularization and frequency-domain patch alignment while keeping DDP ranks synchronized.
    """
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        
        _st = float(getattr(args, "student_temp", 0.1))
        self.dino_loss_fn = SequenceStateLoss(
            out_dim=args.dino_head_n_prototypes, student_temp=_st
        ).to(device)
        self.ibot_loss_fn = PatchStateLoss(
            patch_out_dim=args.ibot_head_n_prototypes, student_temp=_st
        ).to(device)
        f_ref_cfg = int(getattr(args, "fft_align_gram_mse_f_ref_bins", 0))
        if f_ref_cfg < 0:
            gram_f_ref = None
        elif f_ref_cfg == 0:
            pl, st = int(args.patch_len), int(args.stride)
            sq = int(getattr(args, "seq_len", 732))
            n_p = int(math.ceil((sq - pl + st) / st))
            # matches _build_freq_tokens: rfft len n_p//2+1, no DC -> n_p//2
            gram_f_ref = max(1, n_p // 2)
        else:
            gram_f_ref = max(1, f_ref_cfg)
        self.fft_gram_align_fn = FrequencyDomainPatchAlignment(
            alpha=1.0,
            gram_mse_f_ref_bins=gram_f_ref,
            freq_keep_ratio=float(getattr(args, "fft_align_freq_keep_ratio", 1.0)),
            freq_min_bins=int(getattr(args, "fft_align_freq_min_bins", 1)),
            freq_max_bins=int(getattr(args, "fft_align_freq_max_bins", 0)),
        ).to(device)
        self.koleo_loss_fn = FeatureSpreadLoss().to(device)

    def forward(self, outputs):
        # unpack outputs
        valid_idx = outputs.get('valid_sample_indices', [])
        teacher_temp = outputs.get('teacher_temp', self.args.teacher_temp)
        lambda_weights = outputs.get('lambda_weights', {})
        
        # weights
        l_cls = lambda_weights.get('lambda_cls_proto', self.args.lambda_cls_proto)
        l_patch = lambda_weights.get('lambda_patch_proto', self.args.lambda_patch_proto)
        l_fft_align = lambda_weights.get('lambda_fft_align', getattr(self.args, 'lambda_fft_align', 0.05))
        l_koleo = lambda_weights.get('lambda_koleo', self.args.lambda_koleo)
        l_temp = lambda_weights.get('lambda_temporal', self.args.lambda_temporal)
        l_cls_cons = lambda_weights.get('lambda_cls_cons', getattr(self.args, 'lambda_cls_cons', 0.05))

        # ----------------------------------------------------------------
        # 1. Sequence-state loss - includes deadlock-avoidance logic
        # ----------------------------------------------------------------
        loss_cls = torch.tensor(0.0, device=self.device)
        cls_global_overlap_ratio_log = None
        cls_global_cross_weight_log = None
        cls_local_cross_weight_log = None
        cls_global_ce_log = None
        cls_short_cond_ce_log = None
        cls_short_crop_cond_ce_log = None
        cls_short_random_cond_ce_log = None
        cls_short_anchor_cond_ce_log = None
        if l_cls > 0:
            cls_data = outputs.get('cls_data', {})
            s_logits_global = cls_data.get('s_logits_global_valid') # [N, B_valid, K]
            s_logits_local = cls_data.get('s_logits_local_valid')
            t_logits_global = cls_data.get('t_logits_global_valid') # [N, B_valid, K]
            
            # whether this rank has valid data
            has_data = (s_logits_global is not None and len(valid_idx) > 0)
            
            # --- prepare balanced-assignment input ---
            if has_data:
                # normal case: flatten for balanced assignment
                # Convert to float32 only; no extra NaN checks on this path.
                t_in = t_logits_global.flatten(0, 1).detach().float() # [Total_Crops, K]
            else:
                # edge case: dummy input [1, K]
                # zeros are sufficient; assignment runs and the result is discarded
                dummy_k = self.args.dino_head_n_prototypes
                t_in = torch.zeros((1, dummy_k), device=self.device, dtype=torch.float32)

            # --- run synchronized balanced assignment ---
            # all ranks must run this step regardless of data
            t_out = self.dino_loss_fn.sinkhorn.forward(t_in, teacher_temp)

            # --- compute loss (only when data present) ---
            if has_data:
                # restore shape: [N_crops, B_valid, K]
                t_probs = t_out.unflatten(0, t_logits_global.shape[:2])
                
                # Global context sequence-state loss
                global_loss_mode = getattr(self.args, "cls_global_loss_mode", "dino")
                if cls_data.get("cls_loss_mode", None) == "evidence_gap":
                    if bool(cls_data.get("evidence_gap_pairwise", False)):
                        teacher_row_indices = cls_data.get(
                            "evidence_gap_teacher_row_indices", None
                        )
                        if teacher_row_indices is not None:
                            teacher_row_indices = torch.as_tensor(
                                teacher_row_indices,
                                device=t_probs.device,
                                dtype=torch.long,
                            )
                            if int(teacher_row_indices.numel()) != int(s_logits_global.shape[0]):
                                raise ValueError(
                                    "evidence_gap_teacher_row_indices length must match "
                                    f"student rows, got {int(teacher_row_indices.numel())} "
                                    f"and {int(s_logits_global.shape[0])}"
                                )
                            t_probs_pair = t_probs.index_select(0, teacher_row_indices)
                        elif tuple(s_logits_global.shape) == tuple(t_probs.shape):
                            t_probs_pair = t_probs
                        else:
                            raise ValueError(
                                "evidence_gap_pairwise requires matched student/teacher "
                                "rows or evidence_gap_teacher_row_indices, got "
                                f"{tuple(s_logits_global.shape)} and {tuple(t_probs.shape)}"
                            )
                        log_p = F.log_softmax(
                            s_logits_global.float() / self.dino_loss_fn.student_temp,
                            dim=-1,
                        )
                        ce_rows = -(t_probs_pair.detach().float() * log_p).sum(dim=-1)
                        n_global_cls_rows = int(
                            cls_data.get("evidence_gap_n_global_cls_rows", 1) or 1
                        )
                        n_global_cls_rows = max(0, min(n_global_cls_rows, int(ce_rows.shape[0])))
                        drop_global_cls = bool(
                            cls_data.get("evidence_gap_drop_global_cls", False)
                        ) or bool(
                            int(getattr(self.args, "evidence_gap_drop_global_cls", 0) or 0)
                        )
                        if drop_global_cls and n_global_cls_rows > 0:
                            if int(ce_rows.shape[0]) > n_global_cls_rows:
                                loss_g = ce_rows[n_global_cls_rows:].mean()
                            else:
                                # Only global rows present: no short CLS to supervise.
                                loss_g = (ce_rows * 0).sum()
                        else:
                            loss_g = ce_rows.mean()
                        if ce_rows.shape[0] > 0 and n_global_cls_rows > 0:
                            cls_global_ce_log = float(
                                ce_rows[:n_global_cls_rows].detach().mean().item()
                            )
                        if ce_rows.shape[0] > n_global_cls_rows:
                            n_short_orig = cls_data.get("evidence_gap_n_short_original", None)
                            n_short_anchor = int(
                                cls_data.get("evidence_gap_n_short_anchor", 0) or 0
                            )
                            short_ce_all = ce_rows[n_global_cls_rows:]
                            if n_short_orig is not None:
                                n_short_orig = int(n_short_orig)
                                short_ce_orig = short_ce_all[:n_short_orig]
                                short_ce = short_ce_orig
                                if n_short_anchor > 0 and short_ce_all.shape[0] > n_short_orig:
                                    short_ce_anchor = short_ce_all[n_short_orig:]
                                    cls_short_anchor_cond_ce_log = float(
                                        short_ce_anchor.detach().mean().item()
                                    )
                            else:
                                short_ce = short_ce_all
                            cls_short_cond_ce_log = float(short_ce.detach().mean().item())
                            short_view_types = cls_data.get("condition_short_view_types", None)
                            if short_view_types is not None:
                                short_view_types = torch.as_tensor(
                                    short_view_types,
                                    device=ce_rows.device,
                                    dtype=torch.long,
                                ).view(-1)
                                if int(short_view_types.numel()) == int(short_ce.shape[0]):
                                    crop_mask = short_view_types == 0
                                    random_mask = short_view_types == 1
                                    if bool(crop_mask.any()):
                                        cls_short_crop_cond_ce_log = float(
                                            short_ce[crop_mask].detach().mean().item()
                                        )
                                    if bool(random_mask.any()):
                                        cls_short_random_cond_ce_log = float(
                                            short_ce[random_mask].detach().mean().item()
                                        )
                    else:
                        loss_g = self.dino_loss_fn(
                            s_logits_global, t_probs, ignore_diagonal=False
                        )
                elif (
                    global_loss_mode == "overlap_compat"
                    and int(s_logits_global.shape[0]) == 2
                    and int(t_probs.shape[0]) == 2
                ):
                    overlap_ratio = cls_data.get("cls_global_overlap_ratio", 1.0)
                    if t.is_tensor(overlap_ratio):
                        overlap_ratio = float(overlap_ratio.detach().mean().item())
                    else:
                        overlap_ratio = float(overlap_ratio)

                    min_overlap = float(
                        getattr(self.args, "cls_global_compat_min_overlap", -1.0)
                    )
                    if min_overlap < 0.0:
                        min_overlap = float(
                            getattr(self.args, "global_shift_min_overlap_ratio", 0.6)
                        )
                    min_overlap = max(0.0, min(1.0, min_overlap))

                    cross_floor = float(
                        getattr(self.args, "cls_global_compat_cross_floor", 0.25)
                    )
                    cross_floor = max(0.0, min(1.0, cross_floor))
                    self_weight = max(
                        0.0,
                        float(getattr(self.args, "cls_global_compat_self_weight", 1.0)),
                    )

                    if min_overlap >= 1.0:
                        overlap_alpha = 1.0 if overlap_ratio >= 1.0 else 0.0
                    else:
                        overlap_alpha = (overlap_ratio - min_overlap) / (1.0 - min_overlap)
                        overlap_alpha = max(0.0, min(1.0, overlap_alpha))
                    cross_weight = cross_floor + (1.0 - cross_floor) * overlap_alpha
                    cls_global_overlap_ratio_log = float(overlap_ratio)
                    cls_global_cross_weight_log = float(cross_weight)

                    pair_weights = s_logits_global.new_tensor(
                        [[self_weight, cross_weight], [cross_weight, self_weight]]
                    )
                    loss_g = self.dino_loss_fn.weighted_forward(
                        s_logits_global, t_probs, pair_weights
                    )
                else:
                    loss_g = self.dino_loss_fn(
                        s_logits_global, t_probs, ignore_diagonal=True
                    )
                
                # Local-Global Loss
                loss_l = torch.tensor(0.0, device=self.device)
                if s_logits_local is not None and s_logits_local.shape[0] > 0:
                    st_temp = self.dino_loss_fn.student_temp
                    lc = getattr(self.args, "lambda_cls_local_contrib", 0.0)
                    cm = getattr(self.args, "cls_local_contrib_margin", 0.0)
                    n_crop_cd = cls_data.get("cls_local_n_crop_views")
                    n_rand_cd = cls_data.get("cls_local_n_random_views")
                    crop_parent_ids = cls_data.get("cls_local_crop_parent_ids")
                    random_parent_ids = cls_data.get("cls_local_random_parent_ids")
                    crop_cross_overlaps = cls_data.get("cls_local_crop_cross_overlaps")
                    random_cross_overlaps = cls_data.get("cls_local_random_cross_overlaps")
                    L_loc = int(s_logits_local.shape[0])
                    local_cross_beta = max(
                        0.0,
                        float(getattr(self.args, "cls_local_cross_teacher_beta", 0.0)),
                    )
                    local_cross_normalize = bool(
                        int(getattr(self.args, "cls_local_cross_teacher_normalize", 1))
                    )
                    if local_cross_beta > 0.0:
                        _cross_log_terms = []
                        for _ovs in (crop_cross_overlaps, random_cross_overlaps):
                            if _ovs is None:
                                continue
                            if torch.is_tensor(_ovs):
                                _cross_log_terms.append(
                                    _ovs.to(device=s_logits_local.device, dtype=torch.float32).view(-1)
                                )
                            else:
                                _cross_log_terms.append(
                                    torch.as_tensor(
                                        _ovs,
                                        device=s_logits_local.device,
                                        dtype=torch.float32,
                                    ).view(-1)
                                )
                        if len(_cross_log_terms) > 0:
                            cls_local_cross_weight_log = float(
                                (
                                    local_cross_beta
                                    * torch.cat(_cross_log_terms).mean().clamp(0.0, 1.0)
                                )
                                .detach()
                                .item()
                            )

                    _loc_mode = getattr(
                        self.args, "cls_local_loss_mode", "per_view"
                    )
                    crop_gamma = float(
                        getattr(self.args, "cls_local_crop_gamma", 2.0)
                    )
                    crop_lambda_set = float(
                        getattr(self.args, "cls_local_crop_lambda_set", 0.5)
                    )
                    crop_lambda_ind = float(
                        getattr(self.args, "cls_local_crop_lambda_ind", 1.0)
                    )

                    def _per_view_local_loss(s_local, t_local=None):
                        if t_local is None:
                            t_local = t_probs
                        return self.dino_loss_fn(
                            s_local, t_local, ignore_diagonal=False
                        )

                    def _crop_loss_single_target(s_crop_group, t_local):
                        if _loc_mode == "crop_set":
                            return crop_view_loss(
                                s_crop_group,
                                t_local,
                                student_temp=st_temp,
                                n_crop=int(s_crop_group.shape[0]),
                                gamma=crop_gamma,
                                lambda_set=crop_lambda_set,
                                lambda_ind=crop_lambda_ind,
                            )
                        if _loc_mode == "bag":
                            return cls_local_bag_loss(
                                s_crop_group,
                                t_local,
                                st_temp,
                                lambda_contrib=lc,
                                contrib_margin=cm,
                            )
                        return _per_view_local_loss(s_crop_group, t_local=t_local)

                    def _parent_aware_local_loss(
                        s_local, parent_ids_local, cross_overlaps_local=None
                    ):
                        Lp = int(s_local.shape[0])
                        if (
                            parent_ids_local is not None
                            and len(parent_ids_local) == Lp
                            and t_probs.shape[0] > 0
                        ):
                            parent_ids_t = torch.as_tensor(
                                parent_ids_local, device=s_local.device, dtype=torch.long
                            )
                            cross_overlap_t = None
                            if cross_overlaps_local is not None:
                                if torch.is_tensor(cross_overlaps_local):
                                    cross_overlap_t = cross_overlaps_local.to(
                                        device=s_local.device, dtype=torch.float32
                                    ).view(-1)
                                else:
                                    cross_overlap_t = torch.as_tensor(
                                        cross_overlaps_local,
                                        device=s_local.device,
                                        dtype=torch.float32,
                                    ).view(-1)
                                if int(cross_overlap_t.numel()) != Lp:
                                    cross_overlap_t = None
                            weighted_terms = []
                            for parent_id in sorted(set(int(x) for x in parent_ids_local)):
                                if parent_id < 0 or parent_id >= int(t_probs.shape[0]):
                                    continue
                                mask = parent_ids_t == parent_id
                                if not bool(mask.any()):
                                    continue
                                s_group = s_local[mask]
                                t_group = t_probs[parent_id:parent_id + 1]
                                group_loss = _crop_loss_single_target(s_group, t_group)
                                if (
                                    local_cross_beta > 0.0
                                    and cross_overlap_t is not None
                                    and int(t_probs.shape[0]) == 2
                                ):
                                    other_id = 1 - int(parent_id)
                                    if 0 <= other_id < int(t_probs.shape[0]):
                                        cross_weight = (
                                            cross_overlap_t[mask].mean().clamp(0.0, 1.0)
                                            * local_cross_beta
                                        )
                                        cross_loss = _crop_loss_single_target(
                                            s_group, t_probs[other_id:other_id + 1]
                                        )
                                        if local_cross_normalize:
                                            group_loss = (
                                                group_loss + cross_weight * cross_loss
                                            ) / (1.0 + cross_weight)
                                        else:
                                            group_loss = group_loss + cross_weight * cross_loss
                                weighted_terms.append(
                                    (
                                        int(s_group.shape[0]) * int(t_group.shape[0]),
                                        group_loss,
                                    )
                                )
                            if len(weighted_terms) > 0:
                                denom = sum(w for w, _ in weighted_terms)
                                return sum(w * ell for w, ell in weighted_terms) / max(denom, 1)
                        return _crop_loss_single_target(s_local, t_probs)

                    def _crop_local_loss(
                        s_local,
                        n_crop_views,
                        crop_parent_ids_local=None,
                        cross_overlaps_local=None,
                    ):
                        nc = int(n_crop_views)
                        s_crop = s_local[:nc]
                        return _parent_aware_local_loss(
                            s_crop, crop_parent_ids_local, cross_overlaps_local
                        )

                    has_crop_rand_split = (
                        n_crop_cd is not None
                        and n_rand_cd is not None
                        and int(n_crop_cd) + int(n_rand_cd) == L_loc
                    )
                    use_crop_rand_split = has_crop_rand_split and _loc_mode in (
                        "crop_set",
                        "bag",
                        "per_view",
                    )

                    if use_crop_rand_split:
                        nc = int(n_crop_cd)
                        nr = int(n_rand_cd)
                        weighted_terms = []
                        if nc > 0:
                            weighted_terms.append(
                                (
                                    nc,
                                    _crop_local_loss(
                                        s_logits_local,
                                        nc,
                                        crop_parent_ids_local=crop_parent_ids,
                                        cross_overlaps_local=crop_cross_overlaps,
                                    ),
                                )
                            )
                        if nr > 0:
                            weighted_terms.append(
                                (
                                    nr,
                                    _parent_aware_local_loss(
                                        s_logits_local[nc:],
                                        random_parent_ids,
                                        random_cross_overlaps,
                                    ),
                                )
                            )
                        denom = sum(w for w, _ in weighted_terms)
                        loss_l = (
                            sum(w * ell for w, ell in weighted_terms) / denom
                            if denom > 0
                            else torch.tensor(0.0, device=self.device)
                        )
                    elif _loc_mode == "crop_set":
                        loss_l = _crop_local_loss(
                            s_logits_local,
                            L_loc,
                            crop_parent_ids_local=crop_parent_ids,
                            cross_overlaps_local=crop_cross_overlaps,
                        )
                    elif _loc_mode == "bag":
                        loss_l = _crop_local_loss(
                            s_logits_local,
                            L_loc,
                            crop_parent_ids_local=crop_parent_ids,
                            cross_overlaps_local=crop_cross_overlaps,
                        )
                    else:
                        loss_l = _per_view_local_loss(s_logits_local)
                
                # use fixed global/local scales so varying local-window counts do not shift gradient share
                dino_global_scale = cls_data.get("dino_global_scale", None)
                dino_local_scale = cls_data.get("dino_local_scale", None)

                if dino_global_scale is None or dino_local_scale is None:
                    # fallback: old behavior if scales not provided
                    loss_cls = (loss_g + loss_l) / 2.0
                else:
                    loss_cls = dino_global_scale * loss_g + dino_local_scale * loss_l

        # ----------------------------------------------------------------
        # 2. Patch-state loss - includes deadlock-avoidance logic
        # ----------------------------------------------------------------
        loss_patch = torch.tensor(0.0, device=self.device)
        if l_patch > 0:
            patch_data = outputs.get('patch_data', {})
            s_masked = patch_data.get('s_patch_masked')  # [N_masked, K]
            t_masked = patch_data.get('t_patch_masked')  # [N_masked, K]

            has_patch_data = s_masked is not None and len(s_masked) > 0

            # --- prepare balanced-assignment input ---
            if has_patch_data:
                # normal: real teacher patch logits and this rank's masked count
                t_in = t_masked.detach().float()
                n_masked_patches_tensor = torch.tensor(
                    len(s_masked), device=self.device, dtype=torch.long
                )
            else:
                # no valid masked patches on this rank: dummy vector and count 0,
                # so global B_total comes only from ranks with data, avoiding numeric jitter
                dummy_k = self.args.ibot_head_n_prototypes
                t_in = torch.zeros((1, dummy_k), device=self.device, dtype=torch.float32)
                n_masked_patches_tensor = torch.tensor(0, device=self.device, dtype=torch.long)

            # --- run synchronized balanced assignment ---
            t_out = self.ibot_loss_fn.sinkhorn.forward(
                t_in,
                teacher_temp,
                n_masked_patches_tensor=n_masked_patches_tensor,
            )

            # --- compute loss ---
            if has_patch_data:
                cm = patch_data.get('collated_masks_global_valid', None)
                student_masks_flat = cm.flatten(0, 1) if cm is not None else None
                loss_patch = self.ibot_loss_fn.forward_masked(
                    student_patch_tokens_masked=s_masked,
                    teacher_patch_tokens_masked=t_out,
                    student_masks_flat=student_masks_flat,
                    n_masked_patches=len(s_masked),
                    masks_weight=patch_data.get('masks_weight_global_valid'),
                    ibot_denom_rows=patch_data.get('ibot_denom_rows'),
                )
                all_valid_tokens = patch_data.get('all_valid_tokens', None)
                if all_valid_tokens is not None:
                    # forward_masked already row-averaged with ibot_denom_rows.
                    # here denom_rows / sum(n_i) equals divide by mean(n_i),
                    # keep n_i/m_i missing reweight while staying near the patch-state loss scale.
                    denom_rows = patch_data.get('ibot_denom_rows', None)
                    if denom_rows is None:
                        denom_rows = 1.0
                    if torch.is_tensor(all_valid_tokens):
                        all_valid_tokens_t = all_valid_tokens.to(
                            device=loss_patch.device, dtype=loss_patch.dtype
                        ).clamp(min=1.0)
                        denom_rows_t = torch.tensor(
                            float(denom_rows), device=loss_patch.device, dtype=loss_patch.dtype
                        )
                        loss_patch = loss_patch * (denom_rows_t / all_valid_tokens_t)
                    else:
                        loss_patch = loss_patch * (float(denom_rows) / max(float(all_valid_tokens), 1.0))

        # ----------------------------------------------------------------
        # 3. Other losses (local calculation, no DDP sync processing required)
        # ----------------------------------------------------------------
        
        # Frequency-domain patch alignment: masked patches by default; --fft_align_all_patches uses full windows.
        loss_fft_align = torch.tensor(0.0, device=self.device)
        if l_fft_align > 0:
            spec_data = outputs.get('spectral_data', {})
            all_patches_mode = bool(getattr(self.args, 'fft_align_all_patches', False))
            if not all_patches_mode and l_patch > 0:
                loss_fft_align = fft_gram_align_masked_patch_rows(
                    self.fft_gram_align_fn,
                    spec_data.get('fft_masked_s_pre'),
                    spec_data.get('fft_masked_t_pre'),
                    spec_data.get('fft_masked_row_ids'),
                    spec_data.get('fft_masked_patch_idx'),
                )
            elif all_patches_mode:
                s_patch = spec_data.get('s_patch_tokens_all')
                t_patch = spec_data.get('t_patch_tokens_all')
                if s_patch is not None and t_patch is not None and len(s_patch) > 0:
                    loss_fft_align = self.fft_gram_align_fn(s_patch, t_patch)

        # Feature-spread regularization
        loss_koleo = torch.tensor(0.0, device=self.device)
        if l_koleo > 0:
            k_data = outputs.get('koleo_data', {})
            if k_data.get('s_z_cls_flat') is not None:
                loss_koleo = self.koleo_loss_fn(k_data['s_z_cls_flat'])

        # Temporal Loss
        loss_temporal = torch.tensor(0.0, device=self.device)
        if l_temp > 0:
            tm_data = outputs.get('temporal_data', {})
            if tm_data.get('s_z_patch_enc_valid') is not None:
                loss_temporal = temporal_neighbor_loss(tm_data['s_z_patch_enc_valid'])

        # Raw-imputed CLS consistency: cosine alignment of CLS vectors for raw vs imputed views of same sample
        loss_cls_cons = torch.tensor(0.0, device=self.device)
        if l_cls_cons > 0:
            ccd = outputs.get('cls_consistency_data')
            if ccd is not None and ccd.get('s_z_cls_per_view') is not None:
                per_view = ccd['s_z_cls_per_view']
                use_imp = ccd.get('use_imputator_per_view', [True, False])
                if (
                    per_view.dim() >= 2
                    and per_view.shape[0] >= 2
                    and per_view.shape[1] > 0
                    and len(use_imp) >= 2
                    and sum(use_imp) == 1
                ):
                    loss_cls_cons = (1.0 - F.cosine_similarity(per_view[0], per_view[1], dim=-1)).mean()

        # ----------------------------------------------------------------
        # aggregate
        # ----------------------------------------------------------------
        total_loss = (
            l_cls * loss_cls +
            l_patch * loss_patch +
            l_fft_align * loss_fft_align +
            l_koleo * loss_koleo +
            l_temp * loss_temporal +
            l_cls_cons * loss_cls_cons
        )

        # when all gated losses on a sub-batch are constant, weighted sum may lack requires_grad,
        # breaking scaler.backward; anchor graph with zero-multiply on student tensors.
        if not total_loss.requires_grad:
            for anchor in (
                outputs.get("z_global"),
                (outputs.get("spectral_data") or {}).get("s_patch_tokens_all"),
                (outputs.get("spectral_data") or {}).get("fft_masked_s_pre"),
                (outputs.get("koleo_data") or {}).get("s_z_cls_flat"),
            ):
                if anchor is not None and t.is_tensor(anchor) and anchor.requires_grad:
                    total_loss = total_loss + (anchor * 0).sum()
                    break

        loss_dict = {
            'cls': loss_cls.item(),
            'patch': loss_patch.item(),
            'fft_align': loss_fft_align.item(),
            'koleo': loss_koleo.item(),
            'temporal': loss_temporal.item(),
            'cls_cons': loss_cls_cons.item(),
        }
        if cls_global_overlap_ratio_log is not None:
            loss_dict['cls_global_overlap'] = cls_global_overlap_ratio_log
        if cls_global_cross_weight_log is not None:
            loss_dict['cls_global_cross_w'] = cls_global_cross_weight_log
        if cls_local_cross_weight_log is not None:
            loss_dict['cls_local_cross_w'] = cls_local_cross_weight_log
        if cls_global_ce_log is not None:
            loss_dict['cls_global_ce'] = cls_global_ce_log
        if cls_short_cond_ce_log is not None:
            loss_dict['cls_short_cond_ce'] = cls_short_cond_ce_log
        if cls_short_crop_cond_ce_log is not None:
            loss_dict['cls_short_crop_cond_ce'] = cls_short_crop_cond_ce_log
        if cls_short_random_cond_ce_log is not None:
            loss_dict['cls_short_random_cond_ce'] = cls_short_random_cond_ce_log
        if cls_short_anchor_cond_ce_log is not None:
            loss_dict['cls_short_anchor_cond_ce'] = cls_short_anchor_cond_ce_log
        if l_cls > 0:
            cls_data = outputs.get('cls_data', {})
            if cls_data.get("cls_loss_mode", None) == "evidence_gap":
                for src, dst in (
                    ("gap_cls_id", "gap_cls"),
                    ("gap_tokens", "gap_tokens"),
                    ("teacher_tokens", "teacher_tokens"),
                    ("student_tokens", "student_tokens"),
                    ("teacher_len", "teacher_len"),
                    ("student_len", "student_len"),
                    ("condition_ratio", "condition_ratio"),
                    ("condition_ratio_scaled_mean", "condition_ratio_scaled_mean"),
                    ("condition_ratio_scaled_min", "condition_ratio_scaled_min"),
                    ("condition_ratio_scaled_max", "condition_ratio_scaled_max"),
                    ("condition_relative_position_mean", "condition_relative_position_mean"),
                    ("condition_relative_position_min", "condition_relative_position_min"),
                    ("condition_relative_position_max", "condition_relative_position_max"),
                    ("condition_view_type_mean", "condition_view_type_mean"),
                    ("condition_crop_views", "condition_crop_views"),
                    ("condition_random_views", "condition_random_views"),
                    ("condition_anchor_views", "condition_anchor_views"),
                    ("evidence_gap_n_short_anchor", "evidence_gap_n_short_anchor"),
                    ("condition_direction_raw_norm_mean", "condition_direction_raw_norm_mean"),
                    ("condition_direction_step_norm_mean", "condition_direction_step_norm_mean"),
                    ("condition_direction_step_to_z_norm_mean", "condition_direction_step_to_z_norm_mean"),
                    ("condition_gate_mean", "condition_gate_mean"),
                    ("condition_gate_abs_mean", "condition_gate_abs_mean"),
                    ("condition_gate_raw_std", "condition_gate_raw_std"),
                    ("condition_gate_delta_norm_mean", "condition_gate_delta_norm_mean"),
                    ("condition_gate_delta_to_z_norm_mean", "condition_gate_delta_to_z_norm_mean"),
                ):
                    val = cls_data.get(src, None)
                    if val is not None:
                        loss_dict[dst] = float(val)
        
        return total_loss, loss_dict
