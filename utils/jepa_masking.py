"""
Context/target patch masking helpers for TED latent-prediction ablations.

Isolated from the default block-biased patch-state masking path. Enable only via:
  --ibot_target_mode jepa_block
  --ibot_context_target_attn disjoint
Default training keeps using the block-biased masking helper in utils.tools
and full bidirectional attention (no call into this module).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


def random_patch_masking_jepa_blocks(
    B: int,
    mask_ratio_tuple: Tuple[float, float],
    mask_sample_probability: float,
    num_patches: int,
    device,
    n_blocks: int = 1,
    min_context_patches: int = 1,
):
    """
    I-JEPA-like contiguous target block(s) within a 1D patch timeline.

    Returns the same triple as random_patch_masking_dinov3_style:
      masks [B, N] bool (True = target / masked),
      mask_indices_list,
      masks_weight
    """
    N = int(num_patches)
    if N <= 0:
        raise ValueError(f"num_patches must be > 0, got {N}")

    rmin, rmax = float(mask_ratio_tuple[0]), float(mask_ratio_tuple[1])
    if rmax < rmin:
        rmin, rmax = rmax, rmin
    rmin = max(0.0, min(1.0, rmin))
    rmax = max(0.0, min(1.0, rmax))

    p = max(0.0, min(1.0, float(mask_sample_probability)))
    n_samples_masked = max(0, min(B, int(B * p + 0.5)))
    n_blocks = max(1, int(n_blocks))
    min_context = max(1, int(min_context_patches))
    max_target = max(0, N - min_context)

    masks_tensor = torch.zeros(B, N, dtype=torch.bool, device=device)
    if n_samples_masked == 0 or max_target <= 0:
        mask_indices_list = masks_tensor.flatten().nonzero().flatten()
        masks_weight = (
            1 / masks_tensor.sum(-1).clamp(min=1.0)
        ).unsqueeze(-1).expand_as(masks_tensor)[masks_tensor]
        return masks_tensor, mask_indices_list, masks_weight

    ratios = torch.linspace(rmin, rmax, n_samples_masked, device=device)
    for i in range(n_samples_masked):
        n_masked = int((N * float(ratios[i].item())) + 0.5)
        n_masked = max(1, min(max_target, n_masked))
        mask = torch.zeros(N, dtype=torch.bool, device=device)

        # Split target budget across contiguous blocks (last block takes remainder).
        base = n_masked // n_blocks
        rem = n_masked - base * n_blocks
        block_lens = [base + (1 if b < rem else 0) for b in range(n_blocks)]
        block_lens = [L for L in block_lens if L > 0]
        # Place blocks without overlap when possible; allow overlap fallback if packed.
        occupied = torch.zeros(N, dtype=torch.bool, device=device)
        for L in block_lens:
            L = min(L, N)
            max_start = N - L
            if max_start < 0:
                continue
            free = (~occupied).float()
            csum = torch.cat(
                [torch.zeros(1, device=device, dtype=free.dtype), free.cumsum(0)], dim=0
            )
            cover = csum[L:] - csum[:-L]  # [max_start + 1]
            best = float(cover.max().item())
            cands = torch.arange(0, max_start + 1, device=device)[cover >= best - 1e-6]
            start = int(cands[torch.randint(0, cands.numel(), (1,), device=device)].item())
            mask[start : start + L] = True
            occupied[start : start + L] = True

        # Enforce exact n_masked if overlap/underfill.
        cur = int(mask.sum().item())
        if cur < n_masked:
            avail = (~mask).nonzero().flatten()
            take = min(n_masked - cur, int(avail.numel()))
            if take > 0:
                sel = avail[torch.randperm(avail.numel(), device=device)[:take]]
                mask[sel] = True
        elif cur > n_masked:
            on = mask.nonzero().flatten()
            drop = cur - n_masked
            sel = on[torch.randperm(on.numel(), device=device)[:drop]]
            mask[sel] = False

        # Final context floor.
        if int((~mask).sum().item()) < min_context:
            on = mask.nonzero().flatten()
            need = min_context - int((~mask).sum().item())
            if on.numel() > 0 and need > 0:
                sel = on[torch.randperm(on.numel(), device=device)[:need]]
                mask[sel] = False

        masks_tensor[i] = mask

    indices = torch.randperm(B, device=device)
    collated_masks = masks_tensor[indices]
    mask_indices_list = collated_masks.flatten().nonzero().flatten()
    masks_weight = (
        1 / collated_masks.sum(-1).clamp(min=1.0)
    ).unsqueeze(-1).expand_as(collated_masks)[collated_masks]
    return collated_masks, mask_indices_list, masks_weight


def build_jepa_disjoint_attn_mask(
    mask_map: torch.Tensor,
    n_prefix: int,
    num_heads: int,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Additive attention mask [B * num_heads, L, L] for nn.MultiheadAttention.

    Layout: [prefix (CLS/storage) | patches].
    Rule: prefix + context queries cannot attend to target (masked) keys.
    Target queries may attend to everything (predict from context).
    """
    if mask_map.dim() != 2:
        raise ValueError(f"mask_map must be [B, N], got {tuple(mask_map.shape)}")
    B, N = mask_map.shape
    n_prefix = max(0, int(n_prefix))
    L = n_prefix + N
    device = mask_map.device
    if dtype is None:
        dtype = torch.float32

    is_target = torch.zeros(B, L, dtype=torch.bool, device=device)
    is_target[:, n_prefix:] = mask_map.bool()
    is_visible = ~is_target  # prefix + context patches

    # Block visible->target.
    blocked = is_visible.unsqueeze(2) & is_target.unsqueeze(1)  # [B, L, L]
    attn = torch.zeros(B, L, L, device=device, dtype=dtype)
    neg = torch.tensor(float("-inf"), device=device, dtype=dtype)
    attn = torch.where(blocked, neg, attn)

    # MHA expects (B * num_heads, L, L) for per-sample masks.
    H = max(1, int(num_heads))
    return attn.unsqueeze(1).expand(-1, H, -1, -1).reshape(B * H, L, L)
