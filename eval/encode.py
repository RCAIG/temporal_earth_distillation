"""Checkpoint loading and embedding encode helpers for downstream eval."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.cuda.amp import autocast

def build_model_args(cli_args: SimpleNamespace) -> SimpleNamespace:
    """Build TED model init args (aligned with historical eval checkpoint helpers)."""
    return SimpleNamespace(
        model="TED",
        seq_len=cli_args.seq_len,
        patch_len=cli_args.patch_len,
        stride=cli_args.stride,
        enc_in=cli_args.enc_in,
        c_out=cli_args.c_out,
        d_model=cli_args.d_model,
        n_heads=cli_args.n_heads,
        e_layers=cli_args.e_layers,
        d_layers=cli_args.d_layers,
        d_ff=cli_args.d_ff,
        dropout=0.1,
        n_storage_tokens=int(getattr(cli_args, "n_storage_tokens", 2)),
        n_cls_tokens=int(getattr(cli_args, "n_cls_tokens", 1)),
        evidence_gap_distill=int(getattr(cli_args, "evidence_gap_distill", 0)),
        evidence_gap_condition=int(getattr(cli_args, "evidence_gap_condition", 0)),
        evidence_gap_condition_readout=str(
            getattr(cli_args, "evidence_gap_condition_readout", "adapter")
        ),
        evidence_gap_condition_alpha=float(
            getattr(cli_args, "evidence_gap_condition_alpha", 0.1)
        ),
        evidence_gap_condition_view_embed_dim=int(
            getattr(cli_args, "evidence_gap_condition_view_embed_dim", 8)
        ),
        evidence_gap_condition_scalar_embed_dim=int(
            getattr(cli_args, "evidence_gap_condition_scalar_embed_dim", 8)
        ),
        evidence_gap_condition_scalar_n_freqs=int(
            getattr(cli_args, "evidence_gap_condition_scalar_n_freqs", 4)
        ),
        evidence_gap_condition_hidden_dim=int(
            getattr(cli_args, "evidence_gap_condition_hidden_dim", 0)
        ),
        evidence_gap_version=str(getattr(cli_args, "evidence_gap_version", "v4")),
        evidence_gap_condition_backbone_inject=int(
            getattr(cli_args, "evidence_gap_condition_backbone_inject", 0)
        ),
        dino_head_n_prototypes=int(getattr(cli_args, "dino_head_n_prototypes", 2048)),
        dino_head_hidden_dim=1536,
        dino_head_bottleneck_dim=256,
        dino_head_nlayers=3,
        ibot_head_n_prototypes=int(getattr(cli_args, "ibot_head_n_prototypes", 2048)),
        ibot_head_hidden_dim=1536,
        ibot_head_bottleneck_dim=256,
        ibot_head_nlayers=3,
        fft_head_n_prototypes=int(getattr(cli_args, "fft_head_n_prototypes", 2048)),
        fft_head_hidden_dim=1536,
        fft_head_bottleneck_dim=256,
        fft_head_nlayers=3,
        fft_freq_bins_cap=int(getattr(cli_args, "fft_freq_bins_cap", 0)),
        fft_feature_mode=str(getattr(cli_args, "fft_feature_mode", "mag")),
        use_fft_freq_pos_embed=1,
        use_fft_resolution_scale=1,
        no_attn_checkpoint=False,
        compile_backbone=False,
        use_rope=0,
        use_lon_lat_embed=int(getattr(cli_args, "use_lon_lat_embed", 1)),
        lon_lat_n_fourier_freqs=int(getattr(cli_args, "lon_lat_n_fourier_freqs", 4)),
        geo_dropout_p=float(getattr(cli_args, "geo_dropout_p", 0.5)),
        missing_mask_embed_dropout=float(getattr(cli_args, "missing_mask_embed_dropout", 0.5)),
        use_missing_mask_embed=int(getattr(cli_args, "use_missing_mask_embed", 1)),
        local_view_patch_divisor=max(1, int(getattr(cli_args, "local_view_patch_divisor", 8))),
        lambda_recon=0.0,
        lambda_fft_align=float(getattr(cli_args, "lambda_fft_align", 0.0)),
        lambda_cls_proto=1.0,
        lambda_patch_proto=1.0,
        lambda_koleo=0.1,
        lambda_temporal=0.0,
        lambda_cls_cons=0.0,
        lambda_fft_proto=0.0,
        fft_proto_warm_epochs=0,
        block_mask_ratio=0.8,
        mask_sample_probability=0.5,
        imputed_patch_weight=1.0,
        teacher_temp=0.07,
        curriculum_strategy="mixed_batch",
        curriculum_length_jitter=60,
        curriculum_jitter_probability=0.5,
        imputator_mode="full",
        imputator_segment_stride=244,
        root_path=getattr(cli_args, "root_path", "./dataset/"),
        freq="rs",
        sampling_stride=int(getattr(cli_args, "sampling_stride", getattr(cli_args, "seq_len", 732))),
        batch_size=int(getattr(cli_args, "batch_size", 64)),
        num_workers=int(getattr(cli_args, "num_workers", 0)),
        downstream_data_root=getattr(
            cli_args, "downstream_data_root", "./dataset/downstream/classification"
        ),
        use_multi_gpu=False,
        local_rank=0,
        comment="none",
        des="Exp",
        factor=1,
    )


def filter_small_classes(
    y_all: np.ndarray, min_samples_per_class: int
) -> tuple[np.ndarray, np.ndarray]:
    """Drop rare classes and remap labels to 0..C-1."""
    unique_vals, counts = np.unique(y_all, return_counts=True)
    valid_classes = unique_vals[counts >= min_samples_per_class]
    if len(valid_classes) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    keep_mask = np.isin(y_all, valid_classes)
    keep_idx = np.where(keep_mask)[0]
    remap_dict = {c: i for i, c in enumerate(valid_classes)}
    y_remap = np.array([remap_dict[v] for v in y_all[keep_idx]], dtype=np.int64)
    return keep_idx, y_remap


def load_checkpoint_state_dict(ckpt_path: Path) -> dict:
    """Load a checkpoint and return a normalized state_dict."""
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")

    normalized: dict = {}
    for key, val in checkpoint.items():
        norm_key = key
        if norm_key.startswith("module."):
            norm_key = norm_key[len("module.") :]
        norm_key = norm_key.replace("._orig_mod.", ".")
        if norm_key.startswith("_orig_mod."):
            norm_key = norm_key[len("_orig_mod.") :]
        normalized[norm_key] = val
    return normalized


def encode_cls_embeddings(
    model: torch.nn.Module,
    x_all: np.ndarray,
    tm_all: np.ndarray,
    device: torch.device,
    batch_size: int,
    imputator: torch.nn.Module | None = None,
    *,
    msm_pool_mode: str = "patch_avg",
    return_patch_avg: bool = False,
    patch_pool: str | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Batch-encode and extract CLS embeddings. If imputator is set, impute in full mode first.

    MSM defaults to ``msm_pool_mode=patch_avg``.
    If ``patch_pool`` in {avg,max} (or ``return_patch_avg=True`` → avg), also return pooled patches.
    """
    if patch_pool is None and return_patch_avg:
        patch_pool = "avg"
    if patch_pool is not None:
        patch_pool = str(patch_pool).lower().strip()
        if patch_pool not in ("avg", "max"):
            raise ValueError(f"patch_pool must be 'avg' or 'max', got {patch_pool!r}")

    model.eval()
    if imputator is not None:
        imputator.eval()
    # Detect Patch_Masked-style encode(pool_mode=...)
    encode_kwargs: dict = {"imputator": imputator}
    try:
        import inspect

        sig = inspect.signature(model.encode)
        if "pool_mode" in sig.parameters:
            encode_kwargs["pool_mode"] = msm_pool_mode
    except (TypeError, ValueError):
        pass

    emb_list: list[np.ndarray] = []
    patch_list: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_all), batch_size):
            end = min(start + batch_size, len(x_all))
            x_tensor = torch.from_numpy(x_all[start:end]).to(device=device, dtype=torch.float32)
            tm_tensor = torch.from_numpy(tm_all[start:end]).to(device=device, dtype=torch.float32)
            with autocast():
                outputs = model.encode(x_tensor, tm_tensor, **encode_kwargs)
            cls_token = outputs["cls_token"]
            if cls_token.ndim == 3:
                cls_token = cls_token.mean(dim=1)
            emb_list.append(cls_token.detach().float().cpu().numpy())
            if patch_pool is not None:
                patch_tokens = outputs.get("patch_tokens", None)
                if patch_tokens is None:
                    raise RuntimeError("encode returned no patch_tokens (needed for cls+patch)")
                if patch_tokens.ndim == 4:
                    # [B, views?, N, D] → pool views then time
                    patch_tokens = patch_tokens.mean(dim=1)
                if patch_tokens.ndim != 3:
                    raise RuntimeError(f"Unexpected patch_tokens ndim={patch_tokens.ndim}")
                if patch_pool == "avg":
                    patch_pooled = patch_tokens.mean(dim=1)
                else:
                    patch_pooled = patch_tokens.amax(dim=1)
                patch_list.append(patch_pooled.detach().float().cpu().numpy())
    cls_all = np.concatenate(emb_list, axis=0)
    if patch_pool is not None:
        return cls_all, np.concatenate(patch_list, axis=0)
    return cls_all


