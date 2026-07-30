"""Load TED / MSM / NTP encoders from the local HF-style zoo for downstream use."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from eval.checkpoint_meta import (
    infer_model_family,
    infer_model_flags,
    infer_n_storage_tokens,
    merge_flags_with_config,
)
from eval.encode import build_model_args, load_checkpoint_state_dict
from models import msm, ntp, ted
from utils.pretrained import resolve_pretrained


def from_pretrained(
    name_or_path: str,
    device: str | torch.device | None = None,
    *,
    eval_mode: bool = True,
) -> tuple[torch.nn.Module, dict, str]:
    """Load a zoo checkpoint for downstream encoding.

    Example::

        model, config, family = from_pretrained("ted", device="cuda:0")
        # family in {"TED", "MSM", "NTP"}

    Returns ``(model, config, family)``.
    """
    weight, config, model_dir = resolve_pretrained(name_or_path)
    state = load_checkpoint_state_dict(Path(weight))
    model_id = str(config.get("model_id") or model_dir.name)
    family = infer_model_family(str(config.get("model_type") or ""), model_id)
    flags = merge_flags_with_config(infer_model_flags(model_id, state), config)

    d_model = int(config.get("d_model", 768))
    n_heads = int(config.get("n_heads", 12))
    d_ff = int(config.get("d_ff", 3072))
    e_layers = int(config.get("e_layers", 12))
    seq_len = int(config.get("seq_len", 732))

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if family == "TED":
        ns = build_model_args(
            SimpleNamespace(
                seq_len=seq_len,
                patch_len=3,
                stride=3,
                enc_in=7,
                c_out=7,
                d_model=d_model,
                n_heads=n_heads,
                e_layers=e_layers,
                d_layers=2,
                d_ff=d_ff,
                n_storage_tokens=flags["n_storage_tokens"],
                n_cls_tokens=1,
                evidence_gap_distill=flags["evidence_gap_distill"],
                evidence_gap_condition=flags["evidence_gap_condition"],
                evidence_gap_condition_readout=flags["evidence_gap_condition_readout"],
                evidence_gap_condition_alpha=flags["evidence_gap_condition_alpha"],
                evidence_gap_version=flags["evidence_gap_version"],
                evidence_gap_condition_view_embed_dim=8,
                evidence_gap_condition_scalar_embed_dim=8,
                evidence_gap_condition_scalar_n_freqs=4,
                evidence_gap_condition_hidden_dim=0,
                evidence_gap_condition_backbone_inject=0,
                dino_head_n_prototypes=flags["dino_head_n_prototypes"],
                ibot_head_n_prototypes=flags["ibot_head_n_prototypes"],
                fft_head_n_prototypes=flags["fft_head_n_prototypes"],
                use_lon_lat_embed=flags["use_lon_lat_embed"],
                geo_dropout_p=flags["geo_dropout_p"],
                use_missing_mask_embed=flags["use_missing_mask_embed"],
                lon_lat_n_fourier_freqs=4,
                missing_mask_embed_dropout=0.0,
                local_view_patch_divisor=8,
                lambda_fft_align=flags["lambda_fft_align"],
            )
        )
        for k, v in flags.items():
            setattr(ns, k, v)
        ns.no_attn_checkpoint = True
        model = ted.Model(ns)
    elif family == "MSM":
        ns = SimpleNamespace(
            seq_len=seq_len,
            patch_len=3,
            stride=3,
            enc_in=7,
            c_out=7,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=0.1,
            drop_path=0.15,
            n_storage_tokens=infer_n_storage_tokens(model_id, state),
            use_lon_lat_embed=flags["use_lon_lat_embed"],
            geo_dropout_p=flags["geo_dropout_p"],
            use_missing_mask_embed=flags["use_missing_mask_embed"],
            missing_mask_embed_dropout=0.0,
            lon_lat_n_fourier_freqs=4,
            no_attn_checkpoint=True,
        )
        model = msm.Model(ns)
    else:
        ns = SimpleNamespace(
            seq_len=seq_len,
            patch_len=3,
            stride=3,
            enc_in=7,
            c_out=7,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=0.1,
            drop_path=0.15,
            use_lon_lat_embed=flags["use_lon_lat_embed"],
            geo_dropout_p=flags["geo_dropout_p"],
            use_missing_mask_embed=flags["use_missing_mask_embed"],
            missing_mask_embed_dropout=0.0,
            lon_lat_n_fourier_freqs=4,
            no_attn_checkpoint=True,
        )
        model = ntp.Model(ns)

    model.load_state_dict(state, strict=False)
    model.to(device)
    if eval_mode:
        model.eval()
    return model, config, family
