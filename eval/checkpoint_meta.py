"""Shared checkpoint / model-id metadata helpers for downstream loading."""
from __future__ import annotations

import re

from utils.evidence_gap_checkpoint import infer_evidence_gap_readout


def infer_model_seq_len(model_id: str) -> int:
    mid = str(model_id)
    if (
        "seq122last" in mid
        or "yearWin122" in mid
        or re.search(r"(?:^|[_-])sl122(?:[_-]|$)", mid)
        or re.search(r"(?:^|[_-])pe122(?:[_-]|$)", mid)
    ):
        return 122
    return 732


def infer_arch_dims(model_id: str) -> tuple[int, int, int, int]:
    if "12b768d12h" in model_id or "dm768" in model_id:
        return 768, 12, 3072, 12
    if "6b128d8h" in model_id or "dm128" in model_id:
        return 128, 8, 512, 6
    return 384, 6, 1536, 12


def infer_model_family(model: str, model_id: str = "") -> str:
    m = (model or "").lower()
    mid = (model_id or "").lower()
    if m in ("msm", "patch_masked") or mid.startswith("msm") or "patch_masked" in mid:
        return "MSM"
    if m in ("ntp", "patch_ntp_ted") or "patch_ntp" in mid or mid.startswith("ntp"):
        return "NTP"
    return "TED"


def infer_n_storage_tokens(model_id: str, state_dict: dict | None = None) -> int:
    if state_dict is not None:
        for key in (
            "backbone.storage_tokens",
            "teacher.storage_tokens",
            "encoder.storage_tokens",
            "storage_tokens",
        ):
            if key in state_dict:
                return int(state_dict[key].shape[1])
    m = re.search(r"reg(\d+)", model_id, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 4


def infer_n_prototypes(state_dict: dict, default: int = 8192) -> int:
    for key in (
        "backbone.dino_head.last_layer.weight",
        "teacher.dino_head.last_layer.weight",
        "backbone.ibot_head.last_layer.weight",
    ):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    return int(default)


def infer_model_flags(model_id: str, state_dict: dict) -> dict:
    use_lon_lat = 0 if "noLonLat" in model_id else 1
    geo_dropout = 0.0 if "noLonLat" in model_id else 0.5
    use_missing_mask = 0 if "noMissingMaskEmbed" in model_id else 1
    evidence_gap_condition = int(
        "evidenceGapRatioPosGate" in model_id or "Gate-alpha" in model_id
    )
    has_evidence = any(
        ("evidence_gap" in k) or ("condition" in k and "backbone" in k)
        for k in state_dict
    ) or ("evidenceGap" in model_id)
    if "v25-ratioTimelineOffset" in model_id or "v2.5" in model_id or "-v25-" in model_id:
        version = "v2.5"
    elif "v4-scaleOffset" in model_id:
        version = "v4"
    else:
        version = "v2"
    readout = infer_evidence_gap_readout(state_dict)
    if readout is None and "cond_xattn" in model_id:
        readout = "cond_xattn_bottleneck"
    if readout is None:
        readout = "gate" if evidence_gap_condition else "adapter"
    n_proto = infer_n_prototypes(state_dict, default=8192)
    return {
        "use_lon_lat_embed": use_lon_lat,
        "geo_dropout_p": geo_dropout,
        "use_missing_mask_embed": use_missing_mask,
        "evidence_gap_distill": int(has_evidence or "evidenceGap" in model_id),
        "evidence_gap_condition": evidence_gap_condition,
        "evidence_gap_condition_readout": readout,
        "evidence_gap_version": version,
        "evidence_gap_condition_alpha": 0.1,
        "n_storage_tokens": infer_n_storage_tokens(model_id, state_dict),
        "dino_head_n_prototypes": n_proto,
        "ibot_head_n_prototypes": n_proto,
        "fft_head_n_prototypes": n_proto,
        "lambda_fft_align": 1.0 if any("fft" in k for k in state_dict) else 0.0,
    }


def merge_flags_with_config(flags: dict, config: dict | None) -> dict:
    """Override inferred flags with explicit values from ``config.json`` when present."""
    if not config:
        return flags
    out = dict(flags)
    for key in (
        "use_lon_lat_embed",
        "use_missing_mask_embed",
        "geo_dropout_p",
        "evidence_gap_distill",
        "evidence_gap_condition",
        "evidence_gap_condition_readout",
        "evidence_gap_version",
        "evidence_gap_condition_alpha",
        "n_storage_tokens",
        "dino_head_n_prototypes",
        "ibot_head_n_prototypes",
        "fft_head_n_prototypes",
        "lambda_fft_align",
    ):
        if key in config and config[key] is not None:
            val = config[key]
            if key in (
                "use_lon_lat_embed",
                "use_missing_mask_embed",
                "evidence_gap_distill",
                "evidence_gap_condition",
                "n_storage_tokens",
                "dino_head_n_prototypes",
                "ibot_head_n_prototypes",
                "fft_head_n_prototypes",
            ):
                out[key] = int(val)
            elif key in ("geo_dropout_p", "evidence_gap_condition_alpha", "lambda_fft_align"):
                out[key] = float(val)
            else:
                out[key] = val
    return out
