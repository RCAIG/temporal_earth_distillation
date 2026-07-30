"""Helpers for matching evidence-gap readout type to checkpoint weights."""

from __future__ import annotations

from typing import Iterable


def infer_evidence_gap_disjoint_pred(state_dict: dict) -> int:
    """Return 1 if checkpoint uses disjoint-pred condition layout (4 view types, 5 scalars)."""
    for key in (
        "evidence_gap_condition_view_embed.weight",
        "module.evidence_gap_condition_view_embed.weight",
    ):
        weight = state_dict.get(key)
        if weight is not None:
            return int(weight.shape[0] >= 4)
    return 0


def infer_evidence_gap_readout(state_dict: dict) -> str | None:
    """Return evidence-gap condition readout type, or None if not present."""
    keys = set(state_dict.keys())
    has_direction = any(k.startswith("evidence_gap_condition_direction.") for k in keys)
    has_gate = any(k.startswith("evidence_gap_condition_z_to_basis.") for k in keys)
    has_cond_res = any(k.startswith("evidence_gap_cond_res_mlp.") for k in keys)
    has_cond_res_film = any(k.startswith("evidence_gap_cond_res_film_mlp.") for k in keys)
    has_cond_gate_b = any(k.startswith("evidence_gap_cond_b_to_basis.") for k in keys)
    has_cond_mul = any(k.startswith("evidence_gap_cond_mul_mlp.") for k in keys)
    has_cond_xattn = any(k.startswith("evidence_gap_cond_xattn.") for k in keys)
    has_cond_blend = any(k.startswith("evidence_gap_cond_blend_mlp.") for k in keys)
    has_cond_sum = any(k.startswith("evidence_gap_cond_z_mlp.") for k in keys)
    has_cond_film = any(k.startswith("evidence_gap_cond_film_mlp.") for k in keys)
    has_cond_mlp = any(k.startswith("evidence_gap_cond_mlp.") for k in keys)
    has_adapter = any(k.startswith("evidence_gap_condition_adapter.") for k in keys)
    has_film = any(k.startswith("evidence_gap_condition_mlp.") for k in keys)
    has_adaln = any(k.startswith("evidence_gap_adaln_blocks.") for k in keys)
    if has_adaln:
        raise ValueError("AdaLN evidence-gap checkpoints are no longer supported.")
    if has_direction:
        return "direction"
    if has_cond_blend:
        return "cond_blend_mlp"
    if has_cond_xattn:
        return "cond_xattn_bottleneck"
    if has_cond_mul:
        return "cond_mul_bottleneck"
    if has_cond_gate_b:
        return "cond_gate_bottleneck"
    if has_cond_res_film:
        return "cond_res_film_mlp"
    if has_cond_res:
        return "cond_res_mlp"
    if has_cond_sum:
        return "cond_sum_mlp"
    if has_cond_film:
        return "cond_film_mlp"
    if has_cond_mlp:
        return "cond_mlp"
    if has_gate:
        return "gate"
    if has_adapter:
        return "adapter"
    if has_film:
        return "film"
    return None


def evidence_gap_key_prefixes(readout: str) -> tuple[str, ...]:
    if readout == "adapter":
        return (
            "evidence_gap_condition_adapter.",
            "evidence_gap_condition_norm.",
            "evidence_gap_condition_position_embed.",
            "evidence_gap_condition_ratio_embed.",
            "evidence_gap_condition_view_embed.",
        )
    if readout == "direction":
        return (
            "evidence_gap_condition_direction.",
            "evidence_gap_condition_norm.",
            "evidence_gap_condition_position_embed.",
            "evidence_gap_condition_ratio_embed.",
            "evidence_gap_condition_view_embed.",
        )
    if readout == "gate":
        return (
            "evidence_gap_condition_z_to_basis.",
            "evidence_gap_condition_cond_to_gate.",
            "evidence_gap_condition_basis_to_z.",
            "evidence_gap_condition_norm_no_affine.",
            "evidence_gap_condition_position_embed.",
            "evidence_gap_condition_ratio_embed.",
            "evidence_gap_condition_view_embed.",
        )
    if readout == "cond_mlp":
        return (
            "evidence_gap_cond_mlp.",
            "evidence_gap_condition_position_embed.",
            "evidence_gap_condition_ratio_embed.",
            "evidence_gap_condition_view_embed.",
        )
    if readout == "cond_sum_mlp":
        return (
            "evidence_gap_cond_z_mlp.",
            "evidence_gap_cond_c_mlp.",
            "evidence_gap_condition_position_embed.",
            "evidence_gap_condition_ratio_embed.",
            "evidence_gap_condition_view_embed.",
        )
    if readout == "cond_film_mlp":
        return (
            "evidence_gap_cond_film_mlp.",
            "evidence_gap_condition_position_embed.",
            "evidence_gap_condition_ratio_embed.",
            "evidence_gap_condition_view_embed.",
        )
    if readout == "cond_res_mlp":
        return (
            "evidence_gap_cond_res_mlp.",
            "evidence_gap_condition_position_embed.",
            "evidence_gap_condition_ratio_embed.",
            "evidence_gap_condition_view_embed.",
        )
    if readout == "cond_res_film_mlp":
        return (
            "evidence_gap_cond_res_film_mlp.",
            "evidence_gap_condition_position_embed.",
            "evidence_gap_condition_ratio_embed.",
            "evidence_gap_condition_view_embed.",
        )
    if readout == "cond_gate_bottleneck":
        return (
            "evidence_gap_cond_b_to_basis.",
            "evidence_gap_cond_b_cond_to_gate.",
            "evidence_gap_cond_b_basis_to_b.",
            "evidence_gap_cond_b_norm_no_affine.",
            "evidence_gap_condition_position_embed.",
            "evidence_gap_condition_ratio_embed.",
            "evidence_gap_condition_view_embed.",
        )
    if readout == "cond_mul_bottleneck":
        return (
            "evidence_gap_cond_mul_mlp.",
            "evidence_gap_condition_position_embed.",
            "evidence_gap_condition_ratio_embed.",
            "evidence_gap_condition_view_embed.",
        )
    if readout == "cond_xattn_bottleneck":
        return (
            "evidence_gap_cond_xattn.",
            "evidence_gap_condition_position_embed.",
            "evidence_gap_condition_ratio_embed.",
            "evidence_gap_condition_view_embed.",
        )
    if readout == "cond_blend_mlp":
        return (
            "evidence_gap_cond_blend_mlp.",
            "evidence_gap_cond_blend_gate.",
            "evidence_gap_condition_position_embed.",
            "evidence_gap_condition_ratio_embed.",
            "evidence_gap_condition_view_embed.",
        )
    if readout == "film":
        return ("evidence_gap_condition_mlp.", "evidence_gap_condition_view_embed.")
    raise ValueError(f"Unknown readout: {readout!r}")


def filter_evidence_gap_keys(keys: Iterable[str]) -> list[str]:
    prefixes = (
        "evidence_gap_condition_adapter.",
        "evidence_gap_condition_direction.",
        "evidence_gap_condition_z_to_basis.",
        "evidence_gap_condition_cond_to_gate.",
        "evidence_gap_condition_basis_to_z.",
        "evidence_gap_condition_norm_no_affine.",
        "evidence_gap_condition_norm.",
        "evidence_gap_condition_position_embed.",
        "evidence_gap_condition_ratio_embed.",
        "evidence_gap_condition_mlp.",
        "evidence_gap_cond_mlp.",
        "evidence_gap_cond_z_mlp.",
        "evidence_gap_cond_c_mlp.",
        "evidence_gap_cond_film_mlp.",
        "evidence_gap_cond_res_mlp.",
        "evidence_gap_cond_res_film_mlp.",
        "evidence_gap_cond_b_to_basis.",
        "evidence_gap_cond_b_cond_to_gate.",
        "evidence_gap_cond_b_basis_to_b.",
        "evidence_gap_cond_b_norm_no_affine.",
        "evidence_gap_cond_mul_mlp.",
        "evidence_gap_cond_xattn.",
        "evidence_gap_cond_blend_mlp.",
        "evidence_gap_cond_blend_gate.",
        "evidence_gap_adaln_blocks.",
        "evidence_gap_condition_view_embed.",
    )
    return sorted(k for k in keys if k.startswith(prefixes))


def validate_evidence_gap_load(
    cli_readout: str,
    missing_keys: list[str],
    unexpected_keys: list[str],
    state_dict: dict | None = None,
) -> None:
    """Raise if CLI readout does not match checkpoint evidence-gap weights."""
    inferred = infer_evidence_gap_readout(state_dict) if state_dict is not None else None
    eg_missing = filter_evidence_gap_keys(missing_keys)
    eg_unexpected = filter_evidence_gap_keys(unexpected_keys)

    if inferred is not None and inferred != cli_readout:
        raise ValueError(
            f"evidence_gap_condition_readout={cli_readout!r} but checkpoint weights "
            f"indicate readout={inferred!r}. "
            f"Use --evidence_gap_condition_readout {inferred}."
        )

    if eg_missing or eg_unexpected:
        lines = [
            f"Evidence-gap weight mismatch for readout={cli_readout!r}:",
        ]
        if inferred is not None:
            lines.append(f"  checkpoint readout: {inferred}")
        if eg_missing:
            lines.append(f"  missing ({len(eg_missing)}): {eg_missing[:8]}{' ...' if len(eg_missing) > 8 else ''}")
        if eg_unexpected:
            lines.append(
                f"  unexpected ({len(eg_unexpected)}): {eg_unexpected[:8]}{' ...' if len(eg_unexpected) > 8 else ''}"
            )
        raise ValueError("\n".join(lines))
