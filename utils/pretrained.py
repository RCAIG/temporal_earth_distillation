"""Resolve HuggingFace-style pretrained checkpoints under ``pretrained/``.

Layout (DINOv2 / Transformers convention)::

    pretrained/
      ted-hls-12b768/
        config.json
        pytorch_model.bin
      msm-hls-12b768/
      ntp-hls-12b768/

Callers may pass a short alias (``ted``), a folder name (``ted-hls-12b768``),
a directory path, or a direct ``.bin`` / ``.pth`` file.
"""
from __future__ import annotations

import json
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PRETRAINED_ROOT = PACKAGE_ROOT / "pretrained"

# Short names → directory under pretrained/
ALIASES: dict[str, str] = {
    "ted": "ted-hls-12b768",
    "ted-base": "ted-hls-12b768",
    "ted-hls-12b768": "ted-hls-12b768",
    "ted-hls-12b768-cxattnb": "ted-hls-12b768",
    "msm": "msm-hls-12b768",
    "msm-base": "msm-hls-12b768",
    "msm-hls-12b768": "msm-hls-12b768",
    "msm-hls-12b768-reg4": "msm-hls-12b768",
    "ntp": "ntp-hls-12b768",
    "ntp-base": "ntp-hls-12b768",
    "ntp-hls-12b768": "ntp-hls-12b768",
}

WEIGHT_CANDIDATES = (
    "pytorch_model.bin",
    "model.safetensors",
    "checkpoint.pth",
    "model.pth",
)


def list_pretrained() -> list[str]:
    """Return available zoo folder names (those with config.json)."""
    if not PRETRAINED_ROOT.is_dir():
        return []
    out = []
    for p in sorted(PRETRAINED_ROOT.iterdir()):
        if p.is_dir() and (p / "config.json").is_file():
            out.append(p.name)
    return out


def _load_config(dir_path: Path) -> dict:
    cfg_path = dir_path / "config.json"
    if not cfg_path.is_file():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config.json (expected object): {cfg_path}")
    return data


def _find_weight_file(dir_path: Path) -> Path:
    for name in WEIGHT_CANDIDATES:
        cand = dir_path / name
        if cand.is_file() or cand.is_symlink():
            return cand.resolve() if cand.exists() else cand
    # fallback: first checkpoint_epoch_*.pth
    epochs = sorted(dir_path.glob("checkpoint_epoch_*.pth"))
    if epochs:
        return epochs[-1]
    raise FileNotFoundError(
        f"No weight file in {dir_path} "
        f"(expected one of {WEIGHT_CANDIDATES} or checkpoint_epoch_*.pth)"
    )


def resolve_pretrained(name_or_path: str) -> tuple[Path, dict, Path]:
    """Resolve a zoo name or filesystem path to ``(weight_path, config, model_dir)``.

    ``config`` may be empty if only a raw ``.pth`` / ``.bin`` path was given.
    """
    raw = (name_or_path or "").strip()
    if not raw:
        raise ValueError("Empty pretrained / checkpoint path")

    key = raw.replace("\\", "/").rstrip("/")
    # alias / zoo id
    if key in ALIASES:
        model_dir = PRETRAINED_ROOT / ALIASES[key]
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"Pretrained '{key}' → {model_dir} not found. "
                f"Available: {list_pretrained() or '(none)'}"
            )
        return _find_weight_file(model_dir), _load_config(model_dir), model_dir

    path = Path(raw).expanduser()
    if not path.is_absolute():
        # try under package pretrained/ first, then CWD
        under_zoo = PRETRAINED_ROOT / path
        if under_zoo.exists():
            path = under_zoo
        else:
            path = path.resolve()

    if path.is_file():
        model_dir = path.parent
        cfg = _load_config(model_dir) if (model_dir / "config.json").is_file() else {}
        return path, cfg, model_dir

    if path.is_dir():
        return _find_weight_file(path), _load_config(path), path

    # last chance: basename as alias
    base = Path(key).name
    if base in ALIASES:
        return resolve_pretrained(base)

    raise FileNotFoundError(
        f"Cannot resolve pretrained checkpoint: {name_or_path!r}. "
        f"Try one of {sorted(set(ALIASES)) } or a path under {PRETRAINED_ROOT}"
    )
