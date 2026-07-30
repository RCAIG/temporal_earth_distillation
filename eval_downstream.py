#!/usr/bin/env python3
"""Downstream classification eval aligned with prior TED/MSM/NTP tables.

Protocol (paper default):
  - TerraFM-style 2:1:1 split
  - spatial 1280 m grid + proximity filter
  - max_train_per_class=300 (cap300); set 0 for nocap
  - linear probe: LR grid on val BA, SGD 50 epochs, no feature scaler
  - kNN: k grid on val BA
  - MSM encode uses pool_mode=patch_avg
  - primary metric: test balanced accuracy (ba)
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.downstream_factory import PROVIDERS  # noqa: E402
from eval.downstream_geo import (  # noqa: E402
    add_spatial_split_args,
    load_classification_lonlat,
    split_description,
    split_train_val_test_211_spatial,
)
from eval.downstream_split import split_train_val_test_211  # noqa: E402
from eval.encode import (  # noqa: E402
    build_model_args,
    encode_cls_embeddings,
    filter_small_classes,
    load_checkpoint_state_dict,
)
from eval.checkpoint_meta import (  # noqa: E402
    infer_arch_dims,
    infer_model_family,
    infer_model_flags,
    infer_model_seq_len,
    infer_n_storage_tokens,
    merge_flags_with_config,
)
from eval.knn_tfm import DEFAULT_K_GRID, eval_knn_tfm  # noqa: E402
from eval.linear_probe_tfm import linear_probe_tfm_prebuilt  # noqa: E402
from models import msm, ntp, ted  # noqa: E402
from utils.pretrained import list_pretrained, resolve_pretrained  # noqa: E402

ALL5 = (
    "LCMAP_Classification",
    "GlanceTraining_Classification",
    "GlobalTree_Classification",
    "CDL_Classification",
    "CropHarvest_Classification",
)

SHORT = {
    "lcmap": "LCMAP_Classification",
    "glancetraining": "GlanceTraining_Classification",
    "glance": "GlanceTraining_Classification",
    "globaltree": "GlobalTree_Classification",
    "cdl": "CDL_Classification",
    "cropharvest": "CropHarvest_Classification",
}


def dataset_seq_stride(dataset: str, force_data_seq_len: int | None = None) -> tuple[int, int]:
    if force_data_seq_len is not None and int(force_data_seq_len) > 0:
        sl = int(force_data_seq_len)
        return sl, sl
    if dataset in ("CropHarvest_Classification", "CDL_Classification"):
        return 122, 122
    if dataset == "GlobalTree_Classification":
        return 244, 244
    return 732, 732


def _chunk_starts(time_steps: int, win: int) -> list[int]:
    t, w = int(time_steps), int(win)
    if t <= w:
        return [0]
    starts = list(range(0, t - w + 1, w))
    if starts[-1] + w < t:
        last = t - w
        if last != starts[-1]:
            starts.append(last)
    return starts


def encode_cls(
    model,
    x_all,
    tm_all,
    device,
    *,
    batch_size: int,
    model_seq_len: int | None = None,
    msm_pool_mode: str = "patch_avg",
):
    t = int(x_all.shape[1])
    canvas = int(model_seq_len) if model_seq_len is not None else t
    if t <= canvas:
        return encode_cls_embeddings(
            model, x_all, tm_all, device, batch_size, msm_pool_mode=msm_pool_mode
        )
    starts = _chunk_starts(t, canvas)
    chunks = []
    for s0 in starts:
        chunks.append(
            encode_cls_embeddings(
                model,
                x_all[:, s0 : s0 + canvas],
                tm_all[:, s0 : s0 + canvas],
                device,
                batch_size,
                msm_pool_mode=msm_pool_mode,
            )
        )
    return np.mean(np.stack(chunks, axis=0), axis=0).astype(np.float32, copy=False)


def load_task_arrays(ns: SimpleNamespace, dataset: str):
    _, loader = PROVIDERS[dataset](ns, flag=dataset, disable_ddp_split=True)
    xs, tms, ys = [], [], []
    for batch in loader:
        xs.append(batch[0].detach().cpu().numpy().astype(np.float32, copy=False))
        tms.append(batch[1].detach().cpu().numpy().astype(np.float32, copy=False))
        ys.append(batch[2].detach().cpu().numpy().reshape(-1))
    return (
        np.concatenate(xs, axis=0),
        np.concatenate(tms, axis=0),
        np.concatenate(ys, axis=0).astype(np.int64, copy=False),
    )


def parse_datasets(spec: str) -> list[str]:
    spec = (spec or "all").strip().lower()
    if spec in ("all", "*"):
        return list(ALL5)
    out = []
    for tok in re.split(r"[,\s]+", spec):
        if not tok:
            continue
        if tok in SHORT:
            out.append(SHORT[tok])
        elif tok.endswith("_classification") or tok in ALL5:
            name = tok if tok in ALL5 else tok
            # normalize
            for full in ALL5:
                if full.lower() == tok or full.lower().startswith(tok):
                    out.append(full)
                    break
            else:
                raise ValueError(f"Unknown dataset token: {tok}")
        else:
            raise ValueError(f"Unknown dataset token: {tok}")
    # unique preserve order
    seen = set()
    uniq = []
    for d in out:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Downstream classification eval (TED/MSM/NTP)")
    zoo = ", ".join(list_pretrained()) or "ted-hls-12b768, msm-hls-12b768, ntp-hls-12b768"
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "pretrained zoo id (e.g. ted / ted-hls-12b768), a pretrained/ directory, "
            f"or a .pth/.bin weight file. Available zoo: {zoo}"
        ),
    )
    p.add_argument(
        "--model",
        type=str,
        default="",
        help="TED | MSM | NTP (optional if config.json is present next to the weights)",
    )
    p.add_argument(
        "--model_id",
        type=str,
        default="",
        help="optional id for arch/flag inference (defaults from config.json / path)",
    )
    p.add_argument("--eval_mode", type=str, default="linear", choices=["linear", "knn"])
    p.add_argument(
        "--downstream_data_root",
        type=str,
        default="./dataset/downstream/classification",
    )
    p.add_argument("--root_path", type=str, default="./dataset/")
    p.add_argument("--datasets", type=str, default="all")
    p.add_argument("--encode_batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--max_train_per_class", type=int, default=300)
    p.add_argument("--min_samples_per_class", type=int, default=10)
    p.add_argument("--force_data_seq_len", type=int, default=0)
    p.add_argument("--seq_window_align", type=str, default="start", choices=["start", "end"])
    p.add_argument("--msm_pool_mode", type=str, default="patch_avg", choices=["patch_avg", "cls"])
    p.add_argument("--output_csv", type=str, default="results/downstream_eval.csv")
    p.add_argument("--device", type=str, default="cuda:0")
    add_spatial_split_args(p)
    # paper default spatial
    p.set_defaults(spatial_block_m=1280.0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    ckpt, ckpt_cfg, model_dir = resolve_pretrained(args.checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    model_id = (
        args.model_id
        or str(ckpt_cfg.get("model_id") or "")
        or model_dir.name
        or ckpt.stem
    )
    model_name = args.model or str(ckpt_cfg.get("model_type") or "")
    family = infer_model_family(model_name, model_id)
    print(f"[pretrained] dir={model_dir} weight={ckpt.name} family={family}")
    state = load_checkpoint_state_dict(ckpt)
    flags = merge_flags_with_config(infer_model_flags(model_id, state), ckpt_cfg)
    # Prefer explicit dims from config.json when present
    if all(k in ckpt_cfg for k in ("d_model", "n_heads", "d_ff", "e_layers")):
        d_model = int(ckpt_cfg["d_model"])
        n_heads = int(ckpt_cfg["n_heads"])
        d_ff = int(ckpt_cfg["d_ff"])
        e_layers = int(ckpt_cfg["e_layers"])
    else:
        d_model, n_heads, d_ff, e_layers = infer_arch_dims(model_id)
    model_seq_len = int(ckpt_cfg.get("seq_len") or infer_model_seq_len(model_id))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build model
    if family == "TED":
        ns = build_model_args(
            SimpleNamespace(
                seq_len=model_seq_len,
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
                use_pretrained_imputator=0,
            )
        )
        # merge flags not in build_model_args
        for k, v in flags.items():
            setattr(ns, k, v)
        ns.d_model, ns.n_heads, ns.d_ff, ns.e_layers = d_model, n_heads, d_ff, e_layers
        ns.seq_len = model_seq_len
        model = ted.Model(ns).to(device)
    elif family == "MSM":
        ns = SimpleNamespace(
            seq_len=model_seq_len,
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
        model = msm.Model(ns).to(device)
    else:
        ns = SimpleNamespace(
            seq_len=model_seq_len,
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
        model = ntp.Model(ns).to(device)

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] family={family} missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    datasets = parse_datasets(args.datasets)
    split_tag = split_description(args, max_train_per_class=args.max_train_per_class)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "model",
        "model_id",
        "dataset",
        "checkpoint",
        "split",
        "eval_mode",
        "n_train",
        "n_val",
        "n_test",
        "oa",
        "ba",
        "f1_macro",
        "val_ba",
        "best_lr",
        "best_k",
        "feature_scaler",
    ]
    rows = []

    for dataset in datasets:
        print(f"\n===== {dataset} =====")
        data_seq, data_stride = dataset_seq_stride(
            dataset, force_data_seq_len=args.force_data_seq_len or None
        )
        data_ns = SimpleNamespace(
            root_path=args.root_path,
            downstream_data_root=args.downstream_data_root,
            freq="rs",
            use_multi_gpu=False,
            local_rank=0,
            seq_len=data_seq,
            sampling_stride=data_stride,
            batch_size=args.encode_batch_size,
            num_workers=args.num_workers,
            seq_window_align=args.seq_window_align,
        )
        x_all, tm_all, y_raw = load_task_arrays(data_ns, dataset)
        keep, y_all = filter_small_classes(y_raw, args.min_samples_per_class)
        if keep.size == 0:
            print(f"[skip] {dataset}: no classes with >= {args.min_samples_per_class} samples")
            continue
        x_all, tm_all = x_all[keep], tm_all[keep]

        emb = encode_cls(
            model,
            x_all,
            tm_all,
            device,
            batch_size=args.encode_batch_size,
            model_seq_len=model_seq_len,
            msm_pool_mode=args.msm_pool_mode,
        )

        if args.spatial_block_m is not None and float(args.spatial_block_m) > 0:
            lon, lat = load_classification_lonlat(args.downstream_data_root, dataset)
            lon, lat = np.asarray(lon)[keep], np.asarray(lat)[keep]
            idx_train, idx_val, idx_test = split_train_val_test_211_spatial(
                y_all,
                lon,
                lat,
                seed=args.split_seed,
                max_train_per_class=args.max_train_per_class,
                block_size_m=float(args.spatial_block_m),
                spatial_mode=str(args.spatial_split_mode),
                proximity_filter=bool(args.spatial_proximity_filter),
            )
        else:
            idx_train, idx_val, idx_test = split_train_val_test_211(
                y_all,
                seed=args.split_seed,
                max_train_per_class=args.max_train_per_class,
            )

        X_tr, y_tr = emb[idx_train], y_all[idx_train]
        X_va, y_va = emb[idx_val], y_all[idx_val]
        X_te, y_te = emb[idx_test], y_all[idx_test]

        row = {
            "model": family,
            "model_id": model_id,
            "dataset": dataset,
            "checkpoint": str(ckpt),
            "split": split_tag,
            "eval_mode": args.eval_mode,
            "n_train": int(len(y_tr)),
            "n_val": int(len(y_va)),
            "n_test": int(len(y_te)),
            "oa": "",
            "ba": "",
            "f1_macro": "",
            "val_ba": "",
            "best_lr": "",
            "best_k": "",
            "feature_scaler": False,
        }

        if args.eval_mode == "linear":
            metrics = linear_probe_tfm_prebuilt(
                X_tr,
                y_tr,
                X_va,
                y_va,
                X_te,
                y_te,
                device=device,
                use_feature_scaler=False,
            )
            row.update(
                {
                    "oa": metrics["oa"],
                    "ba": metrics["ba"],
                    "f1_macro": metrics["f1_macro"],
                    "val_ba": metrics["val_ba"],
                    "best_lr": metrics["best_lr"],
                    "feature_scaler": metrics.get("feature_scaler", False),
                }
            )
        else:
            metrics = eval_knn_tfm(
                X_tr, y_tr, X_va, y_va, X_te, y_te, k_grid=DEFAULT_K_GRID
            )
            row.update(
                {
                    "oa": metrics["oa"],
                    "ba": metrics["ba"],
                    "f1_macro": metrics["f1_macro"],
                    "val_ba": metrics["val_ba"],
                    "best_k": metrics["best_k"],
                }
            )

        print(
            f"[{dataset}] ba={row['ba']:.4f} oa={row['oa']:.4f} "
            f"n={row['n_train']}/{row['n_val']}/{row['n_test']} split={split_tag}"
        )
        rows.append(row)

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
