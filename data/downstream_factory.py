"""Downstream classification data providers."""
from __future__ import annotations

import os
from typing import Any

from torch.utils.data import DataLoader

from data.downstream_datasets import (
    Dataset_CDL_Classification,
    Dataset_CropHarvest_Classification,
    Dataset_GlanceTraining_Classification,
    Dataset_GlobalTree_Classification,
    Dataset_LCMAP_Classification,
)

# Shareable package layout:
#   dataset/downstream/classification/
#     hls_composite_nc/*.nc
#     *_hls_classification.npz  (lon/lat helpers for spatial splits)
DOWNSTREAM_SUBPATHS = {
    "LCMAP_Classification": ("hls_composite_nc/", ""),
    "GlanceTraining_Classification": ("hls_composite_nc/", ""),
    "GlobalTree_Classification": ("hls_composite_nc/", ""),
    "CDL_Classification": ("hls_composite_nc/", ""),
    "CropHarvest_Classification": ("hls_composite_nc/", ""),
}

_DATASETS = {
    "LCMAP_Classification": Dataset_LCMAP_Classification,
    "GlanceTraining_Classification": Dataset_GlanceTraining_Classification,
    "GlobalTree_Classification": Dataset_GlobalTree_Classification,
    "CDL_Classification": Dataset_CDL_Classification,
    "CropHarvest_Classification": Dataset_CropHarvest_Classification,
}


def probe_seq_window_align(args) -> str:
    align = str(getattr(args, "seq_window_align", "start") or "start").lower().strip()
    if align == "random_year":
        return "end"
    return align


def resolve_downstream_root(args, task_name: str) -> str:
    """Return directory containing processed NC files for a classification task."""
    base = getattr(
        args,
        "downstream_data_root",
        "./dataset/downstream/classification",
    )
    base = os.path.abspath(base)
    # If user pointed at parent `dataset/downstream`, descend into classification/
    if os.path.basename(base.rstrip("/")) == "downstream":
        cand = os.path.join(base, "classification")
        if os.path.isdir(cand):
            base = cand
    rel_candidates = DOWNSTREAM_SUBPATHS[task_name]
    for rel in rel_candidates:
        candidate = os.path.join(base, rel) if rel else base
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(base, rel_candidates[0]) if rel_candidates[0] else base


def _dataloader(args, data_set) -> DataLoader:
    prefetch = max(4, min(16, args.num_workers * 2)) if args.num_workers > 0 else None
    return DataLoader(
        data_set,
        batch_size=None,
        shuffle=False,
        pin_memory=True,
        sampler=None,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=prefetch,
        num_workers=args.num_workers,
        drop_last=False,
    )


def _build_provider(task_name: str, default_stride: int):
    Data = _DATASETS[task_name]

    def provider(args, flag, dataset_dir=None, disable_ddp_split=False):
        ds_root = dataset_dir or resolve_downstream_root(args, task_name)
        seq_len = int(args.seq_len)
        init_kwargs: dict[str, Any] = dict(
            root_path=ds_root,
            flag=flag,
            size=[seq_len, seq_len, 0],
            scale=True,
            timeenc=1,
            freq=getattr(args, "freq", "rs"),
            sampling_stride=getattr(args, "sampling_stride", default_stride),
            batch_size=args.batch_size,
            disable_ddp_split=disable_ddp_split,
        )
        # CDL class has no seq_window_align
        if task_name != "CDL_Classification":
            init_kwargs["seq_window_align"] = probe_seq_window_align(args)
        data_set = Data(**init_kwargs)
        try:
            import torch.distributed as dist

            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"{flag}: {len(data_set)} batches")
        except Exception:
            try:
                print(f"{flag}: {len(data_set)} batches")
            except Exception:
                pass
        return data_set, _dataloader(args, data_set)

    return provider


data_provider_LCMAP_Classification = _build_provider("LCMAP_Classification", 366)
data_provider_GlanceTraining_Classification = _build_provider(
    "GlanceTraining_Classification", 366
)
data_provider_GlobalTree_Classification = _build_provider(
    "GlobalTree_Classification", 244
)
data_provider_CDL_Classification = _build_provider("CDL_Classification", 122)
data_provider_CropHarvest_Classification = _build_provider(
    "CropHarvest_Classification", 732
)

PROVIDERS = {
    "LCMAP_Classification": data_provider_LCMAP_Classification,
    "GlanceTraining_Classification": data_provider_GlanceTraining_Classification,
    "GlobalTree_Classification": data_provider_GlobalTree_Classification,
    "CDL_Classification": data_provider_CDL_Classification,
    "CropHarvest_Classification": data_provider_CropHarvest_Classification,
}
