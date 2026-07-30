"""Data factory for SSL pretraining (HLS only)."""
from data.hls_dataset import Dataset_HLS
from torch.utils.data import DataLoader


def _get_common_dataloader(args, data_set, flag):
    """Unified DataLoader builder for IterableDataset + DDP."""
    sampler = None
    shuffle = False
    prefetch_factor = max(4, min(16, args.num_workers * 2)) if args.num_workers > 0 else None
    return DataLoader(
        data_set,
        batch_size=None,
        shuffle=shuffle,
        pin_memory=True,
        sampler=sampler,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=prefetch_factor,
        num_workers=args.num_workers,
        drop_last=False,
    )


def data_provider(args, flag, dataset_dir=None):
    """Build HLS pretraining IterableDataset + DataLoader."""
    data_set = Dataset_HLS(
        root_path=args.root_path if dataset_dir is None else dataset_dir,
        flag=flag,
        seq_len=args.seq_len,
        sampling_stride=args.sampling_stride,
        batch_size=args.batch_size,
        train_data_ratio=getattr(args, "train_data_ratio", 1.0),
        seq_window_align=getattr(args, "seq_window_align", "start"),
        freq=getattr(args, "freq", "rs"),
    )
    try:
        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(flag, len(data_set))
    except Exception:
        try:
            print(flag, len(data_set))
        except Exception:
            pass
    return data_set, _get_common_dataloader(args, data_set, flag)
