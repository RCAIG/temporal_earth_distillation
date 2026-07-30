"""Downstream classification datasets (HLS composite NC / NPZ)."""
from __future__ import annotations

import glob
import os
import re
import sys

import h5py
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from netCDF4 import Dataset
from torch.utils.data import IterableDataset

from utils.timefeatures import time_features

def _is_hdf5_file(path: str) -> bool:
    try:
        return h5py.is_hdf5(path)
    except Exception:
        return False

class _H5NetCDFView:
    """Minimal netCDF4-like view over an h5py File for classification loaders."""

    def __init__(self, handle):
        self._f = handle
        self.variables = {k: handle[k] for k in handle.keys()}
        self.dimensions = {}
        for k, v in handle.items():
            if hasattr(v, 'shape') and len(v.shape) == 1:
                self.dimensions[k] = type('D', (), {'size': int(v.shape[0])})()
        # Common aliases used by loaders
        if 'data' in handle:
            # data is (time, bands, samples)
            t, c, n = handle['data'].shape
            self.dimensions.setdefault('time', type('D', (), {'size': int(t)})())
            self.dimensions.setdefault('bands', type('D', (), {'size': int(c)})())
            self.dimensions.setdefault('samples', type('D', (), {'size': int(n)})())
            self.dimensions.setdefault('pixels', type('D', (), {'size': int(n)})())

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def open_classification_file(path: str):
    """Open classification file; use h5py for HDF5-backed *.nc."""
    if _is_hdf5_file(path):
        return _H5NetCDFView(h5py.File(path, 'r', swmr=True))
    return Dataset(path, 'r')

def _fixed_window_bounds(time_steps: int, seq_len: int, align: str = "start") -> tuple[int, int]:
    """Single-window slice for classification probes. align='end' takes the trailing seq_len."""
    align = str(align).lower().strip() if align is not None else "start"
    if align not in ("start", "end"):
        raise ValueError(f"seq_window_align must be 'start' or 'end', got {align!r}")
    if align == "end":
        s_begin = max(0, int(time_steps) - int(seq_len))
    else:
        s_begin = 0
    return s_begin, s_begin + int(seq_len)



class Dataset_LCMAP_Classification(IterableDataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='', scale=True,
                 timeenc=1, freq='d', sampling_stride=None, batch_size=1000, disable_ddp_split=False,
                 seq_window_align: str = 'start'):
        super().__init__()
        """
        Overall logic (kNN / downstream classification):
        1) Sequence length
           - seq_len comes from size[0] (usually args.seq_len from training, e.g. 366 / 732);
           - If size is empty, default seq_len = 366.
        2) Time axis and windowing
           - Read the full series from HDF5 / NetCDF as [B, time_steps, C];
           - For kNN probe: no sliding window; take a single window:
               * windows_per_sample = 1
               * window_indices = [0]
               * Default window [0:seq_len]; with seq_window_align='end' take the trailing seq_len.
        3) Length alignment
           - If seq_len <= time_steps: slice by align, no padding;
           - If seq_len > time_steps: take all steps, then pad by repeating the last step to seq_len.
        4) Time encoding
           - Use the same window on self.data_stamp; pad by repeating the last stamp if needed;
           - Broadcast [seq_len, time_feat] to [B, seq_len, time_feat].
        5) Model outputs
           - Return (batch_x, batch_x_mark, labels), where:
               * batch_x: [B, seq_len, C] (matches args.seq_len);
               * batch_x_mark: [B, seq_len, time_feat]；
               * labels: [B] (per-pixel static labels).
        6) kNN usage
           - Feed batch_x / batch_x_mark into model.encode(...),
             then use outputs['cls_token'] features for kNN.
        """
        # size [seq_len, label_len, pred_len] -> classification only uses seq_len (window size)
        # Use size if provided; otherwise default seq_len=366
        if size is not None and len(size) > 0:
            self.seq_len = size[0]
        else:
            self.seq_len = 366
        
        # Initialization
        self.flag = flag
        self.features = features
        self.scale = scale
        self.disable_ddp_split = disable_ddp_split  # Disable DDP sharding for kNN probe / single-process eval
        align = str(seq_window_align).lower().strip() if seq_window_align is not None else 'start'
        if align not in ('start', 'end'):
            raise ValueError(f"seq_window_align must be 'start' or 'end', got {seq_window_align!r}")
        self.seq_window_align = align
        # Keep the historical hard-coded scaler
        self.pre_scaler = {
            'mean': np.array([4.0530856e+02, 6.7968939e+02, 7.3541718e+02, 2.5394734e+03,
                              2.0182101e+03, 1.2844141e+03, 5.2847379e-01], dtype=np.float32),
            'std': np.array([2.7406531e+02, 3.4935846e+02, 5.2149530e+02, 9.8295978e+02,
                             9.3158044e+02, 8.0511346e+02, 2.8968227e-01], dtype=np.float32)
        }
        self.timeenc = timeenc
        self.freq = freq
        # sampling_stride defaults to seq_len (non-overlapping) when omitted
        self.stride = sampling_stride if sampling_stride is not None else self.seq_len
        print(f">>> [Dataset_LCMAP_Classification] seq_len: {self.seq_len}, stride: {self.stride}, sampling_stride param: {sampling_stride}")
        self.root_path = root_path
        # self.data_path = data_path
        self.batch_size = batch_size
        
        self.__read_data__()

    def __read_data__(self):
        # 1. Resolve file path / type (legacy + new names)
        h5_candidates = [
            os.path.join(self.root_path, "lcmap_classification_dataset.h5"),
        ]
        nc_candidates = [
            os.path.join(self.root_path, "lcmap_hls_classification_processed.nc"),
            os.path.join(self.root_path, "lcmap_classification_dataset.nc"),
        ]

        self.file_type = None
        self.data_x = None
        for p in h5_candidates:
            if os.path.exists(p):
                self.file_type = 'h5'
                self.data_x = p
                break
        if self.data_x is None:
            for p in nc_candidates:
                if os.path.exists(p):
                    self.file_type = 'nc'
                    self.data_x = p
                    break
        if self.data_x is None:
            raise FileNotFoundError(f"File not found in {self.root_path}")


        # 2. Load metadata and all labels (labels are small enough to keep in memory)
        if self.file_type == 'h5':
            with h5py.File(self.data_x, 'r', swmr=True) as f:
                shape = f['metadata/shape'][:]
                self.num_pixels, self.time_steps, self.bands = shape
                df_stamp = f['time'][:].astype(str)
                self.labels = f['labels'][:] # load labels [num_pixels]
        else: # netCDF4
            with open_classification_file(self.data_x) as f:
                # Note: NC writers use data dims (time, bands, samples); confirm whether a transpose is needed
                # Following Dataset_HLS: transpose or select by dimension names as needed
                self.time_steps = f.dimensions['time'].size
                self.bands = f.dimensions['bands'].size
                self.num_pixels = f.dimensions['samples'].size
                
                df_stamp = f.variables['time'][:].astype(str)
                self.labels = f.variables['labels'][:] # load labels [num_pixels]

        print(f"Dataset ({self.flag}): {self.num_pixels} pixels, {self.time_steps} steps, Labels loaded.")

        # 3. Time-feature encoding
        if self.timeenc == 1:
            try:
                df_stamp = pd.to_datetime(df_stamp, format='%Y%j')
            except:
                df_stamp = pd.to_datetime(df_stamp) # fall back to automatic datetime parsing
            data_stamp = time_features(df_stamp, freq=self.freq).transpose(1, 0)
        else:
            data_stamp = np.zeros((self.time_steps, 4))
        
        self.data_stamp = data_stamp
        
        # 4. Iteration / batching parameters
        # Probe tasks use a fixed-length window (no sliding windows)
        # Each sample returns one window of length seq_len
        self.windows_per_sample = 1  # fixed to 1: no sliding window
        self.window_indices = [0]  # only the first window
        self.num_batches = (self.num_pixels + self.batch_size - 1) // self.batch_size
        self.batch_indices = list(range(self.num_batches))
        
        print(f">>> [Dataset_LCMAP_Classification] fixed-length read mode:")
        print(f"    time_steps={self.time_steps}, seq_len={self.seq_len}")
        print(f"    seq_window_align={self.seq_window_align}")
        print(f"    windows_per_sample=1 (no sliding window)")
        print(f"    num_pixels={self.num_pixels}, batch_size={self.batch_size}")
        print(f"    num_batches={self.num_batches}")
        print(f"    Total samples = {self.num_batches} batches × 1 window = {self.num_batches}")

    def __iter__(self):

        # 1. All batch indices
        indices = self.batch_indices[:]

        # -----------------------------------------------------------
        # DDP process-level sharding
        # Skip DDP sharding when disable_ddp_split=True (kNN probe / eval)
        # -----------------------------------------------------------
        if dist.is_initialized() and not self.disable_ddp_split:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            
            # Keep equal batch counts across ranks
            # Truncate when length is not divisible by world_size
            batches_per_rank = len(indices) // world_size
            if batches_per_rank > 0:
                # Truncate to a multiple of world_size
                indices = indices[:batches_per_rank * world_size]
                # Strided shard so ranks are disjoint
                # e.g. rank0 -> [0,3,6...], rank1 -> [1,4,7...]
                indices = indices[rank::world_size]
            else:
                # If too few batches, only rank 0 iterates
                indices = indices if rank == 0 else []
        # -----------------------------------------------------------

        # -----------------------------------------------------------
        # DataLoader worker sharding
        # Continue sharding the already rank-filtered indices
        # -----------------------------------------------------------
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            # Further shard this rank's data across workers
            indices = indices[worker_id::num_workers]

        # -----------------------------------------------------------
        # Optional shuffle
        # Historical default is sequential reads (eval/inference).
        # Shuffle for train; keep order for test/val.
        # -----------------------------------------------------------
        # No window_indices needed without sliding windows
        if self.flag == 'train':
            random.shuffle(indices)

        # Open file handle
        if self.file_type == 'h5':
            # SWMR enables multi-process reads
            f = h5py.File(self.data_x, 'r', swmr=True)
        else:
            f = open_classification_file(self.data_x)

        try:
            # Iterate final shard indices
            for batch_idx in indices:
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, self.num_pixels)
                batch_pixels = list(range(start, end))
                
                # Labels for this batch
                # self.labels is an in-memory numpy array from __init__
                batch_labels_raw = self.labels[start:end] # [Batch_Size]

                # Read data (IO)
                if self.file_type == 'h5':
                    seq_x = f['data'][batch_pixels, :self.time_steps, :] # [B, T, C]
                else:
                    # NetCDF: (T, C, B) -> transpose to (B, T, C)
                    seq_x = f.variables['data'][:, :, batch_pixels]
                    seq_x = np.transpose(seq_x, (2, 0, 1))

                # Normalize
                if self.scale:
                    seq_x = (seq_x - self.pre_scaler['mean']) / self.pre_scaler['std']

                # Fixed-length single window: leading seq_len by default; align=end uses trailing
                s_begin, s_end = _fixed_window_bounds(
                    self.time_steps, self.seq_len, self.seq_window_align
                )
                
                # Bound checks
                if s_end > self.time_steps:
                    # If seq_len exceeds available steps, read available steps then pad
                    actual_len = self.time_steps
                    batch_seq_x = seq_x[:, 0:actual_len, :]  # [B, actual_len, C]
                    seq_x_mark = self.data_stamp[0:actual_len, :]  # [actual_len, time_feat]
                    
                    # Pad when actual length < seq_len
                    if actual_len < self.seq_len:
                        pad_len = self.seq_len - actual_len
                        # Pad by repeating the last timestep
                        last_timestep = batch_seq_x[:, -1:, :]  # [B, 1, C]
                        last_stamp = seq_x_mark[-1:, :]  # [1, time_feat]
                        batch_seq_x = np.concatenate([
                            batch_seq_x,
                            np.tile(last_timestep, (1, pad_len, 1))
                        ], axis=1)  # [B, seq_len, C]
                        seq_x_mark = np.concatenate([
                            seq_x_mark,
                            np.tile(last_stamp, (pad_len, 1))
                        ], axis=0)  # [seq_len, time_feat]
                else:
                    batch_seq_x = seq_x[:, s_begin:s_end, :]  # [B, seq_len, C]
                    seq_x_mark = self.data_stamp[s_begin:s_end, :]  # [seq_len, time_feat]
                
                # Expand timestamps to batch dimension
                padded_stamp = np.tile(seq_x_mark[None, :, :], (len(batch_pixels), 1, 1))

                # Return (input, time_mark, label)
                yield (
                    torch.tensor(batch_seq_x, dtype=torch.float32),
                    torch.tensor(padded_stamp, dtype=torch.float32),
                    torch.tensor(batch_labels_raw, dtype=torch.long)
                )

        finally:
            f.close()

    def __len__(self):
        # Length: one window per sample (no sliding window)
        import torch.distributed as dist
        total_len = self.num_batches * self.windows_per_sample  # windows_per_sample = 1
        
        if dist.is_initialized():
            return total_len // dist.get_world_size()
        return total_len




class Dataset_GlobalTree_Classification(IterableDataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='', scale=True,
                 timeenc=1, freq='d', sampling_stride=None, batch_size=1000, disable_ddp_split=False,
                 seq_window_align: str = 'start'):
        super().__init__()
        """
        GlobalTree classification dataset (aligned with LCMAP_Classification):
        - raw netCDF: global_tree_classification_dataset.nc
          Raw shape is like [time, bands, samples]; convert to [samples, time, bands]
        - Single window (no sliding); length from seq_len, default 244 (2 years at daily scale).
        """
        if size is not None and len(size) > 0:
            self.seq_len = size[0]
        else:
            # Default two-year length (244)
            self.seq_len = 244

        self.flag = flag
        self.features = features
        self.scale = scale
        self.disable_ddp_split = disable_ddp_split
        align = str(seq_window_align).lower().strip() if seq_window_align is not None else 'start'
        if align not in ('start', 'end'):
            raise ValueError(f"seq_window_align must be 'start' or 'end', got {seq_window_align!r}")
        self.seq_window_align = align
        self.pre_scaler = {
            'mean': np.array([4.0530856e+02, 6.7968939e+02, 7.3541718e+02, 2.5394734e+03,
                              2.0182101e+03, 1.2844141e+03, 5.2847379e-01], dtype=np.float32),
            'std': np.array([2.7406531e+02, 3.4935846e+02, 5.2149530e+02, 9.8295978e+02,
                             9.3158044e+02, 8.0511346e+02, 2.8968227e-01], dtype=np.float32)
        }
        self.timeenc = timeenc
        self.freq = freq
        self.stride = sampling_stride if sampling_stride is not None else self.seq_len
        print(
            f">>> [Dataset_GlobalTree_Classification] seq_len: {self.seq_len}, "
            f"stride: {self.stride}, sampling_stride param: {sampling_stride}, "
            f"seq_window_align={self.seq_window_align}"
        )

        # Use the resolved downstream root_path
        self.root_path = root_path
        self.batch_size = batch_size

        self.__read_data__()

    def __read_data__(self):
        h5_candidates = [
            os.path.join(self.root_path, "global_tree_classification_dataset.h5"),
        ]
        nc_candidates = [
            os.path.join(self.root_path, "globaltree_hls_classification_processed.nc"),
            os.path.join(self.root_path, "global_tree_classification_dataset.nc"),
        ]

        self.file_type = None
        self.data_x = None
        for p in h5_candidates:
            if os.path.exists(p):
                self.file_type = 'h5'
                self.data_x = p
                break
        if self.data_x is None:
            for p in nc_candidates:
                if os.path.exists(p):
                    self.file_type = 'nc'
                    self.data_x = p
                    break
        if self.data_x is None:
            raise FileNotFoundError(f"GlobalTree classification file not found in {self.root_path}")


        if self.file_type == 'h5':
            with h5py.File(self.data_x, 'r', swmr=True) as f:
                shape = f['metadata/shape'][:]
                self.num_pixels, self.time_steps, self.bands = shape
                df_stamp = f['time'][:].astype(str)

                # Try multiple label field names for compatibility
                if 'labels' in f:
                    self.labels = f['labels'][:]
                elif 'class_ids' in f:
                    self.labels = f['class_ids'][:]
                else:
                    raise KeyError("No labels or class_ids found in GlobalTree classification HDF5 file.")
        else:
            with open_classification_file(self.data_x) as f:
                data_var = f.variables['data'] if 'data' in f.variables else f.variables['images']
                shape = data_var.shape
                # Treat as (time, bands, samples)
                self.time_steps, self.bands, self.num_pixels = shape
                df_stamp = f.variables['time'][:].astype(str)

                if 'labels' in f.variables:
                    self.labels = f.variables['labels'][:]
                elif 'class_ids' in f.variables:
                    self.labels = f.variables['class_ids'][:]
                else:
                    raise KeyError("No labels or class_ids found in GlobalTree classification netCDF file.")

        # Optional per-sample year windows (updated GlobalTree: 3y cube + year_window).
        # Prefer h5py even for *.nc (these processed files are HDF5-backed).
        self.year_window = None
        self.time_years = None
        self.year_to_indices = {}
        try:
            with h5py.File(self.data_x, 'r') as f:
                if 'year_window' in f:
                    self.year_window = np.asarray(f['year_window'], dtype=np.int32)
        except Exception:
            try:
                with open_classification_file(self.data_x) as f:
                    if 'year_window' in f.variables:
                        self.year_window = np.asarray(f.variables['year_window'][:], dtype=np.int32)
            except Exception as e:
                print(f"[GlobalTree_Classification] year_window load skipped: {e}")

        self.time_years = np.array([int(str(t)[:4]) for t in df_stamp], dtype=np.int32)
        for y in np.unique(self.time_years):
            self.year_to_indices[int(y)] = np.where(self.time_years == int(y))[0]

        print(f"[GlobalTree_Classification-{self.flag}] pixels={self.num_pixels}, time_steps={self.time_steps}")
        if self.year_window is not None:
            n_pair = int(np.sum((self.year_window[:, 0] > 0) & (self.year_window[:, 1] > 0)))
            n_one = int(np.sum((self.year_window[:, 0] > 0) & (self.year_window[:, 1] < 0)))
            print(
                f"[GlobalTree_Classification-{self.flag}] year_window enabled: "
                f"2y={n_pair}, 1y={n_one}, years={sorted(self.year_to_indices)}"
            )

        if self.timeenc == 1:
            try:
                df_stamp = pd.to_datetime(df_stamp, format='%Y%j')
            except Exception:
                df_stamp = pd.to_datetime(df_stamp)
            data_stamp = time_features(df_stamp, freq=self.freq).transpose(1, 0)
        else:
            data_stamp = np.zeros((self.time_steps, 4))

        self.data_stamp = data_stamp

        self.windows_per_sample = 1
        self.window_indices = [0]
        self.num_batches = (self.num_pixels + self.batch_size - 1) // self.batch_size
        self.batch_indices = list(range(self.num_batches))

    def _gather_year_window_sample(self, seq_full: np.ndarray, sample_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Build [T,C] + stamp [T,F] for one sample using year_window (122 steps/year)."""
        yw = self.year_window[sample_idx]
        years = [int(y) for y in yw if int(y) > 0]
        if not years:
            s_begin, s_end = _fixed_window_bounds(
                seq_full.shape[0], self.seq_len, self.seq_window_align
            )
            s_end = min(s_end, seq_full.shape[0])
            return seq_full[s_begin:s_end], self.data_stamp[s_begin:s_end]

        blocks_x = []
        blocks_t = []
        for y in years:
            idx = self.year_to_indices.get(y)
            if idx is None or len(idx) == 0:
                raise KeyError(f"GlobalTree year {y} not found in time axis for sample {sample_idx}")
            blocks_x.append(seq_full[idx])
            blocks_t.append(self.data_stamp[idx])
        x = np.concatenate(blocks_x, axis=0)
        tm = np.concatenate(blocks_t, axis=0)
        if x.shape[0] > self.seq_len:
            if self.seq_window_align == "end":
                x = x[-self.seq_len :]
                tm = tm[-self.seq_len :]
            else:
                x = x[: self.seq_len]
                tm = tm[: self.seq_len]
        elif x.shape[0] < self.seq_len:
            pad = self.seq_len - x.shape[0]
            x = np.concatenate([x, np.repeat(x[-1:, :], pad, axis=0)], axis=0)
            tm = np.concatenate([tm, np.repeat(tm[-1:, :], pad, axis=0)], axis=0)
        return x, tm

    def __iter__(self):
        indices = self.batch_indices[:]

        if dist.is_initialized() and not self.disable_ddp_split:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            batches_per_rank = len(indices) // world_size
            if batches_per_rank > 0:
                indices = indices[:batches_per_rank * world_size]
                indices = indices[rank::world_size]
            else:
                indices = indices if rank == 0 else []

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            indices = indices[worker_id::num_workers]

        if self.flag == 'train':
            random.shuffle(indices)

        if self.file_type == 'h5':
            f = h5py.File(self.data_x, 'r', swmr=True)
        else:
            f = open_classification_file(self.data_x)

        try:
            for batch_idx in indices:
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, self.num_pixels)
                batch_pixels = list(range(start, end))

                batch_labels_raw = self.labels[start:end]

                if self.file_type == 'h5':
                    seq_x = f['data'][batch_pixels, :self.time_steps, :]
                else:
                    data_var = f.variables['data'] if 'data' in f.variables else f.variables['images']
                    seq_x = data_var[:, :, batch_pixels]
                    seq_x = np.transpose(seq_x, (2, 0, 1))

                if self.scale:
                    seq_x = (seq_x - self.pre_scaler['mean']) / self.pre_scaler['std']

                if self.year_window is not None:
                    xs, tms = [], []
                    for local_i, pix in enumerate(batch_pixels):
                        x_i, tm_i = self._gather_year_window_sample(seq_x[local_i], pix)
                        xs.append(x_i)
                        tms.append(tm_i)
                    batch_seq_x = np.stack(xs, axis=0)
                    padded_stamp = np.stack(tms, axis=0)
                else:
                    # Legacy: single fixed window (start or end)
                    s_begin, s_end = _fixed_window_bounds(
                        self.time_steps, self.seq_len, self.seq_window_align
                    )
                    if s_end > self.time_steps:
                        actual_len = self.time_steps
                        batch_seq_x = seq_x[:, 0:actual_len, :]
                        seq_x_mark = self.data_stamp[0:actual_len, :]
                        if actual_len < self.seq_len:
                            pad_len = self.seq_len - actual_len
                            last_timestep = batch_seq_x[:, -1:, :]
                            last_stamp = seq_x_mark[-1:, :]
                            batch_seq_x = np.concatenate(
                                [batch_seq_x, np.tile(last_timestep, (1, pad_len, 1))],
                                axis=1,
                            )
                            seq_x_mark = np.concatenate(
                                [seq_x_mark, np.tile(last_stamp, (pad_len, 1))],
                                axis=0,
                            )
                    else:
                        batch_seq_x = seq_x[:, s_begin:s_end, :]
                        seq_x_mark = self.data_stamp[s_begin:s_end, :]
                    padded_stamp = np.tile(seq_x_mark[None, :, :], (len(batch_pixels), 1, 1))

                yield (
                    torch.tensor(batch_seq_x, dtype=torch.float32),
                    torch.tensor(padded_stamp, dtype=torch.float32),
                    torch.tensor(batch_labels_raw, dtype=torch.long),
                )
        finally:
            f.close()

    def __len__(self):
        import torch.distributed as dist

        total_len = self.num_batches * self.windows_per_sample
        if dist.is_initialized():
            return total_len // dist.get_world_size()
        return total_len




class Dataset_CDL_Classification(IterableDataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='', scale=True,
                 timeenc=1, freq='d', sampling_stride=None, batch_size=1000, disable_ddp_split=False):
        super().__init__()
        """
        CDL single-year classification dataset:
        - Example path:/intelnvme01/ziyun/DownStreamTasks/CDL/outdir/classification_dataset_processed.nc
        - Sequence length ~122 (1 year).
        - Label fields: 'labels' / 'class_ids'.
        """
        if size is not None and len(size) > 0:
            self.seq_len = size[0]
        else:
            self.seq_len = 122

        self.flag = flag
        self.features = features
        self.scale = scale
        self.disable_ddp_split = disable_ddp_split
        self.pre_scaler = {
            'mean': np.array([4.0530856e+02, 6.7968939e+02, 7.3541718e+02, 2.5394734e+03,
                              2.0182101e+03, 1.2844141e+03, 5.2847379e-01], dtype=np.float32),
            'std': np.array([2.7406531e+02, 3.4935846e+02, 5.2149530e+02, 9.8295978e+02,
                             9.3158044e+02, 8.0511346e+02, 2.8968227e-01], dtype=np.float32)
        }
        self.timeenc = timeenc
        self.freq = freq
        self.stride = sampling_stride if sampling_stride is not None else self.seq_len
        print(f">>> [Dataset_CDL_Classification] seq_len: {self.seq_len}, stride: {self.stride}, sampling_stride param: {sampling_stride}")

        self.root_path = root_path
        self.batch_size = batch_size

        self.__read_data__()

    def __read_data__(self):
        h5_candidates = [
            os.path.join(self.root_path, "cdl_classification_dataset.h5"),
        ]
        nc_candidates = [
            os.path.join(self.root_path, "cdl_hls_classification_processed.nc"),
            os.path.join(self.root_path, "cdl_classification_dataset.nc"),
        ]

        self.file_type = None
        self.data_x = None
        for p in h5_candidates:
            if os.path.exists(p):
                self.file_type = 'h5'
                self.data_x = p
                break
        if self.data_x is None:
            for p in nc_candidates:
                if os.path.exists(p):
                    self.file_type = 'nc'
                    self.data_x = p
                    break
        if self.data_x is None:
            raise FileNotFoundError(f"CDL classification file not found in {self.root_path}")


        if self.file_type == 'h5':
            with h5py.File(self.data_x, 'r', swmr=True) as f:
                shape = f['metadata/shape'][:]
                self.num_pixels, self.time_steps, self.bands = shape
                df_stamp = f['time'][:].astype(str)

                if 'labels' in f:
                    self.labels = f['labels'][:]
                elif 'class_ids' in f:
                    self.labels = f['class_ids'][:]
                else:
                    raise KeyError("No labels or class_ids found in CDL classification HDF5 file.")
        else:
            with open_classification_file(self.data_x) as f:
                data_var = f.variables['data'] if 'data' in f.variables else f.variables['images']
                self.time_steps, self.bands, self.num_pixels = data_var.shape
                df_stamp = f.variables['time'][:].astype(str)

                if 'labels' in f.variables:
                    self.labels = f.variables['labels'][:]
                elif 'class_ids' in f.variables:
                    self.labels = f.variables['class_ids'][:]
                else:
                    raise KeyError("No labels or class_ids found in CDL classification netCDF file.")

        print(f"[CDL_Classification-{self.flag}] pixels={self.num_pixels}, time_steps={self.time_steps}")

        if self.timeenc == 1:
            try:
                df_stamp = pd.to_datetime(df_stamp, format='%Y%j')
            except Exception:
                df_stamp = pd.to_datetime(df_stamp)
            data_stamp = time_features(df_stamp, freq=self.freq).transpose(1, 0)
        else:
            data_stamp = np.zeros((self.time_steps, 4))

        self.data_stamp = data_stamp

        self.windows_per_sample = 1
        self.window_indices = [0]
        self.num_batches = (self.num_pixels + self.batch_size - 1) // self.batch_size
        self.batch_indices = list(range(self.num_batches))

    def __iter__(self):
        indices = self.batch_indices[:]

        if dist.is_initialized() and not self.disable_ddp_split:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            batches_per_rank = len(indices) // world_size
            if batches_per_rank > 0:
                indices = indices[:batches_per_rank * world_size]
                indices = indices[rank::world_size]
            else:
                indices = indices if rank == 0 else []

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            indices = indices[worker_id::num_workers]

        if self.flag == 'train':
            random.shuffle(indices)

        if self.file_type == 'h5':
            f = h5py.File(self.data_x, 'r', swmr=True)
        else:
            f = open_classification_file(self.data_x)

        try:
            for batch_idx in indices:
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, self.num_pixels)
                batch_pixels = list(range(start, end))

                batch_labels_raw = self.labels[start:end]

                if self.file_type == 'h5':
                    seq_x = f['data'][batch_pixels, :self.time_steps, :]
                else:
                    data_var = f.variables['data'] if 'data' in f.variables else f.variables['images']
                    seq_x = data_var[:, :, batch_pixels]
                    seq_x = np.transpose(seq_x, (2, 0, 1))

                if self.scale:
                    seq_x = (seq_x - self.pre_scaler['mean']) / self.pre_scaler['std']

                s_begin = 0
                s_end = self.seq_len

                if s_end > self.time_steps:
                    actual_len = self.time_steps
                    batch_seq_x = seq_x[:, s_begin:actual_len, :]
                    seq_x_mark = self.data_stamp[s_begin:actual_len, :]
                    if actual_len < self.seq_len:
                        pad_len = self.seq_len - actual_len
                        last_timestep = batch_seq_x[:, -1:, :]
                        last_stamp = seq_x_mark[-1:, :]
                        batch_seq_x = np.concatenate(
                            [batch_seq_x, np.tile(last_timestep, (1, pad_len, 1))],
                            axis=1,
                        )
                        seq_x_mark = np.concatenate(
                            [seq_x_mark, np.tile(last_stamp, (pad_len, 1))],
                            axis=0,
                        )
                else:
                    batch_seq_x = seq_x[:, s_begin:s_end, :]
                    seq_x_mark = self.data_stamp[s_begin:s_end, :]

                padded_stamp = np.tile(seq_x_mark[None, :, :], (len(batch_pixels), 1, 1))

                yield (
                    torch.tensor(batch_seq_x, dtype=torch.float32),
                    torch.tensor(padded_stamp, dtype=torch.float32),
                    torch.tensor(batch_labels_raw, dtype=torch.long),
                )
        finally:
            f.close()

    def __len__(self):
        import torch.distributed as dist

        total_len = self.num_batches * self.windows_per_sample
        if dist.is_initialized():
            return total_len // dist.get_world_size()
        return total_len




class Dataset_GlanceTraining_Classification(IterableDataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='', scale=True,
                 timeenc=1, freq='d', sampling_stride=None, batch_size=1000, disable_ddp_split=False,
                 seq_window_align: str = 'start'):
        super().__init__()
        """
        Overall logic (GlanceTraining classification + kNN probe):
        1) Sequence length
           - seq_len from size[0] (aligned with backbone args.seq_len, e.g. 366 / 732);
           - If size is empty, default seq_len = 366.
        2) Time axis and windowing
           - Read the full series from HDF5 / NetCDF as [B, time_steps, C];
           - Same as Dataset_LCMAP_Classification: no sliding window:
               * windows_per_sample = 1
               * window_indices = [0]
               * Default [0:seq_len]; seq_window_align='end' uses the trailing window.
           - Older code used stride/sliding windows; probes now use one fixed-length window.
        3) Length alignment
           - Assume time_steps >= seq_len (typical annual HLS/Glance cubes) and slice by align;
           - If time_steps < seq_len, follow Dataset_LCMAP_Classification padding logic.
        4) Time encoding
           - Use matching time features from self.data_stamp;
           - Broadcast [seq_len, time_feat] to [B, seq_len, time_feat].
        5) Model outputs
           - Return (batch_x, batch_x_mark, labels), where:
               * batch_x: [B, seq_len, C]；
               * batch_x_mark: [B, seq_len, time_feat]；
               * labels: [B], remapped class_ids for GlanceTraining.
        6) kNN usage
           - Feed batch_x / batch_x_mark into model.encode(...),
             use outputs['cls_token'] for kNN (and optionally a raw-mean baseline).
        """
        # size [seq_len, label_len, pred_len] -> classification only uses seq_len (window size)
        # Use size if provided; otherwise default seq_len=366
        if size is not None and len(size) > 0:
            self.seq_len = size[0]
        else:
            self.seq_len = 366
        
        # Initialization
        self.flag = flag
        self.features = features
        self.scale = scale
        self.disable_ddp_split = disable_ddp_split  # Disable DDP sharding for kNN probe / single-process eval
        align = str(seq_window_align).lower().strip() if seq_window_align is not None else 'start'
        if align not in ('start', 'end'):
            raise ValueError(f"seq_window_align must be 'start' or 'end', got {seq_window_align!r}")
        self.seq_window_align = align
        # Keep the historical hard-coded scaler
        self.pre_scaler = {
            'mean': np.array([4.0530856e+02, 6.7968939e+02, 7.3541718e+02, 2.5394734e+03,
                              2.0182101e+03, 1.2844141e+03, 5.2847379e-01], dtype=np.float32),
            'std': np.array([2.7406531e+02, 3.4935846e+02, 5.2149530e+02, 9.8295978e+02,
                             9.3158044e+02, 8.0511346e+02, 2.8968227e-01], dtype=np.float32)
        }
        self.timeenc = timeenc
        self.freq = freq
        # sampling_stride defaults to seq_len (non-overlapping) when omitted
        self.stride = sampling_stride if sampling_stride is not None else self.seq_len
        print(f">>> [Dataset_GlanceTraining_Classification] seq_len: {self.seq_len}, stride: {self.stride}, sampling_stride param: {sampling_stride}")
        self.root_path = root_path
        # self.data_path = data_path
        self.batch_size = batch_size
        
        self.__read_data__()

    def __read_data__(self):
        # 1. Resolve file path / type (legacy + new names)
        h5_candidates = [
            os.path.join(self.root_path, "glancetraining_classification_dataset.h5"),
        ]
        nc_candidates = [
            os.path.join(self.root_path, "glancetraining_hls_classification_processed.nc"),
            os.path.join(self.root_path, "glancetraining_classification_dataset.nc"),
        ]

        self.file_type = None
        self.data_x = None
        for p in h5_candidates:
            if os.path.exists(p):
                self.file_type = 'h5'
                self.data_x = p
                break
        if self.data_x is None:
            for p in nc_candidates:
                if os.path.exists(p):
                    self.file_type = 'nc'
                    self.data_x = p
                    break
        if self.data_x is None:
            raise FileNotFoundError(f"File not found in {self.root_path}")


        # 2. Load metadata and all labels (labels are small enough to keep in memory)
        if self.file_type == 'h5':
            with h5py.File(self.data_x, 'r', swmr=True) as f:
                shape = f['metadata/shape'][:]
                self.num_pixels, self.time_steps, self.bands = shape
                df_stamp = f['time'][:].astype(str)
                self.labels = f['class_ids'][:] # load labels [num_pixels]
                unique_ids = np.unique(self.labels)

                # e.g. 1->0, 5->1, 24->2
                self.labels = np.searchsorted(unique_ids, self.labels)

                # 4. Infer class count automatically
                self.num_classes = len(unique_ids)
        else: # netCDF4
            with open_classification_file(self.data_x) as f:
                # Note: NC writers use data dims (time, bands, samples); confirm whether a transpose is needed
                # Following Dataset_HLS: transpose or select by dimension names as needed
                self.time_steps = f.dimensions['time'].size
                self.bands = f.dimensions['bands'].size
                self.num_pixels = f.dimensions['samples'].size

                # for var_name, var_obj in f.variables.items():
                #     # var_obj.dimensions -> names like ('time', 'bands', 'samples')
                #     # var_obj.shape -> concrete sizes like (365, 10, 1000)
                #     print(f"  Name: {var_name:<15} | Shape: {str(var_obj.shape):<20} | Dims: {var_obj.dimensions}")

                df_stamp = f.variables['time'][:].astype(str)
                self.labels = f.variables['class_ids'][:] # load labels [num_pixels]
                unique_ids = np.unique(self.labels)

                # e.g. 1->0, 5->1, 24->2
                self.labels = np.searchsorted(unique_ids, self.labels)

                # 4. Infer class count automatically
                self.num_classes = len(unique_ids)

                # --- debug ---
                # print(f"\n[Label Remap Success]")
                # print(f"Original IDs found: {unique_ids}")
                # print(f"Mapped to:        {np.arange(self.num_classes)}")
                # print(f"Total Classes:    {self.num_classes}")
                # print(f"Label Shape:      {self.labels.shape}")
                # print("-" * 30)
                # print(f"Dataset ({self.flag}): {self.num_pixels} pixels, {self.time_steps} steps, Labels loaded.")

        # 3. Time-feature encoding
        if self.timeenc == 1:
            try:
                df_stamp = pd.to_datetime(df_stamp, format='%Y%j')
            except:
                df_stamp = pd.to_datetime(df_stamp) # fall back to automatic datetime parsing
            data_stamp = time_features(df_stamp, freq=self.freq).transpose(1, 0)
        else:
            data_stamp = np.zeros((self.time_steps, 4))
        
        self.data_stamp = data_stamp
        
        # 4. Iteration / batching parameters
        # Probe: same fixed-length windowing as Dataset_LCMAP_Classification
        # One window per pixel (length seq_len)
        self.windows_per_sample = 1
        self.window_indices = [0]
        self.num_batches = (self.num_pixels + self.batch_size - 1) // self.batch_size
        self.batch_indices = list(range(self.num_batches))
        
        print(f">>> [Dataset_GlanceTraining_Classification] fixed-length read mode:")
        print(f"    time_steps={self.time_steps}, seq_len={self.seq_len}")
        print(f"    seq_window_align={self.seq_window_align}")
        print(f"    windows_per_sample=1 (no sliding window)")
        print(f"    num_pixels={self.num_pixels}, batch_size={self.batch_size}")
        print(f"    num_batches={self.num_batches}")
        print(f"    Total samples = {self.num_batches} batches × 1 window = {self.num_batches}")

    def __iter__(self):
        # No shuffle: sequential reads
        window_indices = self.window_indices[:]

        worker_info = torch.utils.data.get_worker_info()
        
        all_batch_indices = self.batch_indices[:]
        
        # DDP process sharding (unless disabled)
        if dist.is_initialized() and not self.disable_ddp_split:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            
            # Keep equal batch counts across ranks
            # Truncate when length is not divisible by world_size
            batches_per_rank = len(all_batch_indices) // world_size
            if batches_per_rank > 0:
                # Truncate to a multiple of world_size
                all_batch_indices = all_batch_indices[:batches_per_rank * world_size]
                # then shard by rank
                all_batch_indices = all_batch_indices[rank::world_size]
            else:
                # If too few batches, only rank 0 iterates
                all_batch_indices = all_batch_indices if rank == 0 else []
        
        if worker_info is None:  
            batch_indices = all_batch_indices
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            batch_indices = all_batch_indices[worker_id::num_workers]
        # ----------------------------------

        # Open file handle
        if self.file_type == 'h5':
            f = h5py.File(self.data_x, 'r', swmr=True)
        else:
            f = open_classification_file(self.data_x)

        try:
            for batch_idx in batch_indices:
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, self.num_pixels)
                batch_pixels = list(range(start, end))
                
                # Labels for this batch
                batch_labels_raw = self.labels[start:end] # [Batch_Size]

                # Read data (IO)
                if self.file_type == 'h5':
                    seq_x = f['data'][batch_pixels, :self.time_steps, :] # [B, T, C]
                else:
                    # NetCDF: (T, C, B) -> transpose to (B, T, C)
                    seq_x = f.variables['data'][:, :, batch_pixels]
                    seq_x = np.transpose(seq_x, (2, 0, 1))

                # Normalize
                if self.scale:
                    seq_x = (seq_x - self.pre_scaler['mean']) / self.pre_scaler['std']

                # single fixed window: leading seq_len by default; align=end uses trailing
                s_begin, s_end = _fixed_window_bounds(
                    self.time_steps, self.seq_len, self.seq_window_align
                )
                if s_end > self.time_steps:
                    actual_end = self.time_steps
                    batch_seq_x = seq_x[:, 0:actual_end, :]
                    seq_x_mark = self.data_stamp[0:actual_end, :]
                    if actual_end < self.seq_len:
                        pad_len = self.seq_len - actual_end
                        last_timestep = batch_seq_x[:, -1:, :]
                        last_stamp = seq_x_mark[-1:, :]
                        batch_seq_x = np.concatenate(
                            [batch_seq_x, np.tile(last_timestep, (1, pad_len, 1))],
                            axis=1,
                        )
                        seq_x_mark = np.concatenate(
                            [seq_x_mark, np.tile(last_stamp, (pad_len, 1))],
                            axis=0,
                        )
                else:
                    batch_seq_x = seq_x[:, s_begin:s_end, :]
                    seq_x_mark = self.data_stamp[s_begin:s_end, :]
                padded_stamp = np.tile(seq_x_mark[None, :, :], (len(batch_pixels), 1, 1))

                yield (
                    torch.tensor(batch_seq_x, dtype=torch.float32),
                    torch.tensor(padded_stamp, dtype=torch.float32),
                    torch.tensor(batch_labels_raw, dtype=torch.long),
                )

        finally:
            f.close()

    def __len__(self):
        return self.num_batches * self.windows_per_sample




class Dataset_CropHarvest_Classification(IterableDataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='', scale=True,
                 timeenc=1, freq='d', sampling_stride=None, batch_size=1000, disable_ddp_split=False,
                 seq_window_align: str = 'start'):
        super().__init__()
        if size is not None and len(size) > 0:
            self.seq_len = size[0]
        else:
            self.seq_len = 366

        self.flag = flag
        self.features = features
        self.scale = scale
        self.disable_ddp_split = disable_ddp_split
        align = str(seq_window_align).lower().strip() if seq_window_align is not None else 'start'
        if align not in ('start', 'end'):
            raise ValueError(f"seq_window_align must be 'start' or 'end', got {seq_window_align!r}")
        self.seq_window_align = align
        self.pre_scaler = {
            'mean': np.array([4.0530856e+02, 6.7968939e+02, 7.3541718e+02, 2.5394734e+03,
                              2.0182101e+03, 1.2844141e+03, 5.2847379e-01], dtype=np.float32),
            'std': np.array([2.7406531e+02, 3.4935846e+02, 5.2149530e+02, 9.8295978e+02,
                             9.3158044e+02, 8.0511346e+02, 2.8968227e-01], dtype=np.float32)
        }
        self.timeenc = timeenc
        self.freq = freq
        self.stride = sampling_stride if sampling_stride is not None else self.seq_len
        print(
            f">>> [Dataset_CropHarvest_Classification] seq_len: {self.seq_len}, "
            f"stride: {self.stride}, sampling_stride param: {sampling_stride}, "
            f"seq_window_align={self.seq_window_align}"
        )
        self.root_path = root_path
        self.batch_size = batch_size

        self.__read_data__()

    def __read_data__(self):
        h5_candidates = [
            os.path.join(self.root_path, "cropharvest_classification_dataset.h5"),
        ]
        nc_candidates = [
            os.path.join(self.root_path, "cropharvest_hls_classification_processed.nc"),
            os.path.join(self.root_path, "cropharvest_classification_dataset.nc"),
        ]

        self.file_type = None
        self.data_x = None
        for p in h5_candidates:
            if os.path.exists(p):
                self.file_type = 'h5'
                self.data_x = p
                break
        if self.data_x is None:
            for p in nc_candidates:
                if os.path.exists(p):
                    self.file_type = 'nc'
                    self.data_x = p
                    break
        if self.data_x is None:
            raise FileNotFoundError(f"File not found in {self.root_path}")


        if self.file_type == 'h5':
            with h5py.File(self.data_x, 'r', swmr=True) as f:
                shape = f['metadata/shape'][:]
                self.num_pixels, self.time_steps, self.bands = shape
                df_stamp = f['time'][:].astype(str)
                if 'labels' in f:
                    self.labels = f['labels'][:]
                elif 'class_ids' in f:
                    self.labels = f['class_ids'][:]
                else:
                    raise KeyError("No labels or class_ids found in CropHarvest classification HDF5 file.")
        else:
            with open_classification_file(self.data_x) as f:
                self.time_steps = f.dimensions['time'].size
                self.bands = f.dimensions['bands'].size
                self.num_pixels = f.dimensions['samples'].size
                df_stamp = f.variables['time'][:].astype(str)
                if 'labels' in f.variables:
                    self.labels = f.variables['labels'][:]
                elif 'class_ids' in f.variables:
                    self.labels = f.variables['class_ids'][:]
                else:
                    raise KeyError("No labels or class_ids found in CropHarvest classification netCDF file.")

        if self.timeenc == 1:
            try:
                df_stamp = pd.to_datetime(df_stamp, format='%Y%j')
            except Exception:
                df_stamp = pd.to_datetime(df_stamp)
            data_stamp = time_features(df_stamp, freq=self.freq).transpose(1, 0)
        else:
            data_stamp = np.zeros((self.time_steps, 4))

        self.data_stamp = data_stamp
        self.windows_per_sample = 1
        self.window_indices = [0]
        self.num_batches = (self.num_pixels + self.batch_size - 1) // self.batch_size
        self.batch_indices = list(range(self.num_batches))

    def __iter__(self):
        indices = self.batch_indices[:]
        if dist.is_initialized() and not self.disable_ddp_split:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            batches_per_rank = len(indices) // world_size
            if batches_per_rank > 0:
                indices = indices[:batches_per_rank * world_size]
                indices = indices[rank::world_size]
            else:
                indices = indices if rank == 0 else []

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            indices = indices[worker_info.id::worker_info.num_workers]

        if self.file_type == 'h5':
            f = h5py.File(self.data_x, 'r', swmr=True)
        else:
            f = open_classification_file(self.data_x)
        try:
            for batch_idx in indices:
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, self.num_pixels)
                batch_pixels = list(range(start, end))
                batch_labels_raw = self.labels[start:end]

                if self.file_type == 'h5':
                    seq_x = f['data'][batch_pixels, :self.time_steps, :]
                else:
                    seq_x = f.variables['data'][:, :, batch_pixels]
                    seq_x = np.transpose(seq_x, (2, 0, 1))

                if self.scale:
                    seq_x = (seq_x - self.pre_scaler['mean']) / self.pre_scaler['std']

                s_begin, s_end = _fixed_window_bounds(
                    self.time_steps, self.seq_len, self.seq_window_align
                )
                if s_end > self.time_steps:
                    actual_end = self.time_steps
                    batch_seq_x = seq_x[:, 0:actual_end, :]
                    seq_x_mark = self.data_stamp[0:actual_end, :]
                    if actual_end < self.seq_len:
                        pad_len = self.seq_len - actual_end
                        last_timestep = batch_seq_x[:, -1:, :]
                        last_stamp = seq_x_mark[-1:, :]
                        batch_seq_x = np.concatenate(
                            [batch_seq_x, np.tile(last_timestep, (1, pad_len, 1))],
                            axis=1,
                        )
                        seq_x_mark = np.concatenate(
                            [seq_x_mark, np.tile(last_stamp, (pad_len, 1))],
                            axis=0,
                        )
                else:
                    batch_seq_x = seq_x[:, s_begin:s_end, :]
                    seq_x_mark = self.data_stamp[s_begin:s_end, :]
                padded_stamp = np.tile(seq_x_mark[None, :, :], (len(batch_pixels), 1, 1))

                yield (
                    torch.tensor(batch_seq_x, dtype=torch.float32),
                    torch.tensor(padded_stamp, dtype=torch.float32),
                    torch.tensor(batch_labels_raw, dtype=torch.long),
                )
        finally:
            f.close()

    def __len__(self):
        return self.num_batches * self.windows_per_sample
