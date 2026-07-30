"""HLS pretraining IterableDataset (train.nc / train.h5)."""
import os
import warnings

import h5py
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from netCDF4 import Dataset
from numpy import random
from torch.utils.data import IterableDataset

from utils.timefeatures import time_features

warnings.filterwarnings('ignore')


class Dataset_HLS(IterableDataset):
    """Iterable loader for HLS SSL pretraining.

    Looks for ``{flag}.h5``, ``{flag}_optimized.nc``, or ``{flag}.nc`` under ``root_path``.
    Yields ``(x, time_mark, next_x, lon_lat)`` where ``next_x`` is always ``None``
    (legacy forecast next-window reading removed; TED/MSM/NTP do not use it).
    """

    def __init__(
        self,
        root_path,
        flag='train',
        seq_len=732,
        sampling_stride=None,
        batch_size=1000,
        train_data_ratio: float = 1.0,
        seq_window_align: str = 'start',
        freq: str = 'rs',
        scale: bool = True,
    ):
        super().__init__()
        assert flag in ['train', 'test', 'val', 'pred']
        self.flag = flag
        self.set_type = {'train': 0, 'val': 1, 'test': 2, 'pred': 3}[flag]
        self.seq_len = int(seq_len)
        self.scale = bool(scale)
        self.pre_scaler = {
            'mean': np.array(
                [4.0530856e+02, 6.7968939e+02, 7.3541718e+02, 2.5394734e+03,
                 2.0182101e+03, 1.2844141e+03, 5.2847379e-01],
                dtype=np.float32,
            ),
            'std': np.array(
                [2.7406531e+02, 3.4935846e+02, 5.2149530e+02, 9.8295978e+02,
                 9.3158044e+02, 8.0511346e+02, 2.8968227e-01],
                dtype=np.float32,
            ),
        }
        self.freq = freq or 'rs'
        self.stride = int(sampling_stride) if sampling_stride is not None else self.seq_len
        self.root_path = root_path
        self.batch_size = int(batch_size)
        self.train_data_ratio = float(train_data_ratio) if train_data_ratio is not None else 1.0
        align = str(seq_window_align).lower().strip() if seq_window_align is not None else 'start'
        if align not in ('start', 'end', 'random_year'):
            raise ValueError(
                f"seq_window_align must be 'start', 'end', or 'random_year', got {seq_window_align!r}"
            )
        self.seq_window_align = align
        self.__read_data__()

    @staticmethod
    def _is_hdf5(path: str) -> bool:
        try:
            return h5py.is_hdf5(path)
        except Exception:
            return False

    def __read_data__(self):
        # Resolve file path. Production train.nc is HDF5-backed; prefer h5py for those.
        h5_path = os.path.join(self.root_path, f"{self.flag}.h5")
        nc_path = os.path.join(self.root_path, f"{self.flag}.nc")
        nc_opt_path = os.path.join(self.root_path, f"{self.flag}_optimized.nc")

        if os.path.exists(h5_path):
            self.data_x = h5_path
        elif os.path.exists(nc_opt_path):
            self.data_x = nc_opt_path
        elif os.path.exists(nc_path):
            self.data_x = nc_path
        else:
            raise FileNotFoundError(f"Neither HDF5 nor netCDF4 file found: {h5_path} or {nc_path}")

        self.file_type = 'h5' if self._is_hdf5(self.data_x) else 'nc'
        self.has_lonlat = False
        self.nc_layout = 'optimized'

        if self.file_type == 'h5':
            with h5py.File(self.data_x, 'r', swmr=True) as f:
                if 'metadata/shape' in f:
                    shape = f['metadata/shape'][:]
                    self.num_pixels, self.time_steps, self.bands = [int(x) for x in shape]
                elif 'data' in f:
                    shape = f['data'].shape
                    self.num_pixels, self.time_steps, self.bands = [int(x) for x in shape]
                else:
                    raise KeyError(f"No 'data' or 'metadata/shape' in {self.data_x}")
                try:
                    import torch.distributed as dist
                    if not dist.is_initialized() or dist.get_rank() == 0:
                        print(
                            f"HDF5 file: {self.data_x}, num_pixels: {self.num_pixels}, "
                            f"time_steps: {self.time_steps}, bands: {self.bands}"
                        )
                except Exception:
                    print(
                        f"HDF5 file: {self.data_x}, num_pixels: {self.num_pixels}, "
                        f"time_steps: {self.time_steps}, bands: {self.bands}"
                    )
                df_stamp = f['time'][:].astype(str)
                if 'lon' in f and 'lat' in f:
                    self.has_lonlat = True
        else:  # classic netCDF4
            with Dataset(self.data_x, 'r') as f:
                data_var = f.variables['data']
                shape = data_var.shape
                dim_names = data_var.dimensions
                if dim_names[0] == 'pixels' or (shape[0] > shape[1] and shape[0] > shape[2]):
                    self.nc_layout = 'optimized'  # (pixels, time, bands)
                    self.num_pixels, self.time_steps, self.bands = shape
                else:
                    self.nc_layout = 'original'  # (time, bands, pixels)
                    self.time_steps, self.bands, self.num_pixels = shape
                try:
                    import torch.distributed as dist
                    if not dist.is_initialized() or dist.get_rank() == 0:
                        print(
                            f"netCDF4 file: {self.data_x}, layout={self.nc_layout}, "
                            f"num_pixels: {self.num_pixels}, time_steps: {self.time_steps}, "
                            f"bands: {self.bands}"
                        )
                except Exception:
                    print(
                        f"netCDF4 file: {self.data_x}, layout={self.nc_layout}, "
                        f"num_pixels: {self.num_pixels}, time_steps: {self.time_steps}, "
                        f"bands: {self.bands}"
                    )
                df_stamp = f.variables['time'][:].astype(str)
                if 'lon' in f.variables and 'lat' in f.variables:
                    self.has_lonlat = True
        if self.has_lonlat:
            try:
                import torch.distributed as dist
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print("  lon/lat available; will be added to batch output")
            except Exception:
                print("  lon/lat available; will be added to batch output")

        df_stamp = pd.to_datetime(df_stamp, format='%Y%j')
        self.data_stamp = time_features(df_stamp, freq=self.freq).transpose(1, 0)
        self.data_sets = {self.flag: self.data_x}
        # random_year: one sample/location; window is a random aligned year block
        # (0:seq_len, seq_len:2*seq_len, ...). Keeps steps matched to stride>=canvas single-window.
        if self.seq_window_align == 'random_year':
            if self.time_steps % self.seq_len != 0:
                raise ValueError(
                    f"random_year requires time_steps % seq_len == 0, "
                    f"got time_steps={self.time_steps}, seq_len={self.seq_len}"
                )
            self.n_year_blocks = self.time_steps // self.seq_len
            if self.n_year_blocks < 1:
                raise ValueError(
                    f"random_year requires at least one year block, "
                    f"got time_steps={self.time_steps}, seq_len={self.seq_len}"
                )
            self.windows_per_sample = 1
        else:
            self.n_year_blocks = None
            self.windows_per_sample = (self.time_steps - self.seq_len) // self.stride + 1
        self.window_indices = list(range(self.windows_per_sample))
        self.num_batches = (self.num_pixels + self.batch_size - 1) // self.batch_size
        self.batch_indices = list(range(self.num_batches))
        if self.windows_per_sample > 0:
            if self.seq_window_align == 'end':
                example_s_begin = self.time_steps - self.seq_len
            elif self.seq_window_align == 'random_year':
                example_s_begin = 0  # illustrative; train samples Uniform{0..n_year_blocks-1}*seq_len
            else:
                example_s_begin = 0
        else:
            example_s_begin = None
        try:
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(
                    f"  Dataset_HLS: seq_len={self.seq_len}, stride={self.stride}, "
                    f"windows_per_sample={self.windows_per_sample}, "
                    f"seq_window_align={self.seq_window_align}, "
                    f"n_year_blocks={self.n_year_blocks}, example_s_begin={example_s_begin}"
                )
        except Exception:
            print(
                f"  Dataset_HLS: seq_len={self.seq_len}, stride={self.stride}, "
                f"windows_per_sample={self.windows_per_sample}, "
                f"seq_window_align={self.seq_window_align}, "
                f"n_year_blocks={self.n_year_blocks}, example_s_begin={example_s_begin}"
            )

        # Optional: subsample training data by taking the first N batch indices.
        # This is applied BEFORE DDP split so that "ratio" refers to the global training set.
        # Note: we deliberately keep the order deterministic (sequential) to make quick testing easier.
        if self.flag == 'train' and self.train_data_ratio < 1.0:
            if not (0.0 < self.train_data_ratio <= 1.0):
                raise ValueError(f"train_data_ratio must be in (0, 1], got {self.train_data_ratio}")
            target_batches = int(np.floor(self.num_batches * self.train_data_ratio))
            target_batches = max(1, min(target_batches, self.num_batches))
            self.batch_indices = self.batch_indices[:target_batches]
            self.num_batches = len(self.batch_indices)
       

    def __iter__(self):
        total_length = self.time_steps
        seq_len = self.seq_len

        # 1. get all indices
        indices = self.batch_indices[:]

        # 2. DDP process-level split
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            batches_per_rank = len(indices) // world_size
            if batches_per_rank > 0:
                indices = indices[:batches_per_rank * world_size]
                indices = indices[rank::world_size]
            else:
                indices = indices if rank == 0 else []

        # 3. worker-thread-level split
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            indices = indices[worker_id::num_workers]

        # 4. Shuffle
        window_indices = self.window_indices[:]
        if self.flag == 'train':
            random.shuffle(window_indices)
            if not (self.train_data_ratio < 1.0):
                random.shuffle(indices)

        # 5. open file
        if self.file_type == 'h5':
            f = h5py.File(self.data_x, 'r', swmr=True)
        else:
            f = Dataset(self.data_x, 'r')

        try:
            for batch_idx in indices:
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, self.num_pixels)
                batch_len = end - start

                if self.file_type == 'h5':
                    seq_x = f['data'][start:end, :self.time_steps, :]
                elif getattr(self, 'nc_layout', 'original') == 'optimized':
                    seq_x = f.variables['data'][start:end, :, :]
                else:
                    seq_x = f.variables['data'][:, :, start:end]
                    seq_x = np.ascontiguousarray(np.transpose(seq_x, (2, 0, 1)))

                batch_lonlat = None
                if self.has_lonlat:
                    if self.file_type == 'h5':
                        b_lon = f['lon'][start:end]
                        b_lat = f['lat'][start:end]
                    else:
                        b_lon = f.variables['lon'][start:end]
                        b_lat = f.variables['lat'][start:end]
                    batch_lonlat = np.stack([b_lon, b_lat], axis=-1).astype(np.float32)

                if self.scale:
                    seq_x = (seq_x - self.pre_scaler['mean']) / self.pre_scaler['std']

                for w_idx in window_indices:
                    if self.seq_window_align == 'random_year':
                        if self.flag == 'train':
                            year_idx = int(random.randint(0, self.n_year_blocks))
                        else:
                            year_idx = self.n_year_blocks - 1
                        s_begin = year_idx * seq_len
                    elif self.seq_window_align == 'end':
                        s_begin = self.time_steps - seq_len - w_idx * self.stride
                    else:
                        s_begin = w_idx * self.stride
                    if s_begin < 0:
                        raise ValueError(
                            f"seq_window_align={self.seq_window_align!r} produced s_begin={s_begin} "
                            f"(time_steps={self.time_steps}, seq_len={seq_len}, w_idx={w_idx}, stride={self.stride})"
                        )
                    s_end = s_begin + seq_len
                    batch_seq_x = seq_x[:, s_begin:s_end, :]
                    seq_x_mark = self.data_stamp[s_begin:s_end, :]

                    padded_x = batch_seq_x
                    padded_stamp = np.tile(seq_x_mark[None, :, :], (batch_len, 1, 1))

                    if batch_lonlat is not None:
                        ll_expanded = np.tile(batch_lonlat[:, None, :], (1, padded_x.shape[1], 1))
                        batch_ll_tensor = torch.from_numpy(np.ascontiguousarray(ll_expanded)).float()
                    else:
                        batch_ll_tensor = None

                    # next_x kept as None for trainer unpacking compatibility
                    yield (
                        torch.from_numpy(np.ascontiguousarray(padded_x)).float(),
                        torch.from_numpy(np.ascontiguousarray(padded_stamp)).float(),
                        None,
                        batch_ll_tensor,
                    )
        finally:
            f.close()


    def __len__(self):
        import torch.distributed as dist
        total_len = self.num_batches * self.windows_per_sample
        
        # If it is DDP, each process length should be total_length / num_gpus
        if dist.is_initialized():
            return total_len // dist.get_world_size()
        return total_len

    def inverse_transform(self, data):
        if self.pre_scaler is not None:
            return data * self.pre_scaler['std'] + self.pre_scaler['mean']
        else:
            return data
