"""Lon/lat I/O and AEF-style spatial splits for downstream classification."""

from __future__ import annotations

import argparse
import os
from argparse import ArgumentParser, Namespace
from types import SimpleNamespace
from typing import Literal

import numpy as np

try:
    from pyproj import Transformer

    _HAS_PYPROJ = True
except ImportError:
    _HAS_PYPROJ = False

try:
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix
    from scipy.spatial import cKDTree

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

import h5py

from data.downstream_factory import resolve_downstream_root as _resolve_downstream_root

_CLASSIFICATION_FILES: dict[str, tuple[list[str], list[str]]] = {
    "LCMAP_Classification": (
        ["lcmap_classification_dataset.h5"],
        ["lcmap_hls_classification_processed.nc", "lcmap_classification_dataset.nc"],
    ),
    "GlanceTraining_Classification": (
        ["glancetraining_classification_dataset.h5"],
        [
            "glancetraining_hls_classification_processed.nc",
            "glancetraining_classification_dataset.nc",
        ],
    ),
    "GlobalTree_Classification": (
        ["global_tree_classification_dataset.h5"],
        [
            "globaltree_hls_classification_processed.nc",
            "global_tree_classification_dataset.nc",
        ],
    ),
    "CDL_Classification": (
        ["cdl_classification_dataset.h5"],
        ["cdl_hls_classification_processed.nc", "cdl_classification_dataset.nc"],
    ),
    "CropHarvest_Classification": (
        ["cropharvest_classification_dataset.h5"],
        [
            "cropharvest_hls_classification_processed.nc",
            "cropharvest_classification_dataset.nc",
        ],
    ),
}

# Composite ``*_processed.nc`` files are HDF5-backed; some datasets keep lon/lat only in NPZ / AEF files.
_LONLAT_NPZ_FALLBACK: dict[str, list[str]] = {
    "LCMAP_Classification": [
        "lcmap_gse_classification.npz",
        "lcmap_hls_classification.npz",
    ],
    "GlanceTraining_Classification": [
        "glancetraining_gse_classification.npz",
        "glancetraining_hls_classification.npz",
    ],
    "GlobalTree_Classification": [
        "globaltree_hls_classification.npz",
    ],
    "CDL_Classification": [
        "cdl_hls_classification.npz",
    ],
    "CropHarvest_Classification": [
        "cropharvest_hls_classification.npz",
    ],
}


def add_spatial_split_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--spatial_block_m",
        type=float,
        default=None,
        help="Enable AEF-style spatial split: block size in meters (commonly 1280 = 1.28 km)",
    )
    parser.add_argument(
        "--spatial_split_mode",
        type=str,
        choices=("grid", "components"),
        default="grid",
        help="grid=UTM grid blocks; components=connected components within 1.28 km radius",
    )
    parser.add_argument(
        "--spatial_proximity_filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply AEF spatial proximity filtering (min pairwise distance >= block_m) before split",
    )


def split_description(args: Namespace, *, max_train_per_class: int = 300) -> str:
    cap_tag = "nocap" if int(max_train_per_class) <= 0 else f"cap{int(max_train_per_class)}"
    if args.spatial_block_m is not None and float(args.spatial_block_m) > 0:
        bm = int(float(args.spatial_block_m))
        pf = "prox" if args.spatial_proximity_filter else "noprox"
        return f"2:1:1_spatial{bm}m_{args.spatial_split_mode}_{pf}_{cap_tag}"
    return f"2:1:1_per_class_{cap_tag}"


def resolve_classification_file(downstream_data_root: str, dataset: str) -> tuple[str, str]:
    """Return (absolute_path, file_type) where file_type is 'h5' or 'nc'."""
    if dataset not in _CLASSIFICATION_FILES:
        raise ValueError(f"Unsupported dataset for lon/lat: {dataset}")
    ns = SimpleNamespace(downstream_data_root=downstream_data_root)
    root = _resolve_downstream_root(ns, dataset)
    h5_names, nc_names = _CLASSIFICATION_FILES[dataset]
    for name in h5_names:
        path = os.path.join(root, name)
        if os.path.isfile(path):
            return path, "h5"
    for name in nc_names:
        path = os.path.join(root, name)
        if os.path.isfile(path):
            return path, "nc"
    raise FileNotFoundError(
        f"No classification file for {dataset} under {root} (tried h5={h5_names}, nc={nc_names})"
    )


def _read_lonlat_hdf5(path: str) -> tuple[np.ndarray, np.ndarray] | None:
    with h5py.File(path, "r", swmr=True) as f:
        if "lon" not in f or "lat" not in f:
            return None
        lon = np.asarray(f["lon"][:], dtype=np.float64)
        lat = np.asarray(f["lat"][:], dtype=np.float64)
    if lon.shape != lat.shape:
        raise ValueError(f"lon/lat shape mismatch in {path}: {lon.shape} vs {lat.shape}")
    return lon, lat


def _read_lonlat_npz(path: str) -> tuple[np.ndarray, np.ndarray] | None:
    z = np.load(path)
    try:
        if "lon" not in z.files or "lat" not in z.files:
            return None
        lon = np.asarray(z["lon"], dtype=np.float64)
        lat = np.asarray(z["lat"], dtype=np.float64)
    finally:
        z.close()
    if lon.shape != lat.shape:
        raise ValueError(f"lon/lat shape mismatch in {path}: {lon.shape} vs {lat.shape}")
    return lon, lat


def load_classification_lonlat(downstream_data_root: str, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """
  Load lon/lat aligned with classification DataLoader rows.

  Prefer composite HDF5 (``*_processed.nc``), then classification-root ``*.npz``.
  Fall back to GSE/AEF npz when HLS composite has no lon/lat.
    """
    path, _file_type = resolve_classification_file(downstream_data_root, dataset)
    out = _read_lonlat_hdf5(path)
    if out is not None:
        return out

    # Shareable layout: NPZ files live at classification root (and optional legacy subdir).
    search_roots = [
        downstream_data_root,
        os.path.join(downstream_data_root, "downstream_classification_task"),
        os.path.dirname(os.path.abspath(downstream_data_root)),
    ]
    tried: list[str] = [path]
    for name in _LONLAT_NPZ_FALLBACK.get(dataset, []):
        found = False
        for root in search_roots:
            npz_path = os.path.join(root, name)
            if not os.path.isfile(npz_path):
                continue
            tried.append(npz_path)
            out = _read_lonlat_npz(npz_path)
            if out is not None:
                return out
            found = True
        if not found:
            tried.append(os.path.join(search_roots[0], name))

    raise FileNotFoundError(
        f"No lon/lat for {dataset}; tried: {tried}"
    )


def lonlat_to_xy_m(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project WGS84 lon/lat to local meters (UTM if pyproj available)."""
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    if _HAS_PYPROJ:
        lon0 = float(np.nanmedian(lon))
        lat0 = float(np.nanmedian(lat))
        zone = int((lon0 + 180.0) // 6.0) + 1
        zone = max(1, min(60, zone))
        epsg = 32600 + zone if lat0 >= 0 else 32700 + zone
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        x, y = transformer.transform(lon, lat)
        return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)

    lat0 = np.deg2rad(float(np.nanmedian(lat)))
    cos_lat = max(np.cos(lat0), 1e-6)
    x = np.deg2rad(lon - np.nanmedian(lon)) * 6371000.0 * cos_lat
    y = np.deg2rad(lat - np.nanmedian(lat)) * 6371000.0
    return x, y


def spatial_proximity_filter_indices(
    lon: np.ndarray,
    lat: np.ndarray,
    min_dist_m: float,
    *,
    seed: int = 42,
) -> np.ndarray:
    """
    AEF «spatial proximity filtering»: greedy keep so pairwise distance >= min_dist_m.
    Returns indices into the input arrays.
    """
    if not _HAS_SCIPY:
        raise ImportError("spatial_proximity_filter requires scipy")
    x, y = lonlat_to_xy_m(lon, lat)
    n = int(lon.shape[0])
    order = np.random.default_rng(int(seed)).permutation(n)
    kept: list[int] = []
    tree: cKDTree | None = None
    min_d = float(min_dist_m)
    for i in order.tolist():
        if tree is None or tree.n == 0:
            kept.append(i)
            tree = cKDTree(np.column_stack([x[kept], y[kept]]))
            continue
        dist, _ = tree.query([[x[i], y[i]]], k=1, distance_upper_bound=min_d)
        if float(dist[0]) > min_d:
            kept.append(i)
            tree = cKDTree(np.column_stack([x[kept], y[kept]]))
    return np.asarray(kept, dtype=np.int64)


def block_ids_grid(x: np.ndarray, y: np.ndarray, block_size_m: float) -> np.ndarray:
    bs = float(block_size_m)
    bx = np.floor(x / bs).astype(np.int64)
    by = np.floor(y / bs).astype(np.int64)
    bx -= int(np.min(bx))
    by -= int(np.min(by))
    return bx * (int(np.max(by)) + 1) + by


def block_ids_connected_components(
    lon: np.ndarray,
    lat: np.ndarray,
    radius_m: float,
) -> np.ndarray:
    if not _HAS_SCIPY:
        raise ImportError("spatial components mode requires scipy")
    x, y = lonlat_to_xy_m(lon, lat)
    n = int(lon.shape[0])
    tree = cKDTree(np.column_stack([x, y]))
    pairs = tree.query_pairs(r=float(radius_m), output_type="ndarray")
    if pairs is None or len(pairs) == 0:
        return np.arange(n, dtype=np.int64)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    data = np.ones(rows.shape[0], dtype=np.uint8)
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    _, labels = connected_components(graph, directed=False, return_labels=True)
    return labels.astype(np.int64)


def compute_block_ids(
    lon: np.ndarray,
    lat: np.ndarray,
    block_size_m: float,
    mode: Literal["grid", "components"] = "grid",
) -> np.ndarray:
    if mode == "components":
        return block_ids_connected_components(lon, lat, block_size_m)
    x, y = lonlat_to_xy_m(lon, lat)
    return block_ids_grid(x, y, block_size_m)


def split_train_val_test_211_spatial(
    y: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    seed: int = 42,
    max_train_per_class: int = 300,
    block_size_m: float = 1280.0,
    spatial_mode: Literal["grid", "components"] = "grid",
    proximity_filter: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    AEF-style spatial hold-out + existing 2:1:1 protocol:

    1. Optional ~1.28 km proximity filtering;
    2. Assign spatial blocks (grid or components) to train/val/test at ~2:1:1;
    3. Within train blocks: keep all pixels if ``max_train_per_class <= 0``;
       otherwise subsample to at most ``max_train_per_class`` per class;
    4. val/test use all pixels in their assigned blocks.
    """
    y = np.asarray(y, dtype=np.int64)
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    if y.shape[0] != lon.shape[0] or y.shape[0] != lat.shape[0]:
        raise ValueError("y, lon, lat must have the same length")

    rng = np.random.default_rng(int(seed))
    pool_idx = np.arange(y.shape[0], dtype=np.int64)
    if proximity_filter:
        pool_idx = spatial_proximity_filter_indices(
            lon, lat, float(block_size_m), seed=int(seed)
        )

    y_p = y[pool_idx]
    lon_p = lon[pool_idx]
    lat_p = lat[pool_idx]
    block_id = compute_block_ids(lon_p, lat_p, float(block_size_m), spatial_mode)

    unique_blocks = np.unique(block_id)
    rng.shuffle(unique_blocks)
    n_b = int(unique_blocks.size)
    if n_b < 3:
        raise ValueError(
            f"Only {n_b} spatial block(s) after filtering; need >=3 for train/val/test."
        )

    n_train_b = max(1, n_b // 2)
    remaining = n_b - n_train_b
    n_val_b = max(1, remaining // 2)
    n_test_b = remaining - n_val_b

    train_blocks = set(unique_blocks[:n_train_b].tolist())
    val_blocks = set(unique_blocks[n_train_b : n_train_b + n_val_b].tolist())
    test_blocks = set(unique_blocks[n_train_b + n_val_b :].tolist())

    in_train = np.isin(block_id, list(train_blocks))
    in_val = np.isin(block_id, list(val_blocks))
    in_test = np.isin(block_id, list(test_blocks))

    pool_train = np.where(in_train)[0]
    pool_val = np.where(in_val)[0]
    pool_test = np.where(in_test)[0]

    idx_train_sub: list[int] = []
    for c in np.unique(y_p):
        idx_c = pool_train[y_p[pool_train] == c]
        if idx_c.size == 0:
            continue
        idx_c = idx_c.copy()
        rng.shuffle(idx_c)
        if int(max_train_per_class) <= 0:
            take = int(idx_c.size)
        else:
            take = min(int(max_train_per_class), int(idx_c.size))
        idx_train_sub.extend(idx_c[:take].tolist())

    if not idx_train_sub:
        raise ValueError("Empty train split after spatial block assignment.")
    if pool_val.size == 0 or pool_test.size == 0:
        raise ValueError("Empty val or test split after spatial block assignment.")

    idx_train = pool_idx[np.asarray(idx_train_sub, dtype=np.int64)]
    idx_val = pool_idx[pool_val]
    idx_test = pool_idx[pool_test]
    return idx_train, idx_val, idx_test


def get_train_val_test_indices(
    y: np.ndarray,
    lon: np.ndarray | None,
    lat: np.ndarray | None,
    args: Namespace,
    *,
    split_fn_random,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use spatial split when ``args.spatial_block_m`` is set, else ``split_fn_random(y, ...)``."""
    if args.spatial_block_m is not None and float(args.spatial_block_m) > 0:
        if lon is None or lat is None:
            raise ValueError("spatial split requires lon/lat arrays")
        return split_train_val_test_211_spatial(
            y,
            lon,
            lat,
            seed=int(args.split_seed),
            max_train_per_class=int(args.max_train_per_class),
            block_size_m=float(args.spatial_block_m),
            spatial_mode=str(args.spatial_split_mode),
            proximity_filter=bool(args.spatial_proximity_filter),
        )
    return split_fn_random(
        y,
        seed=int(args.split_seed),
        max_train_per_class=int(args.max_train_per_class),
    )
