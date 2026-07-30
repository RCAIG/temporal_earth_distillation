"""Downstream classification splits shared by probe / kNN scripts.

Random per-class 2:1:1: ``split_train_val_test_211``.
AEF-style spatial blocks (e.g. 1.28 km): see ``downstream_geo.split_train_val_test_211_spatial``.
"""

from __future__ import annotations

import numpy as np


def split_train_val_test_211(
    y: np.ndarray,
    *,
    seed: int = 42,
    max_train_per_class: int = 300,
    min_val_per_class: int = 1,
    min_test_per_class: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-class **train : val : test ≈ 2 : 1 : 1** (50% / 25% / 25%).

    - ``train`` capped at ``max_train_per_class`` (default 300); ``<= 0`` means no cap (use full per-class ``n//2``).
    - Each class has at least ``min_val_per_class`` / ``min_test_per_class`` in val/test
      when ``n_c`` is large enough (use ``min_samples_per_class >= 4`` globally).
  """
    rng = np.random.default_rng(int(seed))
    idx_train: list[int] = []
    idx_val: list[int] = []
    idx_test: list[int] = []

    for c in np.unique(y):
        idx_c = np.where(y == c)[0].copy()
        rng.shuffle(idx_c)
        n = int(idx_c.size)
        if n < min_val_per_class + min_test_per_class + 1:
            raise ValueError(
                f"class {c}: n={n} too small for 2:1:1 with min_val={min_val_per_class}, "
                f"min_test={min_test_per_class} (need at least "
                f"{min_val_per_class + min_test_per_class + 1})"
            )

        if int(max_train_per_class) <= 0:
            n_train = max(1, n // 2)
        else:
            n_train = min(int(max_train_per_class), max(1, n // 2))
        remaining = n - n_train
        n_val = max(int(min_val_per_class), remaining // 2)
        n_test = remaining - n_val

        if n_test < int(min_test_per_class):
            need = int(min_test_per_class) - n_test
            take = min(need, n_val - int(min_val_per_class))
            n_val -= take
            n_test += take
        if n_test < int(min_test_per_class) and n_train > 1:
            need = int(min_test_per_class) - n_test
            take = min(need, n_train - 1)
            n_train -= take
            n_test += take
        if n_val < int(min_val_per_class) and n_train > 1:
            need = int(min_val_per_class) - n_val
            take = min(need, n_train - 1)
            n_train -= take
            n_val += take

        assert n_train + n_val + n_test == n
        idx_train.extend(idx_c[:n_train].tolist())
        idx_val.extend(idx_c[n_train : n_train + n_val].tolist())
        idx_test.extend(idx_c[n_train + n_val :].tolist())

    return (
        np.asarray(idx_train, dtype=np.int64),
        np.asarray(idx_val, dtype=np.int64),
        np.asarray(idx_test, dtype=np.int64),
    )
