"""TerraFM-style kNN: select k on validation BA, fit on train, report test."""

from __future__ import annotations

import os

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

# TerraFM §5.2
DEFAULT_K_GRID: tuple[int, ...] = (3, 5, 7, 10, 15, 20, 30, 50, 100)


def eval_knn_tfm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    k_grid: tuple[int, ...] = DEFAULT_K_GRID,
) -> dict:
    best_val_ba = -1.0
    best_k = 1
    n_train = int(X_train.shape[0])

    for k in k_grid:
        k = int(k)
        if k < 1 or k > n_train:
            continue
        knn = KNeighborsClassifier(
            n_neighbors=k,
            weights="uniform",
            n_jobs=min(8, os.cpu_count() or 4),
        )
        knn.fit(X_train, y_train)
        pred_va = knn.predict(X_val)
        val_ba = float(balanced_accuracy_score(y_val, pred_va))
        if val_ba > best_val_ba:
            best_val_ba = val_ba
            best_k = k

    knn = KNeighborsClassifier(
        n_neighbors=best_k,
        weights="uniform",
        n_jobs=min(8, os.cpu_count() or 4),
    )
    knn.fit(X_train, y_train)
    pred_te = knn.predict(X_test)
    return {
        "oa": float(accuracy_score(y_test, pred_te)),
        "ba": float(balanced_accuracy_score(y_test, pred_te)),
        "f1_macro": float(f1_score(y_test, pred_te, average="macro", zero_division=0)),
        "val_ba": float(best_val_ba),
        "best_k": int(best_k),
    }
