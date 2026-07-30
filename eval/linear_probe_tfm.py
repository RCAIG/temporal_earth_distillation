"""TerraFM-style linear probe: optional StandardScaler + nn.Linear + SGD, LR grid on val."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# TerraFM / DINOv2 §5.2 learning-rate sweep (subset ok for tabular embeddings)
DEFAULT_LR_GRID: tuple[float, ...] = (
    1e-5,
    2e-5,
    5e-5,
    1e-4,
    2e-4,
    5e-4,
    1e-3,
    2e-3,
    5e-3,
    1e-2,
    2e-2,
    5e-2,
    1.0,
)


class _LinearHead(nn.Module):
    def __init__(self, d_in: int, n_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_in, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@torch.no_grad()
def _predict(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    ds = TensorDataset(torch.from_numpy(X.astype(np.float32, copy=False)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    outs: list[np.ndarray] = []
    for (xb,) in loader:
        logits = model(xb.to(device))
        outs.append(logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(outs, axis=0)


def _train_epochs(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    device: torch.device,
    lr: float,
    epochs: int,
    batch_size: int,
    weight_decay: float,
    momentum: float,
    seed: int = 42,
) -> None:
    model.train()
    ds = TensorDataset(
        torch.from_numpy(X.astype(np.float32, copy=False)),
        torch.from_numpy(y.astype(np.int64, copy=False)),
    )
    g = torch.Generator()
    g.manual_seed(int(seed))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False, generator=g)
    opt = torch.optim.SGD(
        model.parameters(),
        lr=float(lr),
        momentum=float(momentum),
        weight_decay=float(weight_decay),
    )
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(int(epochs)):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss_fn(model(xb), yb).backward()
            opt.step()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "oa": float(accuracy_score(y_true, y_pred)),
        "ba": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _prepare_splits(
    X: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    *,
    use_feature_scaler: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler | None]:
    if use_feature_scaler:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[idx_train])
        X_va = scaler.transform(X[idx_val])
        X_te = scaler.transform(X[idx_test])
        return X_tr, X_va, X_te, scaler
    X_tr = X[idx_train].astype(np.float32, copy=False)
    X_va = X[idx_val].astype(np.float32, copy=False)
    X_te = X[idx_test].astype(np.float32, copy=False)
    return X_tr, X_va, X_te, None


def linear_probe_tfm(
    X: np.ndarray,
    y: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    *,
    device: torch.device,
    lr_grid: tuple[float, ...] = DEFAULT_LR_GRID,
    epochs: int = 50,
    batch_size: int = 256,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    retrain_on_train_val: bool = False,
    use_feature_scaler: bool = False,
    seed: int = 42,
) -> dict:
    """
    1) Optional global ``StandardScaler`` on train (default off; TED CLS is already LayerNorm-ed).
    2) For each LR: SGD train linear head on train; pick best **val BA**.
    3) Load best train checkpoint (default); optional retrain on train+val if enabled.
    4) Evaluate on test.
    """
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    X_tr, X_va, X_te, scaler = _prepare_splits(
        X, idx_train, idx_val, idx_test, use_feature_scaler=use_feature_scaler
    )
    y_tr, y_va, y_te = y[idx_train], y[idx_val], y[idx_test]

    n_classes = int(y.max()) + 1
    d_in = int(X.shape[1])

    best_val_ba = -1.0
    best_lr = float(lr_grid[0])
    best_state: dict | None = None

    for lr in lr_grid:
        model = _LinearHead(d_in, n_classes).to(device)
        _train_epochs(
            model,
            X_tr,
            y_tr,
            device=device,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            weight_decay=weight_decay,
            momentum=momentum,
            seed=seed,
        )
        pred_va = _predict(model, X_va, device, batch_size=batch_size)
        val_ba = float(balanced_accuracy_score(y_va, pred_va))
        if val_ba > best_val_ba:
            best_val_ba = val_ba
            best_lr = float(lr)
            best_state = deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("linear_probe_tfm: no LR trial succeeded")

    if retrain_on_train_val:
        X_tv_raw = np.concatenate([X[idx_train], X[idx_val]], axis=0)
        if use_feature_scaler:
            assert scaler is not None
            X_tv = scaler.fit_transform(X_tv_raw)
        else:
            X_tv = X_tv_raw.astype(np.float32, copy=False)
        y_tv = np.concatenate([y[idx_train], y[idx_val]], axis=0)
        model = _LinearHead(d_in, n_classes).to(device)
        _train_epochs(
            model,
            X_tv,
            y_tv,
            device=device,
            lr=best_lr,
            epochs=epochs,
            batch_size=batch_size,
            weight_decay=weight_decay,
            momentum=momentum,
            seed=seed,
        )
    else:
        model = _LinearHead(d_in, n_classes).to(device)
        model.load_state_dict(best_state)

    pred_te = _predict(model, X_te, device, batch_size=batch_size)
    out = _metrics(y_te, pred_te)
    out["val_ba"] = float(best_val_ba)
    out["best_lr"] = float(best_lr)
    out["feature_scaler"] = bool(use_feature_scaler)
    return out


def linear_probe_tfm_prebuilt(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    device: torch.device,
    lr_grid: tuple[float, ...] = DEFAULT_LR_GRID,
    epochs: int = 50,
    batch_size: int = 256,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    retrain_on_train_val: bool = False,
    use_feature_scaler: bool = False,
    seed: int = 42,
) -> dict:
    """Features already split (e.g. per-block z-score); no extra global scaler by default."""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if use_feature_scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        scaler_tv: StandardScaler | None = scaler
    else:
        scaler_tv = None
    n_classes = int(max(y_train.max(), y_val.max(), y_test.max())) + 1
    d_in = int(X_train.shape[1])
    best_val_ba = -1.0
    best_lr = float(lr_grid[0])
    best_state: dict | None = None

    for lr in lr_grid:
        model = _LinearHead(d_in, n_classes).to(device)
        _train_epochs(
            model,
            X_train,
            y_train,
            device=device,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            weight_decay=weight_decay,
            momentum=momentum,
            seed=seed,
        )
        pred_va = _predict(model, X_val, device, batch_size=batch_size)
        val_ba = float(balanced_accuracy_score(y_val, pred_va))
        if val_ba > best_val_ba:
            best_val_ba = val_ba
            best_lr = float(lr)
            best_state = deepcopy(model.state_dict())

    if retrain_on_train_val:
        X_tv = np.concatenate([X_train, X_val], axis=0)
        if use_feature_scaler and scaler_tv is not None:
            X_tv = scaler_tv.fit_transform(X_tv)
        y_tv = np.concatenate([y_train, y_val], axis=0)
        model = _LinearHead(d_in, n_classes).to(device)
        _train_epochs(
            model,
            X_tv,
            y_tv,
            device=device,
            lr=best_lr,
            epochs=epochs,
            batch_size=batch_size,
            weight_decay=weight_decay,
            momentum=momentum,
            seed=seed,
        )
    else:
        model = _LinearHead(d_in, n_classes).to(device)
        assert best_state is not None
        model.load_state_dict(best_state)

    pred_te = _predict(model, X_test, device, batch_size=batch_size)
    out = _metrics(y_test, pred_te)
    out["val_ba"] = float(best_val_ba)
    out["best_lr"] = float(best_lr)
    out["feature_scaler"] = bool(use_feature_scaler)
    return out
