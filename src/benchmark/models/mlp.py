from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.benchmark.utils.precision import autocast_context


def load_split_arrays(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split_dir = data_dir / split
    x = np.load(split_dir / "x.npy")
    y = np.load(split_dir / "y.npy")
    md = np.load(split_dir / "md.npy", allow_pickle=True)
    y_mask = np.load(split_dir / "y_mask.npy")
    return x, y, md, y_mask


class MLPDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> None:
        self.x = torch.from_numpy(np.ascontiguousarray(x.astype(np.float32, copy=False)))
        self.y = torch.from_numpy(np.ascontiguousarray(y.astype(np.float32, copy=False)))
        self.mask = torch.from_numpy(np.ascontiguousarray(mask.astype(bool, copy=True)))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx], self.mask[idx]


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d_in = input_dim
        for d_out in hidden_dims:
            layers.extend([nn.Linear(d_in, d_out), nn.ReLU(), nn.Dropout(dropout)])
            d_in = d_out
        layers.append(nn.Linear(d_in, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def predict_in_batches(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device)
            yb = model(xb).cpu().numpy()
            preds.append(yb)
    return np.vstack(preds)


def split_train_arrays(
    *,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    md: np.ndarray,
    val_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if val_fraction <= 0.0:
        return (
            x,
            y,
            mask,
            x[:0],
            y[:0],
            mask[:0],
        )
    snapshot_ts = pd.DatetimeIndex(pd.to_datetime(md[:, 0]))
    unique_snapshots = np.sort(snapshot_ts.unique())
    n_val = max(1, int(round(len(unique_snapshots) * val_fraction)))
    n_val = min(n_val, len(unique_snapshots) - 1)
    val_snapshots = set(unique_snapshots[-n_val:])
    val_mask = np.asarray(snapshot_ts.isin(val_snapshots), dtype=bool)
    train_mask = ~val_mask
    return (
        x[train_mask],
        y[train_mask],
        mask[train_mask],
        x[val_mask],
        y[val_mask],
        mask[val_mask],
    )


@torch.no_grad()
def evaluate_val_set(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    target_use_sqrt: torch.Tensor,
    precision: str,
) -> tuple[float, float]:
    model.eval()
    loss_running = 0.0
    n_batches = 0
    abs_err_sum = 0.0
    n_values = 0
    for xb, yb, mb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        mb = mb.to(device, non_blocking=True)
        if not torch.any(mb):
            continue
        with autocast_context(device=device, precision=precision):
            pred = model(xb)
            loss = criterion(pred[mb], yb[mb])
        loss_running += float(loss.item())
        n_batches += 1
        pred_t = pred * target_std + target_mean
        true_t = yb * target_std + target_mean
        pred_sec = torch.where(
            target_use_sqrt,
            torch.sign(pred_t) * torch.square(torch.abs(pred_t)),
            pred_t,
        )
        true_sec = torch.where(
            target_use_sqrt,
            torch.sign(true_t) * torch.square(torch.abs(true_t)),
            true_t,
        )
        abs_err_sum += float(torch.abs(pred_sec[mb] - true_sec[mb]).sum().item())
        n_values += int(mb.sum().item())
    return loss_running / max(n_batches, 1), abs_err_sum / max(n_values, 1)


def build_prediction_eval_table(
    *,
    md: np.ndarray,
    y_pred: np.ndarray,
    y_columns: list[str],
    eval_table: pd.DataFrame,
    eval_target_columns: list[str],
    target_stats: dict[str, tuple[float, float, bool]],
) -> pd.DataFrame:
    md_df = pd.DataFrame(md, columns=["snapshot_ts", "train_id", "service_date"]).rename(
        columns={"snapshot_ts": "ts"}
    )
    pred_df = pd.DataFrame(y_pred, columns=y_columns)
    baseline = eval_table[["ts", "train_id", "service_date", "last_known_delay"]]
    out = md_df.merge(baseline, on=["ts", "train_id", "service_date"], how="inner")
    pred_seconds: dict[str, np.ndarray] = {}
    for y_col, eval_col in zip(y_columns, eval_target_columns):
        mean_t, std_t, use_sqrt = target_stats[y_col]
        t_hat = pred_df[y_col].to_numpy(dtype=np.float64) * std_t + mean_t
        if use_sqrt:
            delta_hat = np.sign(t_hat) * np.square(np.abs(t_hat))
        else:
            delta_hat = t_hat
        pred_seconds[eval_col] = delta_hat + out["last_known_delay"].to_numpy(dtype=np.float64)
    for eval_col in eval_target_columns:
        out[eval_col] = pred_seconds[eval_col]
    return out[["ts", "train_id", "service_date"] + eval_target_columns]
