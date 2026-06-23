"""Standalone training script for S2 tagging with masked pooling plus p_main hit/miss diagnostics.

This version trains a masked set-context model and writes event-level diagnostic CSVs
showing where the model correctly identifies the p_main cluster and where it misses.

Main diagnostic definition:
    main_correct = argmax(predicted p_main over clusters in an event)
                   == argmax(true p_main over clusters in that event)

Notes:
- Features used for training are z-scored.
- Diagnostics include both z-scored feature summaries and raw physical-unit feature summaries.
- The cache format includes raw and scaled arrays. If an older cache is present, it is ignored
  and rebuilt automatically.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import wandb
except Exception:
    wandb = None


# -----------------------------
# User-editable parameters
# -----------------------------

RAW_DATA_PATH = "/home/adeeps/projects/xenon_deepset_nn/data/s2_tag_training_clusters.npy"
CACHE_FILE = "pooling_processed_data_v2.npz"
CHECKPOINT_PATH = "best_pooling_model.pt"

USE_WANDB = True
WANDB_PROJECT = "xenon-pooling"
WANDB_ENTITY = None
WANDB_RUN_NAME = None

LATENT_DIM = 256
PHI_HIDDEN = 512
HEAD_HIDDEN = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
BATCH_SIZE = 512
MAX_EPOCHS = 500
EARLY_STOPPING_PATIENCE = 100
NUM_WORKERS = 4
PIN_MEMORY = True
SHUFFLE_TRAIN = True

SCHEDULER = "reduce_on_plateau"  # "reduce_on_plateau", "cosine", or "none"
SCHEDULER_PATIENCE = 8
SCHEDULER_FACTOR = 0.5
SCHEDULER_MIN_LR = 1e-6
SCHEDULER_T_MAX = 40

RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURES = [
    "x",
    "y",
    "n_electrons_interface",
    "drift_time_mean",
    "drift_time_spread",
]
TARGETS = ["p_main", "p_alt"]
EVENT_COL = "event_number"
TOP13_NS = 192_600


# -----------------------------
# Reproducibility
# -----------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# -----------------------------
# Data prep
# -----------------------------

def build_event_groups(event_ids_array: np.ndarray) -> List[np.ndarray]:
    """Return index arrays, one per event, grouped by event_number."""
    order = np.argsort(event_ids_array, kind="mergesort")
    sorted_events = event_ids_array[order]
    boundaries = np.flatnonzero(sorted_events[1:] != sorted_events[:-1]) + 1
    return list(np.split(order, boundaries))


def _cache_has_required_keys(cached: np.lib.npyio.NpzFile) -> bool:
    required = {
        "X_train",
        "X_train_raw",
        "Y_train",
        "E_train",
        "train_groups",
        "X_val",
        "X_val_raw",
        "Y_val",
        "E_val",
        "val_groups",
        "X_test",
        "X_test_raw",
        "Y_test",
        "E_test",
        "test_groups",
        "scaler_mean",
        "scaler_scale",
    }
    return required.issubset(set(cached.files))


def prepare_data() -> Tuple[np.ndarray, ...]:
    """Load cached processed arrays, or build them once and cache them."""
    if os.path.exists(CACHE_FILE):
        cached = np.load(CACHE_FILE, allow_pickle=True)
        if _cache_has_required_keys(cached):
            X_train = cached["X_train"]
            X_train_raw = cached["X_train_raw"]
            Y_train = cached["Y_train"]
            E_train = cached["E_train"]
            train_groups = [np.asarray(g, dtype=np.int64) for g in cached["train_groups"]]

            X_val = cached["X_val"]
            X_val_raw = cached["X_val_raw"]
            Y_val = cached["Y_val"]
            E_val = cached["E_val"]
            val_groups = [np.asarray(g, dtype=np.int64) for g in cached["val_groups"]]

            X_test = cached["X_test"]
            X_test_raw = cached["X_test_raw"]
            Y_test = cached["Y_test"]
            E_test = cached["E_test"]
            test_groups = [np.asarray(g, dtype=np.int64) for g in cached["test_groups"]]

            return (
                X_train,
                X_train_raw,
                Y_train,
                E_train,
                train_groups,
                X_val,
                X_val_raw,
                Y_val,
                E_val,
                val_groups,
                X_test,
                X_test_raw,
                Y_test,
                E_test,
                test_groups,
            )

        print(
            f"Found {CACHE_FILE}, but it does not contain the v2 diagnostic cache keys. "
            "Rebuilding cache."
        )

    data = np.load(RAW_DATA_PATH)
    df = pd.DataFrame(data)

    # Remove events with very short drift-time clusters, matching the existing cleanup.
    event_min_drift = df.groupby(EVENT_COL)["drift_time_mean"].min()
    bad_event_ids = event_min_drift[event_min_drift < TOP13_NS].index.to_numpy()
    df = df[~df[EVENT_COL].isin(bad_event_ids)].copy()

    # Brief says p_alt values above 1 are a small simulator artifact. Clip and proceed.
    df["p_alt"] = df["p_alt"].clip(0, 1)

    event_ids = df[EVENT_COL].unique()

    train_events, temp_events = train_test_split(
        event_ids,
        test_size=0.30,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    val_events, test_events = train_test_split(
        temp_events,
        test_size=0.50,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    train_df = df[df[EVENT_COL].isin(train_events)].copy()
    val_df = df[df[EVENT_COL].isin(val_events)].copy()
    test_df = df[df[EVENT_COL].isin(test_events)].copy()

    for d in [train_df, val_df, test_df]:
        d[FEATURES] = d[FEATURES].astype(np.float32)

    # Preserve raw physical-unit features for diagnostics.
    X_train_raw = train_df[FEATURES].to_numpy(dtype=np.float32, copy=True)
    X_val_raw = val_df[FEATURES].to_numpy(dtype=np.float32, copy=True)
    X_test_raw = test_df[FEATURES].to_numpy(dtype=np.float32, copy=True)

    scaler = StandardScaler()
    scaler.fit(train_df[FEATURES])

    train_df[FEATURES] = scaler.transform(train_df[FEATURES]).astype(np.float32)
    val_df[FEATURES] = scaler.transform(val_df[FEATURES]).astype(np.float32)
    test_df[FEATURES] = scaler.transform(test_df[FEATURES]).astype(np.float32)

    X_train = train_df[FEATURES].to_numpy(dtype=np.float32, copy=True)
    Y_train = train_df[TARGETS].to_numpy(dtype=np.float32, copy=True)
    E_train = train_df[EVENT_COL].to_numpy(copy=True)

    X_val = val_df[FEATURES].to_numpy(dtype=np.float32, copy=True)
    Y_val = val_df[TARGETS].to_numpy(dtype=np.float32, copy=True)
    E_val = val_df[EVENT_COL].to_numpy(copy=True)

    X_test = test_df[FEATURES].to_numpy(dtype=np.float32, copy=True)
    Y_test = test_df[TARGETS].to_numpy(dtype=np.float32, copy=True)
    E_test = test_df[EVENT_COL].to_numpy(copy=True)

    train_groups = build_event_groups(E_train)
    val_groups = build_event_groups(E_val)
    test_groups = build_event_groups(E_test)

    np.savez_compressed(
        CACHE_FILE,
        X_train=X_train,
        X_train_raw=X_train_raw,
        Y_train=Y_train,
        E_train=E_train,
        train_groups=np.array(train_groups, dtype=object),
        X_val=X_val,
        X_val_raw=X_val_raw,
        Y_val=Y_val,
        E_val=E_val,
        val_groups=np.array(val_groups, dtype=object),
        X_test=X_test,
        X_test_raw=X_test_raw,
        Y_test=Y_test,
        E_test=E_test,
        test_groups=np.array(test_groups, dtype=object),
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
    )

    return (
        X_train,
        X_train_raw,
        Y_train,
        E_train,
        train_groups,
        X_val,
        X_val_raw,
        Y_val,
        E_val,
        val_groups,
        X_test,
        X_test_raw,
        Y_test,
        E_test,
        test_groups,
    )


class S2EventDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        X_raw: np.ndarray,
        Y: np.ndarray,
        E: np.ndarray,
        event_groups: List[np.ndarray],
    ):
        self.X = X
        self.X_raw = X_raw
        self.Y = Y
        self.E = E
        self.event_groups = event_groups

    def __len__(self) -> int:
        return len(self.event_groups)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rows = self.event_groups[idx]
        x = self.X[rows]
        x_raw = self.X_raw[rows]
        y = self.Y[rows]
        event_id = self.E[rows][0]

        return {
            "x": torch.from_numpy(x.copy()),
            "x_raw": torch.from_numpy(x_raw.copy()),
            "y": torch.from_numpy(y.copy()),
            "event_id": torch.tensor(event_id, dtype=torch.long),
            "n_clusters": len(rows),
        }


def collate_events(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_clusters = max(int(item["n_clusters"]) for item in batch)

    x_dim = batch[0]["x"].shape[1]
    y_dim = batch[0]["y"].shape[1]

    x_padded = torch.zeros(batch_size, max_clusters, x_dim, dtype=torch.float32)
    x_raw_padded = torch.zeros(batch_size, max_clusters, x_dim, dtype=torch.float32)
    y_padded = torch.zeros(batch_size, max_clusters, y_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_clusters, dtype=torch.bool)
    event_ids = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        n = int(item["n_clusters"])
        x_padded[i, :n] = item["x"]
        x_raw_padded[i, :n] = item["x_raw"]
        y_padded[i, :n] = item["y"]
        mask[i, :n] = True
        event_ids[i] = item["event_id"]

    return {
        "x": x_padded,
        "x_raw": x_raw_padded,
        "y": y_padded,
        "mask": mask,
        "event_id": event_ids,
    }


# -----------------------------
# Model
# -----------------------------

class MaskedPoolingNet(nn.Module):
    """Per-cluster encoder + masked mean/max pooling + per-cluster head."""

    def __init__(
        self,
        input_dim: int = 5,
        latent_dim: int = 64,
        phi_hidden: int = 64,
        head_hidden: int = 64,
        output_dim: int = 2,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, phi_hidden),
            nn.ReLU(),
            nn.Linear(phi_hidden, latent_dim),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(latent_dim * 3, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, output_dim),
        )

    @staticmethod
    def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).float()
        summed = (x * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return summed / denom

    @staticmethod
    def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1)
        x_masked = x.masked_fill(~mask_f, float("-inf"))
        pooled = x_masked.max(dim=1).values
        return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, K, D], mask: [B, K]
        h = self.encoder(x)  # [B, K, L]
        pooled_mean = self.masked_mean(h, mask)  # [B, L]
        pooled_max = self.masked_max(h, mask)    # [B, L]

        context = torch.cat([pooled_mean, pooled_max], dim=-1)  # [B, 2L]
        context_expanded = context.unsqueeze(1).expand(-1, x.shape[1], -1)  # [B, K, 2L]

        out = self.head(torch.cat([h, context_expanded], dim=-1))  # [B, K, 2]
        return out


# -----------------------------
# Loss and metrics
# -----------------------------

def masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss_per_entry = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )
    mask_f = mask.unsqueeze(-1).float()
    loss_per_entry = loss_per_entry * mask_f
    denom = mask_f.sum() * logits.shape[-1]
    return loss_per_entry.sum() / denom.clamp_min(1.0)


def compute_soft_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["mse"] = float(((y_prob - y_true) ** 2).mean())

    for i, name in enumerate(TARGETS):
        err = np.abs(y_true[:, i] - y_prob[:, i])
        sq_err = (y_true[:, i] - y_prob[:, i]) ** 2
        metrics[f"{name}_mae"] = float(err.mean())
        metrics[f"{name}_brier"] = float(sq_err.mean())

        y_bin = (y_true[:, i] > 0.5).astype(np.int32)
        try:
            metrics[f"{name}_auc"] = float(roc_auc_score(y_bin, y_prob[:, i]))
        except ValueError:
            metrics[f"{name}_auc"] = float("nan")

    metrics["mean_mae"] = 0.5 * (metrics["p_main_mae"] + metrics["p_alt_mae"])
    metrics["mean_brier"] = 0.5 * (metrics["p_main_brier"] + metrics["p_alt_brier"])
    metrics["mean_auc"] = 0.5 * (metrics["p_main_auc"] + metrics["p_alt_auc"])
    return metrics


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer] = None,
    collect_metrics: bool = False,
) -> Tuple[float, Dict[str, float]]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    n_batches = 0
    all_targets: List[torch.Tensor] = []
    all_probs: List[torch.Tensor] = []

    event_correct = 0
    event_total = 0

    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for batch in loader:
            x = batch["x"].to(DEVICE, non_blocking=True)
            y = batch["y"].to(DEVICE, non_blocking=True)
            mask = batch["mask"].to(DEVICE, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(x, mask)
            loss = masked_bce_loss(logits, y, mask)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

            if collect_metrics:
                probs = torch.sigmoid(logits)

                flat_mask = mask.view(-1)
                flat_y = y.view(-1, 2)[flat_mask]
                flat_p = probs.view(-1, 2)[flat_mask]

                all_targets.append(flat_y.detach().cpu())
                all_probs.append(flat_p.detach().cpu())

                for b in range(x.shape[0]):
                    valid = mask[b]
                    if valid.sum().item() == 0:
                        continue

                    pred_main = probs[b, valid, 0]
                    true_main = y[b, valid, 0]

                    event_correct += int(
                        torch.argmax(pred_main).item() == torch.argmax(true_main).item()
                    )
                    event_total += 1

    avg_loss = total_loss / max(n_batches, 1)
    metrics: Dict[str, float] = {}

    if collect_metrics and len(all_targets) > 0:
        all_targets_np = torch.cat(all_targets, dim=0).numpy()
        all_probs_np = torch.cat(all_probs, dim=0).numpy()
        metrics.update(compute_soft_metrics(all_targets_np, all_probs_np))
        metrics["event_main_accuracy"] = float(event_correct / max(event_total, 1))

    return avg_loss, metrics


# -----------------------------
# Event-level diagnostics
# -----------------------------

def collect_main_event_diagnostics(
    model: nn.Module,
    loader: DataLoader,
    split_name: str = "val",
) -> pd.DataFrame:
    """
    Build one row per event, comparing events where the model correctly identifies
    the p_main cluster vs events where it misses.

    Correct means:
        argmax(predicted p_main over clusters) == argmax(true p_main over clusters)
    """
    model.eval()
    rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(DEVICE, non_blocking=True)
            y = batch["y"].to(DEVICE, non_blocking=True)
            mask = batch["mask"].to(DEVICE, non_blocking=True)

            logits = model(x, mask)
            probs = torch.sigmoid(logits)

            x_cpu = x.detach().cpu()
            x_raw_cpu = batch["x_raw"].detach().cpu()
            y_cpu = y.detach().cpu()
            p_cpu = probs.detach().cpu()
            mask_cpu = mask.detach().cpu()
            event_ids_cpu = batch["event_id"].detach().cpu()

            for b in range(x_cpu.shape[0]):
                valid = mask_cpu[b]
                n = int(valid.sum().item())
                if n == 0:
                    continue

                xb = x_cpu[b, valid]              # z-scored features, [n_clusters, n_features]
                xb_raw = x_raw_cpu[b, valid]      # raw physical features, [n_clusters, n_features]
                yb = y_cpu[b, valid]              # [n_clusters, 2]
                pb = p_cpu[b, valid]              # [n_clusters, 2]

                true_main = yb[:, 0]
                pred_main = pb[:, 0]
                true_alt = yb[:, 1]
                pred_alt = pb[:, 1]

                true_main_idx = int(torch.argmax(true_main).item())
                pred_main_idx = int(torch.argmax(pred_main).item())
                main_correct = pred_main_idx == true_main_idx

                true_alt_idx = int(torch.argmax(true_alt).item())
                pred_alt_idx = int(torch.argmax(pred_alt).item())
                alt_correct = pred_alt_idx == true_alt_idx

                if n >= 2:
                    pred_main_top2 = torch.topk(pred_main, k=2).values
                    true_main_top2 = torch.topk(true_main, k=2).values
                    pred_alt_top2 = torch.topk(pred_alt, k=2).values
                    true_alt_top2 = torch.topk(true_alt, k=2).values

                    pred_main_margin = float(pred_main_top2[0] - pred_main_top2[1])
                    true_main_margin = float(true_main_top2[0] - true_main_top2[1])
                    pred_alt_margin = float(pred_alt_top2[0] - pred_alt_top2[1])
                    true_alt_margin = float(true_alt_top2[0] - true_alt_top2[1])
                else:
                    pred_main_margin = float("nan")
                    true_main_margin = float("nan")
                    pred_alt_margin = float("nan")
                    true_alt_margin = float("nan")

                # Useful electron-rank diagnostic: did the true main have the most electrons?
                ne_col = FEATURES.index("n_electrons_interface")
                ne_raw = xb_raw[:, ne_col]
                electron_rank_order = torch.argsort(ne_raw, descending=True)
                true_main_electron_rank = int((electron_rank_order == true_main_idx).nonzero()[0].item() + 1)
                pred_main_electron_rank = int((electron_rank_order == pred_main_idx).nonzero()[0].item() + 1)

                row: Dict[str, Any] = {
                    "split": split_name,
                    "event_id": int(event_ids_cpu[b].item()),
                    "n_clusters": n,

                    "main_correct": bool(main_correct),
                    "alt_correct": bool(alt_correct),
                    "main_and_alt_correct": bool(main_correct and alt_correct),

                    "true_main_idx": true_main_idx,
                    "pred_main_idx": pred_main_idx,
                    "true_alt_idx": true_alt_idx,
                    "pred_alt_idx": pred_alt_idx,

                    "true_main_value_at_true": float(true_main[true_main_idx]),
                    "pred_main_value_at_true": float(pred_main[true_main_idx]),
                    "pred_main_value_at_pred": float(pred_main[pred_main_idx]),
                    "true_main_value_at_pred": float(true_main[pred_main_idx]),

                    "true_alt_value_at_true": float(true_alt[true_alt_idx]),
                    "pred_alt_value_at_true": float(pred_alt[true_alt_idx]),
                    "pred_alt_value_at_pred": float(pred_alt[pred_alt_idx]),
                    "true_alt_value_at_pred": float(true_alt[pred_alt_idx]),

                    "pred_main_margin": pred_main_margin,
                    "true_main_margin": true_main_margin,
                    "pred_alt_margin": pred_alt_margin,
                    "true_alt_margin": true_alt_margin,

                    "max_true_main": float(true_main.max()),
                    "max_pred_main": float(pred_main.max()),
                    "max_true_alt": float(true_alt.max()),
                    "max_pred_alt": float(pred_alt.max()),

                    "sum_true_main": float(true_main.sum()),
                    "sum_true_alt": float(true_alt.sum()),
                    "sum_pred_main": float(pred_main.sum()),
                    "sum_pred_alt": float(pred_alt.sum()),

                    "true_main_electron_rank": true_main_electron_rank,
                    "pred_main_electron_rank": pred_main_electron_rank,
                    "true_main_is_largest_electron_cluster": bool(true_main_electron_rank == 1),
                    "pred_main_is_largest_electron_cluster": bool(pred_main_electron_rank == 1),
                }

                # Event-level feature summaries on z-scored training scale.
                for j, feature_name in enumerate(FEATURES):
                    vals = xb[:, j]
                    row[f"z_{feature_name}_mean"] = float(vals.mean())
                    row[f"z_{feature_name}_std"] = float(vals.std(unbiased=False))
                    row[f"z_{feature_name}_min"] = float(vals.min())
                    row[f"z_{feature_name}_max"] = float(vals.max())
                    row[f"z_true_main_{feature_name}"] = float(xb[true_main_idx, j])
                    row[f"z_pred_main_{feature_name}"] = float(xb[pred_main_idx, j])
                    row[f"z_pred_minus_true_main_{feature_name}"] = float(
                        xb[pred_main_idx, j] - xb[true_main_idx, j]
                    )

                # Event-level feature summaries in raw physical units.
                for j, feature_name in enumerate(FEATURES):
                    vals = xb_raw[:, j]
                    row[f"raw_{feature_name}_mean"] = float(vals.mean())
                    row[f"raw_{feature_name}_std"] = float(vals.std(unbiased=False))
                    row[f"raw_{feature_name}_min"] = float(vals.min())
                    row[f"raw_{feature_name}_max"] = float(vals.max())
                    row[f"raw_true_main_{feature_name}"] = float(xb_raw[true_main_idx, j])
                    row[f"raw_pred_main_{feature_name}"] = float(xb_raw[pred_main_idx, j])
                    row[f"raw_pred_minus_true_main_{feature_name}"] = float(
                        xb_raw[pred_main_idx, j] - xb_raw[true_main_idx, j]
                    )

                rows.append(row)

    return pd.DataFrame(rows)


def summarize_main_diagnostics(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      1. Summary statistics grouped by main_correct.
      2. Accuracy by event multiplicity bin.
    """
    summary_cols = [
        "n_clusters",
        "pred_main_margin",
        "true_main_margin",
        "pred_alt_margin",
        "true_alt_margin",
        "max_true_main",
        "max_pred_main",
        "max_true_alt",
        "max_pred_alt",
        "sum_true_main",
        "sum_true_alt",
        "sum_pred_main",
        "sum_pred_alt",
        "true_main_electron_rank",
        "pred_main_electron_rank",
    ]

    for prefix in ["z", "raw"]:
        for feature_name in FEATURES:
            summary_cols.extend(
                [
                    f"{prefix}_{feature_name}_mean",
                    f"{prefix}_{feature_name}_std",
                    f"{prefix}_{feature_name}_min",
                    f"{prefix}_{feature_name}_max",
                    f"{prefix}_true_main_{feature_name}",
                    f"{prefix}_pred_main_{feature_name}",
                    f"{prefix}_pred_minus_true_main_{feature_name}",
                ]
            )

    summary = (
        df.groupby("main_correct")[summary_cols]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .T
    )

    out = df.copy()
    out["n_cluster_bin"] = pd.cut(
        out["n_clusters"],
        bins=[0, 1, 2, 3, 4, 5, 10, 20, 50, 200],
        labels=["1", "2", "3", "4", "5", "6-10", "11-20", "21-50", "51+"],
        include_lowest=True,
    )

    multiplicity_summary = (
        out.groupby("n_cluster_bin", observed=False)
        .agg(
            n_events=("main_correct", "size"),
            main_accuracy=("main_correct", "mean"),
            alt_accuracy=("alt_correct", "mean"),
            main_and_alt_accuracy=("main_and_alt_correct", "mean"),
            mean_pred_main_margin=("pred_main_margin", "mean"),
            median_pred_main_margin=("pred_main_margin", "median"),
            mean_true_main_margin=("true_main_margin", "mean"),
            median_true_main_margin=("true_main_margin", "median"),
            mean_sum_true_main=("sum_true_main", "mean"),
            mean_sum_true_alt=("sum_true_alt", "mean"),
            frac_true_main_largest_electron=("true_main_is_largest_electron_cluster", "mean"),
            frac_pred_main_largest_electron=("pred_main_is_largest_electron_cluster", "mean"),
        )
        .reset_index()
    )

    return summary, multiplicity_summary


def write_diagnostics(
    model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
) -> None:
    """Write validation/test event-level diagnostics and summary CSVs."""
    val_diag = collect_main_event_diagnostics(model, val_loader, split_name="val")
    test_diag = collect_main_event_diagnostics(model, test_loader, split_name="test")

    val_summary, val_multiplicity = summarize_main_diagnostics(val_diag)
    test_summary, test_multiplicity = summarize_main_diagnostics(test_diag)

    val_diag.to_csv("val_main_event_diagnostics.csv", index=False)
    test_diag.to_csv("test_main_event_diagnostics.csv", index=False)

    val_summary.to_csv("val_main_hit_vs_miss_summary.csv")
    test_summary.to_csv("test_main_hit_vs_miss_summary.csv")

    val_multiplicity.to_csv("val_main_accuracy_by_multiplicity.csv", index=False)
    test_multiplicity.to_csv("test_main_accuracy_by_multiplicity.csv", index=False)

    print("\nValidation main hit/miss summary")
    print(val_summary)

    print("\nValidation main accuracy by event multiplicity")
    print(val_multiplicity)

    print("\nTest main hit/miss summary")
    print(test_summary)

    print("\nTest main accuracy by event multiplicity")
    print(test_multiplicity)

    if USE_WANDB and wandb is not None and wandb.run is not None:
        wandb.save("val_main_event_diagnostics.csv")
        wandb.save("test_main_event_diagnostics.csv")
        wandb.save("val_main_hit_vs_miss_summary.csv")
        wandb.save("test_main_hit_vs_miss_summary.csv")
        wandb.save("val_main_accuracy_by_multiplicity.csv")
        wandb.save("test_main_accuracy_by_multiplicity.csv")


# -----------------------------
# Dataloaders
# -----------------------------

def make_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    (
        X_train,
        X_train_raw,
        Y_train,
        E_train,
        train_groups,
        X_val,
        X_val_raw,
        Y_val,
        E_val,
        val_groups,
        X_test,
        X_test_raw,
        Y_test,
        E_test,
        test_groups,
    ) = prepare_data()

    persistent_workers = NUM_WORKERS > 0

    train_dataset = S2EventDataset(X_train, X_train_raw, Y_train, E_train, train_groups)
    val_dataset = S2EventDataset(X_val, X_val_raw, Y_val, E_val, val_groups)
    test_dataset = S2EventDataset(X_test, X_test_raw, Y_test, E_test, test_groups)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_TRAIN,
        collate_fn=collate_events,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_events,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=persistent_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_events,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=persistent_workers,
    )

    return train_loader, val_loader, test_loader


# -----------------------------
# Training
# -----------------------------

def train() -> Dict[str, float]:
    set_seed(RANDOM_SEED)
    train_loader, val_loader, test_loader = make_dataloaders()

    model = MaskedPoolingNet(
        input_dim=len(FEATURES),
        latent_dim=LATENT_DIM,
        phi_hidden=PHI_HIDDEN,
        head_hidden=HEAD_HIDDEN,
        output_dim=len(TARGETS),
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = None
    scheduler_name = SCHEDULER.lower().strip()
    if scheduler_name == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE,
            min_lr=SCHEDULER_MIN_LR,
        )
    elif scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=SCHEDULER_T_MAX,
            eta_min=SCHEDULER_MIN_LR,
        )
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown SCHEDULER={SCHEDULER!r}")

    best_metric = float("-inf")
    best_state: Optional[Dict[str, Any]] = None
    bad_epochs = 0

    if USE_WANDB and wandb is not None:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=WANDB_RUN_NAME,
            config={
                "latent_dim": LATENT_DIM,
                "phi_hidden": PHI_HIDDEN,
                "head_hidden": HEAD_HIDDEN,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "scheduler": SCHEDULER,
                "scheduler_patience": SCHEDULER_PATIENCE,
                "scheduler_factor": SCHEDULER_FACTOR,
                "scheduler_min_lr": SCHEDULER_MIN_LR,
                "scheduler_t_max": SCHEDULER_T_MAX,
                "features": FEATURES,
                "targets": TARGETS,
            },
        )

    for epoch in range(MAX_EPOCHS):
        train_loss, _ = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            collect_metrics=False,
        )
        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            optimizer=None,
            collect_metrics=True,
        )
        val_acc = float(val_metrics["event_main_accuracy"])

        if scheduler is not None:
            if scheduler_name == "reduce_on_plateau":
                scheduler.step(val_acc)
            else:
                scheduler.step()

        if val_acc > best_metric:
            best_metric = val_acc
            bad_epochs = 0
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_metric": best_metric,
            }
            torch.save(best_state, CHECKPOINT_PATH)
        else:
            bad_epochs += 1

        if USE_WANDB and wandb is not None and wandb.run is not None:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_mse": val_metrics["mse"],
                    "val_p_main_mae": val_metrics["p_main_mae"],
                    "val_p_alt_mae": val_metrics["p_alt_mae"],
                    "val_mean_mae": val_metrics["mean_mae"],
                    "val_p_main_brier": val_metrics["p_main_brier"],
                    "val_p_alt_brier": val_metrics["p_alt_brier"],
                    "val_mean_brier": val_metrics["mean_brier"],
                    "val_p_main_auc": val_metrics["p_main_auc"],
                    "val_p_alt_auc": val_metrics["p_alt_auc"],
                    "val_mean_auc": val_metrics["mean_auc"],
                    "val_event_main_accuracy": val_acc,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

        print(
            f"Epoch {epoch + 1:03d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_event_main_accuracy={val_acc:.4f} | best={best_metric:.4f}"
        )

        if bad_epochs >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])

    test_loss, test_metrics = run_epoch(
        model,
        test_loader,
        optimizer=None,
        collect_metrics=True,
    )

    results = {
        "test_loss": float(test_loss),
        "test_mse": float(test_metrics["mse"]),
        "test_p_main_mae": float(test_metrics["p_main_mae"]),
        "test_p_alt_mae": float(test_metrics["p_alt_mae"]),
        "test_mean_mae": float(test_metrics["mean_mae"]),
        "test_p_main_brier": float(test_metrics["p_main_brier"]),
        "test_p_alt_brier": float(test_metrics["p_alt_brier"]),
        "test_mean_brier": float(test_metrics["mean_brier"]),
        "test_p_main_auc": float(test_metrics["p_main_auc"]),
        "test_p_alt_auc": float(test_metrics["p_alt_auc"]),
        "test_mean_auc": float(test_metrics["mean_auc"]),
        "test_event_main_accuracy": float(test_metrics["event_main_accuracy"]),
        "best_val_event_main_accuracy": float(best_metric),
        "best_epoch": int(best_state["epoch"] if best_state is not None else -1),
    }

    print("\nFinal test results")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k}={v:.4f}")
        else:
            print(f"{k}={v}")

    write_diagnostics(model, val_loader, test_loader)

    if USE_WANDB and wandb is not None and wandb.run is not None:
        wandb.log(results)
        wandb.finish()

    return results


# -----------------------------
# Entry point
# -----------------------------

if __name__ == "__main__":
    train()
