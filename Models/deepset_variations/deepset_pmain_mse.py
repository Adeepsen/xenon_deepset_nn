"""Pure per-cluster p_main regression with graph-batched Deep Sets.

This is a standalone, GPU-ready experiment derived from the best prior Optuna
architecture.  It intentionally differs from the legacy runs:

* target: p_main only (one output per cluster)
* loss/model selection: per-cluster MSE / validation p_main MSE
* data: top-13-cm fiducial cut plus event sum(p_main) <= 1 + tolerance
* final reporting: a single held-out test R² and predicted-vs-true plot

The test loader is never used during training or checkpoint selection.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch_geometric.data import Data as PYGData
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import global_add_pool

try:
    import wandb
except ImportError:  # Allows an offline training run when W&B is unavailable.
    wandb = None


# ---------------------------------------------------------------------------
# Configuration: trial_0004 is the architecture/optimizer starting point.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = Path(os.environ.get(
    "XENON_DATA_PATH", PROJECT_ROOT / "data" / "s2_tag_training_clusters.npy"
))
OUTPUT_DIR = Path(os.environ.get(
    "XENON_PMAIN_MSE_OUTPUT_DIR", Path(__file__).resolve().parent / "pmain_mse_output"
))
CACHE_FILE = OUTPUT_DIR / "pmain_mse_sumcut_processed_data.npz"
CHECKPOINT_PATH = OUTPUT_DIR / "best_pmain_mse_model.pt"
RESULTS_PATH = OUTPUT_DIR / "test_pmain_mse_results.json"
PLOT_PATH = OUTPUT_DIR / "test_pmain_predicted_vs_true.png"

USE_WANDB = True
WANDB_PROJECT = "xenon-graph-pooling-pmain-mse"
WANDB_ENTITY = None  # Uses the logged-in default entity.
WANDB_RUN_NAME = None

FEATURES = ["x", "y", "n_electrons_interface", "drift_time_mean", "drift_time_spread"]
TARGET = "p_main"
EVENT_COL = "event_number"
TOP13_NS = 192_600.0
PMAIN_SUM_TOLERANCE = 1e-6
RANDOM_SEED = 42

# Best completed Optuna architecture (trial_0004), now used with pure MSE.
LATENT_DIM = 128
PHI_HIDDEN = 256
HEAD_HIDDEN = 512
ENCODER_DEPTH = 5
HEAD_DEPTH = 4
DROPOUT = 0.04848861422838413
LEARNING_RATE = 0.0016820065354419135
WEIGHT_DECAY = 1.076825417908119e-5
BATCH_SIZE = 1024

MAX_EPOCHS = 300
EARLY_STOPPING_PATIENCE = 60
SCHEDULER_PATIENCE = 16
SCHEDULER_FACTOR = 0.7
SCHEDULER_MIN_LR = 1e-6
NUM_WORKERS = 4
PIN_MEMORY = True
SCATTER_SAMPLE_SIZE = 100_000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_event_groups(event_ids: np.ndarray) -> List[np.ndarray]:
    """Return row indices belonging to each event, even if input is unsorted."""
    order = np.argsort(event_ids, kind="mergesort")
    sorted_events = event_ids[order]
    boundaries = np.flatnonzero(sorted_events[1:] != sorted_events[:-1]) + 1
    return list(np.split(order, boundaries))


def _cache_metadata() -> Dict[str, Any]:
    return {
        "features": FEATURES,
        "target": TARGET,
        "event_col": EVENT_COL,
        "top13_ns": TOP13_NS,
        "pmain_sum_tolerance": PMAIN_SUM_TOLERANCE,
        "random_seed": RANDOM_SEED,
    }


def _load_cached_data() -> Tuple[Tuple[np.ndarray, ...], Dict[str, Any]] | None:
    if not CACHE_FILE.exists():
        return None
    cached = np.load(CACHE_FILE, allow_pickle=True)
    metadata = json.loads(str(cached["metadata"].item()))
    if metadata.get("preprocessing") != _cache_metadata():
        return None
    arrays = (
        cached["X_train"], cached["Y_train"], cached["E_train"], list(cached["train_groups"]),
        cached["X_val"], cached["Y_val"], cached["E_val"], list(cached["val_groups"]),
        cached["X_test"], cached["Y_test"], cached["E_test"], list(cached["test_groups"]),
    )
    return arrays, metadata["cut_stats"]


def prepare_data() -> Tuple[Tuple[np.ndarray, ...], Dict[str, Any]]:
    """Apply quality cuts before event splitting, then fit scaling on train only."""
    cached = _load_cached_data()
    if cached is not None:
        print(f"Using cached preprocessing: {CACHE_FILE}")
        return cached

    print(f"Loading raw data: {RAW_DATA_PATH}")
    data = np.load(RAW_DATA_PATH)
    df = pd.DataFrame(data)
    n_events_raw = int(df[EVENT_COL].nunique())
    n_clusters_raw = int(len(df))

    # Existing fiducial cut: remove an entire event if any cluster lies in the
    # top 13 cm of the TPC.
    event_min_drift = df.groupby(EVENT_COL)["drift_time_mean"].min()
    fiducial_removed_ids = event_min_drift[event_min_drift < TOP13_NS].index
    df = df[~df[EVENT_COL].isin(fiducial_removed_ids)].copy()

    # New advisor-requested label-consistency cut.  This is intentionally an
    # event-level operation and happens before the train/val/test split.
    event_pmain_sum = df.groupby(EVENT_COL)[TARGET].sum()
    pmain_sum_removed_ids = event_pmain_sum[
        event_pmain_sum > 1.0 + PMAIN_SUM_TOLERANCE
    ].index
    df = df[~df[EVENT_COL].isin(pmain_sum_removed_ids)].copy()

    cut_stats = {
        "events_raw": n_events_raw,
        "clusters_raw": n_clusters_raw,
        "events_removed_fiducial": int(len(fiducial_removed_ids)),
        "events_after_fiducial": int(n_events_raw - len(fiducial_removed_ids)),
        "events_removed_pmain_sum_after_fiducial": int(len(pmain_sum_removed_ids)),
        "clusters_after_both_cuts": int(len(df)),
        "events_after_both_cuts": int(df[EVENT_COL].nunique()),
        "pmain_sum_threshold": 1.0 + PMAIN_SUM_TOLERANCE,
    }
    print(json.dumps(cut_stats, indent=2))

    event_ids = df[EVENT_COL].unique()
    train_events, temp_events = train_test_split(
        event_ids, test_size=0.30, random_state=RANDOM_SEED, shuffle=True
    )
    val_events, test_events = train_test_split(
        temp_events, test_size=0.50, random_state=RANDOM_SEED, shuffle=True
    )
    train_df = df[df[EVENT_COL].isin(train_events)].copy()
    val_df = df[df[EVENT_COL].isin(val_events)].copy()
    test_df = df[df[EVENT_COL].isin(test_events)].copy()

    scaler = StandardScaler().fit(train_df[FEATURES])
    for split in (train_df, val_df, test_df):
        split[FEATURES] = scaler.transform(split[FEATURES]).astype(np.float32)

    def as_arrays(split: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        x = split[FEATURES].to_numpy(dtype=np.float32, copy=True)
        # Keep a column dimension so batch.y is [n_clusters, 1].
        y = split[[TARGET]].to_numpy(dtype=np.float32, copy=True)
        event = split[EVENT_COL].to_numpy(copy=True)
        return x, y, event, build_event_groups(event)

    train = as_arrays(train_df)
    val = as_arrays(val_df)
    test = as_arrays(test_df)
    arrays = (*train, *val, *test)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_FILE,
        X_train=train[0], Y_train=train[1], E_train=train[2], train_groups=np.array(train[3], dtype=object),
        X_val=val[0], Y_val=val[1], E_val=val[2], val_groups=np.array(val[3], dtype=object),
        X_test=test[0], Y_test=test[1], E_test=test[2], test_groups=np.array(test[3], dtype=object),
        metadata=json.dumps({"preprocessing": _cache_metadata(), "cut_stats": cut_stats}),
    )
    return arrays, cut_stats


class S2GraphDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, event_ids: np.ndarray, groups: Iterable[np.ndarray]):
        self.x, self.y, self.event_ids = x, y, event_ids
        self.groups = list(groups)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> PYGData:
        rows = self.groups[index]
        return PYGData(
            x=torch.from_numpy(self.x[rows]),
            y=torch.from_numpy(self.y[rows]),
            event_id=torch.tensor([int(self.event_ids[rows[0]])], dtype=torch.long),
        )


def mlp(in_dim: int, hidden_dim: int, depth: int, dropout: float, out_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    current_dim = in_dim
    for _ in range(depth):
        layers += [nn.Linear(current_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, out_dim))
    return nn.Sequential(*layers)


class GraphDeepSetRegressor(nn.Module):
    """Per-node encoder, sum-pool event context, and per-node p_main head."""
    def __init__(self) -> None:
        super().__init__()
        self.encoder = mlp(len(FEATURES), PHI_HIDDEN, ENCODER_DEPTH, DROPOUT, LATENT_DIM)
        self.head = mlp(2 * LATENT_DIM, HEAD_HIDDEN, HEAD_DEPTH, DROPOUT, 1)

    def forward(self, batch: PYGData) -> torch.Tensor:
        node_embedding = self.encoder(batch.x)
        event_embedding = global_add_pool(node_embedding, batch.batch)
        return self.head(torch.cat([node_embedding, event_embedding[batch.batch]], dim=-1))


def mse_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(torch.sigmoid(logits), targets)


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    return {
        "p_main_mse": float(mean_squared_error(target, prediction)),
        "p_main_mae": float(mean_absolute_error(target, prediction)),
        "p_main_r2": float(r2_score(target, prediction)),
    }


def run_epoch(
    model: nn.Module, loader: PyGDataLoader, optimizer: torch.optim.Optimizer | None = None
) -> Tuple[float, Dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    total_loss, batches = 0.0, 0
    targets, predictions = [], []
    with torch.enable_grad() if training else torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = mse_from_logits(logits, batch.y)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item())
            batches += 1
            if not training:
                targets.append(batch.y[:, 0].cpu())
                predictions.append(torch.sigmoid(logits[:, 0]).cpu())

    average_loss = total_loss / max(batches, 1)
    if training:
        return average_loss, {}
    target = torch.cat(targets).numpy()
    prediction = torch.cat(predictions).numpy()
    return average_loss, regression_metrics(target, prediction)


def make_loaders(arrays: Tuple[np.ndarray, ...]) -> Tuple[PyGDataLoader, PyGDataLoader, PyGDataLoader]:
    x_train, y_train, e_train, train_groups, x_val, y_val, e_val, val_groups, x_test, y_test, e_test, test_groups = arrays
    datasets = [
        S2GraphDataset(x_train, y_train, e_train, train_groups),
        S2GraphDataset(x_val, y_val, e_val, val_groups),
        S2GraphDataset(x_test, y_test, e_test, test_groups),
    ]
    common = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)
    return (
        PyGDataLoader(datasets[0], shuffle=True, **common),
        PyGDataLoader(datasets[1], shuffle=False, **common),
        PyGDataLoader(datasets[2], shuffle=False, **common),
    )


def save_prediction_plot(model: nn.Module, loader: PyGDataLoader, r2: float) -> None:
    model.eval()
    targets, predictions = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            targets.append(batch.y[:, 0].cpu())
            predictions.append(torch.sigmoid(model(batch)[:, 0]).cpu())
    target, prediction = torch.cat(targets).numpy(), torch.cat(predictions).numpy()
    sample_size = min(SCATTER_SAMPLE_SIZE, len(target))
    sample = np.random.default_rng(RANDOM_SEED).choice(len(target), size=sample_size, replace=False)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(target[sample], prediction[sample], s=2, alpha=0.08, rasterized=True)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="ideal: y = x")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="True p_main", ylabel="Predicted p_main",
           title=f"Held-out test p_main regression (R² = {r2:.4f})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=220)
    plt.close(fig)


def main() -> None:
    set_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    arrays, cut_stats = prepare_data()
    train_loader, val_loader, test_loader = make_loaders(arrays)
    model = GraphDeepSetRegressor().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR
    )

    run = None
    if USE_WANDB and wandb is not None:
        run = wandb.init(
            project=WANDB_PROJECT, entity=WANDB_ENTITY, name=WANDB_RUN_NAME,
            config={
                "experiment": "pure_pmain_mse_regression", "target": TARGET, "loss": "mse",
                "checkpoint_metric": "val_p_main_mse", "checkpoint_mode": "min",
                "pooling": "sum", "activation": "gelu", "latent_dim": LATENT_DIM,
                "phi_hidden": PHI_HIDDEN, "head_hidden": HEAD_HIDDEN, "encoder_depth": ENCODER_DEPTH,
                "head_depth": HEAD_DEPTH, "dropout": DROPOUT, "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE, "scheduler": "reduce_on_plateau",
                "scheduler_mode": "min", "scheduler_patience": SCHEDULER_PATIENCE,
                "scheduler_factor": SCHEDULER_FACTOR, "scheduler_min_lr": SCHEDULER_MIN_LR,
                **cut_stats,
            },
        )

    best_val_mse, best_epoch, epochs_without_improvement = float("inf"), -1, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, _ = run_epoch(model, train_loader, optimizer)
        val_loss, val_metrics = run_epoch(model, val_loader)
        val_mse = val_metrics["p_main_mse"]
        scheduler.step(val_mse)

        if val_mse < best_val_mse:
            best_val_mse, best_epoch, epochs_without_improvement = val_mse, epoch, 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "best_val_p_main_mse": best_val_mse, "config": dict(wandb.config) if run else {},
                "cut_stats": cut_stats,
            }, CHECKPOINT_PATH)
        else:
            epochs_without_improvement += 1

        metrics = {
            "epoch": epoch, "train_p_main_mse": train_loss, "val_p_main_mse": val_mse,
            "val_p_main_mae": val_metrics["p_main_mae"], "val_p_main_r2": val_metrics["p_main_r2"],
            "best_val_p_main_mse": best_val_mse, "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if run:
            wandb.log(metrics)
        print(
            f"Epoch {epoch:03d} | train_mse={train_loss:.6f} | val_mse={val_mse:.6f} | "
            f"val_r2={val_metrics['p_main_r2']:.5f} | best={best_val_mse:.6f} | lr={metrics['learning_rate']:.2e}"
        )
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch}; restoring epoch {best_epoch}.")
            break

    # Test is evaluated exactly once, after validation-only selection.
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_metrics = run_epoch(model, test_loader)
    save_prediction_plot(model, test_loader, test_metrics["p_main_r2"])
    results = {
        "best_epoch": best_epoch, "best_val_p_main_mse": best_val_mse,
        "test_p_main_mse": test_metrics["p_main_mse"], "test_p_main_mae": test_metrics["p_main_mae"],
        "test_p_main_r2": test_metrics["p_main_r2"], "test_loss": test_loss,
        "checkpoint": str(CHECKPOINT_PATH), "predicted_vs_true_plot": str(PLOT_PATH), **cut_stats,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print("\nFinal held-out test results")
    print(json.dumps(results, indent=2))
    if run:
        wandb.log({**results, "test_predicted_vs_true": wandb.Image(str(PLOT_PATH))})
        wandb.finish()


if __name__ == "__main__":
    main()
