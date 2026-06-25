"""Standalone training script for S2 tagging using PyTorch Geometric graph batching.

This version keeps variable-size events as variable-size graphs.
There is no padding and no masking in the batching path.
Each event is a PyG Data object with:
- x: per-cluster features, shape [num_clusters, 5]
- y: per-cluster soft targets, shape [num_clusters, 2]

The model uses:
- per-cluster MLP encoder
- graph-level sum pooling with global_add_pool
- broadcast of the pooled event embedding back to each cluster
- per-cluster MLP head

That preserves the original DeepSet-style idea while using graph batching
instead of padded tensors.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch_geometric.data import Data as PYGData
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import global_add_pool

try:
    import wandb
except Exception:
    wandb = None


# -----------------------------
# User-editable parameters
# -----------------------------

RAW_DATA_PATH = "/home/adeeps/projects/xenon_deepset_nn/data/s2_tag_training_clusters.npy"
CACHE_FILE = "graph_pooling_processed_data.npz"
CHECKPOINT_PATH = "best_graph_pooling_model.pt"

USE_WANDB = False
WANDB_PROJECT = "xenon-graph-pooling"
WANDB_ENTITY = None
WANDB_RUN_NAME = None

LATENT_DIM = 64
PHI_HIDDEN = 128
HEAD_HIDDEN = 128
ENCODER_DEPTH = 2
HEAD_DEPTH = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
BATCH_SIZE = 512
MAX_EPOCHS = 400
EARLY_STOPPING_PATIENCE = 30
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


# -----------------------------
# Data prep
# -----------------------------

def build_event_groups(event_ids_array: np.ndarray) -> List[np.ndarray]:
    order = np.argsort(event_ids_array, kind="mergesort")
    sorted_events = event_ids_array[order]
    boundaries = np.flatnonzero(sorted_events[1:] != sorted_events[:-1]) + 1
    return list(np.split(order, boundaries))


def prepare_data() -> Tuple[np.ndarray, ...]:
    if os.path.exists(CACHE_FILE):
        cached = np.load(CACHE_FILE, allow_pickle=True)

        X_train = cached["X_train"]
        Y_train = cached["Y_train"]
        E_train = cached["E_train"]
        train_groups = [np.asarray(g, dtype=np.int64) for g in cached["train_groups"]]

        X_val = cached["X_val"]
        Y_val = cached["Y_val"]
        E_val = cached["E_val"]
        val_groups = [np.asarray(g, dtype=np.int64) for g in cached["val_groups"]]

        X_test = cached["X_test"]
        Y_test = cached["Y_test"]
        E_test = cached["E_test"]
        test_groups = [np.asarray(g, dtype=np.int64) for g in cached["test_groups"]]

        return (
            X_train, Y_train, E_train, train_groups,
            X_val, Y_val, E_val, val_groups,
            X_test, Y_test, E_test, test_groups,
        )

    data = np.load(RAW_DATA_PATH)
    df = pd.DataFrame(data)

    event_min_drift = df.groupby(EVENT_COL)["drift_time_mean"].min()
    bad_event_ids = event_min_drift[event_min_drift < TOP13_NS].index.to_numpy()
    df = df[~df[EVENT_COL].isin(bad_event_ids)].copy()
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
        Y_train=Y_train,
        E_train=E_train,
        train_groups=np.array(train_groups, dtype=object),
        X_val=X_val,
        Y_val=Y_val,
        E_val=E_val,
        val_groups=np.array(val_groups, dtype=object),
        X_test=X_test,
        Y_test=Y_test,
        E_test=E_test,
        test_groups=np.array(test_groups, dtype=object),
    )

    return (
        X_train, Y_train, E_train, train_groups,
        X_val, Y_val, E_val, val_groups,
        X_test, Y_test, E_test, test_groups,
    )


class S2GraphDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, event_groups: List[np.ndarray]):
        self.X = X
        self.Y = Y
        self.event_groups = event_groups

    def __len__(self) -> int:
        return len(self.event_groups)

    def __getitem__(self, idx: int) -> PYGData:
        rows = self.event_groups[idx]
        x = torch.from_numpy(self.X[rows].copy())
        y = torch.from_numpy(self.Y[rows].copy())
        return PYGData(x=x, y=y)


# -----------------------------
# Model helpers
# -----------------------------

def build_mlp(input_dim: int, hidden_dim: int, output_dim: int, depth: int) -> nn.Sequential:
    if depth < 1:
        raise ValueError("depth must be >= 1")

    layers: List[nn.Module] = []
    if depth == 1:
        layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)

    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(nn.ReLU())
    for _ in range(depth - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class GraphSetTagger(nn.Module):
    """Per-cluster encoder + graph sum pooling + per-cluster head."""

    def __init__(
        self,
        input_dim: int = 5,
        latent_dim: int = 64,
        phi_hidden: int = 64,
        head_hidden: int = 64,
        encoder_depth: int = 2,
        head_depth: int = 2,
        output_dim: int = 2,
    ):
        super().__init__()

        self.encoder = build_mlp(
            input_dim=input_dim,
            hidden_dim=phi_hidden,
            output_dim=latent_dim,
            depth=encoder_depth,
        )
        self.head = build_mlp(
            input_dim=latent_dim * 2,
            hidden_dim=head_hidden,
            output_dim=output_dim,
            depth=head_depth,
        )

    def forward(self, batch: PYGData) -> torch.Tensor:
        x = batch.x
        graph_index = batch.batch

        node_embed = self.encoder(x)
        graph_embed = global_add_pool(node_embed, graph_index)
        graph_embed_per_node = graph_embed[graph_index]

        out = self.head(torch.cat([node_embed, graph_embed_per_node], dim=-1))
        return out


# -----------------------------
# Loss and metrics
# -----------------------------

def bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(logits, targets)


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


@torch.no_grad()
def run_epoch(
    model: nn.Module,
    loader: PyGDataLoader,
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
            batch = batch.to(DEVICE)
            y = batch.y

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(batch)
            loss = bce_loss(logits, y)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

            if collect_metrics:
                probs = torch.sigmoid(logits)
                all_targets.append(y.detach().cpu())
                all_probs.append(probs.detach().cpu())

                batch_ids = batch.batch.detach().cpu().numpy()
                probs_cpu = probs.detach().cpu()
                y_cpu = y.detach().cpu()

                for graph_id in range(int(batch_ids.max()) + 1):
                    node_idx = np.where(batch_ids == graph_id)[0]
                    if len(node_idx) == 0:
                        continue
                    pred_main = probs_cpu[node_idx, 0]
                    true_main = y_cpu[node_idx, 0]
                    event_correct += int(torch.argmax(pred_main).item() == torch.argmax(true_main).item())
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
# Dataloaders
# -----------------------------

def make_dataloaders() -> Tuple[PyGDataLoader, PyGDataLoader, PyGDataLoader]:
    (
        X_train,
        Y_train,
        E_train,
        train_groups,
        X_val,
        Y_val,
        E_val,
        val_groups,
        X_test,
        Y_test,
        E_test,
        test_groups,
    ) = prepare_data()

    train_dataset = S2GraphDataset(X_train, Y_train, train_groups)
    val_dataset = S2GraphDataset(X_val, Y_val, val_groups)
    test_dataset = S2GraphDataset(X_test, Y_test, test_groups)

    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_TRAIN,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )
    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )
    test_loader = PyGDataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


# -----------------------------
# Training
# -----------------------------

def train() -> Dict[str, float]:
    set_seed(RANDOM_SEED)
    train_loader, val_loader, test_loader = make_dataloaders()

    model = GraphSetTagger(
        input_dim=5,
        latent_dim=LATENT_DIM,
        phi_hidden=PHI_HIDDEN,
        head_hidden=HEAD_HIDDEN,
        encoder_depth=ENCODER_DEPTH,
        head_depth=HEAD_DEPTH,
        output_dim=2,
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
                "encoder_depth": ENCODER_DEPTH,
                "head_depth": HEAD_DEPTH,
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
            },
        )

    for epoch in range(MAX_EPOCHS):
        train_loss, _ = run_epoch(model, train_loader, optimizer=optimizer, collect_metrics=False)
        val_loss, val_metrics = run_epoch(model, val_loader, optimizer=None, collect_metrics=True)
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

    test_loss, test_metrics = run_epoch(model, test_loader, optimizer=None, collect_metrics=True)

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

    if USE_WANDB and wandb is not None and wandb.run is not None:
        wandb.log(results)
        wandb.finish()

    return results


# -----------------------------
# Entry point
# -----------------------------

if __name__ == "__main__":
    train()
