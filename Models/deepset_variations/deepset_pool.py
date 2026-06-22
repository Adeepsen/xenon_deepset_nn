"""Standalone training script for S2 tagging with masked pooling.

This version is meant to be run once directly, with all hyperparameters edited
at the top of the file instead of using sweeps or a config object.

It keeps the same overall architecture as the original DeepSet code:
- per-cluster MLP encoder
- masked set-level pooling
- per-cluster MLP head
- binary cross-entropy with soft targets

The only substantive change is that the set aggregation is done with explicit
masked mean/max pooling rather than the original sum-based DeepSet block.
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
CACHE_FILE = "pooling_processed_data.npz"
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


# -----------------------------
# Data prep
# -----------------------------

def build_event_groups(event_ids_array: np.ndarray) -> List[np.ndarray]:
    order = np.argsort(event_ids_array, kind="mergesort")
    sorted_events = event_ids_array[order]
    boundaries = np.flatnonzero(sorted_events[1:] != sorted_events[:-1]) + 1
    return list(np.split(order, boundaries))


def prepare_data() -> Tuple[np.ndarray, ...]:
    """Load cached processed arrays, or build them once and cache them."""
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
    )


class S2EventDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, event_groups: List[np.ndarray]):
        self.X = X
        self.Y = Y
        self.event_groups = event_groups

    def __len__(self) -> int:
        return len(self.event_groups)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rows = self.event_groups[idx]
        x = self.X[rows]
        y = self.Y[rows]
        return {
            "x": torch.from_numpy(x.copy()),
            "y": torch.from_numpy(y.copy()),
            "n_clusters": len(rows),
        }


def collate_events(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_clusters = max(item["n_clusters"] for item in batch)

    x_dim = batch[0]["x"].shape[1]
    y_dim = batch[0]["y"].shape[1]

    x_padded = torch.zeros(batch_size, max_clusters, x_dim, dtype=torch.float32)
    y_padded = torch.zeros(batch_size, max_clusters, y_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_clusters, dtype=torch.bool)

    for i, item in enumerate(batch):
        n = item["n_clusters"]
        x_padded[i, :n] = item["x"]
        y_padded[i, :n] = item["y"]
        mask[i, :n] = True

    return {"x": x_padded, "y": y_padded, "mask": mask}


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
        logits, targets, reduction="none"
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

def make_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
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

    persistent_workers = NUM_WORKERS > 0

    train_dataset = S2EventDataset(X_train, Y_train, train_groups)
    val_dataset = S2EventDataset(X_val, Y_val, val_groups)
    test_dataset = S2EventDataset(X_test, Y_test, test_groups)

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
        input_dim=5,
        latent_dim=LATENT_DIM,
        phi_hidden=PHI_HIDDEN,
        head_hidden=HEAD_HIDDEN,
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
