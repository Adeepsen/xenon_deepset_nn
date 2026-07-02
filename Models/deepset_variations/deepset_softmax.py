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
instead of padded tensors. This softmax experiment trains p_main as an
event-normalized distribution over clusters, while p_alt remains an optional
independent sigmoid/BCE target. The evaluation reports calibration metrics,
strict argmax main accuracy, and tie-aware main accuracy for soft/fractional
p_main labels.
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
from torch_geometric.utils import softmax

try:
    import wandb
except Exception:
    wandb = None


# -----------------------------
# User-editable parameters
# -----------------------------

RAW_DATA_PATH = "/home/adeeps/projects/xenon_deepset_nn/data/s2_tag_training_clusters.npy"
CACHE_FILE = "graph_pooling_processed_data.npz"
CHECKPOINT_PATH = "best_graph_pooling_softmax_model.pt"

# Tie-aware event-level metric settings.
# Strict accuracy is still logged for backwards compatibility, but the
# tie-aware metric is the default validation/checkpoint metric.
METRIC_TOL = 1e-6
CHECKPOINT_METRIC = "event_p_main_kl"
CHECKPOINT_MODE = "min"  # "max" for accuracy-like metrics, "min" for error/loss-like metrics

# Optional per-event diagnostic CSVs for validation/test. These are useful
# for making multiplicity, margin, and electron-rank plots after training.
WRITE_EVENT_DIAGNOSTICS = False
DIAGNOSTICS_DIR = "event_diagnostics"

USE_WANDB = True
WANDB_PROJECT = "xenon-graph-pooling-3deep-activation-tests"
WANDB_ENTITY = None
WANDB_RUN_NAME = "softmax-test"

LATENT_DIM = 64
PHI_HIDDEN = 128
HEAD_HIDDEN = 128
ENCODER_DEPTH = 3
HEAD_DEPTH = 3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
BATCH_SIZE = 512
MAX_EPOCHS = 400
EARLY_STOPPING_PATIENCE = 30
NUM_WORKERS = 4
PIN_MEMORY = True
SHUFFLE_TRAIN = True

# Softmax experiment settings. p_main is trained as an event-level
# distribution over clusters. p_alt remains an independent sigmoid/BCE target.
LOSS_NAME = "event_softmax_ce_plus_p_alt_bce"
P_ALT_LOSS_WEIGHT = 0.25

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
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        E: np.ndarray,
        event_groups: List[np.ndarray],
    ):
        self.X = X
        self.Y = Y
        self.E = E
        self.event_groups = event_groups

    def __len__(self) -> int:
        return len(self.event_groups)

    def __getitem__(self, idx: int) -> PYGData:
        rows = self.event_groups[idx]
        x = torch.from_numpy(self.X[rows].copy())
        y = torch.from_numpy(self.Y[rows].copy())

        # Keep the original event id as graph-level metadata. PyG will batch
        # this into shape [num_graphs], which lets us write diagnostics later.
        event_id = torch.as_tensor([self.E[rows[0]]], dtype=torch.long)

        return PYGData(x=x, y=y, event_id=event_id)


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

def prediction_probs(logits: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
    """Convert raw model outputs to probabilities for metrics/inference.

    Column 0 is p_main and is normalized with an event-wise softmax so the
    predicted p_main values sum to one within each event. Column 1 is p_alt and
    remains an independent sigmoid probability.
    """

    pred_p_main = softmax(logits[:, 0], batch_index)
    pred_p_alt = torch.sigmoid(logits[:, 1])
    return torch.stack([pred_p_main, pred_p_alt], dim=-1)


def event_softmax_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    batch_index: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Soft-target cross entropy for event-normalized p_main.

    The model predicts one raw p_main score per cluster. Scores are normalized
    with softmax within each event. The true p_main values are also normalized
    within each event for robustness, so this remains well-defined if an event's
    target p_main sum is slightly different from one.
    """

    pred_p_main = softmax(logits[:, 0], batch_index).clamp_min(eps)
    true_p_main = targets[:, 0].clamp_min(0.0)

    n_events = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 0
    if n_events == 0:
        return logits.sum() * 0.0

    true_sum = torch.zeros(n_events, device=logits.device, dtype=logits.dtype)
    true_sum.index_add_(0, batch_index, true_p_main)
    true_dist = true_p_main / true_sum[batch_index].clamp_min(eps)

    return -(true_dist * torch.log(pred_p_main)).sum() / n_events


def hybrid_softmax_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    batch_index: torch.Tensor,
    p_alt_weight: float = P_ALT_LOSS_WEIGHT,
) -> torch.Tensor:
    """Event-softmax p_main loss plus optional independent p_alt BCE loss."""

    p_main_loss = event_softmax_ce_loss(logits, targets, batch_index)

    if p_alt_weight <= 0.0 or logits.shape[1] < 2 or targets.shape[1] < 2:
        return p_main_loss

    p_alt_loss = nn.functional.binary_cross_entropy_with_logits(
        logits[:, 1],
        targets[:, 1],
    )
    return p_main_loss + p_alt_weight * p_alt_loss


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



def add_tie_aware_metrics_to_diagnostic_csv(
    csv_path: str,
    output_path: Optional[str] = None,
    tol: float = METRIC_TOL,
) -> pd.DataFrame:
    """Add tie-aware event-level correctness to an existing diagnostic CSV.

    Required input columns:
    - main_correct or main_correct_strict
    - true_main_value_at_pred
    - max_true_main
    - true_main_margin
    """

    df = pd.read_csv(csv_path)

    if "main_correct_strict" not in df.columns:
        if "main_correct" not in df.columns:
            raise ValueError("CSV needs either main_correct or main_correct_strict")
        df["main_correct_strict"] = df["main_correct"].astype(bool)
    else:
        df["main_correct_strict"] = df["main_correct_strict"].astype(bool)

    required = {"true_main_value_at_pred", "max_true_main", "true_main_margin"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df["main_correct_tie_aware"] = (
        df["true_main_value_at_pred"] >= df["max_true_main"] - tol
    )
    df["true_main_gap_at_pred"] = (
        df["max_true_main"] - df["true_main_value_at_pred"]
    )
    df["true_main_has_tie"] = df["true_main_margin"] <= tol
    df["strict_miss_but_tie_correct"] = (
        ~df["main_correct_strict"] & df["main_correct_tie_aware"]
    )

    if output_path is not None:
        df.to_csv(output_path, index=False)

    return df


def _safe_nanmedian(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def compute_event_main_metrics_from_batch(
    probs: torch.Tensor,
    targets: torch.Tensor,
    batch_index: torch.Tensor,
    event_ids: Optional[torch.Tensor] = None,
    tol: float = METRIC_TOL,
    collect_rows: bool = False,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Compute event-level ranking and calibration metrics for one PyG batch.

    Ranking metrics answer: did the model put a true max-p_main cluster near the
    top of the event? Calibration metrics answer: are all predicted p_main values
    close to their soft targets for a typical event?
    """

    probs_cpu = probs.detach().cpu()
    targets_cpu = targets.detach().cpu()
    batch_cpu = batch_index.detach().cpu()
    event_ids_cpu = event_ids.detach().cpu() if event_ids is not None else None

    n_graphs = int(batch_cpu.max().item()) + 1 if batch_cpu.numel() > 0 else 0

    strict_correct = 0
    tie_aware_correct = 0
    top2_tie_aware_correct = 0
    top3_tie_aware_correct = 0
    true_tie_count = 0
    strict_miss_tie_correct = 0
    event_total = 0

    true_margins: List[float] = []
    pred_margins: List[float] = []
    true_gaps_at_pred: List[float] = []

    # Event-averaged calibration/distribution metrics for p_main. These give
    # every event equal weight, unlike cluster-averaged MAE/MSE.
    event_p_main_maes: List[float] = []
    event_p_main_mses: List[float] = []
    event_p_main_rmses: List[float] = []
    event_p_main_kls: List[float] = []
    event_p_main_sum_errors: List[float] = []
    event_pred_p_main_sums: List[float] = []
    event_true_p_main_sums: List[float] = []

    rows: List[Dict[str, Any]] = []

    eps = 1e-8

    for graph_id in range(n_graphs):
        node_idx = torch.nonzero(batch_cpu == graph_id, as_tuple=False).flatten()
        if node_idx.numel() == 0:
            continue

        pred_main = probs_cpu[node_idx, 0]
        true_main = targets_cpu[node_idx, 0]
        n_clusters = int(node_idx.numel())

        pred_main_idx = int(torch.argmax(pred_main).item())
        true_main_idx = int(torch.argmax(true_main).item())

        max_true_main = float(true_main.max().item())
        true_main_value_at_pred = float(true_main[pred_main_idx].item())
        true_gap_at_pred = max_true_main - true_main_value_at_pred

        strict = pred_main_idx == true_main_idx
        tie_aware = true_main_value_at_pred >= max_true_main - tol

        true_best_mask = true_main >= (true_main.max() - tol)
        k2 = min(2, n_clusters)
        k3 = min(3, n_clusters)
        pred_top2_idx = torch.topk(pred_main, k=k2).indices
        pred_top3_idx = torch.topk(pred_main, k=k3).indices
        top2_tie_aware = bool(true_best_mask[pred_top2_idx].any().item())
        top3_tie_aware = bool(true_best_mask[pred_top3_idx].any().item())

        if n_clusters >= 2:
            true_top2 = torch.topk(true_main, k=2).values
            pred_top2 = torch.topk(pred_main, k=2).values
            true_main_margin = float((true_top2[0] - true_top2[1]).item())
            pred_main_margin = float((pred_top2[0] - pred_top2[1]).item())
        else:
            true_main_margin = float("nan")
            pred_main_margin = float("nan")

        true_has_tie = bool(n_clusters >= 2 and true_main_margin <= tol)

        err = pred_main - true_main
        event_mae = float(torch.mean(torch.abs(err)).item())
        event_mse = float(torch.mean(err * err).item())
        event_rmse = float(np.sqrt(event_mse))

        pred_sum = float(pred_main.sum().item())
        true_sum = float(true_main.sum().item())
        event_sum_error = pred_sum - true_sum

        # Distribution KL is useful if p_main is meant to be an event-level
        # distribution. It normalizes both vectors within the event, so it is
        # still computable for sigmoid outputs whose sums are not exactly one.
        if true_sum > eps and pred_sum > eps:
            true_dist = true_main / true_main.sum().clamp_min(eps)
            pred_dist = pred_main / pred_main.sum().clamp_min(eps)
            event_kl = float((true_dist * torch.log((true_dist + eps) / (pred_dist + eps))).sum().item())
        else:
            event_kl = float("nan")

        strict_correct += int(strict)
        tie_aware_correct += int(tie_aware)
        top2_tie_aware_correct += int(top2_tie_aware)
        top3_tie_aware_correct += int(top3_tie_aware)
        true_tie_count += int(true_has_tie)
        strict_miss_tie_correct += int((not strict) and tie_aware)
        event_total += 1

        true_margins.append(true_main_margin)
        pred_margins.append(pred_main_margin)
        true_gaps_at_pred.append(true_gap_at_pred)
        event_p_main_maes.append(event_mae)
        event_p_main_mses.append(event_mse)
        event_p_main_rmses.append(event_rmse)
        event_p_main_kls.append(event_kl)
        event_p_main_sum_errors.append(event_sum_error)
        event_pred_p_main_sums.append(pred_sum)
        event_true_p_main_sums.append(true_sum)

        if collect_rows:
            event_number = None
            if event_ids_cpu is not None and graph_id < event_ids_cpu.numel():
                event_number = int(event_ids_cpu[graph_id].item())

            rows.append(
                {
                    "event_number": event_number,
                    "n_clusters": n_clusters,
                    "main_correct_strict": bool(strict),
                    "main_correct_tie_aware": bool(tie_aware),
                    "top2_main_tie_aware": bool(top2_tie_aware),
                    "top3_main_tie_aware": bool(top3_tie_aware),
                    "strict_miss_but_tie_correct": bool((not strict) and tie_aware),
                    "true_main_value_at_pred": true_main_value_at_pred,
                    "max_true_main": max_true_main,
                    "true_main_gap_at_pred": true_gap_at_pred,
                    "true_main_margin": true_main_margin,
                    "pred_main_margin": pred_main_margin,
                    "true_main_has_tie": true_has_tie,
                    "pred_main_idx": pred_main_idx,
                    "strict_true_main_idx": true_main_idx,
                    "event_p_main_mae": event_mae,
                    "event_p_main_mse": event_mse,
                    "event_p_main_rmse": event_rmse,
                    "event_p_main_kl": event_kl,
                    "event_true_p_main_sum": true_sum,
                    "event_pred_p_main_sum": pred_sum,
                    "event_p_main_sum_error": event_sum_error,
                }
            )

    metrics = {
        "event_total": float(event_total),
        "event_main_accuracy_strict": float(strict_correct / max(event_total, 1)),
        "event_main_accuracy_tie_aware": float(tie_aware_correct / max(event_total, 1)),
        "event_main_top2_accuracy_tie_aware": float(top2_tie_aware_correct / max(event_total, 1)),
        "event_main_top3_accuracy_tie_aware": float(top3_tie_aware_correct / max(event_total, 1)),
        "event_main_tie_fraction": float(true_tie_count / max(event_total, 1)),
        "event_main_strict_miss_tie_correct_fraction": float(
            strict_miss_tie_correct / max(event_total, 1)
        ),
        "event_main_mean_true_gap_at_pred": float(np.mean(true_gaps_at_pred))
        if true_gaps_at_pred
        else float("nan"),
        "event_main_median_true_gap_at_pred": _safe_nanmedian(true_gaps_at_pred),
        "event_main_median_true_margin": _safe_nanmedian(true_margins),
        "event_main_median_pred_margin": _safe_nanmedian(pred_margins),
        "event_p_main_mae": float(np.mean(event_p_main_maes)) if event_p_main_maes else float("nan"),
        "event_p_main_mse": float(np.mean(event_p_main_mses)) if event_p_main_mses else float("nan"),
        "event_p_main_rmse": float(np.mean(event_p_main_rmses)) if event_p_main_rmses else float("nan"),
        "event_p_main_kl": float(np.nanmean(event_p_main_kls)) if event_p_main_kls else float("nan"),
        "event_p_main_sum_error_mean": float(np.mean(event_p_main_sum_errors)) if event_p_main_sum_errors else float("nan"),
        "event_p_main_sum_error_mae": float(np.mean(np.abs(event_p_main_sum_errors))) if event_p_main_sum_errors else float("nan"),
        "event_true_p_main_sum_mean": float(np.mean(event_true_p_main_sums)) if event_true_p_main_sums else float("nan"),
        "event_pred_p_main_sum_mean": float(np.mean(event_pred_p_main_sums)) if event_pred_p_main_sums else float("nan"),
    }

    return metrics, rows


def _add_electron_rank_diagnostics(
    rows: List[Dict[str, Any]],
    probs: torch.Tensor,
    targets: torch.Tensor,
    features: torch.Tensor,
    batch_index: torch.Tensor,
) -> None:
    """Add rank diagnostics in-place using z-scored n_electrons_interface.

    Feature index 2 is n_electrons_interface after StandardScaler. Since the
    scaler is monotonic, descending z-scored value gives the same rank ordering
    as descending raw electron count within each event.
    """

    if not rows:
        return

    probs_cpu = probs.detach().cpu()
    targets_cpu = targets.detach().cpu()
    features_cpu = features.detach().cpu()
    batch_cpu = batch_index.detach().cpu()

    for graph_id, row in enumerate(rows):
        node_idx = torch.nonzero(batch_cpu == graph_id, as_tuple=False).flatten()
        if node_idx.numel() == 0:
            continue

        pred_main = probs_cpu[node_idx, 0]
        true_main = targets_cpu[node_idx, 0]
        electrons = features_cpu[node_idx, 2]

        pred_main_idx = int(torch.argmax(pred_main).item())
        true_main_idx = int(torch.argmax(true_main).item())

        electron_order = torch.argsort(electrons, descending=True)
        rank_by_local_idx = torch.empty_like(electron_order)
        rank_by_local_idx[electron_order] = torch.arange(
            1, electron_order.numel() + 1, dtype=rank_by_local_idx.dtype
        )

        row["pred_main_electron_rank"] = int(rank_by_local_idx[pred_main_idx].item())
        row["true_main_electron_rank"] = int(rank_by_local_idx[true_main_idx].item())


def run_epoch(
    model: nn.Module,
    loader: PyGDataLoader,
    optimizer: Optional[torch.optim.Optimizer] = None,
    collect_metrics: bool = False,
    diagnostic_csv_path: Optional[str] = None,
) -> Tuple[float, Dict[str, float]]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    n_batches = 0
    all_targets: List[torch.Tensor] = []
    all_probs: List[torch.Tensor] = []
    diagnostic_rows: List[Dict[str, Any]] = []

    event_metric_weighted_sums: Dict[str, float] = {}
    event_metric_median_inputs: Dict[str, List[float]] = {
        "event_main_median_true_gap_at_pred": [],
        "event_main_median_true_margin": [],
        "event_main_median_pred_margin": [],
    }
    event_total = 0

    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for batch in loader:
            batch = batch.to(DEVICE)
            y = batch.y

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(batch)
            loss = hybrid_softmax_loss(logits, y, batch.batch)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

            if collect_metrics:
                probs = prediction_probs(logits, batch.batch)
                all_targets.append(y.detach().cpu())
                all_probs.append(probs.detach().cpu())

                event_ids = getattr(batch, "event_id", None)
                batch_event_metrics, batch_rows = compute_event_main_metrics_from_batch(
                    probs=probs,
                    targets=y,
                    batch_index=batch.batch,
                    event_ids=event_ids,
                    tol=METRIC_TOL,
                    collect_rows=diagnostic_csv_path is not None,
                )

                _add_electron_rank_diagnostics(
                    rows=batch_rows,
                    probs=probs,
                    targets=y,
                    features=batch.x,
                    batch_index=batch.batch,
                )
                diagnostic_rows.extend(batch_rows)

                batch_event_total = int(batch_event_metrics["event_total"])
                event_total += batch_event_total

                for key, value in batch_event_metrics.items():
                    if key == "event_total":
                        continue
                    if key in event_metric_median_inputs:
                        event_metric_median_inputs[key].append(float(value))
                    else:
                        event_metric_weighted_sums[key] = event_metric_weighted_sums.get(key, 0.0) + (
                            float(value) * batch_event_total
                        )

    avg_loss = total_loss / max(n_batches, 1)
    metrics: Dict[str, float] = {}

    if collect_metrics and len(all_targets) > 0:
        all_targets_np = torch.cat(all_targets, dim=0).numpy()
        all_probs_np = torch.cat(all_probs, dim=0).numpy()
        metrics.update(compute_soft_metrics(all_targets_np, all_probs_np))

        for key, weighted_sum in event_metric_weighted_sums.items():
            metrics[key] = float(weighted_sum / max(event_total, 1))

        for key, values in event_metric_median_inputs.items():
            metrics[key] = _safe_nanmedian(values)

        # Backwards-compatible alias for older code/plots. This remains the
        # strict argmax metric, so old runs compare apples-to-apples.
        metrics["event_main_accuracy"] = metrics["event_main_accuracy_strict"]
        metrics["event_total"] = float(event_total)

        if diagnostic_csv_path is not None:
            os.makedirs(os.path.dirname(diagnostic_csv_path) or ".", exist_ok=True)
            pd.DataFrame(diagnostic_rows).to_csv(diagnostic_csv_path, index=False)

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

    train_dataset = S2GraphDataset(X_train, Y_train, E_train, train_groups)
    val_dataset = S2GraphDataset(X_val, Y_val, E_val, val_groups)
    test_dataset = S2GraphDataset(X_test, Y_test, E_test, test_groups)

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
            mode=CHECKPOINT_MODE,
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

    best_metric = float("inf") if CHECKPOINT_MODE == "min" else float("-inf")
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
                "metric_tol": METRIC_TOL,
                "checkpoint_metric": CHECKPOINT_METRIC,
                "checkpoint_mode": CHECKPOINT_MODE,
                "loss_name": LOSS_NAME,
                "p_alt_loss_weight": P_ALT_LOSS_WEIGHT,
            },
        )

    for epoch in range(MAX_EPOCHS):
        train_loss, _ = run_epoch(model, train_loader, optimizer=optimizer, collect_metrics=False)
        val_loss, val_metrics = run_epoch(model, val_loader, optimizer=None, collect_metrics=True)
        val_metric = float(val_metrics[CHECKPOINT_METRIC])
        val_acc_strict = float(val_metrics["event_main_accuracy_strict"])
        val_acc_tie_aware = float(val_metrics["event_main_accuracy_tie_aware"])

        if scheduler is not None:
            if scheduler_name == "reduce_on_plateau":
                scheduler.step(val_metric)
            else:
                scheduler.step()

        is_better = (val_metric < best_metric) if CHECKPOINT_MODE == "min" else (val_metric > best_metric)
        if is_better:
            best_metric = val_metric
            bad_epochs = 0
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_metric": best_metric,
                "checkpoint_metric": CHECKPOINT_METRIC,
                "checkpoint_mode": CHECKPOINT_MODE,
                "metric_tol": METRIC_TOL,
                "loss_name": LOSS_NAME,
                "p_alt_loss_weight": P_ALT_LOSS_WEIGHT,
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
                    "val_event_main_accuracy_strict": val_acc_strict,
                    "val_event_main_accuracy_tie_aware": val_acc_tie_aware,
                    "val_event_main_top2_accuracy_tie_aware": val_metrics["event_main_top2_accuracy_tie_aware"],
                    "val_event_main_top3_accuracy_tie_aware": val_metrics["event_main_top3_accuracy_tie_aware"],
                    "val_event_main_tie_fraction": val_metrics["event_main_tie_fraction"],
                    "val_event_main_strict_miss_tie_correct_fraction": val_metrics["event_main_strict_miss_tie_correct_fraction"],
                    "val_event_main_mean_true_gap_at_pred": val_metrics["event_main_mean_true_gap_at_pred"],
                    "val_event_main_median_true_gap_at_pred": val_metrics["event_main_median_true_gap_at_pred"],
                    "val_event_main_median_true_margin": val_metrics["event_main_median_true_margin"],
                    "val_event_main_median_pred_margin": val_metrics["event_main_median_pred_margin"],
                    "val_event_p_main_mae": val_metrics["event_p_main_mae"],
                    "val_event_p_main_mse": val_metrics["event_p_main_mse"],
                    "val_event_p_main_rmse": val_metrics["event_p_main_rmse"],
                    "val_event_p_main_kl": val_metrics["event_p_main_kl"],
                    "val_event_p_main_sum_error_mae": val_metrics["event_p_main_sum_error_mae"],
                    "val_event_true_p_main_sum_mean": val_metrics["event_true_p_main_sum_mean"],
                    "val_event_pred_p_main_sum_mean": val_metrics["event_pred_p_main_sum_mean"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

        print(
            f"Epoch {epoch + 1:03d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_strict={val_acc_strict:.4f} | "
            f"val_tie_aware={val_acc_tie_aware:.4f} | "
            f"tie_frac={val_metrics['event_main_tie_fraction']:.4f} | "
            f"best_{CHECKPOINT_METRIC}={best_metric:.4f}"
        )

        if bad_epochs >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])

    val_diag_path = os.path.join(DIAGNOSTICS_DIR, "val_event_diagnostics.csv") if WRITE_EVENT_DIAGNOSTICS else None
    test_diag_path = os.path.join(DIAGNOSTICS_DIR, "test_event_diagnostics.csv") if WRITE_EVENT_DIAGNOSTICS else None

    if WRITE_EVENT_DIAGNOSTICS:
        _, _ = run_epoch(
            model,
            val_loader,
            optimizer=None,
            collect_metrics=True,
            diagnostic_csv_path=val_diag_path,
        )

    test_loss, test_metrics = run_epoch(
        model,
        test_loader,
        optimizer=None,
        collect_metrics=True,
        diagnostic_csv_path=test_diag_path,
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
        "test_event_main_accuracy_strict": float(test_metrics["event_main_accuracy_strict"]),
        "test_event_main_accuracy_tie_aware": float(test_metrics["event_main_accuracy_tie_aware"]),
        "test_event_main_top2_accuracy_tie_aware": float(test_metrics["event_main_top2_accuracy_tie_aware"]),
        "test_event_main_top3_accuracy_tie_aware": float(test_metrics["event_main_top3_accuracy_tie_aware"]),
        "test_event_main_tie_fraction": float(test_metrics["event_main_tie_fraction"]),
        "test_event_main_strict_miss_tie_correct_fraction": float(test_metrics["event_main_strict_miss_tie_correct_fraction"]),
        "test_event_main_mean_true_gap_at_pred": float(test_metrics["event_main_mean_true_gap_at_pred"]),
        "test_event_main_median_true_gap_at_pred": float(test_metrics["event_main_median_true_gap_at_pred"]),
        "test_event_main_median_true_margin": float(test_metrics["event_main_median_true_margin"]),
        "test_event_main_median_pred_margin": float(test_metrics["event_main_median_pred_margin"]),
        "test_event_p_main_mae": float(test_metrics["event_p_main_mae"]),
        "test_event_p_main_mse": float(test_metrics["event_p_main_mse"]),
        "test_event_p_main_rmse": float(test_metrics["event_p_main_rmse"]),
        "test_event_p_main_kl": float(test_metrics["event_p_main_kl"]),
        "test_event_p_main_sum_error_mean": float(test_metrics["event_p_main_sum_error_mean"]),
        "test_event_p_main_sum_error_mae": float(test_metrics["event_p_main_sum_error_mae"]),
        "test_event_true_p_main_sum_mean": float(test_metrics["event_true_p_main_sum_mean"]),
        "test_event_pred_p_main_sum_mean": float(test_metrics["event_pred_p_main_sum_mean"]),
        "best_val_metric": float(best_metric),
        "best_val_metric_name": CHECKPOINT_METRIC,
        "best_val_metric_mode": CHECKPOINT_MODE,
        "loss_name": LOSS_NAME,
        "p_alt_loss_weight": float(P_ALT_LOSS_WEIGHT),
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
