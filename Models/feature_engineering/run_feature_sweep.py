"""Validation-only ablation runner for event-relative p_main features.

Example: python Models/feature_engineering/run_feature_sweep.py --feature-set electron_fraction --seed 42
The held-out test split is deliberately never loaded or evaluated here.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, List, Tuple

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

from feature_sets import add_event_relative_features, available_feature_sets, features_for_set

try:
    import wandb
except ImportError:
    wandb = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = Path(os.environ.get("XENON_DATA_PATH", PROJECT_ROOT / "data" / "s2_tag_training_clusters.npy"))
OUTPUT_ROOT = Path(os.environ.get("XENON_FEATURE_SWEEP_OUTPUT_DIR", Path(__file__).resolve().parent / "feature_sweep_output"))
TARGET, EVENT_COL, TOP13_NS = "p_main", "event_number", 192_600.0
LATENT_DIM, PHI_HIDDEN, HEAD_HIDDEN = 128, 256, 512
ENCODER_DEPTH, HEAD_DEPTH, DROPOUT = 5, 4, 0.04848861422838413
LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE = 3e-4, 1.076825417908119e-5, 1024
MAX_EPOCHS, EARLY_STOPPING_PATIENCE = 300, 60
SCHEDULER_PATIENCE, SCHEDULER_FACTOR, SCHEDULER_MIN_LR = 16, 0.7, 1e-6
NUM_WORKERS, PIN_MEMORY = 4, True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_event_groups(event_ids: np.ndarray) -> List[np.ndarray]:
    order = np.argsort(event_ids, kind="mergesort")
    sorted_events = event_ids[order]
    return list(np.split(order, np.flatnonzero(sorted_events[1:] != sorted_events[:-1]) + 1))


def prepare_data(feature_names: List[str], seed: int) -> Tuple[Tuple[np.ndarray, ...], dict[str, Any]]:
    """Apply approved label handling and split/scaling without touching test data."""
    df = pd.DataFrame(np.load(RAW_DATA_PATH))
    raw_events, raw_clusters = int(df[EVENT_COL].nunique()), int(len(df))
    min_drift = df.groupby(EVENT_COL)["drift_time_mean"].min()
    excluded_events = min_drift[min_drift < TOP13_NS].index
    df = df[~df[EVENT_COL].isin(excluded_events)].copy()
    clipped_above, clipped_below = int((df[TARGET] > 1).sum()), int((df[TARGET] < 0).sum())
    df[TARGET] = df[TARGET].clip(0.0, 1.0)
    df = add_event_relative_features(df, EVENT_COL)
    train_events, remainder = train_test_split(df[EVENT_COL].unique(), test_size=0.30, random_state=seed, shuffle=True)
    val_events, _test_events = train_test_split(remainder, test_size=0.50, random_state=seed, shuffle=True)
    train_df, val_df = df[df[EVENT_COL].isin(train_events)].copy(), df[df[EVENT_COL].isin(val_events)].copy()
    scaler = StandardScaler().fit(train_df[feature_names])
    for split in (train_df, val_df):
        split[feature_names] = scaler.transform(split[feature_names]).astype(np.float32)

    def arrays(split: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        events = split[EVENT_COL].to_numpy(copy=True)
        return (split[feature_names].to_numpy(np.float32, copy=True), split[[TARGET]].to_numpy(np.float32, copy=True), events, build_event_groups(events))

    stats = {
        "events_raw": raw_events, "clusters_raw": raw_clusters, "events_removed_fiducial": int(len(excluded_events)),
        "events_after_fiducial": int(df[EVENT_COL].nunique()), "clusters_after_fiducial": int(len(df)),
        "individual_pmain_values_clipped_above_one": clipped_above, "individual_pmain_values_clipped_below_zero": clipped_below,
        "event_split": "70% train / 15% validation / 15% held-out test; event-level", "held_out_test_used": False,
    }
    return (*arrays(train_df), *arrays(val_df)), stats


class EventDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, groups: Iterable[np.ndarray]):
        self.x, self.y, self.groups = x, y, list(groups)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> PYGData:
        rows = self.groups[index]
        return PYGData(x=torch.from_numpy(self.x[rows]), y=torch.from_numpy(self.y[rows]))


def mlp(in_dim: int, hidden_dim: int, depth: int, dropout: float, out_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    for _ in range(depth):
        layers.extend((nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)))
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)


class GraphDeepSetRegressor(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.encoder = mlp(input_dim, PHI_HIDDEN, ENCODER_DEPTH, DROPOUT, LATENT_DIM)
        self.head = mlp(2 * LATENT_DIM, HEAD_HIDDEN, HEAD_DEPTH, DROPOUT, 1)

    def forward(self, batch: PYGData) -> torch.Tensor:
        embedding = self.encoder(batch.x)
        pooled = global_add_pool(embedding, batch.batch)
        return self.head(torch.cat((embedding, pooled[batch.batch]), dim=-1))


def run_epoch(model: nn.Module, loader: PyGDataLoader, optimizer: torch.optim.Optimizer | None = None) -> Tuple[float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    total, batches, targets, predictions = 0.0, 0, [], []
    with torch.enable_grad() if training else torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = nn.functional.mse_loss(torch.sigmoid(logits), batch.y)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total, batches = total + float(loss.item()), batches + 1
            if not training:
                targets.append(batch.y[:, 0].cpu())
                predictions.append(torch.sigmoid(logits[:, 0]).cpu())
    if training:
        return total / max(batches, 1), {}
    target, prediction = torch.cat(targets).numpy(), torch.cat(predictions).numpy()
    return total / max(batches, 1), {"p_main_mse": float(mean_squared_error(target, prediction)), "p_main_mae": float(mean_absolute_error(target, prediction)), "p_main_r2": float(r2_score(target, prediction))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-set", choices=available_feature_sets())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--list-feature-sets", action="store_true")
    args = parser.parse_args()
    if args.list_feature_sets:
        print("\n".join(available_feature_sets()))
        return
    if args.feature_set is None:
        parser.error("--feature-set is required unless --list-feature-sets is used")
    set_seed(args.seed)
    features = features_for_set(args.feature_set)
    arrays, cut_stats = prepare_data(features, args.seed)
    x_train, y_train, _e_train, train_groups, x_val, y_val, _e_val, val_groups = arrays
    common = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)
    train_loader = PyGDataLoader(EventDataset(x_train, y_train, train_groups), shuffle=True, **common)
    val_loader = PyGDataLoader(EventDataset(x_val, y_val, val_groups), shuffle=False, **common)
    output_dir = OUTPUT_ROOT / args.feature_set / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path, summary_path = output_dir / "best_validation_model.pt", output_dir / "validation_summary.json"
    run = None
    if not args.no_wandb and wandb is not None:
        run = wandb.init(project="xenon-graph-pooling-pmain-feature-sweep", name=args.run_name or f"{args.feature_set}-seed{args.seed}", config={
            "experiment": "event_relative_feature_ablation", "feature_set": args.feature_set, "features": features,
            "n_input_features": len(features), "seed": args.seed, "target": TARGET, "loss": "mse",
            "checkpoint_metric": "val_p_main_mse", "checkpoint_mode": "min", "test_policy": "not evaluated during feature screening",
            "pooling": "sum", "latent_dim": LATENT_DIM, "phi_hidden": PHI_HIDDEN, "head_hidden": HEAD_HIDDEN,
            "encoder_depth": ENCODER_DEPTH, "head_depth": HEAD_DEPTH, "dropout": DROPOUT, "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE, "gradient_clip_norm": 1.0, **cut_stats,
        })
    model = GraphDeepSetRegressor(len(features)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR)
    best_mse, best_epoch, stale = float("inf"), -1, 0
    for epoch in range(1, args.max_epochs + 1):
        train_mse, _ = run_epoch(model, train_loader, optimizer)
        _, val = run_epoch(model, val_loader)
        val_mse = val["p_main_mse"]
        scheduler.step(val_mse)
        if val_mse < best_mse:
            best_mse, best_epoch, stale = val_mse, epoch, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "best_val_p_main_mse": best_mse, "feature_set": args.feature_set, "features": features, "seed": args.seed, "cut_stats": cut_stats}, checkpoint_path)
        else:
            stale += 1
        metrics = {"epoch": epoch, "train_p_main_mse": train_mse, "val_p_main_mse": val_mse, "val_p_main_mae": val["p_main_mae"], "val_p_main_r2": val["p_main_r2"], "best_val_p_main_mse": best_mse, "learning_rate": optimizer.param_groups[0]["lr"]}
        if run:
            wandb.log(metrics)
        print(f"{args.feature_set} seed={args.seed} epoch={epoch:03d} train={train_mse:.6f} val={val_mse:.6f} r2={val['p_main_r2']:.5f} best={best_mse:.6f}")
        if stale >= EARLY_STOPPING_PATIENCE:
            break
    summary = {"feature_set": args.feature_set, "features": features, "seed": args.seed, "best_epoch": best_epoch, "best_val_p_main_mse": best_mse, "best_validation_checkpoint": str(checkpoint_path), "held_out_test_evaluated": False, **cut_stats}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if run:
        wandb.log(summary)
        wandb.finish()


if __name__ == "__main__":
    main()
