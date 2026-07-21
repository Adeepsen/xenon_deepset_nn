"""Validation ablation: wider, lightly regularized Deep Sets p_main regressor.

This converts the fractional-overfit probe into a fair generalization test:
all corrected event-split data are used, MSE remains the objective, and both
overall and fractional p_main metrics are logged for train and validation.
Test events are not evaluated during model selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import global_add_pool

from feature_sets import features_for_set
from run_feature_sweep import (
    BATCH_SIZE, DEVICE, EARLY_STOPPING_PATIENCE, EventDataset, NUM_WORKERS, PIN_MEMORY,
    SCHEDULER_FACTOR, SCHEDULER_MIN_LR, SCHEDULER_PATIENCE, prepare_data, set_seed,
)

try:
    import wandb
except ImportError:
    wandb = None


FEATURE_SET = "all_event_relative"
FRACTIONAL_LOW, FRACTIONAL_HIGH = 0.001, 0.999
LATENT_DIM, ENCODER_WIDTH, HEAD_WIDTH = 256, 512, 1024
ENCODER_DEPTH, HEAD_DEPTH = 5, 4
DEFAULT_DROPOUT, DEFAULT_WEIGHT_DECAY, DEFAULT_LEARNING_RATE = 0.01, 1e-6, 3e-4
MAX_EPOCHS, GRADIENT_CLIP_NORM = 300, 1.0
OUTPUT_ROOT = Path(__file__).resolve().parent / "wide_lowreg_output"
WANDB_PROJECT = "xenon-graph-pooling-pmain-wide-lowreg"


def mlp(in_dim: int, hidden_dim: int, depth: int, dropout: float, out_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    for _ in range(depth):
        layers.extend((nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)))
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)


class WideGraphDeepSetRegressor(nn.Module):
    """Same sum-pooling architecture, widened from 128/256/512 to 256/512/1024."""
    def __init__(self, input_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = mlp(input_dim, ENCODER_WIDTH, ENCODER_DEPTH, dropout, LATENT_DIM)
        self.head = mlp(2 * LATENT_DIM, HEAD_WIDTH, HEAD_DEPTH, dropout, 1)

    def forward(self, batch) -> torch.Tensor:
        node_embedding = self.encoder(batch.x)
        event_embedding = global_add_pool(node_embedding, batch.batch)
        return self.head(torch.cat((node_embedding, event_embedding[batch.batch]), dim=-1))


def metrics_for_predictions(target: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    fractional = (target > FRACTIONAL_LOW) & (target < FRACTIONAL_HIGH)
    return {
        "p_main_mse": float(mean_squared_error(target, prediction)),
        "p_main_mae": float(mean_absolute_error(target, prediction)),
        "p_main_r2": float(r2_score(target, prediction)),
        "fractional_mse": float(mean_squared_error(target[fractional], prediction[fractional])),
        "fractional_mae": float(mean_absolute_error(target[fractional], prediction[fractional])),
        "n_fractional_clusters": int(fractional.sum()),
    }


def run_epoch(model: nn.Module, loader: PyGDataLoader, optimizer: torch.optim.Optimizer | None = None) -> Tuple[float, Dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    total_loss, batches, targets, predictions = 0.0, 0, [], []
    with torch.enable_grad() if training else torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = nn.functional.mse_loss(torch.sigmoid(logits), batch.y)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
                optimizer.step()
            total_loss += float(loss.item())
            batches += 1
            # Metrics are needed every epoch to distinguish capacity from
            # generalization; collect predictions in both train and eval mode.
            targets.append(batch.y[:, 0].detach().cpu())
            predictions.append(torch.sigmoid(logits[:, 0]).detach().cpu())
    target, prediction = torch.cat(targets).numpy(), torch.cat(predictions).numpy()
    return total_loss / max(batches, 1), metrics_for_predictions(target, prediction)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    features = features_for_set(FEATURE_SET)
    arrays, cut_stats = prepare_data(features, args.seed)
    x_train, y_train, _e_train, train_groups, x_val, y_val, _e_val, val_groups = arrays
    common = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)
    train_loader = PyGDataLoader(EventDataset(x_train, y_train, train_groups), shuffle=True, **common)
    val_loader = PyGDataLoader(EventDataset(x_val, y_val, val_groups), shuffle=False, **common)
    output_dir = OUTPUT_ROOT / f"dropout_{args.dropout:g}_wd_{args.weight_decay:g}" / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path, summary_path = output_dir / "best_validation_model.pt", output_dir / "validation_summary.json"

    run = None
    if not args.no_wandb and wandb is not None:
        run = wandb.init(project=WANDB_PROJECT, name=f"wide-lowreg-drop{args.dropout:g}-wd{args.weight_decay:g}-seed{args.seed}", config={
            "experiment": "wide_low_regularization", "feature_set": FEATURE_SET, "features": features,
            "target": "p_main", "loss": "mse", "pooling": "sum", "seed": args.seed,
            "latent_dim": LATENT_DIM, "encoder_width": ENCODER_WIDTH, "head_width": HEAD_WIDTH,
            "encoder_depth": ENCODER_DEPTH, "head_depth": HEAD_DEPTH, "dropout": args.dropout,
            "weight_decay": args.weight_decay, "learning_rate": args.learning_rate,
            "gradient_clip_norm": GRADIENT_CLIP_NORM, "checkpoint_metric": "val_p_main_mse",
            "checkpoint_mode": "min", "test_policy": "not evaluated during selection", **cut_stats,
        })
    model = WideGraphDeepSetRegressor(len(features), args.dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR)
    best_val_mse, best_epoch, stale_epochs = float("inf"), -1, 0
    for epoch in range(1, args.max_epochs + 1):
        train_loss, train = run_epoch(model, train_loader, optimizer)
        val_loss, val = run_epoch(model, val_loader)
        scheduler.step(val["p_main_mse"])
        if val["p_main_mse"] < best_val_mse:
            best_val_mse, best_epoch, stale_epochs = val["p_main_mse"], epoch, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "best_val_p_main_mse": best_val_mse, "features": features, "seed": args.seed, "cut_stats": cut_stats}, checkpoint_path)
        else:
            stale_epochs += 1
        metrics = {
            "epoch": epoch, "train_p_main_mse": train["p_main_mse"], "train_p_main_mae": train["p_main_mae"],
            "train_p_main_r2": train["p_main_r2"], "train_fractional_mse": train["fractional_mse"],
            "train_fractional_mae": train["fractional_mae"], "val_p_main_mse": val["p_main_mse"],
            "val_p_main_mae": val["p_main_mae"], "val_p_main_r2": val["p_main_r2"],
            "val_fractional_mse": val["fractional_mse"], "val_fractional_mae": val["fractional_mae"],
            "best_val_p_main_mse": best_val_mse, "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if run:
            wandb.log(metrics)
        print(f"epoch={epoch:03d} train_mse={train_loss:.6f} train_frac={train['fractional_mse']:.6f} val_mse={val_loss:.6f} val_frac={val['fractional_mse']:.6f} val_r2={val['p_main_r2']:.5f} best={best_val_mse:.6f}")
        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            break
    summary = {"feature_set": FEATURE_SET, "pooling": "sum", "seed": args.seed, "best_epoch": best_epoch, "best_val_p_main_mse": best_val_mse, "best_validation_checkpoint": str(checkpoint_path), "held_out_test_evaluated": False, **cut_stats}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    if run:
        wandb.log(summary)
        wandb.finish()


if __name__ == "__main__":
    main()
