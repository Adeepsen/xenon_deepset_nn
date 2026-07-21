"""BCE-with-logits p_main regression/classification-style ablation.

This is intentionally separate from ``run_feature_sweep.py``.  It holds the
current Deep Sets architecture and preprocessing fixed, changing only:

* training loss: binary_cross_entropy_with_logits(logits, p_main)
* scheduler/checkpoint metric: validation BCE
* scheduler: ReduceLROnPlateau(factor=.7, patience=16, threshold=1e-3,
  threshold_mode='rel', min_lr=1e-6)

Because p_main is clipped to [0, 1], BCE supports its fractional labels.  Test
events remain untouched; choose using validation BCE/MSE/R² before a single
final held-out-test evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.loader import DataLoader as PyGDataLoader

from feature_sets import available_feature_sets, features_for_set
from run_feature_sweep import (
    BATCH_SIZE,
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    EventDataset,
    GraphDeepSetRegressor,
    LEARNING_RATE,
    NUM_WORKERS,
    PIN_MEMORY,
    WEIGHT_DECAY,
    prepare_data,
    set_seed,
)

try:
    import wandb
except ImportError:
    wandb = None

BCE_EARLY_STOPPING_PATIENCE = 10_000

MAX_EPOCHS = 1000
SCHEDULER_FACTOR = 0.7
SCHEDULER_PATIENCE = 16
SCHEDULER_THRESHOLD = 1e-3
SCHEDULER_THRESHOLD_MODE = "rel"
SCHEDULER_MIN_LR = 1e-6
GRADIENT_CLIP_NORM = 1.0
OUTPUT_ROOT = Path(__file__).resolve().parent / "bce_output"
WANDB_PROJECT = "xenon-graph-pooling-pmain-bce"


def run_epoch(
    model: nn.Module, loader: PyGDataLoader, optimizer: torch.optim.Optimizer | None = None
) -> Tuple[float, Dict[str, float]]:
    """Return BCE loss; validation additionally returns p_main regression metrics."""
    training = optimizer is not None
    model.train(training)
    total_bce, batches, targets, predictions = 0.0, 0, [], []
    with torch.enable_grad() if training else torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, batch.y)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
                optimizer.step()
            total_bce += float(loss.item())
            batches += 1
            if not training:
                targets.append(batch.y[:, 0].cpu())
                predictions.append(torch.sigmoid(logits[:, 0]).cpu())
    average_bce = total_bce / max(batches, 1)
    if training:
        return average_bce, {}
    target, prediction = torch.cat(targets).numpy(), torch.cat(predictions).numpy()
    return average_bce, {
        "p_main_mse": float(mean_squared_error(target, prediction)),
        "p_main_mae": float(mean_absolute_error(target, prediction)),
        "p_main_r2": float(r2_score(target, prediction)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-set", choices=available_feature_sets(), default="all_event_relative")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    features = features_for_set(args.feature_set)
    arrays, cut_stats = prepare_data(features, args.seed)
    x_train, y_train, _e_train, train_groups, x_val, y_val, _e_val, val_groups = arrays
    common = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)
    train_loader = PyGDataLoader(EventDataset(x_train, y_train, train_groups), shuffle=True, **common)
    val_loader = PyGDataLoader(EventDataset(x_val, y_val, val_groups), shuffle=False, **common)

    output_dir = OUTPUT_ROOT / args.feature_set / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_validation_bce_model.pt"
    summary_path = output_dir / "validation_summary.json"

    run = None
    if not args.no_wandb and wandb is not None:
        run = wandb.init(
            project=WANDB_PROJECT, name=args.run_name or f"{args.feature_set}-seed{args.seed}",
            config={
                "experiment": "pmain_bce_with_logits", "feature_set": args.feature_set, "features": features,
                "n_input_features": len(features), "seed": args.seed, "target": "p_main",
                "loss": "binary_cross_entropy_with_logits", "checkpoint_metric": "val_p_main_bce",
                "checkpoint_mode": "min", "test_policy": "not evaluated during selection",
                "architecture_reference": "event_relative_feature_sweep", "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE, "gradient_clip_norm": GRADIENT_CLIP_NORM,
                "scheduler": "ReduceLROnPlateau", "scheduler_mode": "min", "scheduler_factor": SCHEDULER_FACTOR,
                "scheduler_patience": SCHEDULER_PATIENCE, "scheduler_threshold": SCHEDULER_THRESHOLD,
                "scheduler_threshold_mode": SCHEDULER_THRESHOLD_MODE, "scheduler_min_lr": SCHEDULER_MIN_LR,
                **cut_stats,
            },
        )

    model = GraphDeepSetRegressor(len(features)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE,
        threshold=SCHEDULER_THRESHOLD, threshold_mode=SCHEDULER_THRESHOLD_MODE, min_lr=SCHEDULER_MIN_LR,
    )
    best_val_bce, best_epoch, stale_epochs = float("inf"), -1, 0
    for epoch in range(1, args.max_epochs + 1):
        train_bce, _ = run_epoch(model, train_loader, optimizer)
        val_bce, val_metrics = run_epoch(model, val_loader)
        scheduler.step(val_bce)
        if val_bce < best_val_bce:
            best_val_bce, best_epoch, stale_epochs = val_bce, epoch, 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(), "best_val_p_main_bce": best_val_bce,
                "feature_set": args.feature_set, "features": features, "seed": args.seed, "cut_stats": cut_stats,
            }, checkpoint_path)
        else:
            stale_epochs += 1
        metrics = {
            "epoch": epoch, "train_p_main_bce": train_bce, "val_p_main_bce": val_bce,
            "val_p_main_mse": val_metrics["p_main_mse"], "val_p_main_mae": val_metrics["p_main_mae"],
            "val_p_main_r2": val_metrics["p_main_r2"], "best_val_p_main_bce": best_val_bce,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if run:
            wandb.log(metrics)
        print(
            f"{args.feature_set} seed={args.seed} epoch={epoch:03d} train_bce={train_bce:.6f} "
            f"val_bce={val_bce:.6f} val_mse={val_metrics['p_main_mse']:.6f} "
            f"val_r2={val_metrics['p_main_r2']:.5f} best_bce={best_val_bce:.6f}"
        )
        if stale_epochs >= BCE_EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch}; best validation BCE was epoch {best_epoch}.")
            break

    summary = {
        "feature_set": args.feature_set, "features": features, "seed": args.seed, "best_epoch": best_epoch,
        "best_val_p_main_bce": best_val_bce, "best_validation_checkpoint": str(checkpoint_path),
        "held_out_test_evaluated": False, **cut_stats,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if run:
        wandb.log(summary)
        wandb.finish()


if __name__ == "__main__":
    main()
