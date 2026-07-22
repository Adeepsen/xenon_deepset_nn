"""BCE counterpart of the fraction-enriched sum/mean/max pooling experiment.

It uses the identical event sampler, input features, and pooling architecture
as ``deepset_pmain_fraction_enriched_pooling.py``. The only training changes
are BCE-with-logits and the advisor-specified BCE scheduler. It logs to the
same W&B project for direct comparison with the MSE run.
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
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from feature_sets import features_for_set
from run_feature_sweep import (
    BATCH_SIZE, DEVICE, DROPOUT, ENCODER_DEPTH, EventDataset, HEAD_DEPTH, HEAD_HIDDEN,
    LATENT_DIM, LEARNING_RATE, NUM_WORKERS, PHI_HIDDEN, PIN_MEMORY, WEIGHT_DECAY,
    prepare_data, set_seed,
)

try:
    import wandb
except ImportError:
    wandb = None


FEATURE_SET = "all_event_relative"
FRACTIONAL_LOW, FRACTIONAL_HIGH = 0.001, 0.999
MAX_EPOCHS, EARLY_STOPPING_PATIENCE, GRADIENT_CLIP_NORM = 300, 60, 1.0
SCHEDULER_FACTOR, SCHEDULER_PATIENCE = 0.7, 16
SCHEDULER_THRESHOLD, SCHEDULER_THRESHOLD_MODE, SCHEDULER_MIN_LR = 1e-3, "rel", 1e-6
OUTPUT_ROOT = Path(__file__).resolve().parent / "fraction_enriched_pooling_bce_output"
# Intentionally identical to the MSE experiment's project.
WANDB_PROJECT = "xenon-graph-pooling-pmain-fraction-enriched"


def mlp(in_dim: int, hidden_dim: int, depth: int, dropout: float, out_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    for _ in range(depth):
        layers.extend((nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)))
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)


class SumMeanMaxDeepSetRegressor(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.encoder = mlp(input_dim, PHI_HIDDEN, ENCODER_DEPTH, DROPOUT, LATENT_DIM)
        self.head = mlp(4 * LATENT_DIM, HEAD_HIDDEN, HEAD_DEPTH, DROPOUT, 1)

    def forward(self, batch) -> torch.Tensor:
        embedding = self.encoder(batch.x)
        context = torch.cat((
            global_add_pool(embedding, batch.batch)[batch.batch],
            global_mean_pool(embedding, batch.batch)[batch.batch],
            global_max_pool(embedding, batch.batch)[batch.batch],
        ), dim=-1)
        return self.head(torch.cat((embedding, context), dim=-1))


def event_fractional_mask(groups: List[np.ndarray], y: np.ndarray) -> np.ndarray:
    return np.asarray([np.any((y[rows, 0] > FRACTIONAL_LOW) & (y[rows, 0] < FRACTIONAL_HIGH)) for rows in groups], dtype=bool)


def metrics(target: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    fractional = (target > FRACTIONAL_LOW) & (target < FRACTIONAL_HIGH)
    return {
        "p_main_mse": float(mean_squared_error(target, prediction)),
        "p_main_mae": float(mean_absolute_error(target, prediction)),
        "p_main_r2": float(r2_score(target, prediction)),
        "fractional_mse": float(mean_squared_error(target[fractional], prediction[fractional])),
        "fractional_mae": float(mean_absolute_error(target[fractional], prediction[fractional])),
    }


def run_epoch(model: nn.Module, loader: PyGDataLoader, optimizer: torch.optim.Optimizer | None = None) -> Tuple[float, Dict[str, float]]:
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
    if training:
        return total_bce / max(batches, 1), {}
    target, prediction = torch.cat(targets).numpy(), torch.cat(predictions).numpy()
    return total_bce / max(batches, 1), metrics(target, prediction)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fractional-event-weight", type=float, default=4.0)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    if args.fractional_event_weight < 1.0:
        parser.error("--fractional-event-weight must be at least 1.0")
    set_seed(args.seed)

    features = features_for_set(FEATURE_SET)
    arrays, cut_stats = prepare_data(features, args.seed)
    x_train, y_train, _e_train, train_groups, x_val, y_val, _e_val, val_groups = arrays
    train_dataset, val_dataset = EventDataset(x_train, y_train, train_groups), EventDataset(x_val, y_val, val_groups)
    fraction_event_mask = event_fractional_mask(train_groups, y_train)
    weights = np.where(fraction_event_mask, args.fractional_event_weight, 1.0)
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(train_dataset), replacement=True)
    common = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)
    train_loader = PyGDataLoader(train_dataset, sampler=sampler, shuffle=False, **common)
    val_loader = PyGDataLoader(val_dataset, shuffle=False, **common)
    raw_fractional_event_rate = float(fraction_event_mask.mean())
    sampled_fractional_event_rate = float((fraction_event_mask * weights).sum() / weights.sum())
    output_dir = OUTPUT_ROOT / f"event_weight_{args.fractional_event_weight:g}" / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path, summary_path = output_dir / "best_validation_bce_model.pt", output_dir / "validation_summary.json"

    run = None
    if not args.no_wandb and wandb is not None:
        run = wandb.init(project=WANDB_PROJECT, name=f"bce-sum-mean-max-fracweight{args.fractional_event_weight:g}-seed{args.seed}", config={
            "experiment": "fraction_enriched_event_sampling_sum_mean_max", "loss": "binary_cross_entropy_with_logits",
            "feature_set": FEATURE_SET, "features": features, "target": "p_main", "pooling": "sum_mean_max",
            "seed": args.seed, "fractional_event_weight": args.fractional_event_weight,
            "raw_fractional_event_rate": raw_fractional_event_rate,
            "expected_sampled_fractional_event_rate": sampled_fractional_event_rate,
            "sampling": "WeightedRandomSampler over full events, replacement=True",
            "checkpoint_metric": "val_p_main_bce", "checkpoint_mode": "min", "test_policy": "not evaluated during selection",
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
            "gradient_clip_norm": GRADIENT_CLIP_NORM, "scheduler": "ReduceLROnPlateau", "scheduler_mode": "min",
            "scheduler_factor": SCHEDULER_FACTOR, "scheduler_patience": SCHEDULER_PATIENCE,
            "scheduler_threshold": SCHEDULER_THRESHOLD, "scheduler_threshold_mode": SCHEDULER_THRESHOLD_MODE,
            "scheduler_min_lr": SCHEDULER_MIN_LR, **cut_stats,
        })
    model = SumMeanMaxDeepSetRegressor(len(features)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, threshold=SCHEDULER_THRESHOLD, threshold_mode=SCHEDULER_THRESHOLD_MODE, min_lr=SCHEDULER_MIN_LR)
    best_val_bce, best_epoch, stale_epochs = float("inf"), -1, 0
    for epoch in range(1, args.max_epochs + 1):
        train_bce, _ = run_epoch(model, train_loader, optimizer)
        val_bce, val = run_epoch(model, val_loader)
        scheduler.step(val_bce)
        if val_bce < best_val_bce:
            best_val_bce, best_epoch, stale_epochs = val_bce, epoch, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "best_val_p_main_bce": best_val_bce, "features": features, "seed": args.seed, "fractional_event_weight": args.fractional_event_weight, "cut_stats": cut_stats}, checkpoint_path)
        else:
            stale_epochs += 1
        logged = {"epoch": epoch, "train_p_main_bce": train_bce, "val_p_main_bce": val_bce, "val_p_main_mse": val["p_main_mse"], "val_p_main_mae": val["p_main_mae"], "val_p_main_r2": val["p_main_r2"], "val_fractional_mse": val["fractional_mse"], "val_fractional_mae": val["fractional_mae"], "best_val_p_main_bce": best_val_bce, "learning_rate": optimizer.param_groups[0]["lr"]}
        if run:
            wandb.log(logged)
        print(f"epoch={epoch:03d} train_bce={train_bce:.6f} val_bce={val_bce:.6f} val_mse={val['p_main_mse']:.6f} val_frac={val['fractional_mse']:.6f} val_r2={val['p_main_r2']:.5f} best_bce={best_val_bce:.6f}")
        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            break
    summary = {"feature_set": FEATURE_SET, "pooling": "sum_mean_max", "loss": "binary_cross_entropy_with_logits", "seed": args.seed, "fractional_event_weight": args.fractional_event_weight, "raw_fractional_event_rate": raw_fractional_event_rate, "expected_sampled_fractional_event_rate": sampled_fractional_event_rate, "best_epoch": best_epoch, "best_val_p_main_bce": best_val_bce, "best_validation_checkpoint": str(checkpoint_path), "held_out_test_evaluated": False, **cut_stats}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    if run:
        wandb.log(summary)
        wandb.finish()


if __name__ == "__main__":
    main()
