"""Gated-attention pooling ablations for p_main MSE regression.

Run either a learned gated-sum context or append it to the current
sum/mean/max context. Gates use independent sigmoids—not softmax—so multiple
clusters in a multi-scatter event may contribute simultaneously.
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
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from feature_sets import features_for_set
from run_feature_sweep import (
    BATCH_SIZE, DEVICE, DROPOUT, EARLY_STOPPING_PATIENCE, ENCODER_DEPTH, EventDataset,
    HEAD_DEPTH, HEAD_HIDDEN, LATENT_DIM, LEARNING_RATE, NUM_WORKERS, PHI_HIDDEN,
    PIN_MEMORY, SCHEDULER_FACTOR, SCHEDULER_MIN_LR, SCHEDULER_PATIENCE, WEIGHT_DECAY,
    prepare_data, set_seed,
)

try:
    import wandb
except ImportError:
    wandb = None


FEATURE_SET = "all_event_relative"
FRACTIONAL_LOW, FRACTIONAL_HIGH = 0.001, 0.999
MAX_EPOCHS, GRADIENT_CLIP_NORM = 300, 1.0
OUTPUT_ROOT = Path(__file__).resolve().parent / "attention_pooling_output"
# Intentionally identical to the sum/mean/max pooling-ablation project.
WANDB_PROJECT = "xenon-graph-pooling-pmain-pooling-ablation"


def mlp(in_dim: int, hidden_dim: int, depth: int, dropout: float, out_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    for _ in range(depth):
        layers.extend((nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)))
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)


class GatedAttentionDeepSetRegressor(nn.Module):
    """Gated event context without a softmax single-cluster constraint."""
    def __init__(self, input_dim: int, pooling: str) -> None:
        super().__init__()
        if pooling not in {"gated_sum", "sum_mean_max_gated"}:
            raise ValueError(f"Unsupported pooling mode: {pooling}")
        self.pooling = pooling
        self.encoder = mlp(input_dim, PHI_HIDDEN, ENCODER_DEPTH, DROPOUT, LATENT_DIM)
        self.gate = nn.Sequential(
            nn.Linear(LATENT_DIM, LATENT_DIM), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(LATENT_DIM, 1),
        )
        # node + gated context, or node + sum + mean + max + gated context.
        context_count = 1 if pooling == "gated_sum" else 4
        self.head = mlp((1 + context_count) * LATENT_DIM, HEAD_HIDDEN, HEAD_DEPTH, DROPOUT, 1)

    def forward(self, batch) -> torch.Tensor:
        embedding = self.encoder(batch.x)
        gates = torch.sigmoid(self.gate(embedding))
        gated_sum = global_add_pool(gates * embedding, batch.batch)
        if self.pooling == "gated_sum":
            context = gated_sum[batch.batch]
        else:
            context = torch.cat((
                global_add_pool(embedding, batch.batch)[batch.batch],
                global_mean_pool(embedding, batch.batch)[batch.batch],
                global_max_pool(embedding, batch.batch)[batch.batch],
                gated_sum[batch.batch],
            ), dim=-1)
        return self.head(torch.cat((embedding, context), dim=-1))


def prediction_metrics(target: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
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
            if not training:
                targets.append(batch.y[:, 0].cpu())
                predictions.append(torch.sigmoid(logits[:, 0]).cpu())
    if training:
        return total_loss / max(batches, 1), {}
    target, prediction = torch.cat(targets).numpy(), torch.cat(predictions).numpy()
    return total_loss / max(batches, 1), prediction_metrics(target, prediction)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pooling", choices=("gated_sum", "sum_mean_max_gated"), required=True)
    parser.add_argument("--seed", type=int, default=42)
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
    output_dir = OUTPUT_ROOT / args.pooling / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path, summary_path = output_dir / "best_validation_model.pt", output_dir / "validation_summary.json"

    run = None
    if not args.no_wandb and wandb is not None:
        run = wandb.init(project=WANDB_PROJECT, name=f"{args.pooling}-seed{args.seed}", config={
            "experiment": "gated_attention_pooling_ablation", "pooling": args.pooling,
            "gating": "independent_sigmoid_gates_then_sum", "feature_set": FEATURE_SET, "features": features,
            "target": "p_main", "loss": "mse", "seed": args.seed, "checkpoint_metric": "val_p_main_mse",
            "checkpoint_mode": "min", "test_policy": "not evaluated during selection",
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
            "gradient_clip_norm": GRADIENT_CLIP_NORM, **cut_stats,
        })
    model = GatedAttentionDeepSetRegressor(len(features), args.pooling).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR)
    best_val_mse, best_epoch, stale_epochs = float("inf"), -1, 0
    for epoch in range(1, args.max_epochs + 1):
        train_mse, _ = run_epoch(model, train_loader, optimizer)
        _, val = run_epoch(model, val_loader)
        scheduler.step(val["p_main_mse"])
        if val["p_main_mse"] < best_val_mse:
            best_val_mse, best_epoch, stale_epochs = val["p_main_mse"], epoch, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "best_val_p_main_mse": best_val_mse, "pooling": args.pooling, "features": features, "seed": args.seed, "cut_stats": cut_stats}, checkpoint_path)
        else:
            stale_epochs += 1
        logged = {"epoch": epoch, "train_p_main_mse": train_mse, "val_p_main_mse": val["p_main_mse"], "val_p_main_mae": val["p_main_mae"], "val_p_main_r2": val["p_main_r2"], "val_fractional_mse": val["fractional_mse"], "val_fractional_mae": val["fractional_mae"], "best_val_p_main_mse": best_val_mse, "learning_rate": optimizer.param_groups[0]["lr"]}
        if run:
            wandb.log(logged)
        print(f"{args.pooling} epoch={epoch:03d} train_mse={train_mse:.6f} val_mse={val['p_main_mse']:.6f} val_frac={val['fractional_mse']:.6f} val_r2={val['p_main_r2']:.5f} best={best_val_mse:.6f}")
        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            break
    summary = {"feature_set": FEATURE_SET, "pooling": args.pooling, "seed": args.seed, "best_epoch": best_epoch, "best_val_p_main_mse": best_val_mse, "best_validation_checkpoint": str(checkpoint_path), "held_out_test_evaluated": False, **cut_stats}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    if run:
        wandb.log(summary)
        wandb.finish()


if __name__ == "__main__":
    main()
