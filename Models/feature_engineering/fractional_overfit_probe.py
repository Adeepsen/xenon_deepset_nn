"""Deliberately overfit a small, fraction-enriched training subset.

This is a diagnostic, not a candidate final model. It answers whether a wider,
unregularized Deep Sets model can substantially reduce *training* error on
events that contain at least one fractional p_main label. Test and validation
events are never used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import global_add_pool

from feature_sets import features_for_set
from run_feature_sweep import (
    BATCH_SIZE, DEVICE, EventDataset, NUM_WORKERS, PIN_MEMORY, prepare_data, set_seed,
)

try:
    import wandb
except ImportError:
    wandb = None


FEATURE_SET = "all_event_relative"
FRACTIONAL_LOW, FRACTIONAL_HIGH = 0.001, 0.999
LATENT_DIM, ENCODER_WIDTH, HEAD_WIDTH = 256, 512, 1024
ENCODER_DEPTH, HEAD_DEPTH = 5, 4
LEARNING_RATE, BATCH_SIZE = 1e-3, 1024
MAX_EPOCHS = 500
OUTPUT_ROOT = Path(__file__).resolve().parent / "overfit_probe_output"
WANDB_PROJECT = "xenon-pmain-fractional-overfit-probe"


def mlp(in_dim: int, hidden_dim: int, depth: int, out_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    for _ in range(depth):
        layers.extend((nn.Linear(in_dim, hidden_dim), nn.GELU()))
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)


class WideGraphDeepSetRegressor(nn.Module):
    """Current depth, but wider and deliberately without dropout."""
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.encoder = mlp(input_dim, ENCODER_WIDTH, ENCODER_DEPTH, LATENT_DIM)
        self.head = mlp(2 * LATENT_DIM, HEAD_WIDTH, HEAD_DEPTH, 1)

    def forward(self, batch) -> torch.Tensor:
        node_embedding = self.encoder(batch.x)
        event_embedding = global_add_pool(node_embedding, batch.batch)
        return self.head(torch.cat((node_embedding, event_embedding[batch.batch]), dim=-1))


def choose_fractional_events(groups: List[np.ndarray], y: np.ndarray, max_events: int, seed: int) -> List[np.ndarray]:
    eligible = [rows for rows in groups if np.any((y[rows, 0] > FRACTIONAL_LOW) & (y[rows, 0] < FRACTIONAL_HIGH))]
    if not eligible:
        raise RuntimeError("No training events contain fractional p_main labels.")
    chosen_indices = np.random.default_rng(seed).permutation(len(eligible))[:min(max_events, len(eligible))]
    return [eligible[index] for index in chosen_indices]


def evaluate(model: nn.Module, loader: PyGDataLoader) -> Dict[str, float]:
    model.eval()
    targets, predictions = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            targets.append(batch.y[:, 0].cpu())
            predictions.append(torch.sigmoid(model(batch)[:, 0]).cpu())
    target, prediction = torch.cat(targets).numpy(), torch.cat(predictions).numpy()
    fractional = (target > FRACTIONAL_LOW) & (target < FRACTIONAL_HIGH)
    return {
        "train_p_main_mse": float(mean_squared_error(target, prediction)),
        "train_p_main_mae": float(mean_absolute_error(target, prediction)),
        "train_fractional_mse": float(mean_squared_error(target[fractional], prediction[fractional])),
        "train_fractional_mae": float(mean_absolute_error(target[fractional], prediction[fractional])),
        "n_train_clusters": int(len(target)),
        "n_train_fractional_clusters": int(fractional.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-events", type=int, default=20_000)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)

    features = features_for_set(FEATURE_SET)
    arrays, cut_stats = prepare_data(features, args.seed)
    x_train, y_train, _events, train_groups, *_unused = arrays
    selected_groups = choose_fractional_events(train_groups, y_train, args.max_events, args.seed)
    dataset = EventDataset(x_train, y_train, selected_groups)
    common = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)
    train_loader = PyGDataLoader(dataset, shuffle=True, **common)
    metric_loader = PyGDataLoader(dataset, shuffle=False, **common)
    output_dir = OUTPUT_ROOT / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path, summary_path = output_dir / "lowest_train_fractional_mse.pt", output_dir / "summary.json"

    run = None
    if not args.no_wandb and wandb is not None:
        run = wandb.init(project=WANDB_PROJECT, name=f"fractional-overfit-seed{args.seed}", config={
            "experiment": "fractional_training_overfit_probe", "feature_set": FEATURE_SET, "features": features,
            "target": "p_main", "loss": "mse", "seed": args.seed, "max_events": args.max_events,
            "dropout": 0.0, "weight_decay": 0.0, "latent_dim": LATENT_DIM, "encoder_width": ENCODER_WIDTH,
            "head_width": HEAD_WIDTH, "encoder_depth": ENCODER_DEPTH, "head_depth": HEAD_DEPTH,
            "learning_rate": LEARNING_RATE, "test_policy": "not evaluated", **cut_stats,
        })

    model = WideGraphDeepSetRegressor(len(features)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    best_fractional_mse, best_epoch = float("inf"), -1
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(torch.sigmoid(model(batch)), batch.y)
            loss.backward()
            optimizer.step()
        metrics = {"epoch": epoch, **evaluate(model, metric_loader)}
        if metrics["train_fractional_mse"] < best_fractional_mse:
            best_fractional_mse, best_epoch = metrics["train_fractional_mse"], epoch
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "metrics": metrics, "features": features}, checkpoint_path)
        if run:
            wandb.log(metrics)
        print(f"epoch={epoch:03d} train_mse={metrics['train_p_main_mse']:.6f} fractional_mse={metrics['train_fractional_mse']:.6f} fractional_mae={metrics['train_fractional_mae']:.6f}")

    summary = {"best_epoch": best_epoch, "best_train_fractional_mse": best_fractional_mse, "checkpoint": str(checkpoint_path), "n_selected_events": len(selected_groups), "held_out_test_evaluated": False, **cut_stats}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    if run:
        wandb.log(summary)
        wandb.finish()


if __name__ == "__main__":
    main()
