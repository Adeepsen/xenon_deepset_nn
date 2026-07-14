#!/usr/bin/env python3
"""Create a final-style p_main regression preview from an existing checkpoint.

This is intentionally an evaluation-only script: it does not alter source data,
training caches, checkpoints, or the active Optuna study.  Its output is a
*preliminary* test R²/scatter because the selected checkpoint was trained with
the previous two-target BCE/combined-objective setup and without the proposed
event-sum cut.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader as PyGDataLoader


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA = ROOT / "data" / "s2_tag_training_clusters.npy"
OPTUNA_SCRIPT = ROOT / "Models" / "deepset_variations" / "optuna_sweep" / "deepset_sweep.py"
DEFAULT_CHECKPOINT = ROOT / "Models" / "deepset_variations" / "optuna_sweep" / "optuna_graph_pooling_checkpoints" / "trial_0004" / "best.pt"
OUTPUT_DIR = ROOT / "analysis" / "wandb_output" / "preliminary_pmain_regression"
TOP13_NS = 192_600.0
SEED = 42


def load_training_module():
    spec = importlib.util.spec_from_file_location("optuna_training", OPTUNA_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {OPTUNA_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_original_test_split() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Rebuild the original fiducial event split without writing the training cache."""
    data = np.load(SOURCE_DATA)
    df = pd.DataFrame(data)
    event_min_drift = df.groupby("event_number")["drift_time_mean"].min()
    fiducial_events = event_min_drift[event_min_drift >= TOP13_NS].index.to_numpy()
    df = df[df["event_number"].isin(fiducial_events)].copy()
    # Match the checkpoint's original preprocessing, including its two labels.
    df["p_alt"] = df["p_alt"].clip(0, 1)

    event_ids = df["event_number"].unique()
    train_events, temp_events = train_test_split(event_ids, test_size=0.30, random_state=SEED, shuffle=True)
    _, test_events = train_test_split(temp_events, test_size=0.50, random_state=SEED, shuffle=True)
    train_df = df[df["event_number"].isin(train_events)].copy()
    test_df = df[df["event_number"].isin(test_events)].copy()

    features = ["x", "y", "n_electrons_interface", "drift_time_mean", "drift_time_spread"]
    scaler = StandardScaler().fit(train_df[features])
    x_test = scaler.transform(test_df[features]).astype(np.float32)
    y_test = test_df[["p_main", "p_alt"]].to_numpy(dtype=np.float32, copy=True)
    event_test = test_df["event_number"].to_numpy(copy=True)
    order = np.argsort(event_test, kind="mergesort")
    sorted_events = event_test[order]
    boundaries = np.flatnonzero(sorted_events[1:] != sorted_events[:-1]) + 1
    groups = list(np.split(order, boundaries))
    return x_test, y_test, event_test, groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sample-size", type=int, default=100_000)
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    params = checkpoint["params"]
    module = load_training_module()
    x_test, y_test, event_test, groups = prepare_original_test_split()
    dataset = module.S2GraphDataset(x_test, y_test, event_test, groups)
    loader = PyGDataLoader(dataset, batch_size=int(params["batch_size"]), shuffle=False, num_workers=0)
    model = module.build_model_from_params(params).to(module.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pred_parts, target_parts = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(module.DEVICE)
            pred_parts.append(torch.sigmoid(model(batch))[:, 0].cpu().numpy())
            target_parts.append(batch.y[:, 0].cpu().numpy())
    prediction = np.concatenate(pred_parts)
    target = np.concatenate(target_parts)
    r2 = float(r2_score(target, prediction))
    mse = float(mean_squared_error(target, prediction))

    rng = np.random.default_rng(SEED)
    n = min(args.sample_size, len(target))
    sample = rng.choice(len(target), size=n, replace=False)
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(target[sample], prediction[sample], s=2, alpha=0.08, rasterized=True)
    ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1, label="ideal: y=x")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="True p_main", ylabel="Predicted p_main",
           title=f"Preliminary test p_main regression (R² = {r2:.4f})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    figure_path = OUTPUT_DIR / "test_pmain_predicted_vs_true.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)

    result = {
        "status": "preliminary: old two-target BCE/combined-objective checkpoint; no proposed p_main-sum cut",
        "checkpoint": str(args.checkpoint),
        "trial_number": checkpoint.get("trial_number"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "test_clusters": int(len(target)),
        "test_events": int(len(groups)),
        "test_p_main_r2": r2,
        "test_p_main_mse": mse,
        "scatter_sample_size": int(n),
        "figure": str(figure_path),
    }
    (OUTPUT_DIR / "test_pmain_regression_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
