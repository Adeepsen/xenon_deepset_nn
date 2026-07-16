#!/usr/bin/env python3
"""Diagnostics for a completed pure-p_main-MSE checkpoint.

Produces an electron-argmax baseline on the exact held-out split, per-cluster
metrics by true p_main bin and event multiplicity, plus residual plots against
electron count and drift time.  It does not retrain or change checkpoints.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader as PyGDataLoader


ROOT = Path(__file__).resolve().parents[1]
TRAINING_SCRIPT = ROOT / "Models" / "deepset_variations" / "deepset_pmain_mse.py"
DEFAULT_CHECKPOINT = ROOT / "Models" / "deepset_variations" / "pmain_mse_output" / "best_pmain_mse_model.pt"
DEFAULT_OUTPUT = ROOT / "Models" / "deepset_variations" / "pmain_mse_output" / "diagnostics"
SAMPLE_SIZE = 200_000


def load_training_module() -> Any:
    spec = importlib.util.spec_from_file_location("pmain_mse_training", TRAINING_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {TRAINING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    variance = float(np.var(target))
    return {
        "n_clusters": int(len(target)),
        "mse": float(mean_squared_error(target, prediction)),
        "mae": float(mean_absolute_error(target, prediction)),
        # R² has no useful interpretation in a nearly constant target slice.
        "r2": float(r2_score(target, prediction)) if variance > 1e-8 else None,
    }


def model_predictions(module: Any, checkpoint_path: Path, arrays: tuple[np.ndarray, ...]) -> np.ndarray:
    _, _, _, _, _, _, _, _, x_test, y_test, e_test, test_groups = arrays
    checkpoint = torch.load(checkpoint_path, map_location=module.DEVICE, weights_only=False)
    model = module.GraphDeepSetRegressor().to(module.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataset = module.S2GraphDataset(x_test, y_test, e_test, test_groups)
    loader = PyGDataLoader(dataset, batch_size=module.BATCH_SIZE, shuffle=False, num_workers=0)
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(module.DEVICE)
            predictions.append(torch.sigmoid(model(batch)[:, 0]).cpu())
    return torch.cat(predictions).numpy()


def validate_checkpoint_preprocessing(checkpoint_path: Path, current_cut_stats: dict[str, Any]) -> None:
    """Refuse to score a checkpoint on a data population it was not trained on."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = checkpoint.get("cut_stats", {})
    saved_events = saved.get("events_after_fiducial_and_individual_clip", saved.get("events_after_both_cuts"))
    current_events = current_cut_stats.get("events_after_fiducial_and_individual_clip")
    if saved_events != current_events:
        raise ValueError(
            "Checkpoint/data preprocessing mismatch: checkpoint was trained on "
            f"{saved_events} retained events, but the current pure-MSE split has {current_events}. "
            "Copy the checkpoint from the matching GPU-server run before evaluating it."
        )


def electron_argmax_predictions(x_test: np.ndarray, test_groups: list[np.ndarray]) -> np.ndarray:
    """One-hot electron-count baseline; intentionally permits only one main cluster."""
    prediction = np.zeros(len(x_test), dtype=np.float32)
    electron_column = 2  # n_electrons_interface; rank is unchanged by z-scoring.
    for rows in test_groups:
        prediction[rows[np.argmax(x_test[rows, electron_column])]] = 1.0
    return prediction


def grouped_metrics(
    target: np.ndarray, prediction: np.ndarray, group: np.ndarray, definitions: list[tuple[str, Callable[[np.ndarray], np.ndarray]]]
) -> pd.DataFrame:
    rows = []
    for name, selector in definitions:
        mask = selector(group)
        result = metrics(target[mask], prediction[mask]) if mask.any() else {"n_clusters": 0, "mse": None, "mae": None, "r2": None}
        rows.append({"group": name, **result})
    return pd.DataFrame(rows)


def multiplicity_per_cluster(test_groups: list[np.ndarray], n_clusters: int) -> np.ndarray:
    values = np.empty(n_clusters, dtype=np.int32)
    for rows in test_groups:
        values[rows] = len(rows)
    return values


def raw_test_features(module: Any, cached_event_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild raw-scale features for the exact cached test-event split."""
    data = np.load(module.RAW_DATA_PATH)
    df = pd.DataFrame(data)
    min_drift = df.groupby(module.EVENT_COL)["drift_time_mean"].min()
    df = df[~df[module.EVENT_COL].isin(min_drift[min_drift < module.TOP13_NS].index)].copy()
    df[module.TARGET] = df[module.TARGET].clip(0.0, 1.0)
    event_ids = df[module.EVENT_COL].unique()
    _, temporary_events = train_test_split(
        event_ids, test_size=0.30, random_state=module.RANDOM_SEED, shuffle=True
    )
    _, test_events = train_test_split(
        temporary_events, test_size=0.50, random_state=module.RANDOM_SEED, shuffle=True
    )
    test_df = df[df[module.EVENT_COL].isin(test_events)]
    if not np.array_equal(test_df[module.EVENT_COL].to_numpy(), cached_event_ids):
        raise RuntimeError("Raw test-feature reconstruction does not match the cached test split.")
    return (
        test_df["n_electrons_interface"].to_numpy(dtype=np.float32),
        test_df["drift_time_mean"].to_numpy(dtype=np.float32),
    )


def residual_plot(electrons: np.ndarray, drift_ns: np.ndarray, residual: np.ndarray, output: Path) -> None:
    rng = np.random.default_rng(42)
    sample = rng.choice(len(residual), size=min(SAMPLE_SIZE, len(residual)), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    axes[0].hexbin(electrons[sample], residual[sample], gridsize=75, mincnt=1, bins="log", cmap="viridis")
    axes[0].axhline(0, color="white", linewidth=0.8)
    axes[0].set(xlabel="n_electrons_interface", ylabel="Residual (prediction − true)",
                title="Residual vs electron count")
    axes[1].hexbin(drift_ns[sample] / 1e3, residual[sample], gridsize=75, mincnt=1, bins="log", cmap="viridis")
    axes[1].axhline(0, color="white", linewidth=0.8)
    axes[1].set(xlabel="drift_time_mean (μs)", title="Residual vs drift time")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-only", action="store_true",
                        help="Evaluate the exact-split electron baseline without loading a model checkpoint.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    module = load_training_module()
    arrays, cut_stats = module.prepare_data()
    _, _, _, _, _, _, _, _, x_test, y_test, e_test, test_groups = arrays
    target = y_test[:, 0]
    baseline = electron_argmax_predictions(x_test, test_groups)
    multiplicity = multiplicity_per_cluster(test_groups, len(target))
    if args.baseline_only:
        prediction = baseline
        evaluation_label = "electron_argmax_baseline"
    else:
        validate_checkpoint_preprocessing(args.checkpoint, cut_stats)
        prediction = model_predictions(module, args.checkpoint, arrays)
        evaluation_label = "model"

    target_bin_definitions = [
        ("zero_[0,0.001]", lambda y: y <= 0.001),
        ("low_fractional_(0.001,0.25]", lambda y: (y > 0.001) & (y <= 0.25)),
        ("mid_fractional_(0.25,0.75]", lambda y: (y > 0.25) & (y <= 0.75)),
        ("high_fractional_(0.75,0.999)", lambda y: (y > 0.75) & (y < 0.999)),
        ("one_[0.999,1]", lambda y: y >= 0.999),
    ]
    multiplicity_definitions = [
        ("1", lambda n: n == 1), ("2", lambda n: n == 2), ("3", lambda n: n == 3),
        ("4", lambda n: n == 4), ("5", lambda n: n == 5), ("6-10", lambda n: (n >= 6) & (n <= 10)),
        ("11-20", lambda n: (n >= 11) & (n <= 20)), ("21-50", lambda n: (n >= 21) & (n <= 50)),
        ("51+", lambda n: n >= 51),
    ]
    target_metrics = grouped_metrics(target, prediction, target, target_bin_definitions)
    multiplicity_metrics = grouped_metrics(target, prediction, multiplicity, multiplicity_definitions)
    target_metrics.to_csv(args.output_dir / f"{evaluation_label}_metrics_by_true_pmain_bin.csv", index=False)
    multiplicity_metrics.to_csv(args.output_dir / f"{evaluation_label}_metrics_by_event_multiplicity.csv", index=False)
    raw_electrons, raw_drift_ns = raw_test_features(module, e_test)
    residual_plot(raw_electrons, raw_drift_ns, prediction - target,
                  args.output_dir / f"{evaluation_label}_residuals_vs_electron_count_and_drift_time.png")

    summary = {
        "checkpoint": str(args.checkpoint),
        "evaluation_label": evaluation_label,
        "evaluated_test_metrics": metrics(target, prediction),
        "electron_argmax_baseline_metrics": metrics(target, baseline),
        "n_test_events": int(len(test_groups)),
        "cut_stats": cut_stats,
        "note": "The electron-argmax baseline predicts one main cluster per event; valid multi-scatter targets can have multiple high p_main clusters.",
    }
    (args.output_dir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
