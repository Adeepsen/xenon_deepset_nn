#!/usr/bin/env python3
"""Evaluate the frozen final model and both baselines on one held-out event split.

This script is intentionally evaluation-only: it loads preselected checkpoints
and never trains or selects a model using the test partition.  It reports
per-cluster regression and event-level main-cluster metrics for:

* electron-count argmax (a hard one-main-cluster heuristic),
* the five-raw-feature per-cluster MLP, and
* the all-event-relative, gated-sum Deep Sets model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import DataLoader as PyGDataLoader

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "Models" / "feature_engineering"
if str(FEATURE_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURE_DIR))

from deepset_pmain_attention_pooling import GatedAttentionDeepSetRegressor  # noqa: E402
from feature_sets import add_event_relative_features, features_for_set  # noqa: E402
from run_feature_sweep import EventDataset, build_event_groups  # noqa: E402

DATA_PATH = ROOT / "data" / "s2_tag_training_clusters.npy"
MLP_CHECKPOINT = ROOT / "Models" / "checkpoints" / "best.pt"
DEEPSETS_CHECKPOINT = (
    ROOT / "Models" / "feature_engineering" / "attention_pooling_output" /
    "gated_sum" / "seed_42" / "best_validation_model.pt"
)
OUTPUT_DIR = ROOT / "analysis" / "comparable_baseline_evaluation"
TOP13_NS, SEED, TIE_TOL = 192_600.0, 42, 1e-8
BASE_FEATURES = features_for_set("baseline")
FINAL_FEATURES = features_for_set("all_event_relative")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PerClusterMLP(nn.Module):
    """The architecture used by Models/mlp.py, reconstructed without training."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fiducial_event_ids() -> tuple[np.ndarray, dict[str, int]]:
    """Find retained event IDs without materializing engineered features."""
    frame = pd.DataFrame(np.load(DATA_PATH))
    raw_events, raw_clusters = frame.event_number.nunique(), len(frame)
    min_drift = frame.groupby("event_number")["drift_time_mean"].min()
    excluded = min_drift[min_drift < TOP13_NS].index
    kept_events = frame.loc[~frame.event_number.isin(excluded), "event_number"].unique()
    del frame
    return kept_events, {
        "events_raw": int(raw_events), "clusters_raw": int(raw_clusters),
        "events_removed_fiducial": int(len(excluded)),
    }


def load_event_subset(event_ids: np.ndarray) -> pd.DataFrame:
    """Load and engineer only one split, keeping peak memory bounded."""
    frame = pd.DataFrame(np.load(DATA_PATH))
    min_drift = frame.groupby("event_number")["drift_time_mean"].min()
    excluded = min_drift[min_drift < TOP13_NS].index
    frame = frame[~frame.event_number.isin(excluded) & frame.event_number.isin(event_ids)].copy()
    frame["p_main"] = frame["p_main"].clip(0.0, 1.0)
    return add_event_relative_features(frame)


def prepare_test_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Rebuild train-only scaling and test arrays without holding both splits."""
    event_ids, stats = fiducial_event_ids()
    train_events, remainder = train_test_split(
        event_ids, test_size=0.30, random_state=SEED, shuffle=True,
    )
    _, test_events = train_test_split(remainder, test_size=0.50, random_state=SEED, shuffle=True)
    train = load_event_subset(train_events)
    base_scaler = StandardScaler().fit(train[BASE_FEATURES])
    final_scaler = StandardScaler().fit(train[FINAL_FEATURES])
    checkpoint = torch.load(MLP_CHECKPOINT, map_location="cpu", weights_only=False)
    if not (np.allclose(base_scaler.mean_, checkpoint["scaler_mean"]) and np.allclose(base_scaler.scale_, checkpoint["scaler_scale"])):
        raise RuntimeError("MLP checkpoint scaler differs from the reconstructed event split.")
    del train
    test = load_event_subset(test_events)
    base_x = base_scaler.transform(test[BASE_FEATURES]).astype(np.float32)
    final_x = final_scaler.transform(test[FINAL_FEATURES]).astype(np.float32)
    target = test.p_main.to_numpy(np.float32)
    events = test.event_number.to_numpy(copy=True)
    electrons = test.n_electrons_interface.to_numpy(copy=True)
    stats.update(events_test=int(test.event_number.nunique()), clusters_test=int(len(test)))
    return base_x, final_x, target, events, electrons, stats


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "test_p_main_mse": float(mean_squared_error(target, prediction)),
        "test_p_main_mae": float(mean_absolute_error(target, prediction)),
        "test_p_main_r2": float(r2_score(target, prediction)),
    }


def metrics_by_target_regime(target: np.ndarray, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    regimes = {
        "Endpoint (0 or 1)": (target == 0.0) | (target == 1.0),
        "Fractional (0 < p_main < 1)": (target > 0.0) & (target < 1.0),
    }
    rows = []
    for regime, mask in regimes.items():
        for model, prediction in predictions.items():
            rows.append({
                "model": model,
                "target_regime": regime,
                "clusters": int(mask.sum()),
                "p_main_mse": float(mean_squared_error(target[mask], prediction[mask])),
                "p_main_mae": float(mean_absolute_error(target[mask], prediction[mask])),
            })
    return pd.DataFrame(rows)


def make_regime_plot(metrics: pd.DataFrame) -> None:
    plotted_models = ("Per-cluster MLP", "Gated-sum Deep Sets")
    regimes = ("Endpoint (0 or 1)", "Fractional (0 < p_main < 1)")
    labels = ("Endpoint\n(p_main = 0 or 1)", "Fractional\n(0 < p_main < 1)")
    x = np.arange(len(regimes))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True)
    for axis, metric, title in zip(axes, ("p_main_mse", "p_main_mae"), ("Test MSE", "Test MAE")):
        for index, model in enumerate(plotted_models):
            values = [float(metrics.loc[(metrics.model == model) & (metrics.target_regime == regime), metric].iloc[0]) for regime in regimes]
            axis.bar(x + (index - 0.5) * width, values, width, label=model)
        axis.set(title=title, xticks=x, xticklabels=labels, ylabel=metric.replace("p_main_", "p_main ").upper())
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "test_metrics_by_target_regime.png", dpi=220)
    plt.close(fig)


def event_metrics(target: np.ndarray, prediction: np.ndarray, groups: list[np.ndarray]) -> dict[str, float]:
    strict_correct = tie_aware_correct = ties = 0
    for rows in groups:
        truth, predicted = target[rows], prediction[rows]
        predicted_index = int(np.argmax(predicted))
        truth_index = int(np.argmax(truth))
        strict_correct += predicted_index == truth_index
        max_truth = float(truth[truth_index])
        tie_aware_correct += truth[predicted_index] >= max_truth - TIE_TOL
        ties += int(np.count_nonzero(truth >= max_truth - TIE_TOL) > 1)
    count = len(groups)
    return {
        "test_event_main_accuracy_strict": strict_correct / count,
        "test_event_main_accuracy_tie_aware": tie_aware_correct / count,
        "test_event_main_tie_fraction": ties / count,
    }


def predict_mlp(x: np.ndarray) -> np.ndarray:
    checkpoint = torch.load(MLP_CHECKPOINT, map_location=DEVICE, weights_only=False)
    model = PerClusterMLP().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    data = torch.as_tensor(x, dtype=torch.float32)
    predictions = []
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(data), batch_size=16_384, shuffle=False):
            predictions.append(model(batch.to(DEVICE))[:, 0].cpu())
    return torch.cat(predictions).numpy()


def predict_deepsets(x: np.ndarray, target: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    checkpoint = torch.load(DEEPSETS_CHECKPOINT, map_location=DEVICE, weights_only=False)
    if checkpoint.get("pooling") != "gated_sum" or checkpoint.get("features") != FINAL_FEATURES:
        raise RuntimeError("The selected checkpoint is not the expected gated-sum/all-event-relative model.")
    y = target[:, None]
    model = GatedAttentionDeepSetRegressor(len(FINAL_FEATURES), "gated_sum").to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = PyGDataLoader(EventDataset(x, y, groups), batch_size=1024, shuffle=False, num_workers=0)
    predictions = []
    with torch.no_grad():
        for batch in loader:
            predictions.append(torch.sigmoid(model(batch.to(DEVICE))[:, 0]).cpu())
    return torch.cat(predictions).numpy()


def make_plots(target: np.ndarray, predictions: dict[str, np.ndarray]) -> None:
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(target), size=min(150_000, len(target)), replace=False)
    plotted = (("Per-cluster MLP", predictions["Per-cluster MLP"]), ("Gated-sum Deep Set", predictions["Gated-sum Deep Sets"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    hexbin = None
    for axis, (title, predicted) in zip(axes, plotted):
        hexbin = axis.hexbin(target[sample], predicted[sample], gridsize=75, mincnt=1, bins="log", cmap="viridis")
        axis.plot([0, 1], [0, 1], color="red", linewidth=1.5)
        axis.set(xlabel="True $p_{main}$", xlim=(0, 1), ylim=(0, 1))
        if title:
            axis.set_title(title)
    axes[0].set_ylabel("Predicted $p_{main}$")
    fig.subplots_adjust(right=0.88, wspace=0.12)
    colorbar = fig.colorbar(hexbin, cax=fig.add_axes((0.90, 0.16, 0.02, 0.72)))
    colorbar.set_label("log10(number of clusters)")
    fig.savefig(OUTPUT_DIR / "test_predicted_vs_true_comparison.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_x, final_x, target, events, electrons, split_stats = prepare_test_data()
    groups = build_event_groups(events)

    argmax = np.zeros(len(target), dtype=np.float32)
    for rows in groups:
        argmax[rows[np.argmax(electrons[rows])]] = 1.0
    mlp = predict_mlp(base_x)
    deepsets = predict_deepsets(final_x, target, groups)
    predictions = {"Electron-count argmax": argmax, "Per-cluster MLP": mlp, "Gated-sum Deep Sets": deepsets}

    results = []
    for name, predicted in predictions.items():
        results.append({"model": name, **regression_metrics(target, predicted), **event_metrics(target, predicted, groups)})
    table = pd.DataFrame(results)
    table.to_csv(OUTPUT_DIR / "test_metrics_comparison.csv", index=False)
    regime_table = metrics_by_target_regime(target, predictions)
    regime_table.to_csv(OUTPUT_DIR / "test_metrics_by_target_regime.csv", index=False)
    make_regime_plot(regime_table)
    make_plots(target, predictions)
    summary = {
        "purpose": "One held-out event split; frozen checkpoints only; no test-driven model selection.",
        "split": {**split_stats, "seed": SEED, "event_split": "70% train / 15% validation / 15% test"},
        "checkpoints": {"per_cluster_mlp": str(MLP_CHECKPOINT), "gated_sum_deepsets": str(DEEPSETS_CHECKPOINT)},
        "metrics": table.to_dict(orient="records"),
        "note": "Argmax emits one hard p_main=1 prediction per event; all other clusters receive zero.",
    }
    (OUTPUT_DIR / "test_metrics_comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(table.to_string(index=False))
    print(f"\nWrote comparison artifacts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
