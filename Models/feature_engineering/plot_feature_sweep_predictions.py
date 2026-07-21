"""Plot matched validation predictions for two feature-ablation checkpoints.

The feature sweep reserves test events and never reads them for selection. This
utility therefore visualizes the common validation event split, not test data.

Example:
    python plot_feature_sweep_predictions.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader as PyGDataLoader

from feature_sets import add_event_relative_features, features_for_set
from run_feature_sweep import (
    BATCH_SIZE,
    DEVICE,
    EVENT_COL,
    RAW_DATA_PATH,
    TARGET,
    TOP13_NS,
    EventDataset,
    GraphDeepSetRegressor,
    build_event_groups,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "feature_sweep_output" / "prediction_plots"
POINT_COLOR = "#0066CC"
POINT_ALPHA = 0.28
POINT_SIZE = 3


def split_arrays(
    feature_sets: List[str], seed: int, include_train_feature_set: str | None = None, max_train_clusters: int = 100_000
) -> Dict[str, Dict[str, np.ndarray]]:
    """Recreate train/validation features exactly as the sweep does, loading once."""
    print(f"Loading and engineering train/validation data from {RAW_DATA_PATH}", flush=True)
    df = pd.DataFrame(np.load(RAW_DATA_PATH))
    min_drift = df.groupby(EVENT_COL)["drift_time_mean"].min()
    df = df[~df[EVENT_COL].isin(min_drift[min_drift < TOP13_NS].index)].copy()
    df[TARGET] = df[TARGET].clip(0.0, 1.0)
    df = add_event_relative_features(df, EVENT_COL)
    print(f"Prepared {len(df):,} clusters; recreating seed-{seed} event split", flush=True)

    train_events, remainder = train_test_split(
        df[EVENT_COL].unique(), test_size=0.30, random_state=seed, shuffle=True
    )
    val_events, _test_events = train_test_split(
        remainder, test_size=0.50, random_state=seed, shuffle=True
    )
    train_df = df[df[EVENT_COL].isin(train_events)].copy()
    val_df = df[df[EVENT_COL].isin(val_events)].copy()
    train_plot_df = train_df
    if include_train_feature_set is not None and len(train_df) > max_train_clusters:
        # Select complete events, rather than arbitrary clusters, for a
        # representative and leakage-safe visual comparison.
        event_sizes = train_df.groupby(EVENT_COL, sort=False).size()
        shuffled_events = np.random.default_rng(seed).permutation(event_sizes.index.to_numpy())
        selected, selected_clusters = [], 0
        for event_id in shuffled_events:
            selected.append(event_id)
            selected_clusters += int(event_sizes.loc[event_id])
            if selected_clusters >= max_train_clusters:
                break
        train_plot_df = train_df[train_df[EVENT_COL].isin(selected)].copy()
    outputs: Dict[str, Dict[str, np.ndarray]] = {"validation": {}}
    if include_train_feature_set is not None:
        outputs["train"] = {}
    for feature_set in feature_sets:
        names = features_for_set(feature_set)
        scaler = StandardScaler().fit(train_df[names])
        outputs["validation"][feature_set] = scaler.transform(val_df[names]).astype(np.float32)
        if feature_set == include_train_feature_set:
            outputs["train"][feature_set] = scaler.transform(train_plot_df[names]).astype(np.float32)
    outputs["validation"]["truth"] = val_df[[TARGET]].to_numpy(dtype=np.float32, copy=True)
    outputs["validation"]["event_ids"] = val_df[EVENT_COL].to_numpy(copy=True)
    if include_train_feature_set is not None:
        outputs["train"]["truth"] = train_plot_df[[TARGET]].to_numpy(dtype=np.float32, copy=True)
        outputs["train"]["event_ids"] = train_plot_df[EVENT_COL].to_numpy(copy=True)
    print(f"Train/validation splits contain {len(train_df):,} / {len(val_df):,} clusters; training plot sample: {len(train_plot_df):,}", flush=True)
    return outputs


def predict(feature_set: str, checkpoint_path: Path, x: np.ndarray, y: np.ndarray, event_ids: np.ndarray, num_workers: int) -> np.ndarray:
    print(f"Running {feature_set} checkpoint: {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    expected = checkpoint.get("features")
    actual = features_for_set(feature_set)
    if expected is not None and expected != actual:
        raise ValueError(f"{checkpoint_path} has a different feature definition than {feature_set}.")
    model = GraphDeepSetRegressor(len(actual)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    groups = build_event_groups(event_ids)
    # Keep this at zero by default: on macOS, spawned workers duplicate a large
    # validation array and can exhaust RAM before inference begins.
    loader = PyGDataLoader(
        EventDataset(x, y, groups), batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=DEVICE.type == "cuda",
    )
    predictions = []
    with torch.no_grad():
        for batch in loader:
            predictions.append(torch.sigmoid(model(batch.to(DEVICE))[:, 0]).cpu())
    return torch.cat(predictions).numpy()


def plot_panel(
    ax: plt.Axes, truth: np.ndarray, prediction: np.ndarray, label: str, sample: np.ndarray, metric_split: str = "validation"
) -> Dict[str, float]:
    mse = float(mean_squared_error(truth, prediction))
    mae = float(mean_absolute_error(truth, prediction))
    r2 = float(r2_score(truth, prediction))
    ax.scatter(truth[sample], prediction[sample], s=POINT_SIZE, alpha=POINT_ALPHA, color=POINT_COLOR, rasterized=True)
    ax.plot([0, 1], [0, 1], "black", linewidth=1, label="ideal: y = x")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="True $p_{main}$", ylabel="Predicted $p_{main}$")
    ax.set_title(f"{label}\n{metric_split} R² = {r2:.4f}; MSE = {mse:.5f}")
    return {"mse": mse, "mae": mae, "r2": r2}


def target_regime_rows(truth: np.ndarray, predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Return endpoint/fractional error metrics; within-endpoint R² is undefined."""
    regimes = {
        "zero (p_main <= 0.001)": truth <= 0.001,
        "fractional (0.001 < p_main < 0.999)": (truth > 0.001) & (truth < 0.999),
        "one (p_main >= 0.999)": truth >= 0.999,
    }
    rows = []
    for regime, mask in regimes.items():
        for feature_set, prediction in predictions.items():
            rows.append({
                "target_regime": regime, "feature_set": feature_set, "n_clusters": int(mask.sum()),
                "mse": float(mean_squared_error(truth[mask], prediction[mask])),
                "mae": float(mean_absolute_error(truth[mask], prediction[mask])),
                "mean_true_p_main": float(truth[mask].mean()),
                "mean_predicted_p_main": float(prediction[mask].mean()),
            })
    return pd.DataFrame(rows)


def binned_calibration_rows(truth: np.ndarray, predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Bin by true p_main and summarize predicted values and their spread."""
    bins = [0.0, 0.001, 0.10, 0.25, 0.50, 0.75, 0.90, 0.999, 1.000001]
    labels = ["0", "(0,.1]", "(.1,.25]", "(.25,.5]", "(.5,.75]", "(.75,.9]", "(.9,1)", "1"]
    categories = pd.cut(truth, bins=bins, labels=labels, include_lowest=True, right=True)
    rows = []
    for label in labels:
        mask = np.asarray(categories == label)
        if not mask.any():
            continue
        for feature_set, prediction in predictions.items():
            values = prediction[mask]
            rows.append({
                "true_p_main_bin": label, "feature_set": feature_set, "n_clusters": int(mask.sum()),
                "mean_true_p_main": float(truth[mask].mean()), "mean_predicted_p_main": float(values.mean()),
                "predicted_p10": float(np.quantile(values, 0.10)), "predicted_p90": float(np.quantile(values, 0.90)),
            })
    return pd.DataFrame(rows)


def save_fractional_plot(truth: np.ndarray, predictions: Dict[str, np.ndarray], feature_sets: List[str], sample_seed: int, path: Path) -> None:
    mask = (truth > 0.001) & (truth < 0.999)
    if not mask.any():
        return
    rng = np.random.default_rng(sample_seed)
    fractional_truth = truth[mask]
    sample = rng.choice(len(fractional_truth), size=min(100_000, len(fractional_truth)), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    for ax, name in zip(axes, feature_sets):
        fractional_prediction = predictions[name][mask]
        mse = mean_squared_error(fractional_truth, fractional_prediction)
        mae = mean_absolute_error(fractional_truth, fractional_prediction)
        ax.scatter(fractional_truth[sample], fractional_prediction[sample], s=POINT_SIZE, alpha=POINT_ALPHA, color=POINT_COLOR, rasterized=True)
        ax.plot([0, 1], [0, 1], "black", linewidth=1)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="True $p_{main}$", ylabel="Predicted $p_{main}$",
               title=f"{name.replace('_', ' ').title()}\nfractional MSE = {mse:.5f}; MAE = {mae:.5f}")
    fig.suptitle("Fractional $p_{main}$ targets only (validation events)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_calibration_plot(rows: pd.DataFrame, feature_sets: List[str], path: Path) -> None:
    fig, (ax, count_ax) = plt.subplots(2, 1, figsize=(8.5, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    colors = ["#1f77b4", "#d95f02"]
    for offset, (name, color) in enumerate(zip(feature_sets, colors)):
        subset = rows[rows.feature_set == name]
        x = subset.mean_true_p_main.to_numpy()
        y = subset.mean_predicted_p_main.to_numpy()
        lower = y - subset.predicted_p10.to_numpy()
        upper = subset.predicted_p90.to_numpy() - y
        ax.errorbar(x + (offset - 0.5) * 0.008, y, yerr=np.vstack((lower, upper)), fmt="o-", capsize=3,
                    color=color, label=name.replace("_", " ").title())
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="ideal calibration")
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), ylabel="Predicted $p_{main}$",
           title="Validation calibration by true $p_{main}$ bin")
    ax.legend(loc="upper left")
    counts = rows[rows.feature_set == feature_sets[0]]
    count_ax.bar(counts.mean_true_p_main, counts.n_clusters, width=0.035, color="#777777")
    count_ax.set(yscale="log", ylabel="Clusters\n(log scale)", xlabel="Mean true $p_{main}$ in bin")
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_regime_error_plot(rows: pd.DataFrame, feature_sets: List[str], path: Path) -> None:
    regimes = list(rows.target_regime.unique())
    x = np.arange(len(regimes))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for offset, name in enumerate(feature_sets):
        subset = rows[rows.feature_set == name].set_index("target_regime").loc[regimes]
        for ax, metric, title in zip(axes, ("mse", "mae"), ("MSE by target regime", "MAE by target regime")):
            ax.bar(x + (offset - 0.5) * 0.34, subset[metric], width=0.34, label=name.replace("_", " ").title())
            ax.set(title=title, ylabel=metric.upper(), xticks=x, xticklabels=["zero", "fractional", "one"])
    axes[0].legend()
    fig.suptitle("Validation error: endpoint versus fractional $p_{main}$", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_train_vs_validation_plot(
    feature_set: str, split_data: Dict[str, Dict[str, np.ndarray]], checkpoint: Path, seed: int, max_points: int, path: Path
) -> Dict[str, Dict[str, float]]:
    """Use the same selected checkpoint to compare fitted and validation data."""
    predictions = {}
    truth = {}
    for split_name in ("train", "validation"):
        truth[split_name] = split_data[split_name]["truth"][:, 0]
        predictions[split_name] = predict(
            feature_set, checkpoint, split_data[split_name][feature_set], split_data[split_name]["truth"],
            split_data[split_name]["event_ids"], num_workers=0,
        )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    results = {}
    for ax, split_name in zip(axes, ("train", "validation")):
        sample = np.random.default_rng(seed).choice(len(truth[split_name]), size=min(max_points, len(truth[split_name])), replace=False)
        results[split_name] = plot_panel(
            ax, truth[split_name], predictions[split_name],
            f"{feature_set.replace('_', ' ').title()} — {split_name}{' event sample' if split_name == 'train' else ''}", sample,
            metric_split=split_name,
        )
    fig.suptitle("Same selected checkpoint: training vs. validation predictions", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--comparison", default="all_event_relative")
    parser.add_argument("--max-points", type=int, default=100_000)
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 locally; GPU servers may use 4.")
    parser.add_argument("--train-vs-validation", action="store_true", help="Also plot fit-versus-validation output for the comparison model.")
    parser.add_argument("--max-train-clusters", type=int, default=100_000, help="Event-level training sample size for --train-vs-validation.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    feature_sets = [args.baseline, args.comparison]
    checkpoints = {
        name: Path(__file__).resolve().parent / "feature_sweep_output" / name / f"seed_{args.seed}" / "best_validation_model.pt"
        for name in feature_sets
    }
    missing = [str(path) for path in checkpoints.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoint(s):\n" + "\n".join(missing))

    split_data = split_arrays(feature_sets, args.seed, args.comparison if args.train_vs_validation else None, args.max_train_clusters)
    validation_data = split_data["validation"]
    truth = validation_data["truth"][:, 0]
    event_ids = validation_data["event_ids"]
    predictions = {
        name: predict(name, checkpoints[name], validation_data[name], validation_data["truth"], event_ids, args.num_workers)
        for name in feature_sets
    }
    sample_size = min(args.max_points, len(truth))
    sample = np.random.default_rng(args.seed).choice(len(truth), size=sample_size, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    metrics = {name: plot_panel(ax, truth, predictions[name], name.replace("_", " ").title(), sample) for ax, name in zip(axes, feature_sets)}
    axes[0].text(0.02, 0.98, "Same validation events\n(test set untouched)", transform=axes[0].transAxes, va="top", fontsize=9,
                 bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    fig.suptitle("Pure MSE $p_{main}$ regression: baseline vs. event-relative features", y=1.02)
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.output_dir / f"validation_true_vs_predicted_baseline_vs_{args.comparison}_seed{args.seed}.png"
    summary_path = plot_path.with_suffix(".json")
    fig.savefig(plot_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    fractional_path = args.output_dir / f"validation_fractional_true_vs_predicted_baseline_vs_{args.comparison}_seed{args.seed}.png"
    calibration_path = args.output_dir / f"validation_binned_calibration_baseline_vs_{args.comparison}_seed{args.seed}.png"
    regime_path = args.output_dir / f"validation_error_by_target_regime_baseline_vs_{args.comparison}_seed{args.seed}.png"
    regime_rows = target_regime_rows(truth, predictions)
    calibration_rows = binned_calibration_rows(truth, predictions)
    regime_csv = regime_path.with_suffix(".csv")
    calibration_csv = calibration_path.with_suffix(".csv")
    regime_rows.to_csv(regime_csv, index=False)
    calibration_rows.to_csv(calibration_csv, index=False)
    save_fractional_plot(truth, predictions, feature_sets, args.seed, fractional_path)
    save_calibration_plot(calibration_rows, feature_sets, calibration_path)
    save_regime_error_plot(regime_rows, feature_sets, regime_path)
    train_vs_validation = None
    if args.train_vs_validation:
        train_vs_validation_path = args.output_dir / f"training_vs_validation_true_vs_predicted_{args.comparison}_seed{args.seed}.png"
        train_vs_validation = save_train_vs_validation_plot(args.comparison, split_data, checkpoints[args.comparison], args.seed, args.max_points, train_vs_validation_path)
        print(f"Saved training-vs-validation plot: {train_vs_validation_path}")
    summary_path.write_text(json.dumps({"split": "validation only; test events untouched", "seed": args.seed, "n_validation_clusters": int(len(truth)), "sampled_points_in_plot": int(sample_size), "metrics": metrics, "checkpoints": {name: str(path) for name, path in checkpoints.items()}, "additional_diagnostics": {"fractional_scatter": str(fractional_path), "binned_calibration_plot": str(calibration_path), "binned_calibration_csv": str(calibration_csv), "target_regime_error_plot": str(regime_path), "target_regime_error_csv": str(regime_csv), "train_vs_validation_metrics": train_vs_validation, "n_training_clusters_in_train_vs_validation": int(len(split_data['train']['truth'])) if args.train_vs_validation else None}}, indent=2) + "\n")
    print(f"Saved plot: {plot_path}")
    print(f"Saved fractional-only plot: {fractional_path}")
    print(f"Saved binned calibration plot: {calibration_path}")
    print(f"Saved error-by-regime plot: {regime_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
