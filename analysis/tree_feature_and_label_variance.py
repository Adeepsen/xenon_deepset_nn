#!/usr/bin/env python3
"""Tree benchmark, permutation importance, and local label-variance diagnostics.

This script is validation-only. It reproduces the approved fiducial cut and
event-level split, samples complete training/validation events, and never reads
the held-out test events into a model or diagnostic.

The nearest-neighbor variance is a *local uncertainty proxy*, not a formal
irreducible-noise estimate: large values mean that clusters which are close in
the available feature space have substantially different p_main labels.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "Models" / "feature_engineering"
if str(FEATURE_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(FEATURE_DIR))

from feature_sets import add_event_relative_features, features_for_set  # noqa: E402

try:
    import wandb
except ImportError:
    wandb = None


DEFAULT_DATA = ROOT / "data" / "s2_tag_training_clusters.npy"
DEFAULT_OUTPUT = ROOT / "analysis" / "tree_diagnostics_output"
EVENT_COL = "event_number"
TARGET = "p_main"
TOP13_NS = 192_600.0
FRACTIONAL_LOW = 0.001
FRACTIONAL_HIGH = 0.999


def parse_int_list(value: str) -> list[int]:
    result = sorted({int(item) for item in value.split(",") if item.strip()})
    if not result or result[0] < 1:
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers, for example 10,30,100")
    return result


def sample_complete_events(
    frame: pd.DataFrame,
    eligible_events: np.ndarray,
    cluster_limit: int,
    seed: int,
) -> pd.DataFrame:
    """Sample complete events until at least ``cluster_limit`` rows are selected."""
    eligible = frame[frame[EVENT_COL].isin(eligible_events)]
    sizes = eligible.groupby(EVENT_COL, sort=False).size()
    shuffled = np.random.default_rng(seed).permutation(sizes.index.to_numpy())
    cumulative = sizes.reindex(shuffled).cumsum().to_numpy()
    n_events = min(int(np.searchsorted(cumulative, cluster_limit, side="left")) + 1, len(shuffled))
    selected = shuffled[:n_events]
    return eligible[eligible[EVENT_COL].isin(selected)].copy()


def prepare_samples(
    data_path: Path,
    feature_sets: list[str],
    seed: int,
    max_train_clusters: int,
    max_val_clusters: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    print(f"Loading {data_path}", flush=True)
    frame = pd.DataFrame(np.load(data_path))
    required = {EVENT_COL, TARGET, *features_for_set("baseline")}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")

    raw_events, raw_clusters = int(frame[EVENT_COL].nunique()), int(len(frame))
    min_drift = frame.groupby(EVENT_COL)["drift_time_mean"].min()
    removed_events = min_drift[min_drift < TOP13_NS].index
    frame = frame[~frame[EVENT_COL].isin(removed_events)].copy()
    clipped_above = int((frame[TARGET] > 1.0).sum())
    clipped_below = int((frame[TARGET] < 0.0).sum())
    frame[TARGET] = frame[TARGET].clip(0.0, 1.0)
    frame = add_event_relative_features(frame, EVENT_COL)

    event_ids = frame[EVENT_COL].unique()
    train_events, remainder = train_test_split(event_ids, test_size=0.30, random_state=seed, shuffle=True)
    val_events, test_events = train_test_split(remainder, test_size=0.50, random_state=seed, shuffle=True)
    train = sample_complete_events(frame, train_events, max_train_clusters, seed)
    validation = sample_complete_events(frame, val_events, max_val_clusters, seed + 1)

    if np.intersect1d(train[EVENT_COL].unique(), validation[EVENT_COL].unique()).size:
        raise RuntimeError("Event leakage detected between sampled training and validation data.")

    retained_columns = sorted({
        EVENT_COL, TARGET, *(name for feature_set in feature_sets for name in features_for_set(feature_set))
    })
    train = train[retained_columns]
    validation = validation[retained_columns]
    stats = {
        "seed": seed,
        "events_raw": raw_events,
        "clusters_raw": raw_clusters,
        "events_removed_fiducial": int(len(removed_events)),
        "events_after_fiducial": int(frame[EVENT_COL].nunique()),
        "clusters_after_fiducial": int(len(frame)),
        "individual_pmain_values_clipped_above_one": clipped_above,
        "individual_pmain_values_clipped_below_zero": clipped_below,
        "sampled_train_events": int(train[EVENT_COL].nunique()),
        "sampled_train_clusters": int(len(train)),
        "sampled_validation_events": int(validation[EVENT_COL].nunique()),
        "sampled_validation_clusters": int(len(validation)),
        "held_out_test_events_reserved": int(len(test_events)),
        "held_out_test_used": False,
    }
    print(json.dumps(stats, indent=2), flush=True)
    return train, validation, stats


def build_regressor(args: argparse.Namespace):
    common = {"random_state": args.seed}
    if args.model == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            max_depth=args.max_depth,
            max_features=args.max_features,
            n_jobs=args.n_jobs,
            **common,
        )
    if args.model == "random_forest":
        return RandomForestRegressor(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            max_depth=args.max_depth,
            max_features=args.max_features,
            max_samples=args.max_samples,
            n_jobs=args.n_jobs,
            **common,
        )
    return HistGradientBoostingRegressor(
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2_regularization,
        random_state=args.seed,
    )


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int | None]:
    if len(truth) == 0:
        return {"n_clusters": 0, "mse": None, "mae": None, "r2": None}
    variance = float(np.var(truth))
    return {
        "n_clusters": int(len(truth)),
        "mse": float(mean_squared_error(truth, prediction)),
        "mae": float(mean_absolute_error(truth, prediction)),
        "r2": float(r2_score(truth, prediction)) if variance > 1e-10 else None,
    }


def regime_masks(target: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "all": np.ones(len(target), dtype=bool),
        "zero": target <= FRACTIONAL_LOW,
        "fractional": (target > FRACTIONAL_LOW) & (target < FRACTIONAL_HIGH),
        "one": target >= FRACTIONAL_HIGH,
    }


def evaluate_by_regime(
    feature_set: str,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {"feature_set": feature_set, "target_regime": name, **regression_metrics(truth[mask], prediction[mask])}
        for name, mask in regime_masks(truth).items()
    ]


def fit_tree_benchmarks(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_sets: list[str],
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    truth = validation[TARGET].to_numpy(dtype=np.float32)
    predictions: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed)
    permutation_rows = rng.choice(
        len(validation), size=min(args.permutation_clusters, len(validation)), replace=False
    )

    for feature_set in feature_sets:
        names = features_for_set(feature_set)
        print(f"Fitting {args.model} for {feature_set} with {len(names)} features", flush=True)
        model = build_regressor(args)
        model.fit(train[names].to_numpy(np.float32), train[TARGET].to_numpy(np.float32))
        prediction = np.clip(model.predict(validation[names].to_numpy(np.float32)), 0.0, 1.0)
        predictions[feature_set] = prediction
        metric_rows.extend(evaluate_by_regime(feature_set, truth, prediction))

        print(f"Computing validation permutation importance for {feature_set}", flush=True)
        importance = permutation_importance(
            model,
            validation.iloc[permutation_rows][names].to_numpy(np.float32),
            truth[permutation_rows],
            scoring="neg_mean_squared_error",
            n_repeats=args.permutation_repeats,
            random_state=args.seed,
            n_jobs=args.n_jobs,
        )
        for name, mean, std in zip(names, importance.importances_mean, importance.importances_std):
            importance_rows.append({
                "feature_set": feature_set,
                "feature": name,
                "mse_increase_mean": float(mean),
                "mse_increase_std": float(std),
                "n_validation_clusters": int(len(permutation_rows)),
                "n_repeats": args.permutation_repeats,
            })

    metrics = pd.DataFrame(metric_rows)
    importances = pd.DataFrame(importance_rows).sort_values(
        ["feature_set", "mse_increase_mean"], ascending=[True, False]
    )
    metrics.to_csv(output_dir / "tree_metrics_by_target_regime.csv", index=False)
    importances.to_csv(output_dir / "permutation_importance.csv", index=False)
    return predictions, metrics, importances


def subsample_rows(length: int, limit: int, seed: int) -> np.ndarray:
    if length <= limit:
        return np.arange(length)
    return np.random.default_rng(seed).choice(length, size=limit, replace=False)


def conditional_variance_analysis(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_sets: list[str],
    tree_predictions: dict[str, np.ndarray],
    neighbor_counts: list[int],
    reference_limit: int,
    query_limit: int,
    n_jobs: int,
    seed: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_rows = subsample_rows(len(train), reference_limit, seed + 10)
    query_rows = subsample_rows(len(validation), query_limit, seed + 20)
    train_target = train[TARGET].to_numpy(np.float32)[reference_rows]
    query_target = validation[TARGET].to_numpy(np.float32)[query_rows]
    max_neighbors = min(max(neighbor_counts), len(reference_rows))
    neighbor_counts = [k for k in neighbor_counts if k <= max_neighbors]
    if not neighbor_counts:
        raise ValueError("The KNN reference sample is smaller than every requested neighbor count.")

    summary_rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []
    for feature_set in feature_sets:
        names = features_for_set(feature_set)
        scaler = StandardScaler().fit(train.iloc[reference_rows][names])
        reference_x = scaler.transform(train.iloc[reference_rows][names]).astype(np.float32)
        query_x = scaler.transform(validation.iloc[query_rows][names]).astype(np.float32)
        print(
            f"Finding up to {max_neighbors} neighbors for {len(query_rows):,} "
            f"{feature_set} validation queries in {len(reference_rows):,} training references",
            flush=True,
        )
        neighbors = NearestNeighbors(n_neighbors=max_neighbors, algorithm="auto", n_jobs=n_jobs)
        neighbors.fit(reference_x)
        distances, indices = neighbors.kneighbors(query_x, return_distance=True)
        neighbor_targets = train_target[indices]
        tree_squared_error = (tree_predictions[feature_set][query_rows] - query_target) ** 2

        for k in neighbor_counts:
            local_targets = neighbor_targets[:, :k]
            local_mean = local_targets.mean(axis=1)
            local_variance = local_targets.var(axis=1)
            local_distance = distances[:, :k].mean(axis=1)
            local_mean_squared_error = (local_mean - query_target) ** 2
            finite = np.isfinite(local_variance) & np.isfinite(tree_squared_error)
            correlation = (
                float(np.corrcoef(local_variance[finite], tree_squared_error[finite])[0, 1])
                if finite.sum() > 2 and np.std(local_variance[finite]) > 0 and np.std(tree_squared_error[finite]) > 0
                else None
            )
            for regime, mask in regime_masks(query_target).items():
                count = int(mask.sum())
                summary_rows.append({
                    "feature_set": feature_set,
                    "neighbors": k,
                    "target_regime": regime,
                    "n_queries": count,
                    "mean_neighbor_label_variance": float(local_variance[mask].mean()) if count else None,
                    "median_neighbor_label_variance": float(np.median(local_variance[mask])) if count else None,
                    "neighbor_mean_prediction_mse": float(local_mean_squared_error[mask].mean()) if count else None,
                    "tree_prediction_mse": float(tree_squared_error[mask].mean()) if count else None,
                    "mean_neighbor_distance": float(local_distance[mask].mean()) if count else None,
                    "variance_vs_tree_squared_error_correlation": correlation if regime == "all" else None,
                })
            detail_frames.append(pd.DataFrame({
                "feature_set": feature_set,
                "neighbors": k,
                "validation_row": query_rows,
                "event_number": validation.iloc[query_rows][EVENT_COL].to_numpy(),
                "true_p_main": query_target,
                "tree_prediction": tree_predictions[feature_set][query_rows],
                "tree_squared_error": tree_squared_error,
                "neighbor_label_mean": local_mean,
                "neighbor_label_variance": local_variance,
                "neighbor_mean_squared_error": local_mean_squared_error,
                "mean_neighbor_distance": local_distance,
            }))

    summary = pd.DataFrame(summary_rows)
    details = pd.concat(detail_frames, ignore_index=True)
    summary.to_csv(output_dir / "conditional_label_variance_summary.csv", index=False)
    details.to_csv(output_dir / "conditional_label_variance_queries.csv.gz", index=False, compression="gzip")
    return summary, details


def save_importance_plot(importances: pd.DataFrame, output: Path) -> None:
    feature_sets = list(importances["feature_set"].unique())
    fig, axes = plt.subplots(1, len(feature_sets), figsize=(7 * len(feature_sets), 5.5), squeeze=False)
    for ax, feature_set in zip(axes[0], feature_sets):
        subset = importances[importances.feature_set == feature_set].sort_values("mse_increase_mean")
        ax.barh(
            subset.feature,
            subset.mse_increase_mean,
            xerr=subset.mse_increase_std,
            color="#2878B5",
            alpha=0.85,
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set(
            title=feature_set.replace("_", " ").title(),
            xlabel="Validation MSE increase after permutation",
        )
    fig.suptitle("Permutation feature importance", y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_conditional_variance_plot(summary: pd.DataFrame, output: Path) -> None:
    overall = summary[summary.target_regime == "all"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for feature_set, subset in overall.groupby("feature_set"):
        label = feature_set.replace("_", " ").title()
        axes[0].plot(subset.neighbors, subset.mean_neighbor_label_variance, "o-", label=label)
        axes[1].plot(subset.neighbors, subset.neighbor_mean_prediction_mse, "o-", label=label)
    axes[0].set(
        xlabel="Number of training neighbors",
        ylabel="Mean local label variance",
        title="Nearby inputs with differing labels",
    )
    axes[1].set(
        xlabel="Number of training neighbors",
        ylabel="MSE of neighbor-label mean",
        title="Local-neighbor prediction benchmark",
    )
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    args: argparse.Namespace,
    stats: dict[str, Any],
    metrics: pd.DataFrame,
    importances: pd.DataFrame,
    conditional: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    overall_metrics = metrics[metrics.target_regime == "all"].set_index("feature_set")
    fractional_metrics = metrics[metrics.target_regime == "fractional"].set_index("feature_set")
    top_features = {
        feature_set: group.nlargest(10, "mse_increase_mean")[
            ["feature", "mse_increase_mean", "mse_increase_std"]
        ].to_dict("records")
        for feature_set, group in importances.groupby("feature_set")
    }
    conditional_overall = conditional[conditional.target_regime == "all"].to_dict("records")
    result = {
        "status": "validation-only diagnostic; held-out test not evaluated",
        "model": args.model,
        "feature_sets": args.feature_sets,
        "tree_overall_metrics": overall_metrics.to_dict("index"),
        "tree_fractional_metrics": fractional_metrics.to_dict("index"),
        "top_permutation_features": top_features,
        "conditional_label_variance_overall": conditional_overall,
        "data_stats": stats,
        "interpretation": {
            "tree_matches_neural_model": "Evidence that the current features, rather than Deep Sets capacity, limit performance.",
            "tree_beats_neural_model": "Evidence that the neural model or its optimization leaves usable feature signal unextracted.",
            "large_local_label_variance": "Nearby points in available feature space have differing labels; consistent with missing inputs, stochastic labels, or insufficient local sampling density.",
            "small_local_label_variance_but_large_model_error": "The current inputs contain locally recoverable signal that the fitted model is not using well.",
        },
    }
    (output_dir / "diagnostic_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def maybe_log_wandb(args: argparse.Namespace, summary: dict[str, Any], output_dir: Path) -> None:
    if not args.wandb:
        return
    if wandb is None:
        raise RuntimeError("--wandb was requested, but wandb is not installed.")
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "experiment": "tree_feature_importance_and_conditional_label_variance",
            "model": args.model,
            "feature_sets": args.feature_sets,
            "seed": args.seed,
            "held_out_test_used": False,
            "max_train_clusters": args.max_train_clusters,
            "max_validation_clusters": args.max_validation_clusters,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "knn_reference_clusters": args.knn_reference_clusters,
            "knn_query_clusters": args.knn_query_clusters,
            "neighbor_counts": args.neighbor_counts,
        },
    )
    flat_metrics: dict[str, float] = {}
    for feature_set, values in summary["tree_overall_metrics"].items():
        for metric in ("mse", "mae", "r2"):
            value = values.get(metric)
            if value is not None:
                flat_metrics[f"validation/{feature_set}/{metric}"] = float(value)
    for feature_set, values in summary["tree_fractional_metrics"].items():
        for metric in ("mse", "mae", "r2"):
            value = values.get(metric)
            if value is not None:
                flat_metrics[f"validation_fractional/{feature_set}/{metric}"] = float(value)
    wandb.log(flat_metrics)
    artifact = wandb.Artifact("tree-diagnostic-results", type="analysis")
    artifact.add_dir(str(output_dir))
    run.log_artifact(artifact)
    wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=["baseline", "all_event_relative"],
        choices=["baseline", "all_event_relative"],
    )
    parser.add_argument(
        "--model",
        choices=["extra_trees", "random_forest", "hist_gradient_boosting"],
        default="extra_trees",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-clusters", type=int, default=500_000)
    parser.add_argument("--max-validation-clusters", type=int, default=150_000)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=24)
    parser.add_argument("--max-features", type=float, default=1.0)
    parser.add_argument("--max-samples", type=float, default=0.7)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=1e-5)
    parser.add_argument("--permutation-clusters", type=int, default=50_000)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--knn-reference-clusters", type=int, default=200_000)
    parser.add_argument("--knn-query-clusters", type=int, default=25_000)
    parser.add_argument("--neighbor-counts", type=parse_int_list, default=parse_int_list("10,30,100"))
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="xenon-pmain-tree-diagnostics")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    for name, value in (
        ("max_train_clusters", args.max_train_clusters),
        ("max_validation_clusters", args.max_validation_clusters),
        ("permutation_clusters", args.permutation_clusters),
        ("knn_reference_clusters", args.knn_reference_clusters),
        ("knn_query_clusters", args.knn_query_clusters),
    ):
        if value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_samples <= 0 or args.max_samples > 1:
        parser.error("--max-samples must be in (0, 1]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, validation, stats = prepare_samples(
        args.data_path,
        args.feature_sets,
        args.seed,
        args.max_train_clusters,
        args.max_validation_clusters,
    )
    predictions, metrics, importances = fit_tree_benchmarks(
        train, validation, args.feature_sets, args, args.output_dir
    )
    conditional, _details = conditional_variance_analysis(
        train=train,
        validation=validation,
        feature_sets=args.feature_sets,
        tree_predictions=predictions,
        neighbor_counts=args.neighbor_counts,
        reference_limit=args.knn_reference_clusters,
        query_limit=args.knn_query_clusters,
        n_jobs=args.n_jobs,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    save_importance_plot(importances, args.output_dir / "permutation_importance.png")
    save_conditional_variance_plot(conditional, args.output_dir / "conditional_label_variance.png")
    summary = write_summary(args, stats, metrics, importances, conditional, args.output_dir)
    maybe_log_wandb(args, summary, args.output_dir)
    print(json.dumps(summary["tree_overall_metrics"], indent=2), flush=True)
    print(f"Wrote diagnostics to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
