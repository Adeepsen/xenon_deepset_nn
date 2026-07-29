#!/usr/bin/env python3
"""Permutation importance for the original five inputs of a p_main checkpoint.

The default grouped-source analysis permutes one original detector observable,
recomputes every engineered feature derived from it, and measures the increase
in validation MSE. This is the appropriate interpretation for the final
14-feature model because, for example, electron count is also represented by
electron fractions and rank.

The held-out test split is never evaluated.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader as PyGDataLoader

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "Models" / "feature_engineering"
if str(FEATURE_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURE_DIR))

from deepset_pmain_attention_pooling import GatedAttentionDeepSetRegressor  # noqa: E402
from feature_sets import BASE_FEATURES, add_event_relative_features, features_for_set  # noqa: E402
from run_feature_sweep import EventDataset, GraphDeepSetRegressor, build_event_groups  # noqa: E402


DEFAULT_CHECKPOINT = (
    FEATURE_DIR / "attention_pooling_output" / "gated_sum" / "seed_42" / "best_validation_model.pt"
)
DEFAULT_DATA = ROOT / "data" / "s2_tag_training_clusters.npy"
DEFAULT_OUTPUT = ROOT / "analysis" / "neural_permutation_output"
EVENT_COL = "event_number"
TARGET = "p_main"
TOP13_NS = 192_600.0
FRACTIONAL_LOW = 0.001
FRACTIONAL_HIGH = 0.999
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_feature_set(features: list[str]) -> str:
    if features == features_for_set("baseline"):
        return "baseline"
    if features == features_for_set("all_event_relative"):
        return "all_event_relative"
    raise ValueError(
        "This diagnostic currently supports baseline and all_event_relative checkpoints; "
        f"checkpoint contains {features}."
    )


def load_model(checkpoint_path: Path) -> tuple[torch.nn.Module, dict[str, Any], list[str], str]:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"{checkpoint_path} is not a supported model checkpoint.")
    features = list(checkpoint.get("features", []))
    if not features:
        raise ValueError("Checkpoint does not record its input feature list.")
    feature_set = infer_feature_set(features)
    pooling = checkpoint.get("pooling")
    if pooling in {"gated_sum", "sum_mean_max_gated"}:
        model = GatedAttentionDeepSetRegressor(len(features), pooling)
        architecture = f"gated_attention:{pooling}"
    elif pooling in {None, "sum"}:
        model = GraphDeepSetRegressor(len(features))
        architecture = "graph_deepset:sum"
    else:
        raise ValueError(f"Unsupported checkpoint pooling mode: {pooling!r}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()
    return model, checkpoint, features, architecture


def sample_complete_validation_events(
    frame: pd.DataFrame,
    validation_events: np.ndarray,
    cluster_limit: int,
    seed: int,
) -> pd.DataFrame:
    eligible = frame[frame[EVENT_COL].isin(validation_events)]
    sizes = eligible.groupby(EVENT_COL, sort=False).size()
    shuffled = np.random.default_rng(seed).permutation(sizes.index.to_numpy())
    cumulative = sizes.reindex(shuffled).cumsum().to_numpy()
    n_events = min(int(np.searchsorted(cumulative, cluster_limit, side="left")) + 1, len(shuffled))
    return eligible[eligible[EVENT_COL].isin(shuffled[:n_events])].copy()


def prepare_data(
    data_path: Path,
    features: list[str],
    seed: int,
    max_validation_clusters: int,
) -> tuple[pd.DataFrame, StandardScaler, dict[str, Any]]:
    print(f"Loading and preprocessing {data_path}", flush=True)
    frame = pd.DataFrame(np.load(data_path))
    missing = sorted({EVENT_COL, TARGET, *BASE_FEATURES}.difference(frame.columns))
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
    train_events, remainder = train_test_split(
        event_ids, test_size=0.30, random_state=seed, shuffle=True
    )
    validation_events, test_events = train_test_split(
        remainder, test_size=0.50, random_state=seed, shuffle=True
    )
    train_mask = frame[EVENT_COL].isin(train_events)
    print(f"Fitting the checkpoint-compatible scaler on {int(train_mask.sum()):,} training clusters", flush=True)
    scaler = StandardScaler().fit(frame.loc[train_mask, features])
    validation = sample_complete_validation_events(
        frame, validation_events, max_validation_clusters, seed + 1
    ).sort_values(EVENT_COL, kind="mergesort").reset_index(drop=True)
    stats = {
        "seed": seed,
        "events_raw": raw_events,
        "clusters_raw": raw_clusters,
        "events_removed_fiducial": int(len(removed_events)),
        "events_after_fiducial": int(frame[EVENT_COL].nunique()),
        "clusters_after_fiducial": int(len(frame)),
        "individual_pmain_values_clipped_above_one": clipped_above,
        "individual_pmain_values_clipped_below_zero": clipped_below,
        "validation_events_evaluated": int(validation[EVENT_COL].nunique()),
        "validation_clusters_evaluated": int(len(validation)),
        "held_out_test_events_reserved": int(len(test_events)),
        "held_out_test_used": False,
    }
    print(json.dumps(stats, indent=2), flush=True)
    return validation, scaler, stats


def predict(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    features: list[str],
    scaler: StandardScaler,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    x = scaler.transform(frame[features]).astype(np.float32)
    y = frame[[TARGET]].to_numpy(dtype=np.float32, copy=True)
    events = frame[EVENT_COL].to_numpy(copy=True)
    groups = build_event_groups(events)
    loader = PyGDataLoader(
        EventDataset(x, y, groups),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=DEVICE.type == "cuda",
        drop_last=False,
    )
    parts = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch.to(DEVICE))
            parts.append(torch.sigmoid(logits[:, 0]).cpu())
    return torch.cat(parts).numpy()


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int | None]:
    if len(truth) == 0:
        return {"n_clusters": 0, "mse": None, "mae": None, "r2": None}
    variance = float(np.var(truth))
    return {
        "n_clusters": int(len(truth)),
        "mse": float(mean_squared_error(truth, prediction)),
        "mae": float(mean_absolute_error(truth, prediction)),
        "r2": float(r2_score(truth, prediction)) if variance > 1e-10 else None,
    }


def target_regimes(target: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "all": np.ones(len(target), dtype=bool),
        "zero": target <= FRACTIONAL_LOW,
        "fractional": (target > FRACTIONAL_LOW) & (target < FRACTIONAL_HIGH),
        "one": target >= FRACTIONAL_HIGH,
    }


def shuffled_source_frame(
    validation: pd.DataFrame,
    source_feature: str,
    permutation: np.ndarray,
    feature_set: str,
) -> pd.DataFrame:
    """Permute one raw source and recompute all of its engineered descendants."""
    raw_columns = [EVENT_COL, TARGET, *BASE_FEATURES]
    perturbed = validation[raw_columns].copy()
    perturbed[source_feature] = perturbed[source_feature].to_numpy()[permutation]
    if feature_set == "all_event_relative":
        return add_event_relative_features(perturbed, EVENT_COL)
    return perturbed


def grouped_source_importance(
    model: torch.nn.Module,
    validation: pd.DataFrame,
    features: list[str],
    feature_set: str,
    scaler: StandardScaler,
    baseline_prediction: np.ndarray,
    repeats: int,
    seed: int,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    truth = validation[TARGET].to_numpy(np.float32)
    baseline_metrics = {
        regime: metrics(truth[mask], baseline_prediction[mask])
        for regime, mask in target_regimes(truth).items()
    }
    rows: list[dict[str, Any]] = []
    for feature_index, source_feature in enumerate(BASE_FEATURES):
        print(f"Permuting source feature: {source_feature}", flush=True)
        for repeat in range(repeats):
            permutation = np.random.default_rng(seed + 10_000 * feature_index + repeat).permutation(
                len(validation)
            )
            perturbed = shuffled_source_frame(
                validation, source_feature, permutation, feature_set
            )
            prediction = predict(
                model, perturbed, features, scaler, batch_size, num_workers
            )
            for regime, mask in target_regimes(truth).items():
                result = metrics(truth[mask], prediction[mask])
                baseline_mse = baseline_metrics[regime]["mse"]
                baseline_mae = baseline_metrics[regime]["mae"]
                rows.append({
                    "source_feature": source_feature,
                    "target_regime": regime,
                    "repeat": repeat,
                    "baseline_mse": baseline_mse,
                    "permuted_mse": result["mse"],
                    "mse_increase": (
                        float(result["mse"]) - float(baseline_mse)
                        if result["mse"] is not None and baseline_mse is not None
                        else None
                    ),
                    "baseline_mae": baseline_mae,
                    "permuted_mae": result["mae"],
                    "mae_increase": (
                        float(result["mae"]) - float(baseline_mae)
                        if result["mae"] is not None and baseline_mae is not None
                        else None
                    ),
                    "baseline_r2": baseline_metrics[regime]["r2"],
                    "permuted_r2": result["r2"],
                    "n_clusters": result["n_clusters"],
                })
    return pd.DataFrame(rows)


def direct_column_importance(
    model: torch.nn.Module,
    validation: pd.DataFrame,
    features: list[str],
    scaler: StandardScaler,
    baseline_prediction: np.ndarray,
    repeats: int,
    seed: int,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    """Optional unique-column importance with engineered descendants held fixed."""
    truth = validation[TARGET].to_numpy(np.float32)
    baseline_mse = float(mean_squared_error(truth, baseline_prediction))
    rows = []
    for feature_index, source_feature in enumerate(BASE_FEATURES):
        if source_feature not in features:
            continue
        print(f"Direct-column permutation: {source_feature}", flush=True)
        for repeat in range(repeats):
            permutation = np.random.default_rng(seed + 100_000 + 10_000 * feature_index + repeat).permutation(
                len(validation)
            )
            perturbed = validation.copy()
            perturbed[source_feature] = perturbed[source_feature].to_numpy()[permutation]
            prediction = predict(
                model, perturbed, features, scaler, batch_size, num_workers
            )
            permuted_mse = float(mean_squared_error(truth, prediction))
            rows.append({
                "source_feature": source_feature,
                "repeat": repeat,
                "baseline_mse": baseline_mse,
                "permuted_mse": permuted_mse,
                "mse_increase": permuted_mse - baseline_mse,
            })
    return pd.DataFrame(rows)


def summarize_importance(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["source_feature", "target_regime"], as_index=False)
        .agg(
            repeats=("repeat", "count"),
            n_clusters=("n_clusters", "first"),
            baseline_mse=("baseline_mse", "first"),
            permuted_mse_mean=("permuted_mse", "mean"),
            permuted_mse_std=("permuted_mse", "std"),
            mse_increase_mean=("mse_increase", "mean"),
            mse_increase_std=("mse_increase", "std"),
            mae_increase_mean=("mae_increase", "mean"),
            mae_increase_std=("mae_increase", "std"),
        )
        .sort_values(["target_regime", "mse_increase_mean"], ascending=[True, False])
    )


def save_plot(summary: pd.DataFrame, output: Path) -> None:
    regimes = ["all", "fractional"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, regime in zip(axes, regimes):
        subset = summary[summary.target_regime == regime].sort_values("mse_increase_mean")
        ax.barh(
            subset.source_feature,
            subset.mse_increase_mean,
            xerr=subset.mse_increase_std.fillna(0),
            color="#2878B5",
            alpha=0.88,
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set(
            title=f"{regime.title()} validation targets",
            xlabel="Neural validation MSE increase after grouped permutation",
        )
    fig.suptitle("Original-input importance in the trained checkpoint", y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-validation-clusters", type=int, default=100_000)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--direct-columns",
        action="store_true",
        help="Also permute only each raw model column while keeping engineered descendants unchanged.",
    )
    args = parser.parse_args()
    if args.max_validation_clusters < 1 or args.permutation_repeats < 1:
        parser.error("Validation sample size and permutation repeats must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint, features, architecture = load_model(args.checkpoint)
    seed = int(checkpoint.get("seed", 42))
    set_seed(seed)
    validation, scaler, stats = prepare_data(
        args.data_path, features, seed, args.max_validation_clusters
    )
    baseline_prediction = predict(
        model, validation, features, scaler, args.batch_size, args.num_workers
    )
    truth = validation[TARGET].to_numpy(np.float32)
    baseline_by_regime = {
        regime: metrics(truth[mask], baseline_prediction[mask])
        for regime, mask in target_regimes(truth).items()
    }
    feature_set = infer_feature_set(features)
    raw_rows = grouped_source_importance(
        model=model,
        validation=validation,
        features=features,
        feature_set=feature_set,
        scaler=scaler,
        baseline_prediction=baseline_prediction,
        repeats=args.permutation_repeats,
        seed=seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    summary = summarize_importance(raw_rows)
    raw_rows.to_csv(args.output_dir / "grouped_source_permutation_repeats.csv", index=False)
    summary.to_csv(args.output_dir / "grouped_source_permutation_summary.csv", index=False)
    save_plot(summary, args.output_dir / "grouped_source_permutation_importance.png")

    direct_summary: list[dict[str, Any]] | None = None
    if args.direct_columns:
        direct = direct_column_importance(
            model=model,
            validation=validation,
            features=features,
            scaler=scaler,
            baseline_prediction=baseline_prediction,
            repeats=args.permutation_repeats,
            seed=seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        direct.to_csv(args.output_dir / "direct_column_permutation_repeats.csv", index=False)
        direct_summary = (
            direct.groupby("source_feature", as_index=False)
            .agg(
                repeats=("repeat", "count"),
                baseline_mse=("baseline_mse", "first"),
                permuted_mse_mean=("permuted_mse", "mean"),
                mse_increase_mean=("mse_increase", "mean"),
                mse_increase_std=("mse_increase", "std"),
            )
            .sort_values("mse_increase_mean", ascending=False)
            .to_dict("records")
        )

    result = {
        "status": "validation-only checkpoint permutation; held-out test not evaluated",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_val_p_main_mse": checkpoint.get("best_val_p_main_mse"),
        "architecture": architecture,
        "feature_set": feature_set,
        "features": features,
        "device": str(DEVICE),
        "permutation_definition": (
            "Globally permute one original raw observable across sampled validation clusters, "
            "then recompute all event-relative descendants before applying the training scaler."
        ),
        "baseline_validation_metrics": baseline_by_regime,
        "grouped_source_importance": summary.to_dict("records"),
        "direct_column_importance": direct_summary,
        "data_stats": stats,
    }
    (args.output_dir / "diagnostic_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(summary[summary.target_regime.isin(["all", "fractional"])].to_string(index=False))
    print(f"Wrote checkpoint permutation analysis to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
