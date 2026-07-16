"""Leakage-safe, event-relative input features for p_main regression."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


BASE_FEATURES = ["x", "y", "n_electrons_interface", "drift_time_mean", "drift_time_spread"]

# The screen includes isolated hypotheses and likely-useful combinations.
FEATURE_SET_EXTRAS: Dict[str, List[str]] = {
    "baseline": [],
    "event_multiplicity": ["event_log_cluster_count"],
    "electron_fraction": ["electron_fraction_of_event_sum", "electron_fraction_of_event_max"],
    "electron_rank": ["electron_rank_fraction"],
    "drift_offset": ["drift_time_minus_event_mean", "drift_time_rank_fraction"],
    "relative_geometry": ["x_minus_event_mean", "y_minus_event_mean", "distance_from_event_centroid"],
    "electron_plus_multiplicity": [
        "event_log_cluster_count", "electron_fraction_of_event_sum", "electron_fraction_of_event_max",
        "electron_rank_fraction",
    ],
    "core_event_relative": [
        "event_log_cluster_count", "electron_fraction_of_event_sum", "electron_fraction_of_event_max",
        "electron_rank_fraction", "drift_time_minus_event_mean", "drift_time_rank_fraction",
    ],
    "all_event_relative": [
        "event_log_cluster_count", "electron_fraction_of_event_sum", "electron_fraction_of_event_max",
        "electron_rank_fraction", "drift_time_minus_event_mean", "drift_time_rank_fraction",
        "x_minus_event_mean", "y_minus_event_mean", "distance_from_event_centroid",
    ],
}


def available_feature_sets() -> List[str]:
    return list(FEATURE_SET_EXTRAS)


def features_for_set(name: str) -> List[str]:
    try:
        return [*BASE_FEATURES, *FEATURE_SET_EXTRAS[name]]
    except KeyError as exc:
        choices = ", ".join(available_feature_sets())
        raise ValueError(f"Unknown feature set {name!r}. Choose one of: {choices}") from exc


def add_event_relative_features(df: pd.DataFrame, event_col: str = "event_number") -> pd.DataFrame:
    """Return a copy augmented with input-only features calculated within event.

    The calculations preserve row order. ``p_main`` is never accessed, so the
    inputs are usable at inference time and cannot leak the target.
    """
    out = df.copy()
    grouped = out.groupby(event_col, sort=False)
    counts = grouped[event_col].transform("size").astype(np.float32)
    out["event_log_cluster_count"] = np.log1p(counts)

    electrons = out["n_electrons_interface"].astype(np.float64)
    electron_sum = grouped["n_electrons_interface"].transform("sum").astype(np.float64)
    electron_max = grouped["n_electrons_interface"].transform("max").astype(np.float64)
    out["electron_fraction_of_event_sum"] = np.divide(electrons, electron_sum, out=np.zeros(len(out)), where=electron_sum.to_numpy() != 0)
    out["electron_fraction_of_event_max"] = np.divide(electrons, electron_max, out=np.zeros(len(out)), where=electron_max.to_numpy() != 0)
    electron_rank = grouped["n_electrons_interface"].rank(method="average", pct=True).astype(np.float64)
    out["electron_rank_fraction"] = np.where(counts.to_numpy() > 1, electron_rank, 0.0)

    drift = out["drift_time_mean"].astype(np.float64)
    drift_mean = grouped["drift_time_mean"].transform("mean").astype(np.float64)
    out["drift_time_minus_event_mean"] = drift - drift_mean
    drift_rank = grouped["drift_time_mean"].rank(method="average", pct=True).astype(np.float64)
    out["drift_time_rank_fraction"] = np.where(counts.to_numpy() > 1, drift_rank, 0.0)

    x_mean = grouped["x"].transform("mean").astype(np.float64)
    y_mean = grouped["y"].transform("mean").astype(np.float64)
    out["x_minus_event_mean"] = out["x"].astype(np.float64) - x_mean
    out["y_minus_event_mean"] = out["y"].astype(np.float64) - y_mean
    out["distance_from_event_centroid"] = np.hypot(out["x_minus_event_mean"], out["y_minus_event_mean"])

    engineered = sorted({name for names in FEATURE_SET_EXTRAS.values() for name in names})
    out[engineered] = out[engineered].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return out
