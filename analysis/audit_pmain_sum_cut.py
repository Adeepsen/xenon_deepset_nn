#!/usr/bin/env python3
"""Audit a proposed event-level ``sum(p_main) > 1`` data-quality cut.

This script is read-only with respect to the source data and training caches.
It reports counts for the raw data, the existing top-13-cm fiducial cut, and
the proposed p_main-sum cut, including their overlap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "s2_tag_training_clusters.npy"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "wandb_output" / "pmain_sum_cut_audit.json"
TOP13_NS = 192_600.0


def pct(count: int, total: int) -> float:
    return 100.0 * count / total if total else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sum-tolerance", type=float, default=1e-6,
                        help="Cut events when sum(p_main) > 1 + this tolerance.")
    args = parser.parse_args()

    data = np.load(args.data, mmap_mode="r")
    required = {"event_number", "p_main", "drift_time_mean"}
    missing = required - set(data.dtype.names or ())
    if missing:
        raise ValueError(f"Dataset is missing required fields: {sorted(missing)}")

    event_number = np.asarray(data["event_number"])
    p_main = np.asarray(data["p_main"], dtype=np.float64)
    drift_time = np.asarray(data["drift_time_mean"], dtype=np.float64)

    events, inverse = np.unique(event_number, return_inverse=True)
    n_events = len(events)
    event_pmain_sum = np.bincount(inverse, weights=p_main, minlength=n_events)
    event_min_drift = np.full(n_events, np.inf)
    np.minimum.at(event_min_drift, inverse, drift_time)

    fiducial_removed = event_min_drift < TOP13_NS
    pmain_sum_removed = event_pmain_sum > 1.0 + args.sum_tolerance
    overlap = fiducial_removed & pmain_sum_removed
    retained_after_fiducial = ~fiducial_removed
    retained_after_both = ~(fiducial_removed | pmain_sum_removed)

    # Cluster counts use the same inverse map, so they accurately account for
    # an event-level cut even if the input rows are not event-sorted.
    event_cluster_count = np.bincount(inverse, minlength=n_events).astype(np.int64)
    result = {
        "source": str(args.data),
        "n_clusters_raw": int(len(data)),
        "n_events_raw": int(n_events),
        "top13_ns": TOP13_NS,
        "pmain_sum_threshold": 1.0 + args.sum_tolerance,
        "pmain_sum_tolerance": args.sum_tolerance,
        "events_removed_fiducial": int(fiducial_removed.sum()),
        "events_removed_pmain_sum_raw": int(pmain_sum_removed.sum()),
        "events_removed_by_both_cuts": int(overlap.sum()),
        "events_removed_only_pmain_sum_after_fiducial": int((pmain_sum_removed & retained_after_fiducial).sum()),
        "events_retained_after_fiducial": int(retained_after_fiducial.sum()),
        "events_retained_after_both_cuts": int(retained_after_both.sum()),
        "clusters_removed_only_pmain_sum_after_fiducial": int(event_cluster_count[pmain_sum_removed & retained_after_fiducial].sum()),
        "clusters_retained_after_both_cuts": int(event_cluster_count[retained_after_both].sum()),
        "fraction_events_removed_only_pmain_sum_after_fiducial_pct": pct(
            int((pmain_sum_removed & retained_after_fiducial).sum()), int(retained_after_fiducial.sum())
        ),
        "fraction_events_removed_by_both_cuts_pct": pct(int((~retained_after_both).sum()), n_events),
        "event_pmain_sum_quantiles": {
            str(q): float(np.quantile(event_pmain_sum, q)) for q in [0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0]
        },
        "event_pmain_sum_max_after_fiducial": float(event_pmain_sum[retained_after_fiducial].max()),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print("p_main event-sum cut audit")
    for key in [
        "n_events_raw", "n_clusters_raw", "events_removed_fiducial",
        "events_removed_pmain_sum_raw", "events_removed_by_both_cuts",
        "events_removed_only_pmain_sum_after_fiducial", "events_retained_after_fiducial",
        "events_retained_after_both_cuts", "clusters_removed_only_pmain_sum_after_fiducial",
        "clusters_retained_after_both_cuts", "fraction_events_removed_only_pmain_sum_after_fiducial_pct",
        "fraction_events_removed_by_both_cuts_pct", "event_pmain_sum_max_after_fiducial",
    ]:
        print(f"{key}: {result[key]}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
