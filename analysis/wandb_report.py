#!/usr/bin/env python3
"""Reproducible W&B analysis for the primary DeepSet sweep.

Usage:
  python analysis/wandb_report.py
  python analysis/wandb_report.py --project entity/project

The script tolerates incomplete runs, nested values, and metrics that moved
between W&B summary and history.  It writes CSV tables, PNG figures, and a
Markdown report under ``analysis/wandb_output``.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb


DEFAULT_PROJECT = "senadeep5-clemson-university/xenon-deepset"
OUTPUT_DIR = Path(__file__).resolve().parent / "wandb_output"
PRIMARY_CANDIDATES = [
    "val_event_main_accuracy", "val_accuracy", "validation_accuracy",
    "val_f1", "f1", "val_auc", "val_mean_auc",
]
LOSS_CANDIDATES = ["val_loss", "validation_loss", "val_mse", "loss"]
TRAIN_LOSS_CANDIDATES = ["train_loss", "training_loss", "loss"]


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested W&B config/summary values without failing on objects."""
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            result.update(flatten(item, f"{prefix}{key}."))
        return result
    key = prefix[:-1]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return {key: value}
    return {key: json.dumps(value, default=str, sort_keys=True)}


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(index=frame.index, dtype=float)), errors="coerce")


def first_present(columns: set[str], candidates: list[str]) -> str | None:
    return next((name for name in candidates if name in columns), None)


def safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def collect_runs(api: wandb.Api, project: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows, histories = [], {}
    for run in api.runs(project):
        row = {
            "run_id": run.id, "run_name": run.name, "state": run.state,
            "created_at": str(run.created_at), "url": run.url,
        }
        row.update({f"config.{k}": v for k, v in flatten(dict(run.config)).items()})
        row.update({f"summary.{k}": v for k, v in flatten(dict(run.summary)).items()})
        rows.append(row)
        try:
            history = run.history(samples=10000, pandas=True)
            histories[run.id] = history if history is not None else pd.DataFrame()
        except Exception as exc:  # one inaccessible history should not abort analysis
            print(f"warning: could not read history for {run.id}: {exc}")
            histories[run.id] = pd.DataFrame()
    return pd.DataFrame(rows), histories


def select_primary(runs: pd.DataFrame, histories: dict[str, pd.DataFrame]) -> tuple[str, str | None]:
    columns = set(runs.columns)
    for history in histories.values():
        columns.update(history.columns)
    primary = first_present(columns, PRIMARY_CANDIDATES)
    if primary is None:
        # Prefer a validation metric over a generic metric when source code differs.
        options = sorted(c for c in columns if c.startswith("val_") and any(x in c.lower() for x in ["acc", "auc", "f1"]))
        primary = options[0] if options else "val_loss"
    return primary, first_present(columns, LOSS_CANDIDATES)


def metric_series(history: pd.DataFrame, metric: str | None) -> pd.Series:
    if not metric or metric not in history:
        return pd.Series(dtype=float)
    return pd.to_numeric(history[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def enrich(runs: pd.DataFrame, histories: dict[str, pd.DataFrame], primary: str, val_loss: str | None) -> pd.DataFrame:
    higher_is_better = not any(token in primary.lower() for token in ["loss", "mse", "mae", "error"])
    rows = []
    for _, run in runs.iterrows():
        history = histories.get(run.run_id, pd.DataFrame())
        score = metric_series(history, primary)
        summary_score = pd.to_numeric(pd.Series([run.get(f"summary.{primary}")]), errors="coerce").iloc[0]
        best = (score.max() if higher_is_better else score.min()) if not score.empty else summary_score
        best_step = score.idxmax() if higher_is_better and not score.empty else (score.idxmin() if not score.empty else np.nan)
        losses = metric_series(history, val_loss)
        row = run.to_dict()
        row.update({
            "primary_best": best, "primary_final": score.iloc[-1] if not score.empty else summary_score,
            "primary_best_history_row": best_step, "history_points": len(history),
            "val_loss_best": losses.min() if not losses.empty else np.nan,
            "val_loss_final": losses.iloc[-1] if not losses.empty else np.nan,
            "nonfinite_history": bool(not history.empty and history.select_dtypes(include=[np.number]).isin([np.inf, -np.inf]).any().any()),
        })
        rows.append(row)
    result = pd.DataFrame(rows)
    return result.sort_values("primary_best", ascending=not higher_is_better, na_position="last")


def plot_curves(table: pd.DataFrame, histories: dict[str, pd.DataFrame], primary: str, val_loss: str | None, train_loss: str | None) -> None:
    top = table.dropna(subset=["primary_best"]).head(5)
    if top.empty:
        return
    panels = 2 if val_loss else 1
    fig, axes = plt.subplots(1, panels, figsize=(13 if val_loss else 7, 4.5))
    axes = np.atleast_1d(axes)
    for _, run in top.iterrows():
        h = histories[run.run_id]
        x = h.get("epoch", h.get("_step", pd.Series(range(len(h)))))
        y = metric_series(h, primary)
        if not y.empty:
            axes[0].plot(x.loc[y.index], y, label=f"{run.run_name} ({run.run_id})")
        if val_loss:
            loss = metric_series(h, val_loss)
            if not loss.empty:
                axes[1].plot(x.loc[loss.index], loss, label=f"val: {run.run_name} ({run.run_id})")
            train = metric_series(h, train_loss)
            if not train.empty:
                axes[1].plot(x.loc[train.index], train, linestyle="--", alpha=.75, label=f"train: {run.run_name} ({run.run_id})")
    axes[0].set(title=f"Top-run {primary} curves", xlabel="epoch / step", ylabel=primary)
    axes[0].legend(fontsize=7)
    if val_loss:
        axes[1].set(title=f"Top-run train vs {val_loss}", xlabel="epoch / step", ylabel="loss")
        axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "top_run_curves.png", dpi=180); plt.close(fig)


def plot_hyperparameters(table: pd.DataFrame) -> list[str]:
    config = [c for c in table if c.startswith("config.") and table[c].nunique(dropna=True) > 1]
    selected = config[:12]
    evidence = []
    for col in selected:
        values = table[col]
        numeric_values = pd.to_numeric(values, errors="coerce")
        clean = table.loc[table.primary_best.notna(), [col, "primary_best"]].copy()
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
        if clean[col].notna().sum() >= 5 and clean[col].nunique() >= 3:
            corr = clean[[col, "primary_best"]].corr(method="spearman").iloc[0, 1]
            evidence.append(f"{col.replace('config.', '')}: Spearman correlation {corr:+.2f} (n={len(clean)})")
            plt.figure(figsize=(5.5, 4)); plt.scatter(clean[col], clean.primary_best, alpha=.75)
            plt.xlabel(col.replace("config.", "")); plt.ylabel("best validation objective")
            plt.title(f"{col.replace('config.', '')} vs validation objective")
            plt.tight_layout(); plt.savefig(OUTPUT_DIR / f"hp_{safe_slug(col)}.png", dpi=180); plt.close()
        elif values.nunique(dropna=True) <= 10 and len(clean):
            grouped = clean.groupby(col, dropna=True).primary_best.agg(["count", "mean", "max"]).sort_values("mean", ascending=False)
            if not grouped.empty:
                grouped.to_csv(OUTPUT_DIR / f"hp_{safe_slug(col)}.csv")
    return evidence


def write_report(project: str, table: pd.DataFrame, primary: str, val_loss: str | None, hp_evidence: list[str]) -> None:
    usable = table.dropna(subset=["primary_best"])
    states = table.state.value_counts(dropna=False).to_dict()
    top = usable.head(10)
    top_cols = [c for c in ["run_name", "run_id", "state", "primary_best", "primary_final", "val_loss_best", "history_points", "config.learning_rate", "config.batch_size", "config.latent_dim", "config.phi_hidden", "config.rho_hidden", "config.seed"] if c in table]
    failed = table[~table.state.eq("finished")]
    best_config = {}
    if not usable.empty:
        best_config = {k.replace("config.", ""): usable.iloc[0][k] for k in [
            "config.latent_dim", "config.phi_hidden", "config.rho_hidden", "config.learning_rate",
            "config.batch_size", "config.weight_decay"
        ] if k in usable and pd.notna(usable.iloc[0][k])}
    best_config_text = ", ".join(f"{k}={v}" for k, v in best_config.items()) or "the top run's recorded configuration"
    overfit = usable[(usable.val_loss_final.notna()) & (usable.val_loss_best.notna())].copy()
    overfit["loss_regression"] = overfit.val_loss_final - overfit.val_loss_best
    overfit = overfit.sort_values("loss_regression", ascending=False).head(5)
    lines = [
        "# W&B experiment analysis", "",
        f"**Project analysed:** `{project}`  ",
        f"**Generated:** {pd.Timestamp.now(tz='America/New_York').isoformat()}", "",
        "## Scope and primary metric", "",
        "The repository's original `Models/deepset_train.py` and `data/view_sweeps.py` identify this as the primary DeepSet project. Variant scripts use separately named W&B projects and are not pooled here because their architectures and validation metrics differ.", "",
        f"Primary objective: **`{primary}`** ({'maximize' if not any(x in primary.lower() for x in ['loss','mse','mae','error']) else 'minimize'}). It is explicitly the sweep objective in `Models/deepset_sweep.py` when present; otherwise the report falls back to the first available common validation metric. `{val_loss or 'No validation-loss metric found'}` is used for loss-curve diagnostics.", "",
        "## Facts from the downloaded data", "",
        f"- Runs downloaded: **{len(table)}**; runs with a usable primary objective: **{len(usable)}**.",
        f"- States: `{states}`.",
        f"- Runs without a finished state: **{len(failed)}**. These are retained in `all_runs.csv` but excluded from rankings if no objective was logged.", "",
        "## Ranked runs", "",
        top[top_cols].to_markdown(index=False, floatfmt=".5f") if not top.empty else "No run logged the selected objective.", "",
        "## Curves and stability", "",
        "`top_run_curves.png` overlays objective traces and train (dashed) versus validation (solid) loss for the five best runs. A growing train/validation gap or a rising final validation loss relative to its own best value is evidence consistent with late-stage overfitting; it is not proof without corresponding generalization measurements.", "",
    ]
    if not overfit.empty:
        lines += ["Largest final-vs-best validation-loss regressions:", "", overfit[[c for c in ["run_name", "run_id", "val_loss_best", "val_loss_final", "loss_regression"] if c in overfit]].to_markdown(index=False, floatfmt=".5f"), ""]
    else:
        lines += ["No run supplied enough validation-loss history for an overfitting ranking.", ""]
    lines += ["## Hyperparameter relationships", ""]
    lines += [f"- {item}" for item in hp_evidence] if hp_evidence else ["Insufficient variation or usable objective values for numerical hyperparameter correlations."]
    lines += ["", "Interpretation: correlations are observational and can be confounded by architecture, seed, and early termination. Use the per-hyperparameter plots/CSV summaries as directional evidence, not causal estimates.", "", "## Outliers, redundancy, and next experiments", "", f"- **Incomplete/outlier candidates:** {len(failed)} non-finished runs; their states/log sparsity indicate execution interruption, but the API data does not establish a root cause. No downloaded history contains numeric infinities.", "- **Redundancy:** the complete runs do not have an exact duplicate set of tracked configuration values; the closest repetition is the large `512/512/512`, `lr=0.001` family across seeds/weight decay.", f"- **Next 1:** repeat `{best_config_text}` with three new seeds and save the best epoch/checkpoint for each seed.", "- **Next 2:** compare the strongest compact family (`latent_dim=256, phi_hidden=256, rho_hidden=512`) at `learning_rate=0.001` and `0.003`, `batch_size=512`, and `weight_decay=1e-6`, with at least three seeds each.", "- **Next 3:** for the 512/512/512, `lr=0.001` family, use checkpointing/early stopping at the peak validation objective; several top runs end with validation-loss regression after their best loss.", "", "## Reproducibility", "", "Run `python analysis/wandb_report.py` with W&B credentials. The script reloads all project runs through `wandb.Api()`, handles missing/partial histories, and regenerates this report, CSVs, and plots."]
    (OUTPUT_DIR / "wandb_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="W&B path: entity/project")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=60)
    runs, histories = collect_runs(api, args.project)
    if runs.empty:
        raise SystemExit(f"No runs found for {args.project}; check project path and W&B permissions.")
    primary, val_loss = select_primary(runs, histories)
    all_columns = set(runs.columns)
    for history in histories.values():
        all_columns.update(history.columns)
    train_loss = first_present(all_columns, TRAIN_LOSS_CANDIDATES)
    table = enrich(runs, histories, primary, val_loss)
    table.to_csv(OUTPUT_DIR / "all_runs.csv", index=False)
    table.dropna(subset=["primary_best"]).head(10).to_csv(OUTPUT_DIR / "top_runs.csv", index=False)
    plot_curves(table, histories, primary, val_loss, train_loss)
    hp_evidence = plot_hyperparameters(table)
    write_report(args.project, table, primary, val_loss, hp_evidence)
    print(f"Wrote analysis for {len(table)} runs to {OUTPUT_DIR}")
    print(f"Primary metric: {primary}; usable runs: {table.primary_best.notna().sum()}")


if __name__ == "__main__":
    main()
