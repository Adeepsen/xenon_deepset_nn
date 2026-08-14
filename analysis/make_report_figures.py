"""Create deterministic dataset and architecture figures for the REU report."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "s2_tag_training_clusters.npy"
OUTPUT_DIR = ROOT / "analysis" / "report_figures"
TOP13_NS = 192_600.0


def retained_frame() -> pd.DataFrame:
    frame = pd.DataFrame(np.load(DATA_PATH))
    min_drift = frame.groupby("event_number")["drift_time_mean"].min()
    retained = frame.loc[~frame.event_number.isin(min_drift[min_drift < TOP13_NS].index)].copy()
    retained["p_main"] = retained["p_main"].clip(0.0, 1.0)
    return retained


def make_dataset_profile(frame: pd.DataFrame) -> None:
    multiplicities = frame.groupby("event_number").size().to_numpy()
    target = frame["p_main"].to_numpy()
    endpoint_zero = int((target == 0.0).sum())
    fractional = int(((target > 0.0) & (target < 1.0)).sum())
    endpoint_one = int((target == 1.0).sum())

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bins = np.arange(0.5, min(int(multiplicities.max()), 40) + 1.5, 1)
    axes[0].hist(multiplicities, bins=bins, color="#3979a9", edgecolor="white")
    axes[0].set_yscale("log")
    axes[0].set(xlabel="Clusters per event", ylabel="Number of events (log scale)",
                title="Event multiplicity after the fiducial cut", xlim=(0.5, 25.5))
    axes[0].axvline(np.median(multiplicities), color="#d95f02", linestyle="--", label=f"median = {np.median(multiplicities):.0f}")
    axes[0].legend(frameon=False)

    labels = ["$p_{main}=0$", "fractional", "$p_{main}=1$"]
    counts = [endpoint_zero, fractional, endpoint_one]
    bars = axes[1].bar(labels, counts, color=["#4c78a8", "#f58518", "#54a24b"])
    axes[1].set_yscale("log")
    axes[1].set(ylabel="Number of clusters (log scale)", title="Target-label imbalance")
    for bar, count in zip(bars, counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2, count * 1.25, f"{count:,}", ha="center", va="bottom", fontsize=9)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "dataset_profile.png", dpi=220)
    plt.close(figure)


def box(axis, xy, text, facecolor):
    patch = FancyBboxPatch(xy, 0.21, 0.18, boxstyle="round,pad=0.02,rounding_size=0.025",
                           linewidth=1.3, edgecolor="#333333", facecolor=facecolor)
    axis.add_patch(patch)
    axis.text(xy[0] + 0.105, xy[1] + 0.09, text, ha="center", va="center", fontsize=10, wrap=True)


def make_architecture_schematic() -> None:
    figure, axis = plt.subplots(figsize=(11, 3.4))
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    boxes = [
        ((0.02, 0.42), "Variable-size event\ncluster features\n(14 inputs)", "#dbe9f6"),
        ((0.28, 0.42), "Shared encoder $\\phi$\n5 layers, width 256\n128-d embedding", "#d9f0d3"),
        ((0.54, 0.42), "Independent sigmoid gates\nand gated-sum\nevent context", "#fde0dd"),
        ((0.80, 0.42), "Shared head $\\rho$\n4 layers, width 512\n$\\hat{p}_{main}$ per cluster", "#fff2cc"),
    ]
    for xy, text, color in boxes:
        box(axis, xy, text, color)
    for start, end in ((0.23, 0.28), (0.49, 0.54), (0.75, 0.80)):
        axis.annotate("", xy=(end, 0.51), xytext=(start, 0.51), arrowprops=dict(arrowstyle="->", lw=1.5, color="#333333"))
    axis.text(0.515, 0.22, "The event context is broadcast back to every encoded cluster before prediction.", ha="center", fontsize=10)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "gated_sum_architecture.png", dpi=220)
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = retained_frame()
    make_dataset_profile(frame)
    make_architecture_schematic()
    print(f"Wrote figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
