"""Plot train-versus-validation p_main predictions for the selected BCE run.

The BCE model is evaluated with the same event-level split used during its
training. The training panel is a deterministic event-level sample for memory
safety; all validation clusters are evaluated. Test events are never read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_feature_sweep_predictions import plot_panel, predict, split_arrays


FEATURE_SET = "all_event_relative"
OUTPUT_ROOT = Path(__file__).resolve().parent / "bce_output"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-clusters", type=int, default=100_000)
    parser.add_argument("--max-points", type=int, default=100_000)
    args = parser.parse_args()
    output_dir = OUTPUT_ROOT / FEATURE_SET / f"seed_{args.seed}"
    checkpoint_path = output_dir / "best_validation_bce_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"BCE checkpoint not found: {checkpoint_path}")

    split_data = split_arrays([FEATURE_SET], args.seed, FEATURE_SET, args.max_train_clusters)
    truth, predictions = {}, {}
    for split_name in ("train", "validation"):
        truth[split_name] = split_data[split_name]["truth"][:, 0]
        predictions[split_name] = predict(
            FEATURE_SET, checkpoint_path, split_data[split_name][FEATURE_SET],
            split_data[split_name]["truth"], split_data[split_name]["event_ids"], num_workers=0,
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    metrics = {}
    for ax, split_name in zip(axes, ("train", "validation")):
        sample = np.random.default_rng(args.seed).choice(
            len(truth[split_name]), size=min(args.max_points, len(truth[split_name])), replace=False
        )
        label = f"BCE all-event-relative — {split_name}{' event sample' if split_name == 'train' else ''}"
        metrics[split_name] = plot_panel(ax, truth[split_name], predictions[split_name], label, sample, metric_split=split_name)
    fig.suptitle("BCE-with-logits selected checkpoint: training vs. validation predictions", y=1.02)
    fig.tight_layout()
    plot_path = output_dir / f"bce_training_vs_validation_true_vs_predicted_seed{args.seed}.png"
    summary_path = plot_path.with_suffix(".json")
    fig.savefig(plot_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    summary_path.write_text(json.dumps({
        "split": "training event sample and full validation; test events untouched",
        "checkpoint": str(checkpoint_path), "seed": args.seed,
        "n_training_clusters_in_plot": int(len(truth["train"])),
        "n_validation_clusters": int(len(truth["validation"])), "metrics": metrics,
    }, indent=2) + "\n")
    print(f"Saved: {plot_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
