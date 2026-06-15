

from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import wandb

# Import your actual training entrypoint from deepset_train.py.
# Expected signature:
#   train(config: dict) -> dict
# where the returned dict contains final validation metrics.
from deepset_train import train


SWEEP_CONFIG: Dict[str, Any] = {
    "method": "bayes",
    "metric": {
        "name": "val_event_main_accuracy",
        "goal": "maximize",
    },
    "parameters": {
        # Keep the model family the same, but widen substantially.
        "latent_dim": {"values": [64, 128, 256, 512]},
        "phi_hidden": {"values": [128, 256, 512]},
        "rho_hidden": {"values": [128, 256, 512]},

        # Optional: if your code supports depth as a tunable parameter,
        # you can uncomment this and adapt the model builder.
        # "phi_layers": {"values": [2, 3, 4]},
        # "rho_layers": {"values": [2, 3, 4]},

        # Optimization.
        "learning_rate": {"values": [1e-4, 3e-4, 1e-3, 3e-3]},
        "batch_size": {"values": [256, 512]},
        "weight_decay": {"values": [0.0, 1e-6, 1e-5, 1e-4]},

        # Scheduler settings. The training code should honor these.
        "scheduler": {"values": ["reduce_on_plateau", "cosine"]},
        "scheduler_patience": {"value": 8},
        "scheduler_factor": {"value": 0.5},
        "scheduler_min_lr": {"value": 1e-6},
        "scheduler_t_max": {"value": 40},

        # Training length and early stopping.
        "max_epochs": {"value": 500},
        "early_stopping_patience": {"value": 100},
        "early_stopping_metric": {"value": "val_event_main_accuracy"},

        # Keep the objective aligned with your comparison metric.
        "loss_name": {"value": "masked_bce"},
        "seed": {"values": [1, 2, 3]},

        # Useful for logging and reproducibility.
        "num_workers": {"value": 4},
        "pin_memory": {"value": True},
        "cudnn_benchmark": {"value": True},
    },
}


def run() -> None:
    """W&B agent entrypoint."""
    with wandb.init() as run:
        config = dict(wandb.config)

        # Optional: make the run name more readable in W&B.
        run.name = (
            f"ld{config.get('latent_dim')}_"
            f"phi{config.get('phi_hidden')}_"
            f"rho{config.get('rho_hidden')}_"
            f"lr{config.get('learning_rate')}_"
            f"bs{config.get('batch_size')}_"
            f"seed{config.get('seed')}"
        )

        # Delegate to the training code.
        # The training function should:
        # - build the model with the supplied widths/latent dim
        # - train for up to max_epochs
        # - step the scheduler
        # - stop early when validation metric plateaus
        # - log val_event_main_accuracy each epoch
        train(config)


def main():
    parser = argparse.ArgumentParser(description="Launch a W&B sweep agent for DeepSet training.")
    parser.add_argument(
        "--sweep_id",
        type=str,
        default="",
        help="Existing W&B sweep ID. If omitted, a new sweep will be created.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of runs for this agent invocation.",
    )
    parser.add_argument(
        "--create_sweep",
        action="store_true",
        help="Force creation of a new sweep before launching the agent.",
    )
    args = parser.parse_args()

    if args.create_sweep or not args.sweep_id:
        sweep_id = wandb.sweep(SWEEP_CONFIG, project="xenon-deepset")
        print(f"Created sweep: {sweep_id}")
    else:
        sweep_id = args.sweep_id

    wandb.agent(sweep_id, function=run, count=args.count)


if __name__ == "__main__":
    main()