# W&B experiment analysis

**Project analysed:** `senadeep5-clemson-university/xenon-pooling`  
**Generated:** 2026-07-23T12:46:52.549072-04:00

## Scope and primary metric

The repository's original `Models/deepset_train.py` and `data/view_sweeps.py` identify this as the primary DeepSet project. Variant scripts use separately named W&B projects and are not pooled here because their architectures and validation metrics differ.

Primary objective: **`val_event_main_accuracy`** (maximize). It is explicitly the sweep objective in `Models/deepset_sweep.py` when present; otherwise the report falls back to the first available common validation metric. `val_loss` is used for loss-curve diagnostics.

## Facts from the downloaded data

- Runs downloaded: **4**; runs with a usable primary objective: **4**.
- States: `{'finished': 4}`.
- Runs without a finished state: **0**. These are retained in `all_runs.csv` but excluded from rankings if no objective was logged.

## Ranked runs

| run_name         | run_id   | state    |   primary_best |   primary_final |   val_loss_best |   history_points |   config.learning_rate |   config.batch_size |   config.latent_dim |   config.phi_hidden |
|:-----------------|:---------|:---------|---------------:|----------------:|----------------:|-----------------:|-----------------------:|--------------------:|--------------------:|--------------------:|
| absurd-valley-2  | 3yeab7lu | finished |        0.74079 |         0.73863 |         0.27841 |              134 |                0.00100 |                 512 |                 128 |                 512 |
| jumping-planet-3 | 50g8sd0p | finished |        0.73718 |         0.73618 |         0.27630 |              323 |                0.00100 |                 512 |                 256 |                 512 |
| robust-firefly-1 | c73d7m1u | finished |        0.73701 |         0.73678 |         0.28167 |              202 |                0.00100 |                 512 |                 256 |                 512 |
| fine-sun-4       | 6tvofja8 | finished |        0.73701 |         0.73678 |         0.28167 |              202 |                0.00100 |                 512 |                 256 |                 512 |

## Curves and stability

`top_run_curves.png` overlays objective traces and train (dashed) versus validation (solid) loss for the five best runs. A growing train/validation gap or a rising final validation loss relative to its own best value is evidence consistent with late-stage overfitting; it is not proof without corresponding generalization measurements.

Largest final-vs-best validation-loss regressions:

| run_name         | run_id   |   val_loss_best |   val_loss_final |   loss_regression |
|:-----------------|:---------|----------------:|-----------------:|------------------:|
| jumping-planet-3 | 50g8sd0p |         0.27630 |          0.28171 |           0.00542 |
| absurd-valley-2  | 3yeab7lu |         0.27841 |          0.27844 |           0.00002 |
| robust-firefly-1 | c73d7m1u |         0.28167 |          0.28167 |           0.00000 |
| fine-sun-4       | 6tvofja8 |         0.28167 |          0.28167 |           0.00000 |

## Hyperparameter relationships

Insufficient variation or usable objective values for numerical hyperparameter correlations.

Interpretation: correlations are observational and can be confounded by architecture, seed, and early termination. Use the per-hyperparameter plots/CSV summaries as directional evidence, not causal estimates.

## Outliers, redundancy, and next experiments

- **Incomplete/outlier candidates:** 0 non-finished runs; their states/log sparsity indicate execution interruption, but the API data does not establish a root cause. No downloaded history contains numeric infinities.
- **Redundancy:** the complete runs do not have an exact duplicate set of tracked configuration values; the closest repetition is the large `512/512/512`, `lr=0.001` family across seeds/weight decay.
- **Next 1:** repeat `latent_dim=128, phi_hidden=512, learning_rate=0.001, batch_size=512, weight_decay=0` with three new seeds and save the best epoch/checkpoint for each seed.
- **Next 2:** compare the strongest compact family (`latent_dim=256, phi_hidden=256, rho_hidden=512`) at `learning_rate=0.001` and `0.003`, `batch_size=512`, and `weight_decay=1e-6`, with at least three seeds each.
- **Next 3:** for the 512/512/512, `lr=0.001` family, use checkpointing/early stopping at the peak validation objective; several top runs end with validation-loss regression after their best loss.

## Reproducibility

Run `python analysis/wandb_report.py` with W&B credentials. The script reloads all project runs through `wandb.Api()`, handles missing/partial histories, and regenerates this report, CSVs, and plots.
