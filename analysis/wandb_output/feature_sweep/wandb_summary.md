# W&B experiment analysis

**Project analysed:** `senadeep5-clemson-university/xenon-graph-pooling-pmain-feature-sweep`  
**Generated:** 2026-07-23T12:44:15.909408-04:00

## Scope and primary metric

The repository's original `Models/deepset_train.py` and `data/view_sweeps.py` identify this as the primary DeepSet project. Variant scripts use separately named W&B projects and are not pooled here because their architectures and validation metrics differ.

Primary objective: **`val_p_main_mse`** (minimize). It is explicitly the sweep objective in `Models/deepset_sweep.py` when present; otherwise the report falls back to the first available common validation metric. `No validation-loss metric found` is used for loss-curve diagnostics.

## Facts from the downloaded data

- Runs downloaded: **9**; runs with a usable primary objective: **9**.
- States: `{'finished': 9}`.
- Runs without a finished state: **0**. These are retained in `all_runs.csv` but excluded from rankings if no objective was logged.

## Ranked runs

| run_name                          | run_id   | state    |   primary_best |   primary_final |   val_loss_best |   history_points |   config.learning_rate |   config.batch_size |   config.latent_dim |   config.phi_hidden |   config.seed |
|:----------------------------------|:---------|:---------|---------------:|----------------:|----------------:|-----------------:|-----------------------:|--------------------:|--------------------:|--------------------:|--------------:|
| all_event_relative-seed42         | wew4yp3f | finished |        0.06073 |         0.06180 |             nan |              140 |                0.00030 |                1024 |                 128 |                 256 |            42 |
| core_event_relative-seed42        | dkn0c7mo | finished |        0.06165 |         0.06253 |             nan |              137 |                0.00030 |                1024 |                 128 |                 256 |            42 |
| drift_offset-seed42               | otvyrs1d | finished |        0.06256 |         0.06333 |             nan |              145 |                0.00030 |                1024 |                 128 |                 256 |            42 |
| electron_plus_multiplicity-seed42 | 7wm220ao | finished |        0.06437 |         0.06548 |             nan |              165 |                0.00030 |                1024 |                 128 |                 256 |            42 |
| electron_fraction-seed42          | fyuwl3ed | finished |        0.06454 |         0.06541 |             nan |              161 |                0.00030 |                1024 |                 128 |                 256 |            42 |
| relative_geometry-seed42          | r1dcc719 | finished |        0.06467 |         0.06561 |             nan |              210 |                0.00030 |                1024 |                 128 |                 256 |            42 |
| event_multiplicity-seed42         | uvgle01t | finished |        0.06506 |         0.06611 |             nan |              179 |                0.00030 |                1024 |                 128 |                 256 |            42 |
| baseline-seed42                   | o2v5hzui | finished |        0.06520 |         0.06600 |             nan |              163 |                0.00030 |                1024 |                 128 |                 256 |            42 |
| electron_rank-seed42              | 0mt8kfrp | finished |        0.06589 |         0.06683 |             nan |              182 |                0.00030 |                1024 |                 128 |                 256 |            42 |

## Curves and stability

`top_run_curves.png` overlays objective traces and train (dashed) versus validation (solid) loss for the five best runs. A growing train/validation gap or a rising final validation loss relative to its own best value is evidence consistent with late-stage overfitting; it is not proof without corresponding generalization measurements.

No run supplied enough validation-loss history for an overfitting ranking.

## Hyperparameter relationships

- n_input_features: Spearman correlation -0.87 (n=9)

Interpretation: correlations are observational and can be confounded by architecture, seed, and early termination. Use the per-hyperparameter plots/CSV summaries as directional evidence, not causal estimates.

## Outliers, redundancy, and next experiments

- **Incomplete/outlier candidates:** 0 non-finished runs; their states/log sparsity indicate execution interruption, but the API data does not establish a root cause. No downloaded history contains numeric infinities.
- **Redundancy:** the complete runs do not have an exact duplicate set of tracked configuration values; the closest repetition is the large `512/512/512`, `lr=0.001` family across seeds/weight decay.
- **Next 1:** repeat `latent_dim=128, phi_hidden=256, learning_rate=0.0003, batch_size=1024, weight_decay=1.076825417908119e-05` with three new seeds and save the best epoch/checkpoint for each seed.
- **Next 2:** compare the strongest compact family (`latent_dim=256, phi_hidden=256, rho_hidden=512`) at `learning_rate=0.001` and `0.003`, `batch_size=512`, and `weight_decay=1e-6`, with at least three seeds each.
- **Next 3:** for the 512/512/512, `lr=0.001` family, use checkpointing/early stopping at the peak validation objective; several top runs end with validation-loss regression after their best loss.

## Reproducibility

Run `python analysis/wandb_report.py` with W&B credentials. The script reloads all project runs through `wandb.Api()`, handles missing/partial histories, and regenerates this report, CSVs, and plots.
