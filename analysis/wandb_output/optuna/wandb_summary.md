# W&B experiment analysis

**Project analysed:** `senadeep5-clemson-university/xenon-graph-pooling-optuna`  
**Generated:** 2026-07-23T12:45:48.870849-04:00

## Scope and primary metric

The repository's original `Models/deepset_train.py` and `data/view_sweeps.py` identify this as the primary DeepSet project. Variant scripts use separately named W&B projects and are not pooled here because their architectures and validation metrics differ.

Primary objective: **`val_event_main_accuracy_strict`** (maximize). It is explicitly the sweep objective in `Models/deepset_sweep.py` when present; otherwise the report falls back to the first available common validation metric. `val_loss` is used for loss-curve diagnostics.

## Facts from the downloaded data

- Runs downloaded: **32**; runs with a usable primary objective: **32**.
- States: `{'finished': 31, 'crashed': 1}`.
- Runs without a finished state: **1**. These are retained in `all_runs.csv` but excluded from rankings if no objective was logged.

## Ranked runs

| run_name   | run_id   | state    |   primary_best |   primary_final |   val_loss_best |   history_points |   config.learning_rate |   config.batch_size |   config.latent_dim |   config.phi_hidden |
|:-----------|:---------|:---------|---------------:|----------------:|----------------:|-----------------:|-----------------------:|--------------------:|--------------------:|--------------------:|
| trial_0023 | 5y0g85cn | finished |        0.74375 |         0.74175 |         0.48971 |              254 |                0.00100 |                1024 |                 256 |                 256 |
| trial_0031 | 4a29zofn | crashed  |        0.74305 |         0.74244 |         0.48643 |              202 |                0.00057 |                1024 |                 256 |                 256 |
| trial_0017 | 940qrjkd | finished |        0.74268 |         0.74178 |         0.50673 |              300 |                0.00045 |                 256 |                 128 |                 128 |
| trial_0004 | zdqz5o4n | finished |        0.74256 |         0.74056 |         0.50353 |              245 |                0.00168 |                1024 |                 128 |                 256 |
| trial_0019 | nxf5ynrd | finished |        0.74209 |         0.74162 |         0.51410 |              155 |                0.00111 |                1024 |                 128 |                 512 |
| trial_0015 | 7ojc6q4h | finished |        0.74139 |         0.74038 |         0.50859 |              300 |                0.00113 |                1024 |                 128 |                 256 |
| trial_0027 | 7k528608 | finished |        0.74119 |         0.73966 |         0.52026 |              122 |                0.00050 |                1024 |                 128 |                 256 |
| trial_0020 | 09b5n6ql | finished |        0.73994 |         0.73960 |         0.52682 |              136 |                0.00072 |                1024 |                 128 |                 256 |
| trial_0000 | dwizy58s | finished |        0.73953 |         0.73922 |         0.52752 |              300 |                0.00008 |                 256 |                  64 |                 512 |
| trial_0018 | zf9w6hgd | finished |        0.73902 |         0.73873 |         0.50899 |              212 |                0.00182 |                1024 |                 128 |                 256 |

## Curves and stability

`top_run_curves.png` overlays objective traces and train (dashed) versus validation (solid) loss for the five best runs. A growing train/validation gap or a rising final validation loss relative to its own best value is evidence consistent with late-stage overfitting; it is not proof without corresponding generalization measurements.

Largest final-vs-best validation-loss regressions:

| run_name   | run_id   |   val_loss_best |   val_loss_final |   loss_regression |
|:-----------|:---------|----------------:|-----------------:|------------------:|
| trial_0022 | iof32u1n |         0.77869 |          0.81607 |           0.03737 |
| trial_0027 | 7k528608 |         0.52026 |          0.55591 |           0.03565 |
| trial_0020 | 09b5n6ql |         0.52682 |          0.54486 |           0.01804 |
| trial_0019 | nxf5ynrd |         0.51410 |          0.53061 |           0.01651 |
| trial_0026 | atsit2qw |         0.62335 |          0.63951 |           0.01616 |

## Hyperparameter relationships

- dropout: Spearman correlation -0.48 (n=32)
- batch_size: Spearman correlation +0.27 (n=32)
- head_depth: Spearman correlation +0.24 (n=32)
- latent_dim: Spearman correlation +0.28 (n=32)
- phi_hidden: Spearman correlation +0.46 (n=32)
- head_hidden: Spearman correlation +0.31 (n=32)
- trial_number: Spearman correlation +0.17 (n=32)
- weight_decay: Spearman correlation -0.55 (n=32)
- encoder_depth: Spearman correlation +0.48 (n=32)
- learning_rate: Spearman correlation +0.02 (n=32)

Interpretation: correlations are observational and can be confounded by architecture, seed, and early termination. Use the per-hyperparameter plots/CSV summaries as directional evidence, not causal estimates.

## Outliers, redundancy, and next experiments

- **Incomplete/outlier candidates:** 1 non-finished runs; their states/log sparsity indicate execution interruption, but the API data does not establish a root cause. No downloaded history contains numeric infinities.
- **Redundancy:** the complete runs do not have an exact duplicate set of tracked configuration values; the closest repetition is the large `512/512/512`, `lr=0.001` family across seeds/weight decay.
- **Next 1:** repeat `latent_dim=256, phi_hidden=256, learning_rate=0.0010024149844640624, batch_size=1024, weight_decay=3.166882229533655e-06` with three new seeds and save the best epoch/checkpoint for each seed.
- **Next 2:** compare the strongest compact family (`latent_dim=256, phi_hidden=256, rho_hidden=512`) at `learning_rate=0.001` and `0.003`, `batch_size=512`, and `weight_decay=1e-6`, with at least three seeds each.
- **Next 3:** for the 512/512/512, `lr=0.001` family, use checkpointing/early stopping at the peak validation objective; several top runs end with validation-loss regression after their best loss.

## Reproducibility

Run `python analysis/wandb_report.py` with W&B credentials. The script reloads all project runs through `wandb.Api()`, handles missing/partial histories, and regenerates this report, CSVs, and plots.
