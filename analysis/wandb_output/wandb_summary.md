# W&B experiment analysis

**Project analysed:** `senadeep5-clemson-university/xenon-deepset`  
**Generated:** 2026-07-13T12:40:49.307849-04:00

## Scope and primary metric

The repository's original `Models/deepset_train.py` and `data/view_sweeps.py` identify this as the primary DeepSet project. Variant scripts use separately named W&B projects and are not pooled here because their architectures and validation metrics differ.

Primary objective: **`val_event_main_accuracy`** (maximize). It is explicitly the sweep objective in `Models/deepset_sweep.py` when present; otherwise the report falls back to the first available common validation metric. `val_loss` is used for loss-curve diagnostics.

## Facts from the downloaded data

- Runs downloaded: **57**; runs with a usable primary objective: **44**.
- States: `{'finished': 49, 'killed': 5, 'failed': 2, 'crashed': 1}`.
- Runs without a finished state: **8**. These are retained in `all_runs.csv` but excluded from rankings if no objective was logged.

## Ranked runs

| run_name                                | run_id   | state    |   primary_best |   primary_final |   val_loss_best |   history_points |   config.learning_rate |   config.batch_size |   config.latent_dim |   config.phi_hidden |   config.rho_hidden |   config.seed |
|:----------------------------------------|:---------|:---------|---------------:|----------------:|----------------:|-----------------:|-----------------------:|--------------------:|--------------------:|--------------------:|--------------------:|--------------:|
| ld512_phi512_rho512_lr0.001_bs512_seed3 | zzeg2dq9 | finished |        0.73821 |         0.73776 |         0.26888 |              290 |                0.00100 |                 512 |                 512 |                 512 |                 512 |       3.00000 |
| ld128_phi512_rho512_lr0.001_bs512_seed1 | 7xganm8m | finished |        0.73796 |         0.73750 |         0.27004 |              446 |                0.00100 |                 512 |                 128 |                 512 |                 512 |       1.00000 |
| ld512_phi512_rho512_lr0.001_bs512_seed3 | f3bwe238 | finished |        0.73772 |         0.73514 |         0.27205 |              394 |                0.00100 |                 512 |                 512 |                 512 |                 512 |       3.00000 |
| ld256_phi256_rho512_lr0.003_bs512_seed1 | ia7tiqjp | finished |        0.73768 |         0.73644 |         0.26820 |              501 |                0.00300 |                 512 |                 256 |                 256 |                 512 |       1.00000 |
| ld512_phi512_rho512_lr0.001_bs512_seed2 | v92rrw4y | finished |        0.73765 |         0.73661 |         0.26939 |              387 |                0.00100 |                 512 |                 512 |                 512 |                 512 |       2.00000 |
| ld256_phi512_rho512_lr0.001_bs512_seed2 | 3rirpasl | finished |        0.73765 |         0.73685 |         0.27266 |              495 |                0.00100 |                 512 |                 256 |                 512 |                 512 |       2.00000 |
| ld512_phi512_rho512_lr0.003_bs512_seed3 | lripyoma | finished |        0.73715 |         0.73600 |         0.26899 |              501 |                0.00300 |                 512 |                 512 |                 512 |                 512 |       3.00000 |
| ld256_phi256_rho512_lr0.001_bs256_seed2 | p9bpf0el | finished |        0.73700 |         0.73679 |         0.27036 |              200 |                0.00100 |                 256 |                 256 |                 256 |                 512 |       2.00000 |
| ld256_phi256_rho512_lr0.003_bs512_seed2 | 7279afm9 | finished |        0.73700 |         0.73619 |         0.27066 |              293 |                0.00300 |                 512 |                 256 |                 256 |                 512 |       2.00000 |
| ld128_phi512_rho512_lr0.001_bs512_seed3 | ji979hug | finished |        0.73699 |         0.73621 |         0.27563 |              279 |                0.00100 |                 512 |                 128 |                 512 |                 512 |       3.00000 |

## Curves and stability

`top_run_curves.png` overlays objective traces and train (dashed) versus validation (solid) loss for the five best runs. A growing train/validation gap or a rising final validation loss relative to its own best value is evidence consistent with late-stage overfitting; it is not proof without corresponding generalization measurements.

Largest final-vs-best validation-loss regressions:

| run_name                                | run_id   |   val_loss_best |   val_loss_final |   loss_regression |
|:----------------------------------------|:---------|----------------:|-----------------:|------------------:|
| ld512_phi512_rho512_lr0.001_bs512_seed3 | f3bwe238 |         0.27205 |          0.28145 |           0.00940 |
| ld256_phi512_rho512_lr0.001_bs512_seed2 | ar2895bk |         0.27639 |          0.28564 |           0.00925 |
| ld128_phi512_rho512_lr0.001_bs512_seed2 | 82ttwef9 |         0.27612 |          0.28484 |           0.00871 |
| ld512_phi512_rho512_lr0.003_bs512_seed3 | lripyoma |         0.26899 |          0.27742 |           0.00843 |
| ld256_phi256_rho512_lr0.003_bs512_seed1 | ia7tiqjp |         0.26820 |          0.27539 |           0.00718 |

## Hyperparameter relationships

- latent_dim: Spearman correlation +0.62 (n=44)
- phi_hidden: Spearman correlation +0.64 (n=44)
- rho_hidden: Spearman correlation +0.65 (n=44)
- learning_rate: Spearman correlation +0.47 (n=44)
- seed: Spearman correlation -0.19 (n=44)
- weight_decay: Spearman correlation -0.09 (n=44)

Interpretation: correlations are observational and can be confounded by architecture, seed, and early termination. Use the per-hyperparameter plots/CSV summaries as directional evidence, not causal estimates.

## Outliers, redundancy, and next experiments

- **Incomplete/outlier candidates:** 8 non-finished runs; their states/log sparsity indicate execution interruption, but the API data does not establish a root cause. No downloaded history contains numeric infinities.
- **Redundancy:** the complete runs do not have an exact duplicate set of tracked configuration values; the closest repetition is the large `512/512/512`, `lr=0.001` family across seeds/weight decay.
- **Next 1:** repeat `latent_dim=512, phi_hidden=512, rho_hidden=512, learning_rate=0.001, batch_size=512, weight_decay=1e-06` with three new seeds and save the best epoch/checkpoint for each seed.
- **Next 2:** compare the strongest compact family (`latent_dim=256, phi_hidden=256, rho_hidden=512`) at `learning_rate=0.001` and `0.003`, `batch_size=512`, and `weight_decay=1e-6`, with at least three seeds each.
- **Next 3:** for the 512/512/512, `lr=0.001` family, use checkpointing/early stopping at the peak validation objective; several top runs end with validation-loss regression after their best loss.

## Reproducibility

Run `python analysis/wandb_report.py` with W&B credentials. The script reloads all project runs through `wandb.Api()`, handles missing/partial histories, and regenerates this report, CSVs, and plots.
