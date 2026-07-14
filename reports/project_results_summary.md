# XENONnT S2 tagging — project results summary

**Status:** architecture exploration is complete enough to select a regression-focused next experiment. The current results establish a useful event-context model, but they are not yet the final `p_main` regression result requested for integration.

## Objective

Each neutron-scatter event contains a variable-size set of spatial clusters. For every cluster, the model predicts its continuous contributions `p_main` and `p_alt` to the main and alternate S2 peaks. The central modelling question is whether set-level context improves on cluster-only predictions.

The advisor's current integration target is specifically **per-cluster `p_main` regression**. The final headline evaluation should therefore be test-set `p_main` R² plus a predicted-versus-true `p_main` plot. Event-level tie-aware main-cluster accuracy is still informative, but is a secondary diagnostic rather than the primary regression result.

## Data, event structure, and preprocessing

The initial simulation contains 10,259,121 clusters from 2,704,607 events (mean 3.79 clusters/event; median 2; maximum 118). Event sizes are much smaller than the worst-case Deep Sets representation bound: 95% have at most 11 clusters and 99% at most 21.

Features are transverse position (`x`, `y`), surviving electron count, drift-time mean, and drift-time spread. The target labels are `p_main` and `p_alt`. Splits are made by `event_number`, which keeps all clusters from an event in the same train, validation, or test split. Feature standardization is fit only on the training events before being applied to validation/test data.

### Fiducial cut and label quality

The raw sample had 2,073 `p_alt > 1` labels (about 0.02% of clusters); the journal's drift-time study found these at short drift time, consistent with a simulation artifact near the top of the TPC. Removing every event with a cluster at drift time below 192,600 ns (the top 13 cm fiducial cut) removed about 659,756 of 2,704,607 events, or **24.4%**, and eliminated the remaining `p_alt > 1` labels. `p_main > 1` was reported as zero in the raw inspection.

The resulting fiducial sample has 6,838,889 rows. This cut should be kept fixed and explicitly reported for comparable experiments.

## Baselines and early evidence for event context

| Model / setting | Result | Interpretation |
|---|---:|---|
| Electron-count argmax, pre-fiducial sample | Main accuracy 0.7072; alt accuracy 0.5402 | Electron count is a strong but incomplete heuristic. |
| Electron-count argmax, fiducial sample | Main accuracy 0.7251; alt accuracy 0.5531 | The cut improves the baseline modestly. |
| Per-cluster MLP, fiducial sample | Test MSE 0.1754; `p_main` ROC-AUC 0.7720; event-main accuracy 0.7209 | A cluster-only model approximately matches the argmax main-selection baseline; it cannot directly compare clusters within an event. |

These values are journal-recorded baseline results, not a common W&B benchmark table. They are useful for context but should not be compared numerically to later runs with different losses, data versions, or selection metrics without verifying the split and target definitions.

## W&B experiment analysis

The primary legacy W&B project is `senadeep5-clemson-university/xenon-deepset`. Its sweep explicitly optimized `val_event_main_accuracy`, so the previous W&B ranking used that metric. The live API download contained 57 runs: 44 with a usable objective, 49 finished, 5 killed, 2 failed, and 1 crashed.

The best recorded legacy run was `zzeg2dq9` (`ld512_phi512_rho512_lr0.001_bs512_seed3`) with best validation event-main accuracy **0.73821**. The following configurations were close:

| Rank | Run ID | Best validation event-main accuracy | Key configuration |
|---:|---|---:|---|
| 1 | `zzeg2dq9` | 0.73821 | latent/phi/rho = 512/512/512; LR 0.001; batch 512; seed 3 |
| 2 | `7xganm8m` | 0.73796 | 128/512/512; LR 0.001; batch 512; seed 1 |
| 3 | `f3bwe238` | 0.73772 | 512/512/512; LR 0.001; batch 512; seed 3 |

Across the scored legacy runs, larger latent and hidden widths were positively associated with the objective (Spearman correlations: latent +0.62, phi +0.64, rho +0.65). Learning rate 0.001 had the strongest average result, with 0.003 competitive. These are observational sweep patterns, not causal effects.

Several high-ranked runs ended with validation loss above their own minimum (largest observed increase: 0.00940), so checkpointing at the best validation metric is important. The existing W&B analysis also found no numeric infinities in downloaded histories; the eight incomplete runs do not by themselves establish optimization divergence.

## Interpretation and current decision

The legacy Deep Sets sweep gives evidence that wider set-context models improve the **event-level identification** objective over the electron-count and cluster-only baselines. It does **not** yet establish the advisor's required quantity: calibrated per-cluster `p_main` regression measured by test R².

There is already an MSE-loss graph variant in `Models/deepset_variations/deepset_graph_mse.py`. It preserves event-level splitting, but it currently selects checkpoints and steps `ReduceLROnPlateau` with tie-aware event-main accuracy. That selection rule is inconsistent with the new regression objective, even though its training loss is MSE.

## Recommended next experiment

Use the best architecture family as the starting point, while making the experiment definition match the integration target:

1. Predict `p_main` per cluster; use MSE on `p_main` as the loss. If retaining `p_alt` as an auxiliary target, log its contribution separately and keep `p_main` as the model-selection metric.
2. Configure `ReduceLROnPlateau(mode="min")`, early stopping, and checkpointing on validation `p_main` MSE—not tie-aware accuracy.
3. Run at least three seeds of a 512/512/512, LR 1e-3, batch 512 configuration; compare with the lower-cost 256/256/512 family at LR 1e-3 and 3e-3.
4. Report only the held-out test set once: `p_main` R², test MSE, and a predicted-vs-true `p_main` scatter plot with the identity line. Keep tie-aware accuracy as a secondary event-level diagnostic.

For clarity, an individual residual is `true p_main - predicted p_main`. R² summarizes the squared residuals relative to predicting the test-set mean: R² = 1 - SSE/SST. R² = 1 is perfect, 0 is mean-prediction performance, and negative values are worse than predicting the mean.

## Reproducibility and evidence

The W&B report, all-run table, top-run table, and learning curves are available in `analysis/wandb_output/`. The W&B analysis script is `analysis/wandb_report.py`; it reloads run summaries and histories with `wandb.Api()`.

### Evidence sources

- `journal.txt` — dataset inspection, fiducial-cut study, and baseline results.
- `analysis/wandb_output/wandb_summary.md` and `all_runs.csv` — W&B results downloaded on 2026-07-13.
- `Models/deepset_train.py`, `Models/deepset_sweep.py`, and `Models/deepset_variations/deepset_graph_mse.py` — current implementation and metric configuration.
