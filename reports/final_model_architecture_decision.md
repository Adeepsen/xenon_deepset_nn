# Final `p_main` model architecture decision

**Decision date:** 2026-07-23  
**Primary selection metric:** minimum per-cluster validation `p_main` MSE  
**Recommended final model:** all-event-relative inputs + compact Deep Sets encoder/head + independent sigmoid-gated sum pooling + MSE

## Decision

Freeze the final architecture as:

- 14 input features: the five base cluster features plus event multiplicity,
  electron fraction/rank, drift offset/rank, and relative transverse geometry;
- five-layer, width-256 node encoder with latent dimension 128;
- learned independent sigmoid gate per cluster, followed by gated sum pooling;
- four-layer, width-512 per-cluster head receiving the node embedding and gated
  event embedding;
- sigmoid output trained with per-cluster `p_main` MSE;
- Adam, learning rate `3e-4`, weight decay `1.0768e-5`, dropout `0.04849`,
  batch size 1024, gradient clipping at 1.0;
- checkpoint, scheduler, and early stopping all driven by minimum validation
  `p_main` MSE.

This is W&B run `y057yugt` (`gated_sum-seed42`). Its best validation MSE is
**0.0601372**, the best result among the corrected-data, validation-only
architecture comparisons.

## Controlled evidence against the base model

The relevant base is W&B run `o2v5hzui`: the same corrected data, event split,
seed, loss, optimizer, widths, depth, regularization, and sum pooling, using
only the five raw cluster features. Its best validation MSE is **0.0652006**.

| Change from the five-feature sum-pooling base | W&B run | Best val MSE | Change vs base | Decision |
| --- | --- | ---: | ---: | --- |
| Add all event-relative features; retain sum pooling | `wew4yp3f` | 0.0607347 | **−6.85%** | Keep |
| Add sum/mean/max pooling on the full features | `l7l6qhxo` | 0.0605121 | −7.19% | Do not keep; only −0.37% beyond full features |
| Add independent sigmoid-gated sum pooling on the full features | `y057yugt` | **0.0601372** | **−7.77%** | **Keep** |
| Add sum/mean/max plus gated pooling | `xb6y407g` | 0.0601964 | −7.68% | Reject: larger and slightly worse than gated sum |
| Widen latent/encoder/head from 128/256/512 to 256/512/1024 | `41gsgt5s` | 0.0621136 | −4.73% | Reject |
| Fraction-enriched sampling, weight 2, with sum/mean/max | `2yu9mdk1` | 0.0614177 | −5.80% | Reject for overall regression |
| Fraction-enriched sampling, weight 4, with sum/mean/max | `nprze393` | 0.0631438 | −3.15% | Reject |
| BCE-with-logits instead of MSE | `fi4abq6z` | final MSE 0.0631756 | −3.11% | Reject for an MSE/R² target |

The parameter counts reinforce the choice:

| Architecture | Parameters |
| --- | ---: |
| Five-feature sum base | 1,217,665 |
| Full-feature sum | 1,219,969 |
| Full-feature gated sum | **1,236,610** |
| Full-feature sum/mean/max | 1,351,041 |
| Full-feature sum/mean/max + gated | 1,433,218 |
| Full-feature wide model | 4,864,769 |

The selected model is only 1.4% larger than the full-feature sum model and
1.6% larger than the five-feature base. The wide model has about four times
the parameters and is worse, so capacity is not the limiting factor.

## What specifically helped

1. **Event-relative inputs are the dominant improvement.** They provide almost
   7% relative MSE reduction without materially
   increasing the network. The isolated drift-relative inputs are the
   strongest single family (best MSE 0.0625560), while electron-only and
   geometry-only additions are modest. Combining all event-relative inputs is
   better than the core set (0.0607347 versus 0.0616475), so retain all 14.
2. **Learned gated sum is a small, plausible incremental improvement.** It
   allows several clusters to contribute independently to event context,
   which matches valid multi-scatter events better than softmax attention's
   single-winner normalization. It improves best MSE by 0.98% relative to the
   full-feature sum model.
3. **More pooled statistics do not justify their complexity.** Sum/mean/max
   helps by only 0.37% beyond the full-feature sum model and remains worse than
   the smaller gated-sum model. Adding all four contexts to the gated model is
   also slightly worse.
4. **More width hurts generalization.** Both wide variants peak early and have
   worse best validation MSE. The improvement is therefore architectural and
   representational, not a simple parameter-count effect.
5. **MSE matches the integration target.** BCE and fraction-enriched sampling
   trade away overall MSE/R². They may alter fractional-target behavior, but
   they do not improve the declared final metric.

## Evidence limitations

- The clean feature and pooling ablations currently have only seed 42. The
  0.0607347 → 0.0601372 pooling gain is small and is not yet a demonstrated
  multi-seed effect. The much larger feature gain is more decision-relevant.
- W&B summaries record the last epoch's R² but the minimum-over-training MSE.
  Rankings therefore use the checkpoint metric (`best_val_p_main_mse`), not
  the final logged R².
- Older projects optimize classification/tie-aware event accuracy, use
  two-target losses, or use different selection rules. They support the value
  of event context but are not pooled numerically with the corrected pure
  `p_main` regression ablations.
- Run `35k7jaap` reports test R² 0.8689 only after deleting 784,766 valid
  fiducial multi-scatter events whose event-level `sum(p_main)>1`. That removes
  38.4% of the fiducial events and reduces the sample from 6.84M to 2.70M
  clusters. It is a population change, not an architecture improvement, and
  must not be used as the headline result.
- The corrected five-feature model (`5hcxgulv`) reports held-out test MSE
  0.066012 and R² 0.73062. This is the valid existing test reference, not the
  final gated model's result.

## Final execution rule

Train the frozen gated-sum/all-event-relative configuration. Restore the
minimum-validation-MSE checkpoint and evaluate the held-out test split once
for test MSE, MAE, R², predicted-versus-true, calibration, and fractional-bin
diagnostics. Do not use the test result to choose another architecture.

If there is time for only one risk-reduction action before that final training,
repeat **only** full-feature sum and full-feature gated sum with two additional
seeds. If gated sum does not win on mean validation MSE, fall back to the
full-feature sum model; the event-relative feature decision remains strongly
supported either way.

## Reproducibility

The live W&B snapshots used here are under `analysis/wandb_output/`, separated
by project. `analysis/wandb_report.py` regenerates them with `wandb.Api()`.
The selected architecture is implemented in
`Models/feature_engineering/deepset_pmain_attention_pooling.py`; the feature
definitions are in `Models/feature_engineering/feature_sets.py`.
