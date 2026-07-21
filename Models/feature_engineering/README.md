# Event-relative feature ablation

`run_feature_sweep.py` compares input-feature sets for the pure per-cluster
`p_main` MSE Deep Sets regressor. It keeps the approved preprocessing fixed:

- remove events with a cluster in the top 13 cm;
- clip only individual `p_main` labels to `[0, 1]` (do not reject events whose
  `sum(p_main) > 1`);
- split by event: 70% train, 15% validation, 15% held-out test;
- fit scaling on training events only.

The held-out test set is never evaluated by this runner. Select configurations
using validation MSE, lock the winning configuration, then evaluate test R²
once in a separate final run.

| Set | Added inputs |
| --- | --- |
| `baseline` | none |
| `event_multiplicity` | log cluster count |
| `electron_fraction` | fractions of event electron sum and max |
| `electron_rank` | within-event electron percentile |
| `drift_offset` | drift offset from event mean and drift percentile |
| `relative_geometry` | x/y offset and distance from event centroid |
| `electron_plus_multiplicity` | electron fraction/rank plus event count |
| `core_event_relative` | count, electron, and drift features |
| `all_event_relative` | all features, including relative geometry |

All added quantities derive only from input clusters in the same event, not
from `p_main`; they are inference-time-safe and do not leak the target.

## Run on the GPU server

Screen each feature set with seed 42:

```bash
for set in $(python Models/feature_engineering/run_feature_sweep.py --list-feature-sets); do
  python Models/feature_engineering/run_feature_sweep.py --feature-set "$set" --seed 42
done
```

Rank by `best_val_p_main_mse`. Then repeat only the strongest baseline, the
best isolated family, the best targeted combination, and `all_event_relative`
with seeds 7, 42, and 31415. W&B runs are logged to
`xenon-graph-pooling-pmain-feature-sweep`; local validation-only outputs go to
`Models/feature_engineering/feature_sweep_output/`.

## Predicted vs. true plot

After the baseline and comparison checkpoints are present locally, create a
matched two-panel plot on the common validation events:

```bash
python Models/feature_engineering/plot_feature_sweep_predictions.py
```

This is intentionally labelled as a validation plot; it does not consume the
held-out test set. The default comparison is `all_event_relative`.
It also produces a fractional-target-only scatter, binned calibration plot,
and endpoint-versus-fractional error table/plot.

To compare the selected comparison model on the fitted training events versus
the validation events, without touching test events, add:

```bash
python Models/feature_engineering/plot_feature_sweep_predictions.py --train-vs-validation
```
