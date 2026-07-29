# Tree benchmark and conditional-label-variance analysis

**Run date:** 2026-07-23  
**Data policy:** validation-only; held-out test events were not evaluated  
**Tree model:** Extra Trees, trained on 500,000 clusters from 149,930 complete
training events and evaluated on 150,000 clusters from 44,819 complete
validation events

## Main conclusion

The tree benchmark does **not** find signal that the Deep Sets model failed to
use. The all-event-relative tree reaches validation MSE 0.12173 and R² 0.5029,
whereas the corresponding neural feature-ablation run reached best validation
MSE 0.06073. The tree's MSE is about twice the neural model's.

This leaves the final architecture decision unchanged: use the compact
all-event-relative Deep Sets model with gated-sum pooling. A conventional
tabular ensemble is not a better final model.

The neighbor analysis does show substantial label disagreement in ambiguous
regions of the available feature space. That is evidence consistent with
missing explanatory inputs or stochastic labels, especially for fractional
targets, but it is not a numerical irreducible-error bound.

## Tree benchmark

| Features | Validation MSE | MAE | R² |
| --- | ---: | ---: | ---: |
| Five-feature baseline | 0.18771 | 0.37824 | 0.2335 |
| All event-relative | 0.12173 | 0.25577 | 0.5029 |
| Neural all-event-relative reference | **0.06073** | — | approximately 0.75 |

Event-relative features reduce the tree's MSE by 35.1%, independently
confirming that the feature-engineering gain is real and not peculiar to the
neural architecture. The absolute tree-versus-neural comparison is not
perfectly controlled: the tree used a 500,000-cluster training sample and
depth/leaf limits, while the neural model used the full training population.
The gap is nevertheless too large to suggest that this tree has uncovered a
better modeling route.

For fractional targets alone:

| Features | Fractional MSE | MAE | R² |
| --- | ---: | ---: | ---: |
| Five-feature baseline | 0.15640 | 0.34611 | −0.405 |
| All event-relative | 0.11423 | 0.27559 | −0.026 |

The engineered features remove most of the strongly negative fractional R²,
but the resulting R² remains approximately zero. The tree is essentially no
better than a mean predictor within the fractional subset.

## Feature importance

Permutation importance on 50,000 validation clusters ranks:

| Rank | Feature | Validation MSE increase when permuted |
| ---: | --- | ---: |
| 1 | Electron fraction of event sum | 0.05267 |
| 2 | Electron fraction of event maximum | 0.03633 |
| 3 | Drift time minus event mean | 0.03606 |
| 4 | Drift-time rank within event | 0.01752 |
| 5 | Event cluster count | 0.00738 |

The decisive information is relative to the other clusters in the event, not
the raw cluster measurements alone. In the five-feature tree, raw electron
count is dominant; after adding event context, its importance is much smaller
because electron fractions carry more directly useful information.

This supports keeping both electron-relative and drift-relative inputs. The
lower-ranked geometry features still have positive importance, but their
individual contributions are much smaller.

## Conditional label variance

Using 200,000 sampled training clusters as the neighbor reference:

| Features | Neighbors | Mean neighbor-label variance | Neighbor-mean MSE | Tree MSE on same queries |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 10 | 0.16851 | 0.20525 | 0.18807 |
| All event-relative | 10 | **0.12595** | **0.14530** | **0.12201** |
| Baseline | 100 | 0.18843 | 0.19004 | 0.18807 |
| All event-relative | 100 | **0.14525** | **0.13733** | **0.12201** |

The event-relative features reduce 10-neighbor label variance by 25.3% and the
neighbor-mean MSE by 29.2%. Local label variance also correlates with tree
squared error (0.388 for 10-neighbor all-event-relative neighborhoods), so
ambiguous neighborhoods are genuinely associated with prediction failures.

For the closest 10% of all-event-relative 10-neighbor queries, mean local label
variance is only 0.0238 and tree MSE is 0.0173. In the middle-distance
neighborhoods, label variance rises to roughly 0.15–0.16 and tree MSE to
roughly 0.15–0.17. The farthest neighborhoods become easier again because
they are dominated by endpoint labels rather than fractional cases.

## Why this is not an irreducible-noise estimate

- The nearest neighbors are computed in a standardized 14-dimensional raw
  feature space. Equal Euclidean weighting is not necessarily the
  physics-relevant similarity metric.
- The neighbor reference contains only 200,000 sampled training clusters.
- Mean 10-neighbor distance is 0.915 in the 14-dimensional standardized space;
  many points are not genuinely near duplicates.
- The neighbor predictor's MSE (0.1453) is worse than the tree (0.1220), and
  both are much worse than the neural model (0.0607). Consequently, 0.126 is
  not a defensible lower bound on achievable MSE.
- Endpoint imbalance shapes both distance and variance: 27.8% of query
  neighborhoods have zero 10-neighbor label variance, while ambiguous
  mixed-label regions approach the Bernoulli maximum variance of 0.25.

The correct statement is therefore:

> Available observables contain strong predictive information, especially in
> event-relative electron and drift quantities, but there are substantial
> regions where similar measured inputs have conflicting labels. This is
> consistent with missing information or label stochasticity, not proof of a
> fixed MSE floor.

## Final decision

Keep the selected gated-sum, all-event-relative Deep Sets architecture. No
additional ordinary tree or width/pooling sweep is justified by these results.
If one more scientific diagnostic is desired, compute neighborhood variance
in the trained neural encoder's latent space or audit which additional
simulation truth variables distinguish the mixed-label neighborhoods.

Evidence files are in `analysis/tree_diagnostics_output/`.
