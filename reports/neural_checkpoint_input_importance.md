# Original-input importance in the selected neural checkpoint

**Checkpoint:** gated-sum, all-event-relative, seed 42, epoch 137  
**Evaluation:** 25,002 clusters from 7,445 sampled validation events  
**Replicates:** three independent grouped permutations per source variable  
**Test policy:** held-out test events were not evaluated

The reconstructed unperturbed validation MSE is **0.060109**, consistent with
the checkpoint's recorded full-validation best MSE of **0.060137**.

## All validation clusters

| Rank | Original source | MSE after permutation | MSE increase | Repeat SD |
| ---: | --- | ---: | ---: | ---: |
| 1 | `n_electrons_interface` | 0.22385 | **0.16374** | 0.00207 |
| 2 | `drift_time_mean` | 0.20568 | **0.14557** | 0.00128 |
| 3 | `x` | 0.12217 | **0.06206** | 0.00081 |
| 4 | `y` | 0.11174 | **0.05163** | 0.00133 |
| 5 | `drift_time_spread` | 0.07383 | **0.01372** | 0.00104 |

The model depends most strongly on electron count and mean drift time,
including all event-relative quantities derived from them. Both transverse
coordinates carry clear additional information. Drift-time spread is useful
but much less important for the aggregate endpoint-dominated objective.

## Fractional targets

The 25,002-cluster sample contains 645 fractional targets. Its unperturbed
fractional MSE is 0.13163.

| Rank | Original source | Fractional MSE after permutation | MSE increase | Repeat SD |
| ---: | --- | ---: | ---: | ---: |
| 1 | `n_electrons_interface` | 0.29499 | **0.16336** | 0.01056 |
| 2 | `drift_time_mean` | 0.17497 | **0.04334** | 0.00628 |
| 3 | `drift_time_spread` | 0.16194 | **0.03031** | 0.00390 |
| 4 | `x` | 0.15079 | **0.01916** | 0.01150 |
| 5 | `y` | 0.13825 | **0.00662** | 0.00562 |

Electron information remains dominant for fractional clusters. Unlike the
overall ranking, drift-time spread becomes the third-most important source,
while transverse position contributes less. This suggests that the
within-cluster drift-time distribution is disproportionately relevant to
fractional behavior even though it has modest importance over the full label
distribution.

## Interpretation boundary

These are grouped source importances, not additive causal effects. Permuting a
raw observable also recomputes its descendants:

- electron count → event-sum fraction, event-max fraction, and electron rank;
- drift-time mean → event-relative offset and drift rank;
- x/y → corresponding centroid offsets and centroid distance;
- drift-time spread → no engineered descendants.

Groups overlap through event context and, for x/y, centroid distance. The
numbers therefore answer which original measured information the trained
checkpoint relies on, not how much credit should be assigned independently to
every one of the 14 model columns.

Evidence files are under `analysis/neural_permutation_output/`.
