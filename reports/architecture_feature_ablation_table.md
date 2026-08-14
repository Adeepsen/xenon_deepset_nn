# Architecture and feature ablation table

Primary metric: minimum validation per-cluster `p_main` MSE (lower is better).

| Variant | Input / architecture change | Parameters | Validation MSE | Decision |
| --- | --- | ---: | ---: | --- |
| Raw-feature baseline | Five raw inputs + sum pooling | 1,217,665 | 0.06520 | Reference |
| Event-relative inputs | 14 inputs + sum pooling | 1,219,969 | 0.06073 | Keep features |
| Gated-sum Deep Sets | 14 inputs + independent sigmoid-gated sum | 1,236,610 | 0.06007* | Selected |
| Additional pooled statistics | 14 inputs + sum / mean / max pooling | 1,351,041 | 0.06051 | Not needed |
| Wide model | Encoder / latent / head widened to 512 / 256 / 1024 | 4,864,769 | 0.06211 | Reject |

*The gated-sum value is the mean across paired seeds 42–44. All other entries are seed-42 validation ablations.

## Main takeaway

Event-relative inputs provide the large improvement at nearly unchanged model size. Gated sum provides a smaller but reproducible additional gain. Adding pooled statistics or substantially increasing capacity does not improve validation performance.
