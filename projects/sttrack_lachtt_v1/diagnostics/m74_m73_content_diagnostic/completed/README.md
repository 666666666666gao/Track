# M74 completed fixed-head content diagnostic

All three conditions use the same fixed M73 seed2027 Category final. Category reuses sealed M73 predictions after exact prefix parity; Empty and Swapped each completed 33108 new calls. No training, new seed or new caption.

| Content | Mean IoU | Macro IoU | Low frames | H10 |
|---|---:|---:|---:|---:|
| category | 0.7236888033 | 0.7389346409 | 5072 | 71 |
| empty | 0.7188236800 | 0.7163547064 | 5235 | 64 |
| swapped | 0.6990786779 | 0.7110734008 | 5952 | 81 |

The audit independently recomputes continuous-box IoU and H10 from hash-bound full traces. Its source is in ../audit_preparation/. This is an artifact/scalar check, not an independent learned-model review.

The original M73 gate remains failed. This diagnostic has no low22/full promotion permission. The repeated Train development22 and automatic category labels cannot establish unseen generalization or same-class instance understanding.
