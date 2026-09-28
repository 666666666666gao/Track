# M99 fixed-state visual control

Train-only fixed-state diagnosis: 130 fitting sequences and 22 reused development sequences.
Empty input; IoU localization supervision; no semantic/physical-identity labels, recursive action or official evaluation.
M91/M96 comparisons are descriptive and differ in representation/origin/capacity; they are not a causal matched ablation.

Native reference: 268/495 correct, mean IoU 0.524770723.

| Model | Correct / 495 | Mean IoU | Delta correct | Delta IoU (pp) | Rescues | Breaks |
|---|---:|---:|---:|---:|---:|---:|
| M91 first-use mean pool | 270 | 0.528512902 | +2 | +0.374218 | 3 | 1 |
| M96 t0-search mean pool | 271 | 0.527995017 | +3 | +0.322429 | 4 | 1 |
| M99 token/context Empty control | 275 | 0.530180954 | +7 | +0.541023 | 10 | 3 |

| Predeclared capacity check | Result |
|---|---|
| correct_exceeds_native | PASS |
| mean_iou_exceeds_native | PASS |
| healthy_breaks_zero | FAIL |
| transition_correct_at_least_native | PASS |

All events and all 22 sequences are retained in the accompanying CSVs.
The result SHA-256 was recorded and the final weight SHA-256 was verified against the result. This report does not independently validate input tensors or rerun the model.
These checks alone do not establish semantic benefits, recursive improvements, C effectiveness or official acceptance.
