# M99 Empty-layout replay predeployment review

**PASS — bounded diagnostic source-review gate only.** Both previously reported diagnostic blockers are resolved. No further source change is requested.

- reviewer: fresh Codex gpt-6-astra / max
- review_independence: same-family
- acceptance_status: provisional
- reviewed_at: 2026-09-28 10:47:46 CST
- scope: local source/receipt review and CPU AST parsing; no SSH, GPU, actual cache/model execution, private mapping, production source edit, checkpoint, or evaluation
- supersedes: `M99_EMPTY_REPLAY_REVIEW_20260928_104415.md`, retained as the initial review record

## Actual failure remains preserved

The archived final-training traceback stops at `train_ab_visual_control.py:193`, its exact selection-versus-visual-selection equality assertion. Final training and queue exits are 1. The log supplies no failed batch/step count. The archived receipt records no final checkpoint or M99 performance result.

M98's saved audit establishes 152 sequences / 3502 events / 219194 frames. The preceding GPU sanity completed two updates on the initial 64 examples; it does not establish success of the shuffled first epoch. Current producer and prototype remain byte-identical to the archived failed-run source copies.

The producer's full Empty text is expanded, while the prototype's null Empty text is multiplied by its all-active mask before projection. Layout-dependent FP32 behavior is a hypothesis supported by this source distinction, not a measured root cause.

## Resolved findings

1. **Exact observed assertion is now distinguished.** Per-batch records include `score_equal` and named finite flags for semantic delta, phrase delta, both selection logits and quality logits. The failure record explicitly stores `score_equality_assertion_failed`. The final original-arm gate requires that boolean to be true. A quality-only nonfinite event, finite-loss failure, nonfinite gradient or nonfinite update cannot be reported as a successful reproduction of the observed line-193 assertion.

2. **The nonfinite alternative is recorded.** Each attempted batch records nonfinite parameter names before forward, named forward finite flags, loss finiteness when reached, nonfinite gradient names after backward, and nonfinite parameter names after an executed update. Controlled loss/gradient/update stops have distinct reasons. Both arm records are written to JSON before the final reproduction/fix assertions. These diagnostics add no change to finite forward values, loss, gradient or update mathematics. Unreached checks stay null rather than claiming success.

3. **Initial execution sequence now matches.** Both arms execute the producer's initial no-grad forward on the initial 64 fit examples and check exact native-score equality and candidate-zero selection before the shuffled epoch.

## Replay and scope checks

Each arm resets Python/NumPy/Torch/CUDA seeds to 2027, constructs a fresh identical prototype, applies the same freezing helper, and uses AdamW at 3e-4 with the same seed-2027 first-epoch permutation. The sole arm intervention is making the full encoded Empty text contiguous for the replay batches. All five Empty slots, mask values, model architecture, Train-GT IoU targets and visual-selection-plus-quality BCE remain unchanged.

The existing loader requires the successful M98 terminal/audit and validates 2544 fit events. Batch64 yields 40 batches, with 48 examples in the last batch, so each arm has at most 40 temporary updates. The original assertion stop records affected fit keys, candidate indices, score pairs, both text strides and delta maxima. Per-output finite flags must be consulted when interpreting a nonfinite record; delta magnitude alone cannot establish finiteness.

The contiguous arm passes the terminal diagnostic gate only when it has no recorded failure and completes all 40 updates. The full loader also reads and assembles development inputs/labels, but the replay uses only its fit panel and calls no development/public forward or evaluation. It saves no checkpoint, runs no tracker/C action and neither restarts nor overwrites the original failed run.

All three Python files passed AST parsing. Direct local Torch import was unavailable in the inspected interpreter, so no local model, tensor-layout, numerical, loss or gradient execution is claimed.

**Remaining evidence:** run this separately saved diagnostic. The original exact assertion must actually reproduce, with its finite flags and affected keys inspected. A clean contiguous arm would establish only the measured first-epoch replay result under that single layout intervention. It would not itself establish completed fixed-budget training, development gains, semantic validity, recursion or official performance.

