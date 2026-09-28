# M99 Empty-layout replay predeployment review

**FAIL — revise diagnostic attribution before GPU deployment.** This is a source-review verdict, not a conclusion about the cause of the M99 failure.

- reviewer: fresh Codex gpt-6-astra / max
- review_independence: same-family
- acceptance_status: provisional
- reviewed_at: 2026-09-28 10:44:15 CST
- reviewed scope: `probe_empty_layout.py`, the archived and current producer/prototype, M99 plan, M98 completion audit, and M99 failed-run/sanity receipts
- execution scope: local source/receipt reading and CPU AST parsing only; no SSH, GPU work, tensor-cache loading, private mapping, training, checkpoint, or source edits

## Actual failure and established limits

The preserved `m99_failed_initial/final_train.log` ends at `train_ab_visual_control.py:193`, the exact `torch.equal(selection_logits, visual_selection_logits)` assertion. Both `final_train.exit` and `queue.exit` are 1. The archived queue records the failed final run separately from a successful batch-64/two-update GPU sanity. The log has no completed-epoch row; it does not identify a failed batch or optimizer-step count. The archive receipt records no final checkpoint or M99 performance result.

The saved M98 audit establishes 152 sequences / 3502 events / 219194 prefix frames and zero measured box/score errors. GPU sanity establishes only its two updates on the initial 64 examples. Neither establishes that the shuffled first training epoch is numerically finite or exactly Empty-equivalent.

Current `train_ab_visual_control.py` and `instance_ab_prototype.py` are byte-identical to the preserved failed-run source copies. The producer expands one Empty vector to five slots with an all-active mask. The prototype's null branch multiplies its expanded Empty by that mask before the text projection. A layout-dependent FP32 difference is therefore a reasonable hypothesis to measure. It is not yet a measured explanation.

## Blocking findings

1. **The probe can label a different failure as the observed assertion reproduction.** At `probe_empty_layout.py:41-59`, any nonfinite semantic, phrase, selection, visual-selection, or quality output enters the same `failure` field as unequal selection scores. The record does not retain the equality boolean. Line 74 accepts any such failure as “Original failure reproduced.” For example, nonfinite quality with equal selection scores takes this branch, while the original producer would pass line 193 and instead fail its subsequent finite-loss assertion. Distinguish the exact score-equality failure from nonfinite diagnostic findings and require the original arm to record the former before declaring reproduction. Preserve the affected fit keys/candidate indices and name the finite flags per output.

2. **The requested nonfinite-update alternative is not recorded.** At lines 60-62, nonfinite loss raises before JSON output; no gradient or post-update parameter finite evidence is collected. If an earlier backward/update introduces nonfinite values, a later aggregate forward flag cannot locate or distinguish that event. This matters to this probe because the cause of the actual equality failure is unknown, and the task explicitly includes nonfinite forward/update as an alternative. Record finite loss, named nonfinite gradients/parameters around the temporary updates, and the relevant batch/key context. Write the diagnostic result for these controlled stop conditions instead of using an unrecorded finite-loss assertion. Do not count such a different stop as the original score-equality assertion unless that assertion is actually false.

These are diagnostic correctness defects; they are not evidence that M99 actually encountered nonfinite values.

## Correct scope and replay properties

- Each arm resets Python, NumPy, CPU Torch and CUDA seeds to 2027, creates a fresh identical model, applies the same freeze helper and AdamW learning rate 3e-4, and uses the same seed-2027 first-epoch permutation.
- The reused loader enforces the existing M98 terminal/audit/input checks and 2544 valid fit events. Splitting that fit order by 64 gives exactly 40 batches, with 48 examples in the last batch, so the loop is bounded by 40 temporary updates per arm.
- The only intended arm difference is `data['text'].contiguous()`; the encoded values, mask, model architecture, loss and fit ordering are unchanged.
- The objective matches the producer's visual-selection BCE plus visual-quality BCE against Train-GT candidate IoU. No semantic or identity label is added.
- The probe contains no development/public inference, evaluation function call or checkpoint save. The shared full loader does read and assemble development inputs/labels before returning the selected fit panel; this is not development evaluation.
- The two arm records are written to a separate caller-supplied output and the script refuses to overwrite an existing output. It does not alter the original failed-run archive or call the original queue.

## Replay-fidelity caveat and verification

The probe omits the producer's initial no-grad/native-output check at lines 170-174. The inspected model has no explicit dropout or running-statistics updates, and all attention modules use their default zero dropout, so no source evidence establishes a changed model/RNG state from this omission. Still, including the same initial forward/check would make the replay follow the actual execution sequence more closely. Do not describe the omission as a proven root cause.

All three Python files passed AST parsing. The available direct local interpreter has no Torch installed; attempted Torch import failed with `ModuleNotFoundError`. No model, actual input, numerical-layout, loss, gradient or GPU-success claim is made by this review.

After correcting the two blocking records, re-review the probe, then execute the bounded diagnostic separately from the original failed run. A clean 40-update contiguous arm would support a narrow first-epoch replay result only; the final fixed-budget training and its evaluation remain unexecuted until separately performed.
