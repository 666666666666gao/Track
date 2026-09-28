# M104 local visual geometry source review - corrected source

- verdict: PASS (source gate after one concrete arithmetic fix; no remaining concrete blocker found)
- reviewer_model: gpt-6-astra
- reviewer_reasoning_effort: max
- review_context: fresh / fork_turns none
- review_independence: same-family
- acceptance_status: provisional
- reviewed_at: 2026-09-28T13:01:36+08:00
- scope: source inspection, historical JSON checks, AST-extracted summary execution, standard-library binary32/synthetic arithmetic; real Torch zero/gradient/cache acceptance pending.

This satisfies the experiment-bridge Phase 2.5 source review for proceeding to the real two-update sanity. It does not certify a GPU run or authorize skipping sanity before the fixed-budget training and independent parent readout. Initial FAIL is retained in `M104_GEOMETRY_REVIEW_20260928_130001.md`.

## Reviewed artifacts and factual basis

Read `M104_VISUAL_GEOMETRY_PLAN.md`, `train_local_visual_geometry.py`, the relevant current `EXPERIMENT_PLAN.md` section, `train_ab_visual_control.py`, `instance_ab_prototype.py`, `collect_train_states.py`, `analyze_train_states.py`, `collect_ab_interface_panel.py`, `inspect_ab_candidate_scores.py`, `diagnose_b_geometry.py`, the overlay candidate-set observation helper, and native `diagnostics/m84_centered/started/code/lib/utils/box_ops.py`. Read actual `m103_completed/result.json`, `m101_completed/train_weight1/result.json`, and all `m100_completed/full/events.jsonl` rows. The ongoing M103 events download was not accessed or modified. No SSH, GPU, install, download, or private human review packet was used.

M103 actual counts agree with the plan: 3039 valid events = 2544 fit + 495 development; eligible fit/development 2033/346; 292 fit center-in full256 misses, with full256 GT-size/GT-center hits 218/78, Top-10 counterparts 100/46, and 205 eligible local misses. These are GT diagnostics, not learned recovery.

M101 actual final is `complete_visual_control`, native preservation weight 1, development correct 272/495, mean IoU 0.5307076979870673, healthy 264/264 and transition 4/127. Its fit correct count is 1602/2544. The review does not substitute M99 or the weight-zero parent.

## Resolved blocker

The initial decode rebuilt the extent via float32 `(x+w)-x`, violating the explicit zero-correction identity on real saved boxes. The first example is `cube04_indoor@10`, candidate 0: width 34.62568283081055 becomes 34.62567138671875; height 36.83968734741211 becomes 36.839691162109375. Initial arithmetic changed 25682/30390 actual saved candidates (fit 21392; development 4290).

Final `decode` lines 40-46 retains `raw_right`/`raw_bottom` and computes the extent by adding the right/bottom clipping displacement and subtracting the left/top clipping displacement from the raw extent. All side clamps and the minimum-10 floor remain. In real arithmetic, `raw_w + (right-raw_right) - (left-raw_x) = right-left`; the derivative is also `dright-dleft`. There is no zero-delta branch, detached output, relaxed equality, new fallback, or unrelated refactor. The actual zero equality assertion remains at lines 151-154.

The corrected binary32 arithmetic preserved all 30390 historical boxes exactly. Image dimensions for this local arithmetic check were reconstructed from saved normalized geometry and box-size ratios, rounded to integers, and agreed within every sequence: 640 x 320 or 640 x 360. No original feature tensor was loaded locally. Seven interior, clipped-edge, fully-outside and minimum-size scalar cases matched native clip output; finite-difference derivative discrepancy was at most 2.14e-11 at smooth points. These tests support the arithmetic fix, but do not replace the planned Torch/autograd check.

## Method and implementation checks

- Shapes and masking are consistent. The frozen evidence encoder returns current `[B,10,32,64]` and context `[B,10,24,64]` tokens, with matching validity masks, using current/context roles 2/3. Concatenation preserves their order as 56 tokens. The new 64-to-16 projection is masked after its affine transform, so invalid projected biases are also removed. Flattening gives 896 features, and the same native 13 geometry features yield the 909-input MLP. Selecting one candidate produces `[B,56,64]` / `[B,56]` / `[B,13]`; evaluating all candidates yields `[B,10,4]` deltas. Native geometry and candidate tokens are from the same original candidate state.
- Parameter budget is exactly 59540: projection 1040, 909-to-64 layer 58240, final 64-to-4 layer 260. The final layer is initialized to zero. The optimizer receives only this separate model. The complete loaded M101 parent is in eval mode, requires no gradients, and every state-dict tensor/buffer is compared with a pre-optimization clone. Parent outputs are cached under no-grad and are never rescored or changed by geometry training; the saved selection indices are used for every final readout.
- Decode predicts native-width/height-normalized center movement and log size ratios. `expm1` gives exact zero extent increments; the shift subtracts half the extent increment. The normalized target is the inverse before clipping, verified on a synthetic box/target pair. The training decode remains `[B,1,4]` then `[B,4]`. Both decoded and GT xywh boxes are converted to xyxy before the native aligned `[N,4]` GIoU utility; its first returned value is the scalar mean loss. `SmoothL1(delta, normalized_target) + 2 * giou_loss` is the specified objective.
- Training selects one highest-original-IoU candidate for each of the 2033 eligible fit events. Eligibility is the M103 full-GT-inside AND Top-10-IoU-at-least-0.1 label. Only fit tensors enter the optimizer loop. The forward/decode/evaluate geometry path never reads eligibility, target or IoU to decide whether to refine a candidate; every one of the ten candidates is refined for every event. GT enters assignment/loss and reported comparison metrics only. Empty input is passed to the fixed parent and exact visual/selection-logit equality is asserted. No semantic identity target, online language model, C, recursion, or public dataset evaluation appears.
- Budget and selection policy match the plan: seed 2027, AdamW 3e-4, batches of 64, fixed 12 epochs, `ceil(2033/64) * 12 = 384` updates, and final checkpoint only. Sanity uses the same 64 shuffled eligible events for exactly two steps and checks final-layer gradients on both steps plus projection gradients on the second. Development has pre-training frozen-parent/zero-geometry parity checks, but no post-update development evaluation or checkpoint in sanity.
- Evaluation uses original GT through `overlaps`, not a parent box as the target. Both fit and development preserve every event's original parent selection, before/refined selected IoU, original/refined Top-10 oracle IoU, native IoU, and overlapping strata. Parent-relative rescues/breaks are separate from the inherited native-relative counts. Acceptance compares development correct count/mean IoU, healthy breaks, and transition count against the frozen parent. Failures remain recorded instead of becoming checkpoint selection. A newly loaded geometry head must reproduce every saved row and complete summary exactly. The head has no dropout or train/eval-dependent normalization, so the lack of a separate `eval()` call on the reloaded head does not change behavior here.

## Executed local checks

All checks used `E:/python.exe` with standard-library modules only; Torch is unavailable locally and was not installed.

1. Syntax compilation for the four requested implementation/reuse files and the final patched M104 source passed without importing their ML dependencies.
2. Recomputed the existing M101 development summary from all 495 real rows using the AST-extracted current `summarize` function; every field and overlapping stratum matched exactly.
3. Executed M104's AST-extracted summary augmentation on those 495 rows in reference configuration: parent counts/means and original Top-10 counts matched M101, with zero parent rescues/breaks and zero original-miss/refined-hit events.
4. Synthetic rows distinguished native-relative versus parent-relative rescues/breaks and retained overlapping strata. Expected parent rescues/breaks 2/1 versus native 1/0 passed; a Top-10 original-miss/refined-hit was counted correctly.
5. Parameter/update counts, normalized-target inversion, 30390 native-box zero arithmetic, and seven clipping/finite-difference cases passed as described above.

## Remaining acceptance owned by the executor

No further source change is requested. The actual two-update Torch sanity must still pass on the real cached inputs: exact zero delta and all3039 native geometry equality, original495 parent selection parity, finite losses/gradients, nonzero final-layer and second-step projection gradients, and unchanged parent state. The parent scores/indices remain fixed by construction; GPU1's independent complete parent readout should be checked against the trained arm's parent fields for all3039 events. Full training then needs actual 384-update/59540-parameter receipts, all3039 output rows, final reload equality, and the four predeclared capacity results.

The final source can produce both the planned GPU0 training and useful GPU1 parent-reference modes. Launch order/device binding, elapsed runtime, actual feature/mask values, CUDA execution, independent cross-arm equality and empirical capacity remain runtime responsibilities. A passing source review alone establishes none of those results. Any eventual positive result supports Train fixed-state visual geometry capacity only, not semantic benefit, recursive tracking, Full152, or public-test acceptance.
