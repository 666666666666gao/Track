# M103 cached B geometry predeployment source review

- verdict: PASS (final corrected source; no concrete blocker remains)
- reviewer_model: gpt-6-astra
- reviewer_reasoning_effort: max
- review_context: fresh / fork_turns none
- review_independence: same-family
- acceptance_status: provisional
- reviewed_at: 2026-09-28 12:43:18 +08:00
- scope: source review, real historical JSON arithmetic, executed standard-library summary/group checks; M103 Torch/cache runtime remains pending.

This is the experiment-bridge Phase 2.5 source gate. It permits the intended CPU diagnostic to proceed; it does not claim M103 runtime acceptance, a learned correction, physical-instance truth, or public benchmark improvement. The initial FAIL is preserved in `M103_GEOMETRY_REVIEW_20260928_123854.md`.

## Reviewed inputs

`M103_LOCAL_GEOMETRY_PLAN.md`, `diagnose_b_geometry.py`, `analyze_train_states.py`, `collect_train_states.py`, `m102_completed/RESULT.md`, `m102_completed/B_REACHABILITY.json`, `m102_completed/B_REACHABILITY_EVENTS.jsonl`, `m102_completed/result.json`, `m101_completed/train_weight1/result.json`, existing collection preparation/coverage receipts, the overlay `_search_origin` function, and the existing scalar `clip_box` implementation.

## Initial blockers resolved

1. Rows now preserve `strata`; the exact grouping source appends to every `split:stratum` group and explicitly declares that groups overlap. Executing these final grouping statements over the real M101 rows retained all 495 unique development events and 501 memberships: healthy 264, transition 127, late_low 41, intermediate 69.
2. Local dense misses now retain seven named joint-profile fields: Top-10 partial overlap, full GT inside, candidate support, 2x context support, GT side below 10, Top-10 GT-center hit, and Top-10 GT-size hit. Each profile count is indexed by the explicitly saved field order. This reports the joint observability/Top-10 intervention conditions alongside the existing Top-10/full256 marginals; every row also preserves the full256 intervention IoUs. The eligibility count remains exactly full GT inside AND Top-10 max IoU >= 0.1.

Only these narrow fixes were needed. No fallback, compatibility mechanism, extra exception path, threshold sweep, or new hash scheme is requested. The existing label SHA and M101 parity assertions remain.

## Geometry and scope findings

- GT-center intervention is `(GT center - predicted size / 2, predicted size)`. GT-size intervention is `(predicted center - GT size / 2, GT size)`. For `[N,4]` boxes, the `[1,2]` GT center broadcasts over N and the GT size expands to `[N,2]`; each concatenation remains `[N,4]`. The intended center/size is preserved before clipping. Boundary clipping can subsequently change center or size, as required by the plan.
- The vector clipping equations exactly match the existing collector and scalar `clip_box(..., margin=10)`: left/top clamp to `[0, dimension-10]`, right/bottom clamp to `[10, dimension]`, and resulting width/height clamp to at least 10. Original cached boxes are evaluated directly and never modified in place. Clipped-GT IoU is evaluated against the original GT, so the minimum-size/boundary diagnostic is not hidden by changing the reference target.
- All IoUs use the actual current target in `training_labels.json`, separately from the inference cache. The source skips only missing current GT, consistent with the planned 3039 valid-event denominator rather than the original 3502 total events.
- Actual M101 status/mode are `complete_visual_control` and `train`; the revised code uses these exact names. It checks all 495 development keys' native and Top-10 IoUs with exact equality, enforces 3039 unique output keys and the 2544/495 split, and writes every valid row to JSONL.
- Crop origin reuses `_search_origin`. The source honestly reports reconstruction from cached float32 prior/resize values, not reproduction of historical crop tensors. A center-out flag is only a range proxy and is not treated as complete target invisibility.
- `local_dense_misses` is precisely center inside AND full256 IoU < 0.5. All success comparisons use inclusive IoU >= 0.5; eligibility uses inclusive overlap >= 0.1. Candidate/context support is the union of the existing Top-10 boxes/their 2x contexts.
- No model forward, optimizer, tracker construction/action, checkpoint writing, recursive policy, public-test access, semantic labeling, or next-refiner launch exists in this diagnostic. The outputs explicitly identify GT counterfactuals as undeployable. Counting eligibility does not enact the future single-target assignment or authorize training.

## Verification actually executed

- AST parsing of the three Python sources passed. The final source was re-parsed after the fix.
- Executed the final `summarize()` function through AST extraction using only the standard library: 8258 synthetic aggregation rows, of which 8256 are local dense misses, exercise all 128 seven-field profiles with unequal counts. Every profile, profile total, named marginal, refinement-eligibility intersection, full256 intervention marginal, and clipped-GT marginal matched an independently constructed expected count. Rows at full256 IoU exactly 0.5 and rows with center outside were excluded from the local subset; Top-10 overlap exactly 0.1 and intervention IoU exactly 0.5 were included. Empty-subset output was also checked. These are aggregation tests, not geometric/Torch execution.
- Executed the exact final grouping statements over actual M101 development rows and verified the overlapping stratum counts above.
- Verified the two output write expressions contain real newline literals, not a literal backslash followed by n.
- Recomputed all 495 historical M101/B_REACHABILITY key and IoU equalities. Native 268, selected 272, Top-10 305, full256 324; 223 errors split into 33 selection errors, 19 Top-10 omissions, 65 center-in dense misses and 106 center-out dense misses. Full GT inside is 361. All saved historical totals agree.
- Existing M102 is terminal and both teacher-screening gates are false. It was not rerun and its outputs were not promoted to identity truth.

## Runtime acceptance still required

The working local interpreter is `E:/python.exe`; it has neither Torch nor NumPy, and local collection artifacts do not contain the 152 cached feature files. No dependency was installed. Therefore tensor broadcasting/clipping equivalence above is a source/algebra finding; no real or mocked Torch geometry execution occurred, and this review does not claim M103 has produced 3039 actual result rows.

On the already planned CPU-only cache run, accept the diagnostic only after successful process completion and verification of 3039 unique JSONL rows, 2544 fit/495 development, exact saved M101 parity, the unchanged historical native/Top-10/full256 coverage and 65/106 development dense-miss counts, all overlapping stratum summaries, and joint-profile/marginal/eligibility consistency. Inspect the source's required actual artifacts rather than substituting these synthetic test counts. Remote import success and real intervention values remain unverified here.

No SSH, GPU, downloads, private-v2 human mapping, source edits, or experiment deployment were performed by this reviewer. No non-blocking speculative findings are added.
