# M105 geometry component source recheck

- verdict: **PASS — initial blocker B1 resolved; no remaining blocking source issue found**
- review_independence: **same-family**
- acceptance_status: **provisional**
- review_type: source recheck and local deterministic checks; **not real-GPU acceptance**
- reviewer: `/root/m105_geometry_component_review`, the reviewer that issued the initial FAIL
- reviewer_model: **gpt-6-astra**
- reasoning_effort: **max**
- original_spawn_context: **fork_turns=none**
- attribution_source: parent confirmed the actual spawn settings when requesting this recheck
- reviewed_at: 2026-09-28T13:59:01+08:00

The latest `diagnose_learned_geometry_components.py` closes the sole blocking issue in `M105_GEOMETRY_COMPONENT_REVIEW_20260928_135551.md`. The initial fixed-name and timestamped FAIL reports are preserved without modification. This recheck records the corrected source rather than replacing that history.

**B1 resolution and exact evidence.**

At line 91, each current event's `strata` must equal the corresponding M104 row's `strata`. Together with ordered key equality at line 56 and the unchanged exact selected/native/parent/full/oracle comparisons at lines 92–97, this establishes equality of every field in the existing M104 row representation.

Lines 118–131 now reconstruct all fourteen original M104 summary fields, and line 132 asserts complete dictionary equality with the saved summary:

`valid_gt`, `native_iou50`, `selected_iou50`, `oracle_iou50`, `rescues`, `breaks`, `native_mean_iou`, `selected_mean_iou`, `parent_correct`, `parent_mean_iou`, `parent_rescues`, `parent_breaks`, `original_top10_correct`, `original_miss_refined_hit`.

The field mappings are correct: native metrics use the native IoU already checked against M104; full metrics use the full intervention; parent metrics use the unchanged parent selection; original Top-10 coverage uses the parent candidate maximum. Native-relative rescues/breaks remain distinct from parent-relative rescues/breaks. Python iteration preserves the existing within-group summation order.

The current saved M104 artifacts contain exactly five groups per split: `all`, `healthy`, `intermediate`, `late_low`, and `transition`. Rebuilding groups from the supplied 3039 rows produced precisely that set in both splits. The new per-row strata equality preserves group membership during M105 replay, and the gate checks every one of these groups. There is no `eligible` summary group in the existing M104 result; eligibility remains unused metadata in M105.

**Checks run against the actual updated source.**

- In-memory syntax compilation passed for the updated script, using `E:\python.exe -B` without writing bytecode or test files.
- Extracted the actual updated `summarize`, grouping/joint-profile statements, fourteen-field gate, and strata assertion through Python AST; executed those source statements locally on parent/full observations adapted from the saved M104 rows.
- The complete gate passed for 2544 fit rows and 495 development rows, covering all fourteen fields in each of the five original groups.
- Changed each expected field individually in an in-memory copy of the original result: all 70 field mismatches per split were rejected, for **140/140 rejected mismatches**. This includes `original_miss_refined_hit`, whose mismatch passed the initial source gate.
- Executed the new strata assertion for all 3039 saved rows; all passed. A changed strata list was rejected in each split.
- Joint-profile event totals equal 2544 and 495 respectively.

The arithmetic inputs represented already recorded parent/full selected IoUs and candidate maxima. They were sufficient to exercise the replay gate; they do not constitute new per-candidate GPU output or center-only/size-only predictions.

The latest full source was re-read. The previously accepted forward wiring remains intact: one learned delta per candidate; four readouts from original boxes through the existing decoder; fixed parent-selected index; batches of 64; no GT-dependent geometry gate; exact zero-delta checks; full parent/refiner state-dictionary comparisons; and all/marginal/joint output coverage. No optimizer, tracker action, template write, semantic truth, public evaluation or automatic variant promotion was added.

No code changes, deployment, installation, GPU calls or remote actions were performed by this reviewer. No annotation/private mapping files were accessed. Actual M105 execution must still establish fit on GPU 0 and development on GPU 1, exact checkpoint replay, all 30390 zero-delta boxes, unchanged frozen parameters/buffers, and independent arithmetic acceptance of its generated artifacts. This PASS satisfies the corrected source-review step only; `acceptance_status` remains provisional until that runtime evidence exists.