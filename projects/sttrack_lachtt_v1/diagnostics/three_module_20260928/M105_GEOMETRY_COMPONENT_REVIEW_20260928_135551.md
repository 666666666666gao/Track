# M105 geometry component source review

- verdict: **FAIL — one blocking replay-verification gap**
- review_independence: **same-family**
- acceptance_status: **provisional**
- review_type: fresh-source review plus local syntax and deterministic arithmetic checks; **not real-GPU acceptance**
- reviewer: fresh Codex agent `/root/m105_geometry_component_review`; exact model/effort is not exposed in this child context and is not inferred.
- reviewed_at: 2026-09-28T13:55:51+08:00

The forward wiring is correct for the planned read-only intervention. The blocking issue is the incomplete enforcement of the explicit full-replay contract. No source code was modified. No remote operation or training was performed, and no tracker state, annotation files or private mappings were accessed. Only these review reports were written.

**Blocking issue B1 — the declared complete M104 replay checks only part of the saved contract.**

`M105_GEOMETRY_COMPONENT_PLAN.md:27–29` requires every existing full row and all original stratum summaries to replay exactly before component interpretation. In `diagnose_learned_geometry_components.py:114–120`, only five saved summary fields are compared: `selected_iou50`, `selected_mean_iou`, `parent_rescues`, `parent_breaks`, and `oracle_iou50`. The existing `m104_completed/train/result.json` has fourteen fields in each of five groups for each split. These nine are not compared:

- `valid_gt`, `native_iou50`, `rescues`, `breaks`, `native_mean_iou`;
- `parent_correct`, `parent_mean_iou`, `original_top10_correct`, `original_miss_refined_hit`.

The per-row assertions at `diagnose_learned_geometry_components.py:56,91–96` correctly compare ordered keys, selected index, native IoU, parent-selected IoU, full-selected IoU, original Top-10 maximum IoU, and full Top-10 maximum IoU. They do not compare the row's `strata`; line 97 copies current strata. The summary loop also does not compare the full group key set. Line 123 nevertheless writes `full_replay_original_rows_and_summaries_exact=True`.

This is an implemented check missing a required invariant, not a claim that the completed M104 results are wrong. As a direct check, I extracted and executed the actual current summary gate against both existing splits. It passed. Changing only the in-memory expected `all.original_miss_refined_hit` by one still passed for both splits. Thus the current gate cannot establish the all-summary claim it emits.

Minimal correction: construct M104-compatible replay rows from the computed values, including current `key` and `strata`, and compare them with `original_rows`. Reconstruct all fourteen existing summary fields for every original group, using the existing M104 calculation, and compare the complete summary dictionary with `original_result[args.split]`. Keep the current exact comparisons; no tolerance, new hash scheme, training change, compatibility layer, or fallback is needed. Re-review this focused correction before deployment.

**Verified source behavior.**

| Contract | Evidence and assessment |
|---|---|
| Same completed final and frozen M101 parent | M105 lines 44–64 require the completed M104 train result, the native-preservation-weight-1 parent, and the existing recorded checkpoint identities; both state dictionaries are loaded, put in evaluation mode, and frozen. No optimizer or checkpoint-save call exists in M105. |
| Parent score and candidate index | `train_ab_visual_control.py:62–75` preserves cached candidate ROI, geometry, native scores and boxes by the same event index. `collect_ab_interface_panel.py:52–62` derives the 13 geometry features from native score/rank/cell/box/prior data. `instance_ab_prototype.py:108–125` selects `argmax(base_scores + learned residual)`. M105 line 87 uses the frozen parent's selected index for every variant and never chooses by component IoU. |
| Empty five-slot inputs | `train_ab_visual_control.py:86–94` excludes `iou` and non-tensor key/strata metadata from the model batch and supplies the same Empty vector in all five text slots. `encoded_inputs` in `train_local_visual_geometry.py:70–84` asserts visual and full selection-logit equality. No semantic labels or teacher are consumed. |
| Original local/context encoding | The existing `encoded_inputs` encodes candidate ROI with role 2 and context ring with role 3, then concatenates their 32+24 tokens and validity masks. `LocalVisualRefiner.forward` consumes these tokens, validity, and the 13 native geometry features. |
| GT metadata separation | `metadata` loads targets, original eligibility metadata and image shape before the loop. The existing loader also computes evaluation IoUs before inference. Neither target, eligibility, strata nor cached IoU enters the refiner forward at M105 line 73; target is consumed by `overlaps` after decoding at line 88. Eligibility is unused by M105. Original image dimensions alone enter clipping. There is no GT-dependent geometry gate. |
| Four correct component readouts | M105 lines 73–82 compute one delta per candidate and preserve parent boxes; full uses all four values, center-only zeros the log-size pair, and size-only zeros the center pair. The existing decoder (`train_local_visual_geometry.py:33–46`) implements center shift `delta_xy * original_wh` and size factor `exp(delta_wh)`, with original-image/minimum-10 clipping. All branches start from original boxes. Clipping makes these interventions non-additive, as the plan and result interpretation state. |
| Batch/order preservation and invariants | M105 lines 55–56 enforce split counts 2544/495 and original ordered keys. Both encoding and readout retain within-split batches of 64. Line 83 checks exact zero-delta decoding for every box in every batch. Lines 65–66 and 102–103 compare every parent/refiner state-dictionary tensor, including buffers, before and after processing. Actual 30390-box success remains a runtime requirement. |
| Output coverage | Lines 97–113 retain every event, four per-candidate IoU arrays, four selected boxes, the selected learned four-dimensional delta, all marginal groups and sorted joint-stratum profiles. Joint counts must exhaust the split. `summarize` counts selected/oracle correctness, parent rescues/breaks, and original Top-10 coverage gains/losses at IoU 0.5. No favorable variant is promoted into inference. |
| GPU assignment | The script provides independent `--split` and `--device` arguments. The plan specifies fit on GPU 0 and development on GPU 1. Actual launch mapping and successful execution were not established by this source review and must be recorded by the later two-GPU run. |

**Checks actually run.**

1. In-memory Python parsing/compilation passed for `diagnose_learned_geometry_components.py`, `train_local_visual_geometry.py`, `train_ab_visual_control.py`, `instance_ab_prototype.py`, `analyze_train_states.py`, `collect_ab_interface_panel.py`, and `train_fixed_visual_selector.py`. This used `E:\python.exe -B` and did not write bytecode or test files.
2. Independently recomputed all fourteen M104 fields for all five groups in fit and development from the saved JSONL rows. Both complete dictionaries equal the saved M104 result exactly. All 3039 keys are unique within their split. Paired M104 train/reference rows have exactly matching keys, strata, selected indices and parent/native fields; the reference-selected IoU equals parent IoU and reference oracle equals original Top-10 IoU.
3. Executed the actual M105 `summarize` function on parent/full observations adapted from saved M104 rows. Its parent/full statistics agree with the corresponding existing metrics. The adapted candidate arrays represented known maxima solely for this arithmetic test; no center-only/size-only model results or actual per-candidate readouts were generated.
4. Executed the actual M105 five-field replay gate and the focused missing-field check described in B1. The unverified-field mismatch passed the current gate for both splits.
5. Actual PyTorch inference was not run. The available `E:\python.exe` has no `torch` module; no installation or remote operation was attempted. Syntax and scalar arithmetic checks do not establish decoder behavior, CUDA determinism, checkpoint replay, zero-delta checks or frozen-state checks on the real workload.

| Existing M104 split | Rows | Parent correct | Full correct | Parent rescues | Parent breaks |
|---|---:|---:|---:|---:|---:|
| fit | 2544 | 1602 | 1652 | 51 | 1 |
| development | 495 | 272 | 286 | 15 | 1 |

The fit harm remains `pine01_indoor@1276`: `0.5004430413246155 -> 0.47923150658607483`. The development harm remains `mobilephone02_indoor@65`: `0.5105010271072388 -> 0.44826263189315796`. Both are healthy events. This review does not relax the healthy gate.

Existing joint profiles independently counted from M104 rows:

- Fit: healthy 1560; transition 425; late_low 152; intermediate 380; late_low|transition 26; intermediate|transition 1. Total 2544.
- Development: healthy 264; transition 121; late_low 35; intermediate 69; late_low|transition 6. Total 495.

After B1 is corrected, the required next evidence is the actual fit/GPU-0 and development/GPU-1 M105 readout, complete exact replay, all-zero/frozen-state invariants, and independent local arithmetic acceptance. The completed M104 experiment and old Full152 experiments do not need rerunning. The present verdict authorizes no claim of real-GPU acceptance, official metrics, semantic truth or tracker improvement.