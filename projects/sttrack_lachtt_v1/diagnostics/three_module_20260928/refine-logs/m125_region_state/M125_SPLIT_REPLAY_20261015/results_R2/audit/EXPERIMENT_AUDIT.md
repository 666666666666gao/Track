# Experiment Audit Report: M125 R2 E6 replay

**Overall verdict: WARN. Integrity status: warn. Evaluation: real_gt, bounded posthoc diagnosis.**

Fresh native Codex reviewer `/root/audit_m125_e6_replay_result`. Requested routing: `gpt-6-astra`, reasoning `max`; actual model and effort are **UNATTESTED** because no host attestation was supplied. `review_independence: same-family`; `acceptance_status: provisional`. Deterministic checks are separately accepted within their stated boundaries. Generated 2026-10-15T12:14:31.008942+00:00.

The supplied primary artifacts are internally consistent. **3,798 deterministic checks pass, zero fail** across 55 files: the 49 manifest entries, INPUTS.json and five explicitly supplied supplementary artifacts. All 49 input hashes match, all 48 archive text payloads match their declared bytes/SHA and extracted copies, and all 55 audited inputs remain unchanged. This is an audit of the supplied local evidence, not new tracking execution.

| Check | Status | Finding |
|---|---|---|
| A. Ground truth provenance | PASS | Dataset GT, read after sealed predictions; initializer equals dataset row zero. |
| B. Score normalization | PASS | No prediction-statistic normalization of performance metrics. |
| C. Result existence and consistency | PASS | Raw files, child exits, counts and six-decimal outputs agree. |
| D. Metric invocation and verification | WARN | Actual scoring path exists and produced vectors; local exact VOT recomputation is unavailable. |
| E. Scope | WARN | Four posthoc prefixes, one selected checkpoint; test-metric selection and multiframe aids disclosed in metadata. |
| F. Evaluation classification | PASS / real_gt | GT-grounded geometry diagnosis; saved-output agreement is only a consistency check. |

## A. Ground truth provenance — PASS within supplied-code scope

The CPU analyzer opens `dataset_root/sequence/groundtruth.txt` and verifies it against `case.gt_sha256` ([047_analyze_m125_selected_E6_observation.py:44](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_R2_RESULT_INPUTS_20261015/047_analyze_m125_selected_E6_observation.py:44)). All four actual GT files match the stage/result/manifest hashes. Their row counts are 999, 1,522, 874 and 1,819. Initialization boxes equal dataset row zero and the archived original predictions' initial row. Model candidates are predictions evaluated against these dataset boxes; original saved predictions are only a parity reference.

The replay starts from the bound initialization, calls `actor.trusted.step(image)` using the current image, saves the action and then calculates alternatives. It seals prediction JSON before the CPU process opens subsequent GT ([046_replay_m125_selected_E6_observation.py:61](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_R2_RESULT_INPUTS_20261015/046_replay_m125_selected_E6_observation.py:61), [046_replay_m125_selected_E6_observation.py:74](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_R2_RESULT_INPUTS_20261015/046_replay_m125_selected_E6_observation.py:74), [046_replay_m125_selected_E6_observation.py:121](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_R2_RESULT_INPUTS_20261015/046_replay_m125_selected_E6_observation.py:121), [005_driver.py:24](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_R2_RESULT_INPUTS_20261015/005_driver.py:24)). No current-GT read, GT-based reset, optimizer construction, backward call or optimizer update exists in the supplied replay loop. The entire recorded `previous_bbox → bbox` chain is continuous, initialized once per sequence. This is source-visible and recorded-state evidence; the imported runtime internals are not included and cannot receive a transitive no-GT certificate from this audit.

GT validity was independently recomputed as finite coordinates and positive width/height, matching the locked metric's treatment of invalid rows ([locked_depthtrack_pr.py:49](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_R2_RESULT_INPUTS_20261015/locked_depthtrack_pr.py:49)). Eleven sampled rows are invalid: cart 600/606/624; bottle 560/570/573; jug 280/285/315/325; robot 454. They are excluded from all coverage and identity conclusions. Their stored zero IoUs are placeholders for invalid GT, not evidence of an observed failure. The metric variable `visible` means non-special/valid GT here; it does not establish visible pixels.

## B. Score normalization — PASS

The continuous coverage denominator is dataset GT area inside the image. The locked overlap denominator is governed by VOT region overlap. Neither is the model's maximum, minimum or mean. `response / quality` is stored as an explicitly defined same-state ranking counterfactual, with raw responses and probabilities also retained ([046_replay_m125_selected_E6_observation.py:91](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_R2_RESULT_INPUTS_20261015/046_replay_m125_selected_E6_observation.py:91)). All 25 states' argmax selections and score/quality algebra are consistent with their archived arrays. Dividing a ranking score by quality is not reported as normalized tracking performance. These score alternatives are not a full trained-module ablation.

## C. Result existence and consistency — PASS

The four NN child return codes and four CPU child return codes are zero, agree between the standalone exit files and launches.json, and have matching terminal log records. `controller.exit` is a completion sentinel written by driver.py after its checks; it is **not independent OS-level attestation of the controller process exit** ([005_driver.py:25](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_R2_RESULT_INPUTS_20261015/005_driver.py:25), [005_driver.py:33](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_R2_RESULT_INPUTS_20261015/005_driver.py:33), [005_driver.py:45](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_R2_RESULT_INPUTS_20261015/005_driver.py:45)). The controller result binds each sequence result by SHA. Launch arguments, selected frames, source hashes, E6 checkpoint hash and stage objects agree.

| Sequence | Tracker step calls | Prefix rows, including initialization | Candidate states | Max bbox difference | Max confidence difference |
|---|---:|---:|---:|---:|---:|
| cart_room_occ_1 | 624 | 625 | 8 | 5.00000055581e-07 | 4.99996900556e-07 |
| bottle_room_occ_1 | 580 | 581 | 6 | 4.99999998738e-07 | 4.97783660847e-07 |
| jug | 327 | 328 | 8 | 4.99755856254e-07 | 4.99790191677e-07 |
| robot_corridor_occ_1 | 1363 | 1364 | 3 | 4.99999998738e-07 | 4.99809265087e-07 |

All **11,592 bbox components and 2,898 confidence values**, formatted to six decimals, equal the original saved files exactly; there are zero formatted mismatches. The supplied analyzer uses a 5.01e-7 tolerance and the maximum observed bbox difference is 5.000000555810402e-7. This audit additionally checked exact six-decimal formatting instead of treating only the tolerance flag as proof. Original numeric hidden states, queries and templates were not saved for comparison, so identical formatted outputs do not prove identical original hidden state.

There are 2,894 recorded tracker steps, four initializations, 25 additional Empty decoder evaluations and 6,400 candidate boxes in each geometry (12,800 across native-float32 and refined). These are distinct counters; the CPU archive/summarizer's `NN_calls: 0` describes those CPU stages, not the replay's total computation. The frozen source plan's `prepared_not_reviewed_not_run` and `NN_calls: 0` are historical plan fields; current execution state is established by the separate terminal records. The plan, config, replay source and stage hashes agree.

Every sampled action equals its corresponding prefix action and selected refined candidate. Native candidate geometry is explicitly float32, not the exact native double-precision output ([046_replay_m125_selected_E6_observation.py:93](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_R2_RESULT_INPUTS_20261015/046_replay_m125_selected_E6_observation.py:93)). The record includes 32 C predictions and 30 template writes; these normal own-history updates are not optimizer steps or GT resets. The provided action history and supplied source show no alternative action committed to history.

## D. Metric invocation and verification — WARN

The analyzer actually calls the locked `_vot_overlaps` on replayed/saved prefixes and on both candidate geometries, then writes its returned arrays ([047_analyze_m125_selected_E6_observation.py:59](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_R2_RESULT_INPUTS_20261015/047_analyze_m125_selected_E6_observation.py:59), [047_analyze_m125_selected_E6_observation.py:71](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_R2_RESULT_INPUTS_20261015/047_analyze_m125_selected_E6_observation.py:71)). The four CPU logs and their zero return codes support that execution. From the source loop/counts this is 58 overlap-function invocations covering 18,596 box pairs. It does **not** call the full PR/F-score evaluator; no new complete benchmark PR/F score is established by this diagnostic.

Exact local overlap reproduction is **UNVERIFIED_DEPENDENCIES_UNAVAILABLE**. In the supplied Python 3.12.12 / NumPy 1.26.4 environment, importing the exact metric fails at `import cv2`; directly importing `vot.region` also fails. The supplied standard Python 3.13.12 interpreter has neither package nor NumPy. Actual command, return code and traceback are preserved in [dependency_probes.json](dependency_probes.json). No package was installed and no approximate geometry implementation was substituted.

All per-candidate maxima, counts, chosen-index IoUs, oracle score ranks, quality and semantic-delta summaries were independently recalculated **from archived IoU vectors**, and match. Thus the recorded statement “13 of 14 GT-valid sampled states have no refined candidate with IoU ≥ 0.5” is arithmetically consistent with those arrays. Their underlying raster overlaps remain unverified locally. The native and refined count at robot 909 is 14 and 15 respectively; no other valid sampled state has a recorded candidate ≥ 0.5. Keep this distinction in every downstream numerical claim.

## Independent continuous box coverage — PASS

The audit independently clipped dataset GT to the image, then clipped that rectangle to the recorded search crop. Coverage is `area(GT ∩ image ∩ crop) / area(GT ∩ image)`. All 14 valid rows agree with the supplied summary. The actual tracker source sets `search_size=256` and calls `sample_target(..., output_sz=256)` ([full_dense_tracker.py:29](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/full_dense_tracker.py:29), [full_dense_tracker.py:54](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/full_dense_tracker.py:54)); this file's hash matches all corresponding stage-bundle source entries. All 2,894 recorded `resize_factor × crop_size` values equal 256 within floating-point precision. Crop arithmetic also agrees with every previous predicted box. GT resized width/height below use that source-confirmed 256, not an assumed resolution.

Frame numbers are **zero-based replay indices**; dataset image filenames use `frame + 1` ([046_replay_m125_selected_E6_observation.py:67](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_R2_RESULT_INPUTS_20261015/046_replay_m125_selected_E6_observation.py:67)).

| Sequence | Frame | Continuous coverage | GT width × height after search resize |
|---|---:|---:|---:|
| cart_room_occ_1 | 607 | 1 | 4.263766 × 24.277362 |
| cart_room_occ_1 | 608 | 1 | 5.164265 × 25.729107 |
| cart_room_occ_1 | 610 | 1 | 6.727007 × 25.226277 |
| cart_room_occ_1 | 620 | 1 | 6.206061 × 23.272727 |
| cart_room_occ_1 | 623 | 1 | 5.271676 × 24.305202 |
| bottle_room_occ_1 | 574 | 0 | 37.519610 × 29.914537 |
| bottle_room_occ_1 | 575 | 0 | 37.731137 × 30.127686 |
| bottle_room_occ_1 | 580 | 0 | 37.832512 × 31.211823 |
| jug | 286 | 0 | 73.224649 × 23.928216 |
| jug | 287 | 0 | 102.234280 × 61.423923 |
| jug | 326 | 0 | 110.844041 × 27.711340 |
| jug | 327 | 0 | 116.372995 × 45.037037 |
| robot_corridor_occ_1 | 909 | 1 | 79.118110 × 32.110866 |
| robot_corridor_occ_1 | 1363 | 1 | 28.070175 × 56.589474 |

Seven valid states have full box coverage and seven have zero. The five covered cart boxes are only 4.26–6.73 pixels wide after resize. This describes rectangular geometry and scale only. It does not establish whether the object is occluded, its pixels are visible, depth is valid, a competing instance is present, or the language identifies the target correctly. Image dimensions and RGB/depth hashes are saved in snapshots, but raw images are outside the supplied input set.

## E. Scope — WARN

The source plan explicitly selects posthoc E6 failures plus a positive sequence with robot quartile frames ([PLAN.json:57](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_SPLIT_REPLAY_SOURCE_20261015/PLAN.json:57)). Four sequences and 25 chosen states are not a whole-benchmark sample. There is one replay per sequence at one selected checkpoint; no seed-replicated experiment is present in this package.

E6 was selected by the mean of DepthTrack Test and CDTB F, and the bundle explicitly marks `Test50_CDTB80_used_for_checkpoint_selection: true` and `human_review_used_multiframe_aids: true` ([006_prediction_stage.json:209](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_R2_RESULT_INPUTS_20261015/006_prediction_stage.json:209), [006_prediction_stage.json:218](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_R2_RESULT_INPUTS_20261015/006_prediction_stage.json:218), [006_prediction_stage.json:319](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_R2_RESULT_INPUTS_20261015/006_prediction_stage.json:319)). The two historical F values and their average are selection metadata, not independently revalidated benchmark results in this audit. Describe the panel as posthoc diagnosis of a test-selected checkpoint with the declared human-input protocol; do not imply an untouched test set or independent selection.

At each sampled state, the primary action has already been committed before the Empty decoder call. The code checks the box state, query, selected feature and templates against their before-alternative copies and checks model digests across the run ([046_replay_m125_selected_E6_observation.py:79](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_R2_RESULT_INPUTS_20261015/046_replay_m125_selected_E6_observation.py:79), [046_replay_m125_selected_E6_observation.py:114](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_R2_RESULT_INPUTS_20261015/046_replay_m125_selected_E6_observation.py:114)). This supports the stated read-only comparison within the saved runtime evidence, not an exhaustive proof for every hidden field. Same-state changes cannot establish complete-trajectory gains, a first-error cause, training ablation effects or semantic identity truth. Full crop coverage together with poor candidate overlap is a sampled geometric observation, not proof of a particular model mechanism causing failure.

## F. Classification — PASS / real_gt

Dataset boxes ground the coverage/overlap evaluation. Predictions are model-produced candidates, not model-produced labels. Saved-output parity is an implementation-consistency check. Empty and quality alternatives are fixed-state diagnostics whose scores are evaluated against real GT; they are not synthetic-proxy ground truth. No human semantic-identity evaluation or visibility-mask evaluation is present.

## Verification blockers and claim ceiling

1. **Exact raster IoU reproduction is blocked by absent local cv2/vot dependencies.** The archived numerical results may be quoted only with that verification boundary; they must not be described as independently reproduced official overlaps.
2. **Full transitive execution and original-state attestation are outside this input set.** The runtime/decoder/trusted-tracker imports, raw image bytes, checkpoint tensors, full-suite terminal/best/all_results/receipt files and original hidden-state snapshots are not supplied. Hash references and successful-stage assertions are consistent, but they cannot independently certify those absent artifacts. This audit does not revalidate the full training/VOT terminal claims or the parent suite's nine metrics.

No integrity contradiction was found in the supplied artifacts. Qualified coverage, counts and six-decimal parity findings are supported. Whole-benchmark rates, first-error causality, exact original hidden-state identity, intervention trajectory gains and semantic identity truth are unsupported by this panel. No new NN experiment is needed to state these limited findings.

## Reproducibility and actual reviewer failures

- [deterministic_checks.py](deterministic_checks.py): independent CPU checks, with [deterministic_checks.json](deterministic_checks.json) and [independent_coverage.json](independent_coverage.json).
- [input_inventory.json](input_inventory.json): all 55 SHA-256/size entries; [final_input_integrity.json](final_input_integrity.json): unchanged-input verification.
- [dependency_probes.json](dependency_probes.json): exact missing-dependency commands and failures.
- Reviewer-only first checker run used a tab delimiter for comma-delimited prediction files and exited 1. The failed source is preserved as `deterministic_checks_r0_csv_failure.py`. The next run incorrectly assumed phrase cosine was scalar per candidate; 25 shape checks failed because the actual field is 256×5. Its source/output are preserved as `deterministic_checks_r1_shape_assumption.py/.json`. Correcting these reviewer assumptions produced the final 3,798/0 result; no experiment input was changed. Initial broad text reads were truncated, then all primary content was parsed directly and checked; truncation was not treated as a successful semantic review of omitted text.
- The first exact-module import probe automatically generated one `locked_depthtrack_pr.cpython-312.pyc` file in the input directory. Its creation time was 2026-10-15T12:02:35.0704287Z and its SHA was `5ef2a20f85796e1d6e46cb2d7cbbb16ba1956c09f3ecb71d7889dbc186246f7a`. The sole reviewer-created cache directory was moved intact into this audit directory after checking its exact paths, name and creation time. [reviewer_cache_relocation.json](reviewer_cache_relocation.json) preserves the receipt. Future metric probes use Python `-B`; the final full checker rerun passed and no cache remained in the input directory. Original input contents were never changed. This transient output-location mistake is recorded rather than omitted.
- [TOOL_EXECUTION_LOG.json](TOOL_EXECUTION_LOG.json) records actual tool outcomes. The full reviewer request/response and attribution metadata are under `.aris/traces/experiment-audit/2026-10-15_run01/`.

Key SHA-256 values: archive `ed0181d13e42282cf0469093d10143c02bdc178710602a7a2a8580c4dc0cdd6b`; replay `be2239091b279bebc614a4b77d31887ba4c0f6bd125862e194c502fb1a067cac`; analyzer `81081356adb1af5e5429883aa22a3fcba519204133f70001c013b7e195f5bfe4`; locked metric `05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc`; full dense tracker `0aee7de2932d7294976e2ed512218cd58d281e24bf3fb34b6267167470aa1778`. Full hashes are in the inventory.
