# M124 selected-best CDTB onset replay — R1 source integrity review

**Overall: FAIL — one deterministic source blocker (B1).** No new replay result or terminal suite result is claimed.

Date: 2026-10-12. Fresh Codex review; requested `gpt-6-astra` / `max`; actual model and reasoning effort **UNATTESTED**. `review_independence: same-family`; `acceptance_status: provisional`.

This is a SOURCE ONLY review. It is not GPU release, NN admission or launch authorization. It does not certify raw-server completion, actual NN completion, final nine-metric success, calibrated probabilities or model novelty.

All 13 requested source files were read directly. Twelve supporting source files/functions and five skill/policy files were inspected; exact semantic read scopes and hashes are in the JSON. No executor content summary was used. Original bytes are preserved under the trace directory and were rehashed unchanged before this report was written.

## Blocking finding

### B1: Selected-best metadata is overwritten by a candidate index before result serialization

Line 40 binds best to best.json metadata. Every accepted nonempty --frames request creates at least one snapshot (lines 31, 53-55, 68-80). The diagnostics loop overwrites best with int(refined_iou.argmax()) at line 164. The new selected_epoch=best['epoch'] expression at line 182 therefore raises TypeError: 'int' object is not subscriptable, after predictions have been sealed but before result.json can be written at line 200. Earlier failures do not make the result path valid. The exact two source AST expressions reproduce this error without importing torch, a dataset or a model.

**Evidence:** `replay_m124_selected_best_cdtb_onsets.py:31`; `replay_m124_selected_best_cdtb_onsets.py:40`; `replay_m124_selected_best_cdtb_onsets.py:53-55`; `replay_m124_selected_best_cdtb_onsets.py:68-80`; `replay_m124_selected_best_cdtb_onsets.py:153-182`; `replay_m124_selected_best_cdtb_onsets.py:200`; `prepare_m124_selected_best_onset_replay_source_20261012.py:16`; `prepare_m124_selected_best_onset_replay_source_20261012.py:29`.

**Minimal correction:** Give the selected checkpoint metadata and the per-state oracle candidate index distinct names; make the matching generator substitution consistent. No fallback, retries, model change or training-source change is needed.

The CPU fixture executed only the original overwrite and selected_epoch expressions, with a scalar argmax stub. It reproduced `TypeError: 'int' object is not subscriptable`. It did not import torch or model/dataset code.

## A–F checks

### A ground truth provenance: PASS

Source-level real_gt provenance and causal timing pass. The replay only passes case.init_bbox at initialization; step(image) receives no current GT. Prediction/candidate/action JSON is written and hashed before groundtruth.txt is hashed and parsed. Both replay and saved trajectories are compared against dataset GT, not to each other as a substitute for GT. Train152 truth is independently hashed and loaded from groundtruth.txt; losses access its current frame only after tracker.step. Dataset files and actual run receipts were not supplied, so this is not a raw-data attestation. Human label binding discloses multiframe aids and does not establish strictly first-image-only human labeling.

**Evidence:** `replay_m124_selected_best_cdtb_onsets.py:50-79`; `replay_m124_selected_best_cdtb_onsets.py:123-147`; `depthtrack_pr.py:13-22`; `depthtrack_pr.py:40-60`; `train_m124_joint_epoch.py:113-144`; `train_m122_full_causal.py:14-22`; `bind_m122_official_initializations.py:33-54`.

### B score normalization: PASS

No benchmark metric is divided by the model's own maximum, minimum or mean. Diagnostic IoUs use VOT rectangle intersections; coverage counts and rank are raw values. response/quality removes one internal model factor for a fixed-state readout; odds_residual/native is model score construction, not reported performance normalization. The complete OPE helper uses conventional threshold curves, per-sequence precision/recall and macro-averaging, with percent scaling only. No new numeric metric or suspicious near-perfect result is claimed. Calibration is outside the evidence (W4).

**Evidence:** `replay_m124_selected_best_cdtb_onsets.py:93-111`; `replay_m124_selected_best_cdtb_onsets.py:146-178`; `dense_target_decoder.py:18-22`; `dense_target_decoder.py:97-115`; `depthtrack_pr.py:117-153`.

### C result file existence and binding: WARN

No actual new result, suite terminal result, raw-server completion or nine-metric success was asserted or audited. Source requires terminal suite metadata before model construction and verifies selected-final/base-bank/cases identities, then saved box/score hashes. The saved receipt identity fields and the complete manifest hash chain are not all independently revalidated here (W2). Any actual existence and identity conclusion must wait for the concrete artifacts; formal_nine_metrics means complete metric availability, not any_joint_pass.

**Evidence:** `replay_m124_selected_best_cdtb_onsets.py:35-56`; `replay_m124_selected_best_cdtb_onsets.py:130-143`; `m124_epochwise_official_runtime.py:9-41`; `collect_m124_epochwise_results.py:39-48`; `run_m124_epochwise_suite.py:95-103`.

### D called metrics and reachable output: FAIL

B1 makes the new result.json construction fail deterministically after a successful nonempty replay. The intended IoU helpers are actually called, so there is no phantom computed IoU in source; however no diagnostic result can reach its final write until B1 is fixed. evaluate_depthtrack_results and _determine_thresholds are not invoked by the onset replay; the complete OPE runner invokes them separately. format_metrics and verify_author_reference are not called by this replay path and confer no extra metric or author-reference verification.

**Evidence:** `replay_m124_selected_best_cdtb_onsets.py:136-178`; `replay_m124_selected_best_cdtb_onsets.py:182-200`; `run_m124_epochwise_official.py:58`; `depthtrack_pr.py:117`; `depthtrack_pr.py:157-158`; `dense_region_encoding.py:38-56`.

### E scope assessment: WARN

Only a source-only, caller-specified one-sequence prefix diagnostic is reviewed. The dynamic panel is larger in allowed scope than the old three named sequences, but no sequence, frame or model has actually been evaluated in this review. Test50/CDTB80 selection bias, same-state-only alternatives, unknown saved-history parity and model/effort attribution must remain explicit. No whole-trajectory rescue, full benchmark pass, novelty or calibrated semantics can be inferred.

**Evidence:** `replay_currentC_cdtb_onsets.py:21-24`; `replay_m124_selected_best_cdtb_onsets.py:29-32`; `replay_m124_selected_best_cdtb_onsets.py:53-80`; `replay_m124_selected_best_cdtb_onsets.py:180-198`; `run_m124_epochwise_suite.py:56-69`.

### F evaluation type: PASS

The designed diagnostic is real_gt: post-seal dataset GT measures submitted and candidate IoU. The saved-prefix comparison is separately a replay-consistency check, not ground truth performance. Empty/no-quality/native-score comparisons are model-based fixed-state readouts with posthoc real_gt measurement, not executed alternative trajectories. The audit itself is static/source-only plus stdlib deterministic checks, not a completed real_gt experiment.

**Evidence:** `replay_m124_selected_best_cdtb_onsets.py:123-178`; `replay_m124_selected_best_cdtb_onsets.py:192-198`; `depthtrack_pr.py:40-60`.

## Requested source checks

### exact diff: PASS

Saved exact new-versus-old unified diff. Executing only generator text-transformation/check AST statements before line 35 exactly reconstructs current new bytes. Changes are runtime import/name, removal of fixed FRAMES, explicit frame/suite args and guards, metadata fields and scope wording. Core tracking, candidate diagnostics and GT comparator are unchanged. The new metadata read introduces B1 because the old diagnostic already reused best as an integer.

**Evidence:** `prepare_m124_selected_best_onset_replay_source_20261012.py:4-34`; `replay_currentC_cdtb_onsets.py:154`; `replay_m124_selected_best_cdtb_onsets.py:164`; `replay_m124_selected_best_cdtb_onsets.py:182`.

### python38 syntax: PASS

25 source inputs parse using feature_version=(3,8); syntax only, no runtime imports or model execution.

**Evidence:** `source_validation.json:python38_syntax`.

### terminal guard order: PASS

All explicit suite/selected-final/cases/frame guards at lines 37-55 precede OfficialDenseTracker construction at line 56. Runtime checked_plan validates base and M124 artifacts using hash/JSON reads; constructors are inside the class method, not executed on import. The suite source emits its final status and controller.exit only after full VOT analysis/seal/collection returns. This remains a source-order conclusion, not terminal proof.

**Evidence:** `replay_m124_selected_best_cdtb_onsets.py:35-56`; `m124_epochwise_official_runtime.py:9-45`; `run_m124_epochwise_suite.py:77-103`.

### selected identity: WARN

Selected final and selected plan path plus base bank/binding/cases/evaluator are bound. The hardcoded evaluator digest equals the reviewed snapshot. Saved-prefix receipt identity fields are a separate verification gap (W2).

**Evidence:** `replay_m124_selected_best_cdtb_onsets.py:40-50`; `replay_m124_selected_best_cdtb_onsets.py:130-143`; `m124_epochwise_official_runtime.py:9-41`; `m122_official_runtime.py:7-21`.

### own history no forced parity: PASS

The only initialization is frame zero. The loop advances trusted.step(image) sequentially and appends its submitted action. Saved boxes/scores are opened only after all requested own-history steps and prediction sealing. Differences use 5.01e-7 bbox tolerance matching six-decimal OPE output; score maximum error is separate. No saved boxes, GT reset, parity retry or equality assertion controls tracking.

**Evidence:** `replay_m124_selected_best_cdtb_onsets.py:68-79`; `replay_m124_selected_best_cdtb_onsets.py:123-151`; `replay_m124_selected_best_cdtb_onsets.py:192-198`; `run_m124_epochwise_official.py:32-35`.

### alternative state isolation: PASS

Alternatives execute only after the full tracking action is committed. They call the decoder with a shallow copied data dictionary and a replacement semantic query. State/query/selected-feature/template tensor values are cloned and compared afterward. The decoder has no actor handle and performs read-only tensor operations; no in-place input mutation or actor-state submission was found. The shared action dictionary is freshly made each step and never modified after appending. Language does not enter the geometry branch. Parameter/buffer digests are checked at prefix end. These checks do not certify every possible hidden native object or raw server state.

**Evidence:** `replay_m124_selected_best_cdtb_onsets.py:76-121`; `trusted_template_tracker.py:32-68`; `dense_target_decoder.py:51-123`.

### external gradients and optimizer: PASS

Replay main and trusted initialize/step are no_grad, inference A/B and C models are eval with requires_grad False, and frozen native/CLIP models are eval with requires_grad False. No optimizer or backward call is on the replay path; importing state_digest from the trainer does not call its main. Training code legitimately has Train152 AdamW/backward steps and must not be described as globally zero-gradient. Test50/CDTB80 only influence checkpoint selection and stop decisions here.

**Evidence:** `replay_m124_selected_best_cdtb_onsets.py:18`; `replay_m124_selected_best_cdtb_onsets.py:26-27`; `m124_epochwise_official_runtime.py:61-70`; `full_dense_tracker.py:31-33`; `trusted_template_tracker.py:21-27`; `train_template_write_C.py:31-36`; `train_template_write_C.py:186-187`; `train_m124_joint_epoch.py:85-87`; `train_m124_joint_epoch.py:126-167`.

### spatial mapping and dtypes: PASS

Source uses 256 one-hot spatial cells on a 16x16 grid, row-major xy search positions, and the native size/offset head for per-cell boxes. It retains native pixel mapping in double, gives the decoder float32 boxes and half-to-float observation features, then submits native-double base plus the float32 learned delta cast to double. Native diagnostics explicitly report the float32 observation representation; refined diagnostics report the submitted geometry formula. ROI mapping normalizes coordinates against the same crop origin/side with align_corners=False. Exact remote native implementation and runtime parity are not attested.

**Evidence:** `full_dense_tracker.py:35-71`; `dense_target_decoder.py:47-49`; `dense_target_decoder.py:102-123`; `trusted_template_tracker.py:36-46`; `dense_region_decoder.py:12-27`; `region_patch_evidence.py:8-11`; `replay_m124_selected_best_cdtb_onsets.py:95-114`; `replay_m124_selected_best_cdtb_onsets.py:197`.

### counter semantics: PASS

For a successful invocation, sorted unique frames are positive and bounded, so prefix_frames=max(frames)+1, tracker_calls=max(frames), candidate_states=extra_Empty_decoder_calls=len(frames), candidates_per_state=256. The counts do not equal all model forward calls. No such runtime counters were produced in this audit.

**Evidence:** `replay_m124_selected_best_cdtb_onsets.py:31`; `replay_m124_selected_best_cdtb_onsets.py:53-89`; `replay_m124_selected_best_cdtb_onsets.py:180-187`.

## Warnings and claim limits

### W1: Source guards are not actual terminal or GPU-release evidence

The terminal status/formal_nine_metrics/zero-retry/controller.exit and selected-final checks precede model construction. This audit has no suite files, controller runtime receipt, process-exit proof, raw server bytes, GPU-release proof or new replay result. It cannot certify that any gate has passed. This report is not launch authorization.

**Evidence:** `replay_m124_selected_best_cdtb_onsets.py:35-56`; `run_m124_epochwise_suite.py:77-103`.

### W2: Saved-prefix identity is only partially revalidated by this entry point

The replay final, selected plan pathname, base bank/binding and evaluator path are bound by source checks. For the saved prefix, lines 139-143 only read receipt.sequences and verify the two file hashes. They do not recheck receipt status, plan_sha256, final_sha256, bank_sha256 or bundle_sha256, although the OPE writer supplies them and the OPE analyzer checks the key identities. The replay also does not revalidate all_results row result_sha256 values or selection_sha256. This is a local verification gap, not evidence that an artifact is actually mismatched. Before a selected-best saved-prefix claim, verify the concrete receipt/manifest chain; direct equality checks in the replay are also a small scoped correction.

**Evidence:** `replay_m124_selected_best_cdtb_onsets.py:40-49`; `replay_m124_selected_best_cdtb_onsets.py:137-143`; `m124_epochwise_official_runtime.py:9-41`; `run_m124_epochwise_official.py:40-61`; `collect_m124_epochwise_results.py:13-35`; `collect_m124_epochwise_results.py:46-48`.

### W3: The checkpoint is selected using Test50 and CDTB80 metrics

Training gradients come from Train152, but epoch selection uses the equal mean of DepthTrack Test and CDTB F and the dual-decline stop rule. The source explicitly labels this external metric checkpoint selection. Do not call these datasets untouched holdouts or interpret selected-best OPE as an unbiased held-out estimate. This is distinct from external gradient training, which is absent from the replay path.

**Evidence:** `run_m124_epochwise_suite.py:48-75`; `m124_epochwise_selection.py:4-10`; `train_m124_joint_epoch.py:113-167`; `train_m124_joint_epoch.py:233-234`.

### W4: Readout scores and oracle candidates are not calibrated semantic truth

quality_probability is a sigmoid head value, phrase_cosine is a cosine readout, and semantic_delta is a learned full-minus-Empty branch score. No reliability/calibration test is run. C is a linear current-IoU regressor thresholded at 0.5, not a calibrated future-memory-utility probability. The GT-best candidate is an after-the-fact diagnostic oracle. A same-state ranking improvement cannot establish trajectory recovery, identity correctness, phrase visibility, text truth or model novelty.

**Evidence:** `dense_target_decoder.py:74-123`; `template_write_C.py:13-24`; `train_m124_joint_epoch.py:140-144`; `train_m124_joint_epoch.py:232`; `replay_m124_selected_best_cdtb_onsets.py:153-178`; `replay_m124_selected_best_cdtb_onsets.py:196-198`.

### W5: Onset scope and counters have a limited meaning

Each invocation replays one caller-selected sequence from initialization through max(frames); it samples len(frames) states and 256 candidates per sampled state. These are posthoc panels, not all CDTB80 or the final nine metrics. tracker_calls=max(frames) counts tracking steps, not total NN forward calls: initialization also observes native/CLIP, every step runs native/CLIP/decoder, C is conditionally evaluated, and sampled states add an Empty decoder call. seed_scan=False describes this script, not unknown caller behavior. No actual calls were executed by this audit.

**Evidence:** `replay_m124_selected_best_cdtb_onsets.py:29-32`; `replay_m124_selected_best_cdtb_onsets.py:53-80`; `replay_m124_selected_best_cdtb_onsets.py:89`; `replay_m124_selected_best_cdtb_onsets.py:180-198`; `full_dense_tracker.py:53-85`; `trusted_template_tracker.py:32-67`.

### W6: Compatibility and full state reproduction remain runtime questions

All 25 inspected Python inputs parse with Python 3.8 grammar via CPython 3.12.12 ast.parse(feature_version=(3,8)). This does not attest an installed Python 3.8 environment, torch/NumPy/VOT versions, remote native cal_bbox implementation, or exact NN behavior. Native candidate observations are float32; submitted/refined geometry uses a retained float64 native base with a float32 learned delta. The comparator reports drift and never forces saved boxes. It does not compare the original hidden recurrent/template state. Per-prefix/candidate overlaps use the last loaded image dimensions, as the existing code does; actual sequence dimensions were not examined.

**Evidence:** `full_dense_tracker.py:35-71`; `trusted_template_tracker.py:32-46`; `replay_m124_selected_best_cdtb_onsets.py:95-109`; `replay_m124_selected_best_cdtb_onsets.py:144-159`; `replay_m124_selected_best_cdtb_onsets.py:192-198`.

### W7: Requested reviewer model and effort are not host-attested

Requested gpt-6-astra / max. Actual model and reasoning effort: UNATTESTED. This is a fresh-context Codex review in the same family, acceptance_status provisional. No cross-family acceptance or unavailable model/effort attestation is claimed.

## Metric reachability

| Function | Onset replay | Other inspected source path |
|---|---|---|
| `_load_rows` | GT, saved boxes and saved scores: replay lines 136, 142–143 | Full OPE helper lines 97–99 |
| `_vot_overlaps` | Replay/saved prefixes and native/refined candidates: replay lines 146–159 | Full OPE helper line 111 |
| `evaluate_depthtrack_results` | Not called; no onset P/R/F is produced | `run_m124_epochwise_official.py:58` |
| `_determine_thresholds` | Not called directly or transitively by onset replay | `depthtrack_pr.py:117` within full OPE helper |
| `format_metrics` | Not called | Not called by the inspected M124 OPE runner |
| `verify_author_reference` | Not called | No author-reference validation is earned by this replay |

The final onset `write(result.json, result)` is currently unreachable on the successful nonempty path because of B1. The table describes source call sites, not actual executed metrics.

## Validation and forensic artifacts

- Exact old/new diff: `exact_new_vs_old.diff` in the trace directory.
- Generator reconstruction matched the current new source bytes exactly.
- Python 3.8 syntax: all 25 Python source inputs passed `ast.parse(feature_version=(3, 8))` under CPython 3.12.12. This is not a Python 3.8 runtime/library compatibility test.
- Primary guards at new replay lines 37–55 precede the first model construction at line 56.
- Source-expression CPU reproduction confirms B1; no NN or torch import occurred.
- SSH/network/GPU/training queries, installs, model construction, NN calls, optimizer steps, training-source edits, git operations and recursive delegation: all zero.

The generic local `python` launcher returned `No pyvenv.cfg file`; the already-installed local CPython 3.12.12 interpreter was then used. No installation or network lookup was performed.

Trace: `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/traces/experiment-audit/2026-10-12_run01`.

## Primary input SHA-256 hashes

| Source | SHA-256 |
|---|---|
| `replay_m124_selected_best_cdtb_onsets.py` | `4915b45543e904f02d4ecd0801c43863a01481f3771739cd95587bb58a420fbb` |
| `replay_currentC_cdtb_onsets.py` | `49e314232de10edac76812a56a60886571c1e7cf1734b7bf0175812c2c4033dd` |
| `m124_epochwise_official_runtime.py` | `60de3f27538c76bcba1178d5b24f9064c587bda9ba564a60aae179cbcc023e48` |
| `m122_official_runtime.py` | `25aa67000cf581c454b160c524c8dcf8f55ec30d227b387990cc03ed18be3157` |
| `full_dense_tracker.py` | `0aee7de2932d7294976e2ed512218cd58d281e24bf3fb34b6267167470aa1778` |
| `dense_target_decoder.py` | `055d9ddffe4e332fa9a04fca93a6ec301d031a35b198f93a26809d7baa7104f2` |
| `trusted_template_tracker.py` | `1dd4873192b05f6921cdf729a525ed684ed96a18bfdfc48673cda14d50352e0d` |
| `collect_m124_epochwise_results.py` | `7a48b9af70c96ffc462ce00c32a38ef9cc053aeb0eba2598e1f8a0a2b202374f` |
| `run_m124_epochwise_suite.py` | `63924cc9f7f5e38ab605cb8f40dddf083321bee12248e71a7e8516f5a2d17dfc` |
| `m123_joint_full_tracker.py` | `53b1e0ae9f054b62cf4f7c13d77400453d616f8b62e62b406014f7075ef529a7` |
| `train_m124_joint_epoch.py` | `6bc43f9f113c3a648e9a674ad3f7f18d1114e4f84880edf18fa61d40e804081c` |
| `depthtrack_pr.py` | `05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc` |
| `prepare_m124_selected_best_onset_replay_source_20261012.py` | `98a285a0d82c0fd10e24b8814781c18aa12cd60d510445b6043eb7307fbd94ca` |

The JSON additionally includes exact absolute paths, byte counts, line counts, hashes for all supporting and policy inputs, semantic read scopes and immutable source snapshot paths.

## Required follow-through

Correct B1 in the diagnostic source and its generator, preserve this R1, then obtain a review of the revised bytes. Before any launch or selected-best saved-prefix claim, establish the concrete terminal/artifact identity chain and the separately required GPU-release/runtime admission proof. This review changes no training code and provides no launch authorization.
