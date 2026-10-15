# M125 split replay source review

**Source verdict: PASS — 0 blockers. Overall: WARN.**

No blocking runtime defect was found in the reviewed source. NN execution, actual CPU metric execution, remote startup and replay parity are **UNVERIFIED**. This is a source review with local stdlib/AST fixtures; it is not a replay result or launch authorization.

Requested reviewer: `gpt-6-astra`, reasoning `max`. Actual model and effort: **UNATTESTED**. Review independence: **same-family**. Acceptance: **provisional**. Reviewer: `/root/audit_m125_split_replay_source`.

## Evidence names

- `P` = `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_SPLIT_REPLAY_SOURCE_20261015/replay_m125_selected_E6_observation.py`
- `C` = `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_SPLIT_REPLAY_SOURCE_20261015/analyze_m125_selected_E6_observation.py`
- `O` = `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/replay_m124_selected_best_cdtb_onsets.py`
- `G` = `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/prepare_M125_split_replay_source_20261015.py`
- `PLAN` = `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_SPLIT_REPLAY_SOURCE_20261015/PLAN.json`

Other source basenames refer to `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928`; `depthtrack_pr.py` is in the sibling `native_ope/source_snapshots` directory. Archived review/probe/terminal files are in `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007`; the environment ledger is `.aris/compute/direct_ssh_rgbd.md`.

## Findings

**A. GT provenance and timing: PASS.** The prediction process initializes with case.init_bbox, then advances step(image) on its own submitted history. It has no groundtruth.txt load or metric import. Predictions and candidate snapshots are written and SHA-256 hashed before the stage artifact is written. The CPU process checks the source, plan, suite and prediction hashes and flags before importing the locked metric and hashing/loading the dataset groundtruth.txt. Saved reference boxes are used only for a separate parity comparison. This is real_gt by source design, with replay parity classified as consistency. No dataset GT was loaded in this review.

Evidence: `P:48-77`; `P:120-137`; `C:20-46`; `C:47-60`; `m124_epochwise_official_runtime.py:9-41`; `m122_official_runtime.py:7-21`.

**B. Prediction actions and optimization: PASS.** The complete prediction main body through prediction_sha is AST-identical to the original. Actual actions are appended before the Empty readout; alternatives are never submitted. The retained source checks bbox, query, selected feature, templates and parameter digests around the extra readout. TrustedTemplateTracker commits the selected bbox/query/features and its ordinary C-gated template decision before returning. There is no optimizer or backward call in the split entry points. These are source checks, not observed NN equality.

Evidence: `O:28-127`; `P:26-125`; `P:74-119`; `trusted_template_tracker.py:27-68`; `dense_target_decoder.py:96-125`.

**C. Dependency separation and imports: PASS_WITH_RUNTIME_LIMIT.** Static traversal covers 15 local prediction modules and 2 local CPU modules, without importing any of them. The prediction closure has no VOT metric import; its existing torch/NumPy/CLIP/PIL/native dependencies remain. The CPU entry imports NumPy and the stdlib-only sha helper; the dynamically loaded locked metric adds cv2/NumPy/vot.region, with no torch/model dependency. This matches the existing sttrack NN and mplt metric routing. The archived probe records sttrack missing vot and mplt succeeding; no current remote import was performed.

Evidence: `P:6-16`; `P:58`; `C:2-9`; `C:39-43`; `bind_m122_official_initializations.py:2-10`; `depthtrack_pr.py:3-10`; `direct_ssh_rgbd.md:5-9`; `metric_import_probe_raw_stdout.txt:5-22`.

**D. Numeric formulas, serialization and result reachability: PASS.** Original O:148-181 and C:58-91 are AST-identical, including bounded VOT overlaps, 5.01e-7 bbox parity tolerance, stable score ranking and all candidate readouts. The image dimensions move from the last image into stage.image_shape. The generator exactly reproduces both files in memory. Producer/consumer and both final-write fixtures pass; candidate best assignments leave selected_best intact. All 37 old result keys remain: only status, source_sha256 and elapsed_seconds expressions change, with NN_elapsed_seconds and CPU_elapsed_seconds added. Thirty-four unchanged field values match in each synthetic result fixture. 2,052 finite float64 values, including represented float32 values, round-trip through JSON with identical binary64 bits. No actual NumPy/VOT/NN computation was run.

Evidence: `G:12-79`; `O:147-205`; `P:126-137`; `C:57-115`; `source_review_validation.json:checks`; `source_review_validation.json:result_fields`; `source_review_validation.json:result_construction_fixtures`.

**E. Variable, CLI and artifact binding: PASS_WITH_STATED_LIMITS.** Both entry points have all referenced module globals defined. The producer serializes args, selected_best, plan, bundle, case, frames, shape and prediction identity; the consumer restores every value required by the unchanged scoring tail. Source/plan/suite/prediction bytes and the output directory are checked. The saved-reference receipt retains plan/final/bank/bundle and bbox/confidence identities. Eleven actual-assertion negative fixtures reject changed hashes, status, flags or output directory. The external PLAN.json is not itself consumed by either entry point; its fixed E6 and panel bindings must be enforced by the launcher and result collector.

Evidence: `P:27-53`; `P:126-137`; `C:17-38`; `C:49-54`; `PLAN:3-52`; `source_review_validation.json:symbol_tables`; `source_review_validation.json:guard_negative_fixtures`.

**F. Complete panel bounds and accounting: PASS.** All 25 zero-based requested frames are sorted, unique, at least 1 and below their archived case frame counts. The archived plan, cases and selected-best embedded texts match their recorded hashes; selected epoch 6 and the PLAN final hash agree. The complete panel totals 2,894 tracking steps, 2,898 output rows including four initializations, 25 Empty calls and 6,400 candidates for each geometry representation. This reads archived metadata only, not GT coordinate files, images, checkpoints or live server state.

Evidence: `G:86-104`; `PLAN:6-44`; `P:51-74`; `P:78-113`; `M124_E6_terminal_audit_artifacts_20261015.json:1`; `source_review_validation.json:panel`; `source_review_validation.json:archived_metadata`.

**G. Score normalization, called metrics and claim scope: WARN.** There is no benchmark score divided by a prediction maximum/mean. response/quality is an internal same-state diagnostic, and the reported IoUs, candidate counts and ranks remain raw. _load_rows and _vot_overlaps are called in the CPU scoring tail; full OPE PR/F evaluation is not called. The panel is explicitly selected post hoc from failures and a positive sequence, and selected-best training used Test50/CDTB80 for checkpoint selection. This cannot establish a full-trajectory intervention, full benchmark gain, untouched-test performance, calibrated quality, phrase visibility or semantic instance truth.

Evidence: `P:91-110`; `C:46-91`; `C:97-112`; `PLAN:57`; `m124_epochwise_official_runtime.py:30-33`; `bind_m122_official_initializations.py:54`.

## Panel

| Sequence | Available frames | Sampled zero-based frames | Tracking steps |
|---|---:|---|---:|
| bottle_room_occ_1 | 1522 | 560, 570, 573, 574, 575, 580 | 580 |
| cart_room_occ_1 | 999 | 600, 606, 607, 608, 610, 620, 623, 624 | 624 |
| jug | 874 | 280, 285, 286, 287, 315, 325, 326, 327 | 327 |
| robot_corridor_occ_1 | 1819 | 454, 909, 1363 | 1363 |

Total: **2,894 tracking steps, 2,898 rows including initialization, 25 states and Empty calls, 6,400 candidates per geometry**. These are plan counts, not newly executed neural-call or performance measurements. The selected final in archived metadata is `4d5c7e411a047fa996835647d1efa3bb94f4184fdc3a592380523be8c04156eb` (epoch 6).

## Remaining limits

**W1. Actual runtime remains unverified.** No new entry point was run in either remote interpreter. Existing import receipts support the split but do not prove current startup, tensor assertions, selected checkpoint loading, replay completion or saved-prefix equality. No M125 result artifact is supplied for verification. The local stdlib fixtures do not execute VOT overlaps, so numerical equality across the remote environments is not certified.

Evidence: `direct_ssh_rgbd.md:5-9`; `C:39-60`; `P:54-119`.

**W2. Launch context and fixed-plan identity are external.** The staged files use bare imports from the existing diagnostics directory, which is not their staging directory. A concrete launch must make that unchanged module directory importable and retain the exact prediction source path for the CPU hash check. Use the supplied absolute plan/suite/output paths across processes. Neither entry point reads PLAN.json or enforces epoch==6, its selected_final_sha256, or the exact four-sequence panel; the E6 status string alone is not proof of E6. The archived metadata and intended PLAN agree, with no observed mismatch. This is a launcher/collector requirement, not a request for compatibility code.

Evidence: `P:14-16`; `P:27-43`; `P:126-132`; `C:9`; `C:24-33`; `PLAN:3-52`.

**W3. The serialized stage is a trusted intermediate artifact.** The consumer verifies the external plan hash but uses the serialized plan/bundle/case/selected_best dictionaries. It does not independently reconstruct and compare these copies, reload checked_plan, or hash the whole stage against an external manifest. The broader inherited receipt-completion and collection-manifest chain is also not rebuilt here. Subsequent result auditing should seal the actual stage and verify its copies and CLI against the reviewed PLAN and runtime receipts. No edited stage or broken provenance was observed.

Evidence: `P:126-137`; `C:20-36`; `C:49-54`; `O:139-144`.

**W4. Output equivalence has explicit boundaries.** The scientific diagnostic expressions are preserved, but output JSON is not byte- or schema-identical: status changes, source_sha256 becomes a two-entry object, and two timing fields are added. The timers exclude process setup/model construction and transport; NN_elapsed_seconds also contains source checks and hashing. They are not complete-system latency. Last-prefix-image dimensions still bound all prefix/candidate overlaps, exactly as in the original; image-size constancy was not checked. Actual NumPy/VOT versions and live floating-point results were not tested.

Evidence: `O:67`; `O:147`; `O:183-202`; `P:54-65`; `P:133-136`; `C:38`; `C:57`; `C:93-112`.

**W5. Posthoc and semantic limits remain.** Only four selected sequence prefixes and 25 states are covered. Alternatives are same-state readouts, not changed trajectories. Bbox rounding parity, even if later true, does not prove all hidden recurrent/template states identical to the saved run. Test50/CDTB80 were used for checkpoint selection. The human initialization binding explicitly records multiframe aids; this split does not establish independently blind language annotation. Sigmoid quality and GT-best candidate ranks do not establish calibration, identity truth or future memory benefit.

Evidence: `PLAN:6-44`; `PLAN:57`; `C:107-111`; `m124_epochwise_official_runtime.py:33`; `bind_m122_official_initializations.py:54`.

## Deterministic evidence

The local CPython 3.11.14 stdlib validator passed four Python 3.8 grammar checks, exact in-memory reproduction of both generated files, prediction/math AST equality, a producer/consumer stage fixture, 11 negative integrity-guard fixtures, two final-result write fixtures and 2,052 binary64-preserving JSON round trips. There are no unresolved module globals in either entry point. The analyzer retains all 37 original result keys and adds two timing keys.

16 current source files also match their recorded R2 hashes, with zero mismatches. All 8 supplied/request inputs match their stored initial hashes; all 31 audited inputs were rehashed at report write. Full supplemental hashes and read scopes are in `SOURCE_REVIEW.json`.

Reproducible reviewer artifacts: `source_review_validate.py`, `source_review_validation.json`, `source_review_prediction.diff`, `source_review_analysis.diff`, and `source_review_inputs_before.json`. The generator was executed only through its in-memory text construction, before any source output write. Experiment inputs were never edited.

One reviewer tool attempt failed before Python execution because the PATH `python` alias reported `No pyvenv.cfg file`. The reviewer then selected the already-installed absolute CPython interpreter. No installation, environment change, experiment restart or source workaround occurred.

No NN/model call, third-party numerical/model import, dataset GT/image load, SSH/network call, GPU/progress query, installation, input edit or sub-agent launch was performed.

## Full supplied-input SHA-256

| File | Bytes | SHA-256 |
|---|---:|---|
| `replay_m125_selected_E6_observation.py` | 8016 | `be2239091b279bebc614a4b77d31887ba4c0f6bd125862e194c502fb1a067cac` |
| `analyze_m125_selected_E6_observation.py` | 8514 | `81081356adb1af5e5429883aa22a3fcba519204133f70001c013b7e195f5bfe4` |
| `PLAN.json` | 1852 | `c195814e0223003c00f8d7eadd9e7a11d5185e4cd37da38d56f2fe47c047e318` |
| `prepare_M125_split_replay_source_20261015.py` | 6898 | `3546b98d3934712fb9e85f3a6cf7276f6967162097d0f9b0e4b47e8ce96ee8ec` |
| `replay_m124_selected_best_cdtb_onsets.py` | 13622 | `2b88ddffb9c888ef8d1db8dfff44e1c44cc50086a1bc77b2e99c3c7b984d0538` |
| `M124_selected_best_onset_replay_source_review_R2_20261012.json` | 66621 | `607cfef9a38f7f3a61d44a8ec1b1f80da5854742a54c388ef1732bdff3c8cdcb` |
| `direct_ssh_rgbd.md` | 1341 | `0f6f34aabc512a9da057def850aa543b84d904c9c6fbedc36cdaf8605e1da72e` |
| `SOURCE_REVIEW_REQUEST.txt` | 1886 | `e928f2771b6702ddaa4f882ae90f28bb93b39d12b470449daedca149001db0c8` |

The seven files listed in the request and the request itself are included above. The JSON report additionally contains full absolute paths and hashes for every supporting source, archived metadata/probe and policy file consumed by this audit.
