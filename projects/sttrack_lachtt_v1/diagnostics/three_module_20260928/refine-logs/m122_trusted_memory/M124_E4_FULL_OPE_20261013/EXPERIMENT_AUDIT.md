# M124 epoch-04 captured OPE integrity audit — R1

**Overall verdict: WARN.** Saved bytes, result arithmetic, declared paired bindings and the inspected source path are consistent. Raw GT, predictions, checkpoints and runtime bindings are unavailable in this intake. This is not independently reproduced performance acceptance.

**Reviewer:** gpt-6-astra / max, fresh context (`fork_turns=none`). **review_independence=same-family; acceptance_status=provisional.** Date: 2026-10-13. No reviewer retry was used.

The review performed local source/artifact inspection and CPU standard-library hashing, parsing and arithmetic. It made no SSH/server/GPU calls, imported no torch, executed no model, deserialized no checkpoints and changed no pipeline source. Writes are limited to these reports and the received-task trace requested by the parent.

The observation is from **2026-10-13T02:11:23.216142+00:00**, or 10:11:23 Asia/Shanghai. Later live state is outside this report. The saved request trace confirms model/effort/fork metadata, but its prompt field is opaque; its plaintext could not be compared with the received task. The received task is separately preserved verbatim.

## Evidence notation

- `C/` = `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch04_captured_OPE_20261013/`
- `S/` = `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/`
- `D/` = `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/`
- `O` = `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_thirtyfirst_hourly_original_observation_stdout_20261013.json`; this is one line, so nested keys identify its embedded payloads.

## Deterministic findings

All **18 captured payloads, 379,594 bytes**, match `captured_manifest.json` SHA-256 values and byte-match the corresponding observer strings. All 59 embedded observation payload values were parsed. These matches verify capture fidelity within one capture chain; they are not independent witnesses of experiment execution.

| Epoch-04 dataset | Precision (%) | Recall (%) | F (%) | Threshold | Declared sequences / frames |
|---|---:|---:|---:|---:|---:|
| DepthTrack Test | 67.191841838381 | 63.253862529058 | 65.163411050277 | 0.037617 | 50 / 76373 |
| CDTB | 74.416472197621 | 69.161544989019 | 71.692843941834 | 0.337623 | 80 / 101956 |

Both metric records declare final `8f26f42d663091171b27870d7a77e845c10daeb27fde995c2e4490fe6e6af470` and bundle `d41bad6159ed4d12d87faeeb3b921fe6be3e78ef3b67e626f295af025a9e0f78`. The training result declares the same final. Their distinct plan/receipt hashes are appropriate for distinct datasets. Both local metric source copies are byte-identical and match recorded hash `05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc`. The local trainer and joint tracker bytes match their hashes in the training result. The actual final, bundle, plans, prediction receipts and raw trajectories are absent here.

Evidence: `C/captured_manifest.json:2-19`; both `C/epochs/epoch-04/evaluation/*/predictions/metrics.json:2-18`; `C/epochs/epoch-04/fit/result.json:46-53`; `O:1`.

## Independently recomputed selection

| Epoch | DepthTrack F (%) | CDTB F (%) | Equal mean F (%) | Dual decline | Streak | Best epoch |
|---|---:|---:|---:|---|---:|---:|
| 1 | 64.449641059332 | 72.633539630522 | 68.541590344927 | false | 0 | 1 |
| 2 | 64.992905079815 | 71.453987841364 | 68.223446460589 | false | 0 | 1 |
| 3 | 64.711280784974 | 72.055640887358 | 68.383460836166 | false | 0 | 1 |
| 4 | 65.163411050277 | 71.692843941834 | 68.428127496055 | false | 0 | 1 |

Recalculation used the embedded dataset metric values, then checked every saved `previous_F_percent`, `F_percent`, `score`, `dual_decline`, streak, stop and replacement field. Every field agrees. E4 versus E3 is +0.452130265303 / −0.362796945524 pp, so the strict dual-decline condition is false. E4 mean is 0.113462848872 pp below E1. `best.json` and `current_training_state.json` agree: E1 best-to-date, four completed epochs, streak zero, early stop false, next resume E4/latest.

The starting baseline mean is **69.12734926869535**, above E1 by 0.585758923768 pp. The controller initializes `best=None`; “best” selects among trained epochs and does not include that baseline. Comparisons for stopping use the immediately previous pair, not the best pair. No tie occurs; strict greater-than replacement would retain the earliest tie.

Evidence: `S/m124_epochwise_selection.py:4-10`; `S/run_m124_epochwise_suite.py:24-25,47-76`; `C/epoch_history.json:37-49,86-98,135-147,184-196`; `C/best.json:2-12`; `C/current_training_state.json:2-20`.

## Coverage and stage boundary

All 608 cumulative training records were parsed. Each epoch contains 152 unique sequences, indices 0–151 in the same order, 219,802 calls, 203,376 A+B supervised frames and 203,224 C supervised frames. Every cumulative call count reconciles. Each epoch adds 6,807 A+B and 6,807 C updates; totals are 879,208 calls and 27,228 updates per group. Template-write counts are 1,147 / 1,128 / 1,102 / 1,166. All numeric values and count bounds were checked.

The E4 log has all 152 rows and exactly matches its cumulative-result slice. E1 has 115 continuation log rows plus the carried 37-sequence prefix, which accounts for 54,906 calls and 1,711 updates. E2–E4 declared initial model-state hashes equal the preceding final-state hashes. These are independently reconciled records, not independently verified tensor/optimizer execution.

Training and both OPE-track captured exit codes and process receipts are zero. E4 training finished at 2026-10-12T23:48:25.352931+00:00; tracking receipts finish at 2026-10-13T01:44:08.157944+00:00 and 02:04:25.864013+00:00. The observer contains E5 launch at 02:04:37.902228+00:00 and one saved sequence: 2,575 calls and 81 new updates, cumulative 881,783 / 27,309. There is no completed E5, M124 VOT or complete-suite root result in the intake. Current state is the last completed epoch, not evidence of a stopped run.

Evidence: `C/epochs/epoch-04/fit/result.json:2-17,58-89,7390-7403,9806-9819`; `C/epochs/epoch-04/fit/sequence_log.jsonl:1-152`; `C/epochs/epoch-04/train.receipt.json:31-34`; both track receipt files `:13-16`; `O:1`, keys `files[epochs/epoch-05/train.launch.json]` and `files[epochs/epoch-05/fit/sequence_log.jsonl]`.

## A. GT provenance — WARN

The inspected OPE source uses dataset `groundtruth.txt`, not model outputs, as benchmark truth. Tracking receives initial boxes and RGB/depth frames; subsequent actions come from the tracker. Analysis checks saved prediction hashes and dataset GT hashes before calling the metric. The metric separately loads GT, predicted boxes and confidences, checks their lengths and handles invalid GT as absent regions. Raw GT provenance and bytes cannot be authenticated from this intake.

Training calls `tracker.step` before forming losses from Train152 GT. A+B targets and C selected-box IoU use real GT boxes. Native response margins are a preservation regularizer, not evaluation truth. Initial visual references and current/past C features are model inputs. Historical teacher current/future labels are GT-derived IoU and write-minus-keep IoU after rollouts; fixed-teacher MSE must remain separate from OPE and from changed-policy evaluation.

This is a project DepthTrack/VOT-style PR wrapper using VOT overlaps. The source and its local hash do not establish independent official-protocol conformance or the dependency package actually imported on the server.

Evidence: `S/run_m124_epochwise_official.py:12-42,45-61`; `D/native_ope/source_snapshots/depthtrack_pr.py:95-111`; `S/train_m122_full_causal.py:14-31`; `S/train_m124_joint_epoch.py:113-148`; `S/train_m121_dense_target.py:39-64`; `S/template_write_features.py:11-40`; `S/run_template_write_teacher.py:129-156`; `S/train_template_write_C.py:149-180`.

## B. Score normalization — PASS, bounded to source and arithmetic

No reported performance is divided by its own maximum, minimum or mean. Confidence-derived thresholds define a PR sweep. Precision is selected-frame mean overlap; recall is overlap sum divided by GT-visible frame count. Sequence curves are macro-averaged, harmonic F is maximized, and percentages multiply fractions by 100. All eight observed epoch/dataset metric records have consistent P/R/F arithmetic and percentage conversion. Max-F threshold selection uses evaluation GT and must be described as that PR protocol, not a preselected deployment threshold.

Feature/query `F.normalize` and LayerNorm, model confidence `probability/native`, and normalized preservation margins in the training loss do not normalize the reported benchmark metric. This PASS does not attest the unavailable raw trajectories or package conformance.

Evidence: `D/native_ope/source_snapshots/depthtrack_pr.py:25-37,117-153`; `S/dense_target_decoder.py:51-59,74-85,97-121`; `S/train_m121_dense_target.py:55-64`; both epoch-04 metric files `:4-12`.

## C. Results, hashes, binding, coverage and status — WARN

The deterministic capture checks and numerical/ledger reconciliation above pass. Identical final/bundle declarations are established. Actual loading of those checkpoint bytes and actual per-sequence coverage are not. The saved `bbox_file_count` values are counts reported by the observer; the 130 bbox files were not provided. Analyze launch/exit/log records, prediction `receipt.json`, GT, raw confidence files, image dimensions, plans/bundle/gate, final/latest/best tensors and the human text bank are absent from the captured inventory.

Captured frozen-state, roundtrip, no-GT and parameter-change fields are execution assertions, not independent measurements by this reviewer. Parameter-change counts are relative to the warm final in source, not per-epoch changes or evidence of causal gains. Other runtime source pins cannot be bound without the actual gate/bundle.

The tracker’s last entry, line 195, predates this observation and still has E4 OPE pending. It is a stale historical status, not a contradiction of the later completed captures. Older CurrentC COMPLETE_ALL9 entries and original Future R4/R5 NOT_RUN entries must not be relabeled as M124 completion.

Evidence: `C/captured_manifest.json:2-19`; both epoch-04 metric files `:14-18`; `C/epochs/epoch-04/fit/result.json:46-89`; `S/train_m124_joint_epoch.py:78,203-227`; `S/m124_epochwise_official_runtime.py:9-41`; `S/refine-logs/m122_trusted_memory/EXPERIMENT_TRACKER.md:7-14,52,193-195`.

## D. Metric call path — WARN; static connection verified

`run_m124_epochwise_suite.main` launches both `ope_track` subprocesses, waits for their completion, then invokes `ope_analyze` with `gate.metric_python`. The shared subprocess runner records exit codes, sets `PYTHONOPTIMIZE=0`, and asserts success without automatic retries. CLI dispatch reaches `ope_analyze`, which checks the plan, prediction receipt and hashes, dynamically imports the pinned metric, calls `evaluate_depthtrack_results`, and writes its values to `metrics.json`.

The metric calls `_load_rows`, `_vot_overlaps` and `_determine_thresholds`; its result keys match the captures. `format_metrics` is an unused serialization helper, not a missing scientific metric. VOT is deliberately deferred until selection ends. Missing analyze receipts, raw prediction receipts and the actual dependency environment prevent an executed-call attestation. A supplemental local VOT overlap snapshot was inspected but is not bound to this captured process.

Evidence: `S/run_m124_epochwise_suite.py:34-61,77-103`; `S/run_m123_joint_full_suite.py:14-32`; `S/run_m124_epochwise_official.py:45-61,81-84`; `D/native_ope/source_snapshots/depthtrack_pr.py:63-158`; `D/native_vot_full127/source_snapshot/vot/region/__init__.py:145-146`; `D/native_vot_full127/source_snapshot/vot/region/raster.py:394-416`.

## E. Scope, seed and input/selection protocol — WARN

This is seed 2027 with sequential epoch candidates, not four independent runs. M124 jointly updates 450,888 A+B+C parameters on all Train152 sequences, with frozen STTrack/CLIP observations and warm initialization. It is not a from-scratch update of every backbone weight.

Test50/CDTB80 F selects checkpoints and stopping; this is explicit in source and result. Gradient exclusion does not make these untouched held-out tests. E1 is the best trained epoch to date, not proof of beating the captured pre-M124 baseline. No seed variance or independent A/B/C contribution is established.

Human text consists of a confirmed category and the first four stable attributes. Separate uncertain/later-frame fields are excluded by preparation, but human review explicitly used multiframe/video aids. RGB bytes and legal initialization bbox form the lookup key; text stays fixed online. Thus this is human-assisted text input, not automatic captions or strictly first-frame-only human knowledge. Raw human reviews and bank are not in the intake.

The legacy M123 handoff carries 37 complete sequences, repeats the partial sequence from its first frame and explicitly reports missing original RNG / `legacy_handoff_RNG_exact=false`. Source resets seed for that legacy boundary and saves subsequent RNG. Exact uninterrupted replay or a fresh epoch-1 run is unsupported. Historical fixed-teacher fit120/dev32 diagnostics remain separate from M124 full152 joint training. Complete M124 VOT/full nine metrics and final goal success remain unestablished.

Evidence: `C/epochs/epoch-04/fit/result.json:3-17,58-89`; `S/full_dense_tracker.py:31-33`; `S/train_m124_joint_epoch.py:61-96,186-198`; `S/run_m124_epochwise_suite.py:24-27,47-76`; `S/prepare_m122_external_human_labels.py:24-48`; `S/bind_m122_official_initializations.py:28-54`; `S/m124_epochwise_official_runtime.py:52-74`; `C/actual_launch.json:17-32`; `S/refine-logs/m122_trusted_memory/EXPERIMENT_TRACKER.md:7-14,110-113`.

## F. Evaluation type — WARN; source classification `real_gt`

Epoch-04 DepthTrack/CDTB OPE is `real_gt` by the inspected source path, qualified as human-text-assisted and externally selected. Its raw realization remains unverified. Learned references, teacher response regularizers and human text are not benchmark reference boxes. Human input does not make the metric `human_eval`. Historical fixed-teacher MSE is an auxiliary diagnostic on dataset-GT-derived labels, not benchmark performance. M124 VOT is pending and receives no completed evaluation claim here.

Evidence: `D/native_ope/source_snapshots/depthtrack_pr.py:95-111,117-153`; `S/train_template_write_C.py:149-180`; `S/run_template_write_teacher.py:139-171`; `S/prepare_m122_external_human_labels.py:43-48`; `S/run_m124_epochwise_suite.py:77-103`.

## Blockers and action items

No demonstrated integrity contradiction blocks a carefully qualified account of captured progress. These four limits block stronger acceptance:

1. **Raw metric acceptance unavailable:** obtain the existing bbox/confidence files, prediction receipts, GT/cases with provenance hashes and image-size evidence; recount and recompute all 50/76373 and 80/101956 observations using the recorded metric dependency environment. This can be a CPU check without NN reruns.
2. **Runtime binding/tensor acceptance unavailable:** obtain existing plans, bundle, gate, bank/binding/reviews, final/latest/best bytes or independently captured attestations, plus analysis invocation/environment evidence. Match the actual chain and keep captured booleans distinct from tested tensors.
3. **Protocol qualifiers required:** disclose seed 2027, warm-started full152 adapter training, Test50/CDTB checkpoint selection, multiframe-assisted human text and non-exact legacy RNG. Untouched-test, automatic-text, multi-seed, exact-resume and unique-module-gain claims are unsupported.
4. **Suite completion pending:** report four completed training epochs and paired OPEs, E1 best-to-date, one E5 sequence captured, and VOT/full-suite completion pending. No new launch, retry or restart is requested by this audit.

The supported claim is that these captured records report the listed scores and agree internally. Independently authenticated model execution/performance and final goal success remain unverified. Only B is PASS, strictly for inspected source and arithmetic; missing runtime checks were not converted to PASS.

## Input hashes

The matching JSON records absolute paths, saved-byte SHA-256, sizes and read scopes. Required source files were read completely. Large result/observer files were fully loaded and every repeated JSON/JSONL record parsed and checked. The supplemental VOT raster snapshot was inspected only at lines 281–416; its full bytes were hashed without asserting runtime identity. Hashes below use saved bytes without newline normalization.

| Input relative to repository root | Bytes | SHA-256 |
|---|---:|---|
| `.aris/m122_full_causal_20261007/M124_epoch04_captured_OPE_20261013/captured_manifest.json` | 2016 | `f3e9042c8789322a1b22dfe5d8b45f732b85b1a5e4e15062653e1b0c7cc1b472` |
| `.aris/m122_full_causal_20261007/M124_epoch04_captured_OPE_20261013/actual_launch.json` | 2636 | `8088bbd8cf955924328aca662b179506ed961dad872d114d5128260650b74ed7` |
| `.aris/m122_full_causal_20261007/M124_epoch04_captured_OPE_20261013/controller.supervisor.launch.json` | 377 | `13d3ae397b662acb4eabb0b786d4ad12b46c6ef7fe71d17f4687594093f7352c` |
| `.aris/m122_full_causal_20261007/M124_epoch04_captured_OPE_20261013/current_training_state.json` | 806 | `0dbc7cace9f292f701506035dd436704e559be39dd36d866039d400e1b60752e` |
| `.aris/m122_full_causal_20261007/M124_epoch04_captured_OPE_20261013/best.json` | 505 | `f61469e0addcdff3f83c7776a81935bb2fbd11da349aefcb5e21ca17e1a884c3` |
| `.aris/m122_full_causal_20261007/M124_epoch04_captured_OPE_20261013/epoch_history.json` | 6796 | `74a9fe69bba19ed273e8c16294eda6b93957696ecda2505cbe38bb23892278da` |
| `.aris/m122_full_causal_20261007/M124_epoch04_captured_OPE_20261013/epochs/epoch-04/train.launch.json` | 1220 | `a61b716d5c0d481a3c35e29a0283e034ac7aa4e2a5fbd97d51f2a476bc61e367` |
| `.aris/m122_full_causal_20261007/M124_epoch04_captured_OPE_20261013/epochs/epoch-04/train.exit` | 2 | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `.aris/m122_full_causal_20261007/M124_epoch04_captured_OPE_20261013/epochs/epoch-04/train.receipt.json` | 1291 | `4156358a380ea716bf71e05609a51a0d2b92192760f26a37119bf4af40e09bf6` |
| `.aris/m122_full_causal_20261007/M124_epoch04_captured_OPE_20261013/epochs/epoch-04/fit/result.json` | 303115 | `2ad990415d57b9a35e58010b2ca3d7c406adf6453477a5475ecdd493b33e19b7` |
| `.aris/m122_full_causal_20261007/M124_epoch04_captured_OPE_20261013/epochs/epoch-04/fit/sequence_log.jsonl` | 60269 | `30a1027380b2949c61d02b27984f1d74da5b034f4c41fe6d7d57347e22e72084` |
| `.aris/m122_full_causal_20261007/M124_epoch04_captured_OPE_20261013/epochs/epoch-04/depthtrack_test_track.exit` | 2 | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `.aris/m122_full_causal_20261007/M124_epoch04_captured_OPE_20261013/epochs/epoch-04/depthtrack_test_track.receipt.json` | 507 | `4a74c6b3e53b0739c1bf38b59b443cf7bd830e96f312eff3dbe3d1a371be8d93` |
| `.aris/m122_full_causal_20261007/M124_epoch04_captured_OPE_20261013/epochs/epoch-04/cdtb_track.exit` | 2 | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `.aris/m122_full_causal_20261007/M124_epoch04_captured_OPE_20261013/epochs/epoch-04/cdtb_track.receipt.json` | 485 | `461ad7b804540b2d7f520897adfbaf06b0db006ff051772fb5fba1c9f83e901d` |
| `.aris/m122_full_causal_20261007/M124_epoch04_captured_OPE_20261013/epochs/epoch-04/evaluation/depthtrack_test/predictions/bbox_file_count` | 2 | `1a6562590ef19d1045d06c4055742d38288e9e6dcd71ccde5cee80f1d5a774eb` |
| `.aris/m122_full_causal_20261007/M124_epoch04_captured_OPE_20261013/epochs/epoch-04/evaluation/depthtrack_test/predictions/metrics.json` | 788 | `bc617128bbe82ae54df361ddf836f959bd2c0c657f9e700d9692310a65645803` |
| `.aris/m122_full_causal_20261007/M124_epoch04_captured_OPE_20261013/epochs/epoch-04/evaluation/cdtb/predictions/bbox_file_count` | 2 | `48449a14a4ff7d79bb7a1b6f3d488eba397c36ef25634c111b49baf362511afc` |
| `.aris/m122_full_causal_20261007/M124_epoch04_captured_OPE_20261013/epochs/epoch-04/evaluation/cdtb/predictions/metrics.json` | 789 | `1b270134e1a1840cc61eb85517d88ddbcee97a30b3320d4b1752d5b26c16c4ef` |
| `.aris/m122_full_causal_20261007/M124_thirtyfirst_hourly_original_observation_stdout_20261013.json` | 1118489 | `00abb8809852e884ebfede96b1cb0f0788e6e328d64167ec627ef9404c5cb9ce` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m124_epochwise_official.py` | 6281 | `e8ef46a5c1f1e1debecfe1c0b315ba7c766f49907982c4cc20360a0bcedf916a` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/m124_epochwise_official_runtime.py` | 4419 | `60de3f27538c76bcba1178d5b24f9064c587bda9ba564a60aae179cbcc023e48` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/m122_official_runtime.py` | 2986 | `25aa67000cf581c454b160c524c8dcf8f55ec30d227b387990cc03ed18be3157` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/bind_m122_official_initializations.py` | 3945 | `4023d263cde4e10436f2f2dc78247d2b8875df5316352935ea85b2a1ad71f69a` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/prepare_m124_epochwise_evaluation.py` | 7981 | `ad05b5029b6e27e5c163c8ffa190d1db83312f011331f5d8dd426e89192ee8cb` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m124_epochwise_suite.py` | 6989 | `63924cc9f7f5e38ab605cb8f40dddf083321bee12248e71a7e8516f5a2d17dfc` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/m124_epochwise_selection.py` | 560 | `24d73496437a60dea724bca14b51c8e8f0742657fb242ef4c793ea37ee9c4a24` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/train_m124_joint_epoch.py` | 15536 | `6bc43f9f113c3a648e9a674ad3f7f18d1114e4f84880edf18fa61d40e804081c` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/full_dense_tracker.py` | 7693 | `0aee7de2932d7294976e2ed512218cd58d281e24bf3fb34b6267167470aa1778` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/dense_target_decoder.py` | 7739 | `055d9ddffe4e332fa9a04fca93a6ec301d031a35b198f93a26809d7baa7104f2` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/trusted_template_tracker.py` | 4173 | `1dd4873192b05f6921cdf729a525ed684ed96a18bfdfc48673cda14d50352e0d` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/template_write_C.py` | 966 | `4bf01cd83d6a7ea2a2e8ced6668cecde38f476af61c4af4eeae0db628afb9fc1` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/train_template_write_C.py` | 11018 | `7089754fc017f45c9e80d53ced96d98e624b63a15b5acb808fc5de9f35a9f88e` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/refine-logs/m122_trusted_memory/EXPERIMENT_TRACKER.md` | 39769 | `1c842dcf6f7e64a2d7a2b7fa27d40488a8905b10105ba64772215d2f49165151` |
| `projects/sttrack_lachtt_v1/diagnostics/native_ope/source_snapshots/depthtrack_pr.py` | 7143 | `05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc` |
| `projects/sttrack_lachtt_v1/diagnostics/m82_m84_development_pr/depthtrack_pr.py` | 7143 | `05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/m123_joint_full_tracker.py` | 5028 | `53b1e0ae9f054b62cf4f7c13d77400453d616f8b62e62b406014f7075ef529a7` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/template_write_features.py` | 2028 | `a40ef7c814c1e6b2d6eea779afa62478714c4d5aab6c0c0274f8c72657f2105b` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m123_joint_full_suite.py` | 4821 | `47639fc6802a6c245835196188380d450d65471f647f62236afc18662e4d571c` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/train_m122_full_causal.py` | 11546 | `ffa40a08d167d9a7f185cf83ecf0851e9c5ed548c2b1280bb969d6b8c689f891` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/train_m121_dense_target.py` | 13099 | `40d00f84c9f2220deb6f557a0e516f4385922c86d276da1b5d3b17f8891502ee` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/prepare_m122_evaluation_suite.py` | 10987 | `1428b3488d37cbbb71ae66eeca896b553c14d4156c9ab65657c59f741d321cea` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/dense_region_decoder.py` | 7272 | `66b9317529f741aba818aa9e9feba7fcebdb54c285f0bd9184f98f4b1f202e7b` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/dense_region_encoding.py` | 3120 | `71715a497147eb917e6a93ba6916ebbe01bc4a859cb275401289df2706627f33` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/region_patch_evidence.py` | 2237 | `85b7f245dc79dd5ed8c04b075854cbd8b0863bf5428a01c49b4e4951f44af7e1` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/instance_ab_prototype.py` | 7951 | `4485a7ca29a90fa13cc868c32e4e4083c351cfba7b318d341c0165c32a886e4f` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/m121_dense_inputs.py` | 7918 | `89651f43d40f1b13e16ab586d6c0aaa19e3edfcf91f59d3286f74cf093ef2d85` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/prepare_m122_external_human_labels.py` | 4021 | `0c1bab71df8c5c8f7dff225a4ccfe6b6138ab0033c695bc224e6b9fb9e354bf4` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/encode_m122_official_human_text.py` | 2617 | `6f50da069c5a4a7e902a05dc1d3227032bb67a9952f9687ca4a95279fab3348f` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_template_write_teacher.py` | 11581 | `0db429a83a0f7627a9d486b2658fe07d8ce1571932347a777df0c672633ef169` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_train_states.py` | 4469 | `2439a8d0cdb7a7d6c35bbf68aaacd28850bfb8db2211f07eb9e9e7dac66b6def` |
| `projects/sttrack_lachtt_v1/diagnostics/native_vot_full127/source_snapshot/vot/region/__init__.py` | 3728 | `111e19a38ceae96ed523d927badf0227e68634a18e682d2fdac94f179428f403` |
| `projects/sttrack_lachtt_v1/diagnostics/native_vot_full127/source_snapshot/vot/region/raster.py` | 15220 | `ba6797ae9013cf31700eb4671d73e50480fcda62a65adb529fff3c4d777a23b9` |
| `.aris/traces/experiment-audit/2026-10-13_m124_epoch04_captured_ope/001-captured-epoch04-OPE-review.request.json` | 12336 | `5ac0cb5c6a2d2b3d919ef666b3d90710877d6e81f379c6447e9525e41a3a93e8` |
