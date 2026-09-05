# M52 pre-training experiment audit

Date: 2026-09-05. Auditor: GPT-5.5 xhigh, independent context, GPT-family Type-A advisory review. This is not a cross-family acquittal. The following is the reviewer response, preserved verbatim.

Project root audited: `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1`

Overall verdict: **WARN**. The M52 design and code mostly preserve the intended integrity boundaries: collection avoids GT, labels come from dataset GT after sealed feature checks, paired control/mixed training is wired with matched initialization/order, and recursive metrics are computed against `groundtruth.txt` rather than prediction-derived references. The warning is because the local M52 evidence is still launch/contract/pre-training only: no local `collection_receipt.json`, `data_audit.json`, arm `training_result.json`, `runtime_contract.json`, shard receipts, `recursive_result.json`, or `per_sequence.csv` exists in `diagnostics/m52`. Also, the requested local source file `overlay/lib/test/tracker/sttrack.py` is missing, so the base recursive tracker implementation cannot be independently inspected from this local source bundle.

This is an independent-context GPT-family Type-A review, not a cross-family acquittal.

**A. Ground-Truth Provenance: PASS, with scope qualification**

Evidence:
- `overlay/tools/prepare_sttrack_m44.py:28-33` binds labels to `/root/autodl-tmp/depthtrack/train/sequences` and loads each sequence `groundtruth.txt`.
- `overlay/tools/prepare_sttrack_m44.py:54-57` writes `training_labels.json` from `gt[f]` and `gt[f-1]`, using `None` when GT is invalid/unavailable.
- `overlay/tools/collect_sttrack_m52.py:164-168` writes collection receipt with `labels_opened=False`, `optimizer_steps=0`.
- `overlay/tools/train_sttrack_m52.py:103-106` explicitly verifies sealed feature/trajectory receipts before reading `training_labels.json`.
- `overlay/tools/train_sttrack_m52.py:119-127` computes current/previous candidate IoUs against label GT; candidate 0 is overwritten with `public_bbox`, i.e. the native candidate-zero/default box.
- `overlay/tools/run_sttrack_m52.py:79-83` loads development `groundtruth.txt` and recomputes native baseline metrics before comparing.
- `overlay/tools/run_sttrack_m52.py:99-115` computes arm metrics from recursive boxes against GT via `independent_overlap`.
- `diagnostics/m52/EXPERIMENT_PLAN.md:15` says collection reads manifest/images only before sealed audit, then reads existing fitting labels.
- `diagnostics/m52/EXPERIMENT_PLAN.md:25` says each view uses its own candidate boxes and the same physical-frame GT, and mixed must use actual `previous_choice`.

Details:
- Labels/metrics trace to DepthTrack Train dataset GT, not model-output-derived pseudo-GT.
- Prediction-derived traces are used for state generation, event selection, and consistency checks. That is not fake GT, but it makes the evaluation a Train/development diagnostic rather than an official public benchmark.
- The implementation distinguishes candidate-zero/default supervision from selected-box logging: `collect_sttrack_m52.py:137-140` stores `public_bbox` and `selected_bbox` separately; `train_sttrack_m52.py:119-127` supervises against the current/previous candidate set with candidate 0 bound to `public_bbox`, not blindly to the actually selected output box.
- Official benchmark claims are not supported by this evaluator. The M44 spec says public release requires later frozen low22 and three-dataset validation gates (`diagnostics/m44/spec.json:101`), while the M52 launch report says there are no M52 weights or performance metrics yet (`diagnostics/m52/LAUNCH_REPORT.md:5`).

**B. Score Normalization: PASS**

Evidence:
- `overlay/tools/train_sttrack_m42.py:19-23` defines IoU as intersection over union with GT.
- `overlay/tools/audit_sttrack_m43.py:17-36` computes recursive `mean_iou` as `sum(valid IoU) / len(valid)`, with low-IoU frame and H10 episode counts from valid GT frames.
- `overlay/tools/train_sttrack_m52.py:236-252` static diagnostics select logits, gather GT IoUs, and report event mean/correct/rescue/break counts.
- `overlay/tools/run_sttrack_m52.py:113-115` recursive aggregate mean is `iou_sum / valid_frames`.
- `overlay/tools/run_sttrack_m52.py:120-128` gates compare mixed/control/default metrics directly; no metric is divided by the model’s own max/min/mean output.

Details:
- I found no self-referential normalization.
- Static training diagnostics count unavailable-label rows as zero IoU through `train_sttrack_m52.py:121-123`, which is acceptable for the stated training diagnostic, but it is not an official benchmark denominator.
- Recursive evaluation uses valid GT frames as the denominator.

**C. Result Existence: WARN**

Evidence:
- `diagnostics/m52/launch.json:13-14` records `optimizer_steps: 0` and scope as fixed-policy data collection only.
- `diagnostics/m52/LAUNCH_REPORT.md:5` states there is no M52 new weight or performance metric.
- `diagnostics/m52/MANIFEST.md:6-7` states collection is running, no M52 training/performance result yet, and large features/traces remain in the remote root with hashes in completion receipts.
- `diagnostics/m52/run.sh:5-8` launches only collection and writes `collection.exit`/`controller.exit`.
- `diagnostics/m52/execution_binding.json:6-7` sets `auto_training=false` and `auto_public_evaluation=false`.
- `overlay/tools/collect_sttrack_m52.py:175` would write `collection_receipt.json`; it is not present in the local `diagnostics/m52` tree.
- `overlay/tools/train_sttrack_m52.py:187` would write `data_audit.json`; `overlay/tools/train_sttrack_m52.py:271` would write each arm `training_result.json`; neither exists locally.
- `overlay/tools/run_sttrack_m52.py:143-145` would write `recursive_result.json` and `per_sequence.csv`; neither exists locally.
- Existing M45 static result is complete: `diagnostics/m45/geometry_result.json:8-12` says status complete, 960 optimizer steps, 20 epochs; `diagnostics/m45/geometry_result.json:235-247` and `15741-15753` contain fit/development static metrics.

Details:
- The local M52 artifacts support launch and 240-frame contract only, not completed collection, paired training, runtime loader verification, or recursive results.
- Remote evidence may exist under `/root/autodl-tmp/sttrack_m52_policy_state_augmentation_v1_20260905`, but I did not access that remote root in this bounded local read-only audit. Completion cannot be invented from the prepared scripts.

**D. Dead Code And Execution: WARN**

Evidence:
- The full pipeline is wired in `diagnostics/m52/run_training.sh:16-24`: data audit, control training, mixed training, runtime contract, two recursive shards per arm, then analysis.
- `overlay/tools/train_sttrack_m52.py:183-191` calls `load_data`, writes `data_audit.json` in audit mode, and requires that same audit before training.
- `overlay/tools/train_sttrack_m52.py:198-205` fresh-seeds the model and constructs `control` as original fit duplicated or `mixed` as original fit plus policy fit.
- `overlay/tools/train_sttrack_m52.py:211-220` passes `previous_choice` into the model and trains with `supervised_loss`.
- `overlay/tools/run_sttrack_m52.py:52-63` analysis requires both recursive shard exits/receipts before aggregating.
- `overlay/tools/run_sttrack_m52.py:126-135` separately computes `primary_pass`, `extra_training_control_pass`, and `advancing_arm`, distinguishing data-effect claim from model advancement.
- `diagnostics/m52/PRE_TRAINING_NOTE.md:5-7` states the same distinction: mixed supports state-data evidence only if it beats control under the fixed gates; otherwise a passing control is only extra-training benefit.
- Local source completeness defect: `overlay/lib/test/tracker/sttrack_candidate_set.py:3` imports `lib.test.tracker.sttrack.STTrack`, and `overlay/tools/collect_sttrack_m44.py:24` imports the same base tracker, but the requested local file `overlay/lib/test/tracker/sttrack.py` is missing. `diagnostics/m44/spec.json:7-9` hashes a remote `lib/test/tracker/sttrack.py`, so remote execution may still have had the file; the local overlay bundle cannot verify it.

Details:
- I found the metric functions wired into the prepared pipeline, not dead.
- I did not find a definite metric logic defect in the M52 code paths read.
- Execution evidence is incomplete: the code is prepared, but local M52 result outputs are absent.
- The missing local `sttrack.py` prevents independent inspection of base tracker initialization/reset, native track behavior, and any state carryover assumptions. It would be a local runtime import failure if the overlay tree were used directly as the source tree; it is not proof that the remote run fails, because the scripts point at `/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1`.

**E. Scope: WARN**

Evidence:
- `diagnostics/m52/spec.json:10-12` fixes 1,511 fit events, 63 fit sequences, and 93,362 frames through final events.
- `diagnostics/m52/spec.json:15-30` defines two paired arms: control duplicates original 1,511 records; mixed uses original 1,511 plus the same physical events from M45 own-state trajectories.
- `diagnostics/m52/spec.json:22` limits development to original 590 static events and full 22 recursive sequences.
- `diagnostics/m52/EXPERIMENT_PLAN.md:23-25` says both arms use fresh seed 2026, 20 epochs, batch 32, 1,900 steps, and duplicated views do not mean independent frames/sequences.
- `overlay/tools/train_sttrack_m52.py:146-148` asserts 1,511 original fit, 1,511 policy fit, 590 development, matching physical keys, and sequence-disjoint fit/development.
- `overlay/tools/train_sttrack_m52.py:196-205` uses a single seed 2026 and creates 3,022 logical samples per arm.
- `overlay/tools/run_sttrack_m52.py:35-36` asserts 22 development sequences across the two recursive shards.
- `overlay/tools/run_sttrack_m52.py:136-142` reports 33,130 frames per arm, 66,260 total frames, and describes the scope as repeated Train development.

Details:
- Scope is suitable for an internal Train/development diagnostic and paired ablation.
- It is not enough for strong public benchmark or generalization claims: one seed, one fixed policy checkpoint, 63 Train fit sequences, 22 Train development sequences, duplicated paired views rather than 3,022 independent physical events.
- Train/development separation is explicitly enforced by sequence split, but these are still DepthTrack Train folds with prior experiment exposure, not fresh public test data.

**F. Evaluation Type Classification: PASS for classification, WARN for claim strength**

Evidence:
- Collection/contract: `diagnostics/m52/contract.json:25-32` reports 238 events, 240 frames, labels not opened, optimizer steps 0, observed/plain recursive state exact.
- Static training diagnostics: `overlay/tools/train_sttrack_m52.py:105-126` uses `training_labels.json` derived from dataset GT, and `overlay/tools/train_sttrack_m52.py:245-251` reports event-level selected/default IoU rows.
- Recursive development: `overlay/tools/run_sttrack_m52.py:79-83` and `99-115` use DepthTrack Train `groundtruth.txt` and independent overlap metrics.
- Public benchmark: `diagnostics/m52/spec.json:44` has `public_automatic_launch=false`; `diagnostics/m52/LAUNCH_REPORT.md:5` says no M52 performance metric.

Classification:
- M52 collection contract: `self_supervised_proxy` / implementation-consistency check, not a performance evaluation.
- M52 data audit and static fit/development diagnostics: `real_gt`, but sampled DepthTrack Train diagnostic, not official benchmark.
- M52 recursive development evaluator: `real_gt`, custom DepthTrack Train development evaluation.
- M45 `geometry_result.json`: `real_gt` static Train diagnostic; complete recursion remains decisive per `diagnostics/m45/geometry_result.json:21817`.
- Public VOT / DepthTrack Test / CDTB claims: not executed in the local M52 evidence; cannot be classified as completed evaluations.

**Prioritized Action Items**

1. Publish or provide the M52 completion receipts before making any M52 result claim: `collection.exit`, `collection_receipt.json`, `data_audit.json`, `control/training_result.json`, `mixed/training_result.json`, `runtime_contract.json`, all recursive shard exits/receipts, `recursive_result.json`, and `per_sequence.csv`, with SHA bindings.
2. Restore or publish the exact `lib/test/tracker/sttrack.py` source snapshot matching `diagnostics/m44/spec.json:8`, or provide the remote source bundle used by `/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1`, so the base recursive tracker can be audited.
3. After M52 results exist, verify that `recursive_result.json` contains `primary_pass`, `extra_training_control_pass`, `advancing_arm`, `aggregates.default`, `aggregates.control`, `aggregates.mixed`, and per-sequence metrics, and compare the numbers to the trajectory files through `run_sttrack_m52.py`.
4. Keep M52 claims limited to Train/development paired evidence unless official benchmark scripts/results are provided for VOT/DepthTrack Test/CDTB.

**Claim Impact**

Supported:
- M52 plan freezes one policy-state data augmentation iteration and does not claim first-ever recursive training: `diagnostics/m52/spec.json:1-4`, `diagnostics/m52/EXPERIMENT_PLAN.md:7-9`.
- M52 collection contract passed for 240 frames without labels or optimizer steps: `diagnostics/m52/contract.json:25-32`, `diagnostics/m52/contract.exit:1`.
- M52 collection was launched, not auto-trained: `diagnostics/m52/launch.json:1-14`, `diagnostics/m52/execution_binding.json:5-7`.
- Implementation distinguishes mixed data-effect claim from control extra-training advancement: `diagnostics/m52/PRE_TRAINING_NOTE.md:5-7`, `overlay/tools/run_sttrack_m52.py:126-135`.

Qualified:
- Training labels and recursive metrics use real dataset GT, but only for DepthTrack Train sampled/development evaluations.
- Pairing of physical frame keys, candidate-zero supervision, actual `previous_choice`, and matched initialization/order are implemented in source, but post-training evidence is not present locally.

Unsupported in the available evidence:
- Any M52 completed collection, data audit, paired training, runtime-contract pass, recursive metric, `primary_pass`, or selected `advancing_arm`.
- Any M52 state-data benefit over paired control.
- Any M52 public benchmark improvement on VOT, DepthTrack Test, or CDTB.
- Any claim that this local source bundle alone is fully auditable, because `overlay/lib/test/tracker/sttrack.py` is missing.