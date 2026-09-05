# M52 completed training evidence audit

GPT-5.5 xhigh, independent-context GPT-family Type-A advisory review. Reviewer response preserved verbatim; no cross-family acquittal. The remote running-stage observation is separately recorded in TRAINING_REPORT.md.

Updated verdict: **WARN**.

The newly available artifacts now support **completed M52 collection, sealed data audit, paired static training for both arms, actual tracker-loader spot contract, and descriptive predicted-state comparison**. I found **no concrete integrity defect** in those completed stages. The remaining warning is scope/result availability: **recursive 22-sequence evaluation and public benchmark results are still unavailable locally**, so `primary_pass`, `advancing_arm`, recursive mean-IoU/low-frame/H10 gates, and any public VOT/DepthTrack Test/CDTB claim remain unsupported.

This remains a **GPT-family Type-A advisory review**, not a cross-family acquittal.

**A. Ground-Truth Provenance: PASS**

Collection does not use GT. `diagnostics/m52/spec.json:3-4` freezes collection as GT-free fixed M45 policy-state collection. `diagnostics/m52/collection_receipt.json:763-768` reports 1,511 events, 93,362 frames, `labels_opened=false`, `optimizer_steps=0`, and `source_unchanged=true`.

The data audit enforces sealed input verification before labels. `overlay/tools/train_sttrack_m52.py:46-61` requires `collection.exit=0`, complete receipt, 1,511 events, 93,362 frames, no labels opened, parent inference-input SHA, 85 old receipts, and 63 new receipts. `overlay/tools/train_sttrack_m52.py:103-106` says labels cannot influence collection because feature and trajectory receipts have already been checked, then verifies and reads `training_labels.json`.

Training labels and static diagnostics use dataset-derived labels, not prediction-derived references. `overlay/tools/train_sttrack_m52.py:115-127` reads label rows by physical key and computes IoU for current/previous candidate boxes against the label GT, with candidate zero explicitly replaced by the view’s `public_bbox`. `overlay/tools/train_sttrack_m52.py:137-142` fixes the target rule as max-IoU candidate if at least 0.5, else NONE, with default priority when candidate zero already has IoU at least 0.5.

The audit result confirms the chain: `diagnostics/m52/data_audit.json:2-11` is `PASS`, binds spec/collection/training/labels SHA, verifies 85 native feature files, 63 policy feature files, 63 trace files, and records `labels_read_after_sealed_inputs_verified=true` and `original_targets_match_all_2101_m45_records=true`.

The descriptive predicted-state comparison is correctly labeled as non-GT. `diagnostics/m52/describe_state_difference.py:1` says it is not GT motion; `diagnostics/m52/state_difference_description.json:10612-10613` says `labels_opened=false` and scope is “not GT motion, not adjacent-frame displacement, and not model-advancement gates.”

Official benchmark claims are still not supported by this evaluator. `diagnostics/m52/spec.json:44` sets `public_automatic_launch=false`; `diagnostics/m52/training_launch.json:13` also records `public_evaluation_auto_launch=false`. The recursive evaluator, when run, uses Train development GT via `groundtruth.txt`, not official public benchmark scoring: `overlay/tools/run_sttrack_m52.py:79-83`.

**B. Score Normalization: PASS**

No self-referential normalization was found.

Static diagnostics gather selected candidate IoU and default IoU from GT-overlap arrays, then report means and counts directly: `overlay/tools/train_sttrack_m52.py:236-251`. The denominator is event count, not model-output max/min/mean.

Recursive analysis, when available, recomputes overlap against `groundtruth.txt`: `overlay/tools/run_sttrack_m52.py:79-83` for native baseline and `overlay/tools/run_sttrack_m52.py:99-115` for arm metrics, with aggregate mean computed as `iou_sum / valid_frames`.

The state-difference descriptor normalizes center distance by old predicted scale, but it is explicitly descriptive state comparison, not a performance metric: `diagnostics/m52/describe_state_difference.py:44-51` computes predicted-state deltas, and `diagnostics/m52/state_difference_description.json:10613` limits the result to descriptive statistics outside model-advancement gates.

**C. Result Existence And Key Verification: WARN**

Completed and verified locally:

- Collection completed: `diagnostics/m52/collection.exit:1` is `0`; `diagnostics/m52/collection_receipt.json:2-4` is complete and binds spec/policy checkpoint; `diagnostics/m52/collection_receipt.json:763-769` records 1,511 events, 93,362 frames, 36 nonzero previous-choice events, no labels opened, zero optimizer steps.
- Training launch exists: `diagnostics/m52/training_launch.json:1-13` records actual paired-training launch, collection/training-binding hashes, expected stages, 66,260 expected recursive frames, and no public auto launch.
- Data audit completed: `diagnostics/m52/data_audit.exit:1` is `0`; `diagnostics/m52/data_audit.json:2-23` contains the full expected audit keys and counts.
- Control training completed: `diagnostics/m52/training_control.exit:1` is `0`; `diagnostics/m52/control/training_result.json:2-6` reports complete, arm `control`, 448,739 parameters, 1,900 optimizer steps, 20 epochs.
- Mixed training completed: `diagnostics/m52/training_mixed.exit:1` is `0`; `diagnostics/m52/mixed/training_result.json:2-6` reports complete, arm `mixed`, 448,739 parameters, 1,900 optimizer steps, 20 epochs.
- Runtime contract completed: `diagnostics/m52/runtime_contract.exit:1` is `0`; `diagnostics/m52/runtime_contract.json:2-14` is `PASS` and binds both arm checkpoint hashes plus check-runtime source hash.
- State difference completed: `diagnostics/m52/state_difference_description.exit:1` is `0`; `diagnostics/m52/state_difference_description.json:2-7` records 1,511 events, 1,052 exact-different predicted-state events, 561 center-distance-over-1px events, and 271 over quarter-scale events.

I also parsed the JSON and verified these cross-file relationships:

- Full `initial_state_sha256` objects are identical between control and mixed. The shared object starts at `diagnostics/m52/control/training_result.json:7` and `diagnostics/m52/mixed/training_result.json:7`, and both end with the same `match.weight` at lines `52-53`.
- Logical sample order is identical: both files have `b6981fcc0dbd0dda1b8229173dee33cc758cf41a7654c6dfa0e19c1444d348cd` at `diagnostics/m52/control/training_result.json:55` and `diagnostics/m52/mixed/training_result.json:55`.
- Runtime contract checkpoint hashes match training-result checkpoint hashes: control `0bd04...bcf8` at `diagnostics/m52/runtime_contract.json:5` and `diagnostics/m52/control/training_result.json:227`; mixed `a249...56ce` at `diagnostics/m52/runtime_contract.json:9` and `diagnostics/m52/mixed/training_result.json:227`.
- `state_difference_description.json` exact-different count equals `data_audit.json` changed-event count: 1,052 at `diagnostics/m52/state_difference_description.json:4` and `diagnostics/m52/data_audit.json:17`.

Still unavailable locally:

- `recursive_control_s0.exit`
- `recursive_control_s1.exit`
- `recursive_mixed_s0.exit`
- `recursive_mixed_s1.exit`
- `control/shard0_receipt.json`
- `control/shard1_receipt.json`
- `mixed/shard0_receipt.json`
- `mixed/shard1_receipt.json`
- `analysis.exit`
- `recursive_result.json`
- `per_sequence.csv`

This matters because `overlay/tools/run_sttrack_m52.py:52-63` requires all recursive shard exits and receipts before analysis can aggregate results, and `overlay/tools/run_sttrack_m52.py:133-143` writes `recursive_result.json` only after that completed analysis.

**D. Dead Code, Wiring, And Execution: WARN**

For completed stages, code is wired and executed:

- Data audit is wired to write `data_audit.json` in audit mode: `overlay/tools/train_sttrack_m52.py:183-188`, and the artifact exists with status PASS at `diagnostics/m52/data_audit.json:2`.
- Training requires the audit result to match before either arm trains: `overlay/tools/train_sttrack_m52.py:190-191`.
- Training constructs control vs mixed exactly as planned: `overlay/tools/train_sttrack_m52.py:204-205` uses `original_fit + original_fit` for control and `original_fit + policy_fit` for mixed.
- Previous-choice input is wired into training: `overlay/tools/train_sttrack_m52.py:211-220` appends `previous_choice[index]` to model inputs and trains with `supervised_loss`.
- Final checkpoint save and exact reload-logit check are wired: `overlay/tools/train_sttrack_m52.py:229-259`.
- The training artifacts report `reload_logits_exact=true`: control at `diagnostics/m52/control/training_result.json:227-233`; mixed at `diagnostics/m52/mixed/training_result.json:227-233`.

Actual tracker-loader evidence exists but is limited in scope:

- `diagnostics/m52/check_runtime.py:23-34` loads `data_audit.json`, selects a feature sequence with nonzero previous-choice events, loads feature tensors, and constructs inputs including actual previous choices.
- `diagnostics/m52/check_runtime.py:40-51` loads both trained checkpoints into the actual `STTrackCandidateSet` loader and a direct `CandidateSetAssociation`, then asserts exact logits and affinity equality.
- The runtime artifact records sequence `ball05_indoor` and actual previous choices `[0,1,2,3]`: `diagnostics/m52/runtime_contract.json:15-21`.

Prepared but not executed locally:

- Recursive full22 evaluation code is wired but not evidenced as executed: `overlay/tools/run_sttrack_m52.py:52-63` requires shard exits/receipts; the required local files are absent.
- Result writing and `primary_pass`/`advancing_arm` computation are prepared at `overlay/tools/run_sttrack_m52.py:126-143`, but no local `recursive_result.json` exists to verify completed values.

No dead metric functions or concrete runtime/logic defect was found in the completed stages. The warning is because the recursive/public result stage is still prepared code, not completed evidence.

**E. Scope: WARN**

The scope is internally coherent but limited.

- M52 spec fixes 63 fit sequences, 1,511 fit events, and 93,362 frames through final events: `diagnostics/m52/spec.json:10-12`.
- The arms are paired logical views, not independent additional physical events: `diagnostics/m52/spec.json:15-30`; `diagnostics/m52/spec.json:17-18` defines control as duplicated original records and mixed as original records plus same physical events from M45 own-state trajectories.
- The plan fixes one seed and matched initialization/order: `diagnostics/m52/spec.json:25`.
- Development scope is static 590 events and intended full22 recursive sequences, with no development data in fitting: `diagnostics/m52/spec.json:22`.
- Data audit confirms 1,511 physical fit events, 3,022 paired fit views, 63 fit sequences, 22 development sequences, and 36 nonzero previous-choice events: `diagnostics/m52/data_audit.json:12-16`.
- Training source enforces physical-key pairing and sequence-disjoint fit/development: `overlay/tools/train_sttrack_m52.py:146-148`.
- Static training artifacts explicitly say static diagnostics do not replace full paired recursion: control `diagnostics/m52/control/training_result.json:36393-36394`; mixed `diagnostics/m52/mixed/training_result.json:36393-36394`.

Static diagnostics are completed, but they are not promotion gates. The completed static numbers are:

- Control native fit: mean IoU 0.5163035989 vs default 0.4891679287, 870 correct vs 762 default, 108 changes, 28 rescues, 0 breaks: `diagnostics/m52/control/training_result.json:234-244`.
- Control policy fit: mean IoU 0.5133534074 vs default 0.5035730600, 861 correct vs 813 default, 85 changes, 11 rescues, 1 break: `diagnostics/m52/control/training_result.json:15357-15367`.
- Control static development: mean IoU 0.4461315572 vs default 0.4402737916, 272 correct vs 268 default, 10 changes, 4 rescues, 1 break: `diagnostics/m52/control/training_result.json:30480-30490`.
- Mixed native fit: mean IoU 0.5157520175 vs default 0.4891679287, 863 correct vs 762 default, 102 changes, 29 rescues, 0 breaks: `diagnostics/m52/mixed/training_result.json:234-244`.
- Mixed policy fit: mean IoU 0.5194222927 vs default 0.5035730600, 881 correct vs 813 default, 68 changes, 18 rescues, 0 breaks: `diagnostics/m52/mixed/training_result.json:15357-15367`.
- Mixed static development: mean IoU 0.4466814995 vs default 0.4402737916, 274 correct vs 268 default, 14 changes, 4 rescues, 0 breaks: `diagnostics/m52/mixed/training_result.json:30480-30490`.

Parsed static deltas:
- Mixed static development over control static development: **+0.0005499423 mean IoU**.
- Mixed policy-fit over control policy-fit: **+0.0060688853 mean IoU**.
- Mixed native-fit is slightly below control native-fit: **-0.0005515814 mean IoU**.

These static numbers can support a descriptive diagnostic, not a recursive model-advancement claim. The pre-training rule requires full 22-sequence recursion for both arms and separates mixed state-data benefit from control extra-training benefit: `diagnostics/m52/PRE_TRAINING_NOTE.md:5-11`.

**F. Evaluation Type Classification: WARN**

Completed collection:
- Type: `self_supervised_proxy` / instrumentation consistency and data capture.
- Evidence: collection GT read is false in `diagnostics/m52/spec.json:3`; receipt has `labels_opened=false` and `optimizer_steps=0` in `diagnostics/m52/collection_receipt.json:765-767`.
- Limit: not a performance evaluation.

Completed data audit:
- Type: provenance/integrity audit over sealed data plus later real-GT labels.
- Evidence: `diagnostics/m52/data_audit.json:2-11`.
- Limit: confirms data and label binding; not recursive performance.

Completed paired static training diagnostics:
- Type: `real_gt` static Train diagnostic.
- Evidence: static evaluator uses GT-derived IoU labels at `overlay/tools/train_sttrack_m52.py:115-127` and reports event-level means/counts at `overlay/tools/train_sttrack_m52.py:236-251`.
- Limit: static diagnostics do not replace full paired recursion, stated in both training results at `diagnostics/m52/control/training_result.json:36393-36394` and `diagnostics/m52/mixed/training_result.json:36393-36394`.

Completed runtime contract:
- Type: implementation/loader consistency check.
- Evidence: `diagnostics/m52/check_runtime.py:40-51`; result PASS at `diagnostics/m52/runtime_contract.json:2-14`.
- Limit: sample loader equality on selected records, not full recursive performance.

Completed predicted-state comparison:
- Type: `self_supervised_proxy` descriptive comparison of two predicted-state trajectories at same physical events.
- Evidence: `diagnostics/m52/state_difference_description.json:2-7` and scope at `diagnostics/m52/state_difference_description.json:10612-10613`.
- Limit: not GT motion, not adjacent-frame displacement, not model-advancement gates.

Prepared recursive development evaluation:
- Type when executed: `real_gt` Train development recursive evaluation.
- Evidence for intended evaluator: `overlay/tools/run_sttrack_m52.py:79-83` and `overlay/tools/run_sttrack_m52.py:99-115`.
- Status: unavailable locally; no recursive result artifact to classify as completed.

Public benchmark:
- Type: unavailable.
- Evidence: no public auto launch in `diagnostics/m52/spec.json:44` and `diagnostics/m52/training_launch.json:13`.
- Status: unsupported.

**Concrete Defects**

No concrete defect found in the completed collection, data audit, paired static training, runtime loader contract, or descriptive predicted-state comparison artifacts.

The concrete remaining evidence gap is not a code defect: recursive shard receipts/results and public benchmark results are absent locally, while the analysis code requires them before it can produce `recursive_result.json`.

**Supported Claims**

- M52 collection completed: 63 fit sequences, 1,511 events, 93,362 frames, 36 nonzero previous-choice events, no labels opened, zero optimizer steps.
- Data audit passed after sealed-input verification and before label use; it verified 85 native feature files, 63 policy feature files, and 63 new trace files.
- Original 2,101 M45 target labels were reproduced.
- Paired training completed for both control and mixed: 448,739 parameters, seed 2026, 1,900 optimizer steps, 20 epochs, exact reload logits.
- Control and mixed have identical initial parameter hashes and identical logical sample-order hash.
- Runtime loader contract passed for both trained checkpoints on actual previous-choice inputs.
- Predicted-state descriptive comparison found 1,052/1,511 exact-different event boxes and is correctly labeled as non-GT/non-gate evidence.
- Static diagnostics exist for native fit, policy fit, and 590-event development.

**Qualified Claims**

- Static diagnostics suggest the mixed arm is better than control on policy-fit static mean IoU and slightly better on static development mean IoU, but the gains are small and static-only.
- Completed static development does not establish recursive tracking improvement.
- Runtime contract supports loader consistency for sampled records with actual previous choices; it is not exhaustive recursive validation.

**Unsupported Claims**

- Any M52 recursive full22 improvement.
- Any `primary_pass` value.
- Any `advancing_arm` selection.
- Any state-data benefit over paired control under the frozen recursive gates.
- Any claim that control qualifies as an extra-training advancement under recursive gates.
- Any public VOT/DepthTrack Test/CDTB benchmark improvement.
- Any official benchmark claim from these completed artifacts.