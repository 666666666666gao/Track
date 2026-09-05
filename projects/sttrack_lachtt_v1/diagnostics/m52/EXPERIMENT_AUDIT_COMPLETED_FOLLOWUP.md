# M52 completed recursive and affinity integrity review

GPT-5.5 xhigh, independent-context GPT-family Type-A advisory review. Full reviewer response preserved verbatim; not a cross-family acquittal. The private copies used for recomputation are not included in the public repository.

Audit completed as a read-only fourth-stage integrity review. I did not edit files or launch experiments. This is a GPT-family Type-A advisory review only; it is not a cross-family acquittal, and I did not treat the artifact’s own `integrity_pass` flag as an independent verdict.

Path aliases below:

- `P = C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1`
- `S = C:/Users/gb/.codex_remote_staging/m52_completed_review`

Overall verdict: **WARN for claim readiness; PASS for the integrity of the completed internal M52 recursive artifact set I could verify.** I found no concrete arithmetic, hash-binding, shard-coverage, paired-training, loader-wiring, or metric-wiring defect in the completed internal M52 artifacts. The actual experimental outcome is negative: neither `mixed` nor `control` passes the frozen recursive advancement gates, `primary_pass` is false, and `advancing_arm` is null. The only supported recursive result is a bounded DepthTrack Train development result, not a public/official benchmark result.

### Key recomputation result

Using the independent metric definition from `P/overlay/tools/audit_sttrack_m43.py:17-37` and the private staged trajectory/GT copies bound by `S/review_data_binding.json:2-72`, I recomputed all new-arm recursive per-sequence metrics from the 44 trajectory JSON files and 22 ground-truth files. The recomputation produced:

- 0 trajectory SHA mismatches against `S/review_data_binding.json:3-48`.
- 0 ground-truth SHA mismatches against `S/review_data_binding.json:49-72`.
- 0 per-sequence mismatches against `P/diagnostics/m52/recursive_result.json`.
- 0 CSV row/metric mismatches against `P/diagnostics/m52/per_sequence.csv`.
- 0 aggregate mismatches against `P/diagnostics/m52/recursive_result.json:46-60`.

The recomputed aggregates were:

| arm | valid frames | IoU sum | mean IoU | low-IoU frames | H10 episodes | macro sequence mean |
|---|---:|---:|---:|---:|---:|---:|
| control | 28,897 | 18,605.997579961506 | 0.643872982661228 | 7,429 | 90 | 0.6671082475548341 |
| mixed | 28,897 | 18,230.964731480588 | 0.6308947202644076 | 8,058 | 80 | 0.6781284158048133 |

These match the published artifact values at `P/diagnostics/m52/recursive_result.json:46-60`.

---

## A. Ground-truth provenance — PASS, with bounded-scope limits

The collection stage is documented and checked as GT-free. The M52 spec sets `collection_gt_read` to false at `P/diagnostics/m52/spec.json:3`, and defines collection as a frozen M45 policy-state run using native templates with no M48/M50/text additions at `P/diagnostics/m52/spec.json:4`. The completed collection receipt reports 1,511 events, 93,362 frames, 36 nonzero previous-choice events, `labels_opened: false`, `optimizer_steps: 0`, and `source_unchanged: true` at `P/diagnostics/m52/collection_receipt.json:763-768`.

The training loader verifies sealed inputs before reading labels. It asserts the collection exit/receipt and `labels_opened` state at `P/overlay/tools/train_sttrack_m52.py:46-52`, verifies old/new feature and trace hashes plus policy-state trace consistency at `P/overlay/tools/train_sttrack_m52.py:64-100`, and only then reads the label file at `P/overlay/tools/train_sttrack_m52.py:103-106`. The data audit independently records `labels_read_after_sealed_inputs_verified: true`, `original_targets_match_all_2101_m45_records: true`, 1,511 physical fit events, 3,022 paired fit views, 63 fit sequences, and 22 development sequences at `P/diagnostics/m52/data_audit.json:10-15`.

The policy-state records are prediction-derived state data, not dataset GT. For policy records, the loader checks `previous_choice` against the policy trace previous frame and `current_choice` against the trace current frame at `P/overlay/tools/train_sttrack_m52.py:89-93`, and verifies `selected_bbox` / `previous_selected_bbox` against the policy trace boxes at `P/overlay/tools/train_sttrack_m52.py:94-95`. These are consistency checks against frozen policy traces, not GT labels.

Candidate-zero and selected-box supervision are kept distinct in the training code. Candidate-zero IoU is explicitly computed from `public_bbox` / `previous_public_bbox`, while candidate candidates use `current_boxes` / `previous_boxes`, at `P/overlay/tools/train_sttrack_m52.py:119-127`. The target rule then sets unmatched to 10, and forces candidate zero when candidate-zero IoU is at least 0.5, at `P/overlay/tools/train_sttrack_m52.py:137-141`. This supports the conclusion that candidate-zero supervision is not silently replaced by the selected policy box.

The recursive evaluator uses dataset GT for scoring. It loads `groundtruth.txt`, recomputes overlap for the default/native baseline, and checks it against the prior native result at `P/overlay/tools/run_sttrack_m52.py:74-83`. The metric definition computes framewise IoU, excludes frame 0 and invalid GT boxes, counts `<= .1` low-IoU frames, and counts H10 failure episodes at `P/overlay/tools/audit_sttrack_m43.py:17-37`. This makes the recursive result a real-GT internal Train-development evaluation, not a prediction-only consistency score.

Official/public benchmark claims are not supported by the files read. The spec sets `public_automatic_launch` to false at `P/diagnostics/m52/spec.json:44`, and the recursive result scope is “Repeated Train development” at `P/diagnostics/m52/recursive_result.json:714`. Any claim that M52 has an official public benchmark result remains unsupported by this artifact set.

---

## B. Score normalization — PASS

I found no self-referential denominator in the reported recursive metrics. The metric definition uses valid finite GT frames as the denominator: valid IoUs are collected, `valid_frames` is `len(valid)`, `iou_sum` is `math.fsum(valid)`, and `mean_iou` is `math.fsum(valid)/len(valid)` at `P/overlay/tools/audit_sttrack_m43.py:33-36`.

The M52 recursive analyzer computes arm totals by summing per-sequence `valid_frames`, `iou_sum`, `low_iou_frames`, and `failure_episodes`, then computes aggregate `mean_iou` as `iou_sum / valid_frames` and macro mean as the mean of sequence means at `P/overlay/tools/run_sttrack_m52.py:113-115`. The output reports those denominators and sums explicitly:

- Default: 28,897 valid frames, IoU sum 18,847.382327003917, mean IoU 0.6522262631762438 at `P/diagnostics/m52/recursive_result.json:30-36`.
- M45: 28,897 valid frames, IoU sum 19,768.50630952179, mean IoU 0.6841023742783606 at `P/diagnostics/m52/recursive_result.json:38-44`.
- Control: 28,897 valid frames, IoU sum 18,605.997579961513, mean IoU 0.6438729826612283 at `P/diagnostics/m52/recursive_result.json:46-52`.
- Mixed: 28,897 valid frames, IoU sum 18,230.96473148059, mean IoU 0.6308947202644077 at `P/diagnostics/m52/recursive_result.json:54-60`.

The CSV uses explicit per-sequence raw fields and deltas, not a self-normalized score. Its header is `arm,sequence,valid_frames,iou_sum,mean_iou,low_iou_frames,failure_episodes,changes,mean_iou_gain,low_frame_delta,episode_delta` at `P/diagnostics/m52/per_sequence.csv:1`. Example rows show raw denominators and deltas per sequence at `P/diagnostics/m52/per_sequence.csv:2-8` and `P/diagnostics/m52/per_sequence.csv:39-45`.

The frozen gate calculations are also direct comparisons, not normalized by each arm’s own changed frames. The evaluator compares arm aggregate mean IoU, low frames, and H10 episodes against the default baseline and fixed gate thresholds at `P/overlay/tools/run_sttrack_m52.py:117-124`, then compares mixed against paired control for incremental gates at `P/overlay/tools/run_sttrack_m52.py:126-131`.

---

## C. Result existence and status — PASS for internal M52 artifacts; WARN for public evidence

The completed internal recursive artifacts exist and have zero exits:

- `P/diagnostics/m52/analysis.exit:1` is `0`.
- `P/diagnostics/m52/pipeline.exit:1` is `0`.
- `P/diagnostics/m52/recursive_control_s0.exit:1` is `0`.
- `P/diagnostics/m52/recursive_control_s1.exit:1` is `0`.
- `P/diagnostics/m52/recursive_mixed_s0.exit:1` is `0`.
- `P/diagnostics/m52/recursive_mixed_s1.exit:1` is `0`.
- `P/diagnostics/m52/affinity_readout_diagnostic.exit:1` is `0`.

The recursive result reports `status: complete`, `primary: policy_state_data`, `primary_pass: false`, and `advancing_arm: null` at `P/diagnostics/m52/recursive_result.json:2-5` and `P/diagnostics/m52/recursive_result.json:27-28`. The file’s own `integrity_pass: true` at `P/diagnostics/m52/recursive_result.json:3` is an executor-produced field; I did not use it as independent evidence.

The paired training artifacts exist and report completion. Control has `status: complete`, 448,739 parameters, 1,900 optimizer steps, and 20 epochs at `P/diagnostics/m52/control/training_result.json:2-6`. Mixed reports the same at `P/diagnostics/m52/mixed/training_result.json:2-6`. Control and mixed both report the same logical sample order hash `b6981fcc0dbd0dda1b8229173dee33cc758cf41a7654c6dfa0e19c1444d348cd` at `P/diagnostics/m52/control/training_result.json:55` and `P/diagnostics/m52/mixed/training_result.json:55`. Checkpoints are bound and reload-checked at `P/diagnostics/m52/control/training_result.json:227-233` and `P/diagnostics/m52/mixed/training_result.json:227-233`.

The recursive analysis records 33,130 frames per arm, 66,260 total frames, matched initialization/order, recomputed native baseline, checkpoint hashes, training-result hashes, spec hash, training-binding hash, and data-audit hash at `P/diagnostics/m52/recursive_result.json:699-713`.

Public/official result evidence remains unavailable. The spec explicitly disables automatic public launch at `P/diagnostics/m52/spec.json:44`, and the recursive result is scoped to “Repeated Train development” at `P/diagnostics/m52/recursive_result.json:714`. I cannot verify any completed official benchmark/public evaluation from the files provided.

---

## D. Dead code, execution wiring, and runtime/logic defects — PASS

The recursive metrics are wired into the analysis path and executed over completed shard receipts. The analyzer requires the four recursive shard exit files and receipts at `P/overlay/tools/run_sttrack_m52.py:52-63`, verifies the bound native and M45 recursive result hashes at `P/overlay/tools/run_sttrack_m52.py:64-67`, recomputes the default/native baseline against `groundtruth.txt` at `P/overlay/tools/run_sttrack_m52.py:74-83`, reads each new-arm trajectory, checks trajectory SHA, frame order, finite boxes, init box, valid choice semantics, and default-prefix equality before first override at `P/overlay/tools/run_sttrack_m52.py:89-105`, computes totals/gates at `P/overlay/tools/run_sttrack_m52.py:113-131`, and writes the recursive result at `P/overlay/tools/run_sttrack_m52.py:133-143`.

Shard coverage is complete by binding and result. The fixed shard list contains 11 sequences in shard 0 and 11 in shard 1 at `P/diagnostics/m52/training_binding.json:14-40`. The analyzer asserts each arm has 22 receipt sequences and that the receipt sequence set equals the case set at `P/overlay/tools/run_sttrack_m52.py:55-63`. The result reports `frames_per_arm: 33130` and `total_frames: 66260` at `P/diagnostics/m52/recursive_result.json:699-700`. My private recomputation found 22 control rows and 22 mixed rows in `P/diagnostics/m52/per_sequence.csv`, consistent with its header and rows at `P/diagnostics/m52/per_sequence.csv:1-45`.

I found no concrete dead-code defect in the requested completed M52 recursive pipeline. Earlier prepared/evaluation code is now backed by completed result files and zero exits. I also found no concrete runtime or logic defect in the new recursive result generation path from the files read.

The auxiliary affinity diagnostic is also wired and completed. Its code loads trained checkpoints, recomputes CPU classifier outputs, compares them to recorded GPU choices, and asserts exact equality at `P/diagnostics/m52/inspect_affinity_readout.py:33-43`. It then computes the causal previous-choice affinity readout without GT at `P/diagnostics/m52/inspect_affinity_readout.py:44-47`, computes the explicitly privileged previous-GT oracle readout at `P/diagnostics/m52/inspect_affinity_readout.py:48-51`, and writes a non-policy-changing diagnostic result with `optimizer_steps: 0` and `recursive_policy_changed: false` at `P/diagnostics/m52/inspect_affinity_readout.py:78-81`.

The affinity diagnostic output reports completion at `P/diagnostics/m52/affinity_readout_diagnostic.json:2`, all six classifier CPU-vs-GPU parity flags are true at `P/diagnostics/m52/affinity_readout_diagnostic.json:8`, `P/diagnostics/m52/affinity_readout_diagnostic.json:12147`, `P/diagnostics/m52/affinity_readout_diagnostic.json:24286`, `P/diagnostics/m52/affinity_readout_diagnostic.json:29062`, `P/diagnostics/m52/affinity_readout_diagnostic.json:41201`, and `P/diagnostics/m52/affinity_readout_diagnostic.json:53340`, and it records `optimizer_steps: 0` plus `recursive_policy_changed: false` at `P/diagnostics/m52/affinity_readout_diagnostic.json:58116-58117`.

---

## E. Scope, physical events, split separation, seed count, and claim strength — WARN

The scope is narrow and must stay narrow.

M52 collection/training scope is 1,511 physical fitting events from 63 fit sequences, not 3,022 independent physical events. The spec defines control as duplicated original records and mixed as original records plus the same 1,511 physical events from M45 own-state trajectories at `P/diagnostics/m52/spec.json:15-19`. The data audit confirms `physical_fit_events: 1511` and `paired_fit_views: 3022` at `P/diagnostics/m52/data_audit.json:12-13`.

Train/development separation is documented and checked. The spec says development is “Original590 static events; full22 recursive sequences for both trained arms; no development data in fitting” at `P/diagnostics/m52/spec.json:22`. The loader asserts 1,511 original fit, 1,511 policy fit, 590 development events, 3,612 total loaded keys, exact original/policy physical-key matching, and disjoint fit/development sequence names at `P/overlay/tools/train_sttrack_m52.py:143-148`. The data audit reports 63 fit sequences and 22 development sequences at `P/diagnostics/m52/data_audit.json:14-15`.

Initialization and sample-order matching are fixed for the paired arms. The spec states “Fresh seed2026 for both arms, matched parameter hashes and index orders” at `P/diagnostics/m52/spec.json:25`. The training code asserts seed 2026, seeds Torch/CUDA/NumPy/Python random, initializes the same `CandidateSetAssociation` model, checks the initial state against the paired reference, and uses the paired train construction at `P/overlay/tools/train_sttrack_m52.py:196-205`. Both training results show the same logical sample order hash at `P/diagnostics/m52/control/training_result.json:55` and `P/diagnostics/m52/mixed/training_result.json:55`. The recursive result records `matched_initialization_and_order: true` at `P/diagnostics/m52/recursive_result.json:701`.

The seed count is one. The fixed seed is 2026 at `P/diagnostics/m52/spec.json:25` and `P/overlay/tools/train_sttrack_m52.py:196-198`; I found no multi-seed replication evidence in the requested files.

The claim strength is constrained by failed gates. The frozen advancement rule is fixed before results: mixed advances only if it passes native and incremental gates, otherwise control can advance only as an extra-training benefit, otherwise neither advances, at `P/diagnostics/m52/PRE_TRAINING_NOTE.md:5-9` and `P/diagnostics/m52/training_binding.json:44-45`. The output follows that rule: both native gate sets fail, the incremental mixed-vs-control mean and low-frame gates fail, `extra_training_control_pass` is false, and `advancing_arm` is null at `P/diagnostics/m52/recursive_result.json:6-28`.

Exact gate arithmetic:

| comparison | requirement | observed | status |
|---|---:|---:|---|
| control mean vs default +0.01 | ≥ 0.6622262631762438 | 0.6438729826612283 | fail |
| control low frames vs default | < 7,397 | 7,429 | fail |
| control H10 episodes vs default | ≤ 75 | 90 | fail |
| mixed mean vs default +0.01 | ≥ 0.6622262631762438 | 0.6308947202644077 | fail |
| mixed low frames vs default | < 7,397 | 8,058 | fail |
| mixed H10 episodes vs default | ≤ 75 | 80 | fail |
| mixed mean vs paired control | > 0.6438729826612283 | 0.6308947202644077 | fail |
| mixed low frames vs paired control | ≤ 7,429 | 8,058 | fail |
| mixed H10 episodes vs paired control | ≤ 90 | 80 | pass |

The aggregate inputs are reported at `P/diagnostics/m52/recursive_result.json:30-60`, gate booleans at `P/diagnostics/m52/recursive_result.json:6-28`, and positive sequence / new-failure sequence summaries at `P/diagnostics/m52/recursive_result.json:535-536` for control and `P/diagnostics/m52/recursive_result.json:695-696` for mixed.

---

## F. Evaluation classification — WARN because only internal/proxy evidence exists

Classification by evaluation/artifact:

| artifact or evaluation | classification | evidence and limit |
|---|---|---|
| M52 collection receipt / policy-state trace consistency | `self_supervised_proxy` / data collection, no performance evaluation | GT-free collection is declared at `P/diagnostics/m52/spec.json:3-4`; collection receipt says `labels_opened: false` and `optimizer_steps: 0` at `P/diagnostics/m52/collection_receipt.json:763-768`. It produces prediction-derived policy-state traces/features, not a benchmark score. |
| M52 paired static training metrics on native/policy/development events | `real_gt`, static proxy | Labels are read only after sealed inputs at `P/overlay/tools/train_sttrack_m52.py:103-106`; targets are IoU/GT-derived at `P/overlay/tools/train_sttrack_m52.py:119-142`. Control static development reports 590 events, mean 0.44613155722618103, default 0.44027379155158997 at `P/diagnostics/m52/control/training_result.json:30480-30489`; mixed reports 590 events, mean 0.44668149948120117, default 0.44027379155158997 at `P/diagnostics/m52/mixed/training_result.json:30480-30489`. This is event-level candidate classification, not recursive benchmark tracking. |
| M52 recursive Train-development evaluation | `real_gt` | The analyzer loads `groundtruth.txt` and recomputes overlaps at `P/overlay/tools/run_sttrack_m52.py:74-83`, then computes new-arm metrics against the same GT at `P/overlay/tools/run_sttrack_m52.py:89-115`. The scope is repeated Train development at `P/diagnostics/m52/recursive_result.json:714`. |
| Descriptive predicted-state comparison from earlier M52 artifacts | `self_supervised_proxy` | It compares predicted trajectories/states, not GT motion or adjacent-frame displacement. The prior state-difference description remains descriptive only and should not be converted into an advancement or model-effect claim. |
| Affinity causal previous-choice readout | `real_gt` scoring of a `self_supervised_proxy` decision rule | The decision input uses the actual previous-choice affinity column with no GT at `P/diagnostics/m52/inspect_affinity_readout.py:44-47`; metrics are computed from GT-derived IoUs at `P/diagnostics/m52/inspect_affinity_readout.py:52-60`. Development results show causal readout below default/control-classifier for control and below default/mixed-classifier for mixed at `P/diagnostics/m52/affinity_readout_diagnostic.json:24296-24305` and `P/diagnostics/m52/affinity_readout_diagnostic.json:53350-53359`. |
| Affinity previous-GT oracle subset | `real_gt`, privileged offline diagnostic | The oracle uses `previous_target` as the affinity column at `P/diagnostics/m52/inspect_affinity_readout.py:48-51`; the plan labels it as privileged and never deployable at `P/diagnostics/m52/AFFINITY_DIAGNOSTIC_PLAN.md:15-19`. Development subset metrics are 336-event privileged diagnostics at `P/diagnostics/m52/affinity_readout_diagnostic.json:24306-24332` and `P/diagnostics/m52/affinity_readout_diagnostic.json:53360-53386`. |
| Public/official benchmark result | cannot classify as completed | `public_automatic_launch` is false at `P/diagnostics/m52/spec.json:44`; recursive scope is Train development at `P/diagnostics/m52/recursive_result.json:714`. No public result file was provided. |

No `simulation_only` or `human_eval` evidence appears in the requested files.

---

## Auxiliary affinity diagnostic finding

The affinity diagnostic is valid as an exploratory static diagnostic and does not support a causal improvement claim.

The plan explicitly restricts the diagnostic: it changes no tracker/checkpoints/data/gates at `P/diagnostics/m52/AFFINITY_DIAGNOSTIC_PLAN.md:3-5`, requires CPU classifier choices to equal recorded GPU choices for every event at `P/diagnostics/m52/AFFINITY_DIAGNOSTIC_PLAN.md:7-13`, restricts the causal readout to the actual previous-choice affinity column with no GT at `P/diagnostics/m52/AFFINITY_DIAGNOSTIC_PLAN.md:7-13`, and labels the previous-GT column as a privileged oracle capacity diagnostic that is never deployable at `P/diagnostics/m52/AFFINITY_DIAGNOSTIC_PLAN.md:15-19`.

The report states the diagnostic changed no policy and is not part of the frozen advancement rule at `P/diagnostics/m52/AFFINITY_REPORT.md:3-7`. It also states the causal readout uses no GT and that unmatched maps to candidate zero only for output comparison at `P/diagnostics/m52/AFFINITY_REPORT.md:9-13`.

Development-event results:

| arm/readout | events | mean IoU | default mean IoU | correct | changes | rescues | breaks |
|---|---:|---:|---:|---:|---:|---:|---:|
| control classifier | 590 | 0.44613155722618103 | 0.4402737319469452 | 272 | 10 | 4 | 1 |
| control causal previous-choice affinity | 590 | 0.43926283717155457 | 0.4402737319469452 | 267 | 8 | 1 | 1 |
| mixed classifier | 590 | 0.44668149948120117 | 0.4402737319469452 | 274 | 14 | 4 | 0 |
| mixed causal previous-choice affinity | 590 | 0.4387143850326538 | 0.4402737319469452 | 268 | 14 | 2 | 1 |

Evidence: control development classifier/causal metrics at `P/diagnostics/m52/affinity_readout_diagnostic.json:24285-24305`; mixed development classifier/causal metrics at `P/diagnostics/m52/affinity_readout_diagnostic.json:53339-53359`.

Privileged previous-GT subset results:

| arm/readout | events | mean IoU | limit |
|---|---:|---:|---|
| control classifier subset | 336 | 0.7395433783531189 | deployable classifier on previous-GT-available subset |
| control causal subset | 336 | 0.7324622273445129 | previous-choice affinity column |
| control privileged previous-GT affinity | 336 | 0.7352411150932312 | privileged oracle input |
| mixed classifier subset | 336 | 0.7422884106636047 | deployable classifier on previous-GT-available subset |
| mixed causal subset | 336 | 0.7391106486320496 | previous-choice affinity column |
| mixed privileged previous-GT affinity | 336 | 0.7418895959854126 | privileged oracle input |

Evidence: control subset/oracle metrics at `P/diagnostics/m52/affinity_readout_diagnostic.json:24306-24332`; mixed subset/oracle metrics at `P/diagnostics/m52/affinity_readout_diagnostic.json:53360-53386`.

This diagnostic supports the limited claim that directly substituting the current cached affinity readout for the classifier does not improve the static development cache under this readout protocol. It does not prove anything about online identity propagation, KeepTrack reproduction, calibrated affinities, or a causal mechanism for M52 recursive behavior.

---

## Concrete defects found

I found **no concrete implementation, arithmetic, execution-wiring, shard-coverage, checkpoint-binding, source-binding, paired-order, or staged-copy recomputation defect** in the completed artifacts reviewed.

The concrete negative result is experimental, not a code defect:

- `primary_pass` is false at `P/diagnostics/m52/recursive_result.json:5`.
- Both native gate sets fail at `P/diagnostics/m52/recursive_result.json:6-20`.
- Mixed fails the mean-IoU and low-frame incremental gates against paired control at `P/diagnostics/m52/recursive_result.json:22-26`.
- `extra_training_control_pass` is false and `advancing_arm` is null at `P/diagnostics/m52/recursive_result.json:27-28`.

---

## Supported claims

These claims are supported by the files and my read-only recomputation:

1. **M52 collection was GT-free and optimizer-free.**  
   Evidence: `P/diagnostics/m52/spec.json:3-4`, `P/diagnostics/m52/collection_receipt.json:763-768`, `P/overlay/tools/train_sttrack_m52.py:46-52`.

2. **The paired training setup used 1,511 physical fit events expanded to 3,022 paired views, with Train-development held out.**  
   Evidence: `P/diagnostics/m52/spec.json:15-22`, `P/diagnostics/m52/data_audit.json:12-15`, `P/overlay/tools/train_sttrack_m52.py:143-148`.

3. **Control and mixed were trained with matched initialization/order and 1,900 optimizer steps.**  
   Evidence: `P/diagnostics/m52/spec.json:25-29`, `P/overlay/tools/train_sttrack_m52.py:196-205`, `P/overlay/tools/train_sttrack_m52.py:211-229`, `P/diagnostics/m52/control/training_result.json:2-6`, `P/diagnostics/m52/mixed/training_result.json:2-6`, and matching logical sample-order hashes at `P/diagnostics/m52/control/training_result.json:55` and `P/diagnostics/m52/mixed/training_result.json:55`.

4. **M52 recursive results are complete for the fixed 22 Train-development sequences and 66,260 total arm-frames.**  
   Evidence: shard list at `P/diagnostics/m52/training_binding.json:14-40`, analysis checks at `P/overlay/tools/run_sttrack_m52.py:52-63`, zero exits for all recursive shards and analysis/pipeline exit files, and result totals at `P/diagnostics/m52/recursive_result.json:699-700`.

5. **The recursive aggregate/per-sequence arithmetic is internally consistent and independently recomputed from private trajectory/GT copies.**  
   Evidence: metric definition at `P/overlay/tools/audit_sttrack_m43.py:17-37`, private copy binding at `S/review_data_binding.json:2-72`, per-sequence table schema and rows at `P/diagnostics/m52/per_sequence.csv:1-45`, and aggregate result fields at `P/diagnostics/m52/recursive_result.json:29-60`.

6. **The completed M52 recursive result is negative under the pre-fixed advancement rules.**  
   Evidence: frozen rule at `P/diagnostics/m52/PRE_TRAINING_NOTE.md:5-9` and `P/diagnostics/m52/training_binding.json:44-45`; output gates and selection at `P/diagnostics/m52/recursive_result.json:6-28`.

7. **The affinity diagnostic was exploratory, non-training, and policy-nonchanging.**  
   Evidence: plan restrictions at `P/diagnostics/m52/AFFINITY_DIAGNOSTIC_PLAN.md:3-26`, implementation output fields at `P/diagnostics/m52/inspect_affinity_readout.py:78-81`, and output footer at `P/diagnostics/m52/affinity_readout_diagnostic.json:58113-58118`.

---

## Qualified claims

These claims are partially supported but must be stated with limits:

1. **“M52 static development classifier improves over default.”**  
   Qualified as a static, candidate-set, real-GT proxy result only. Control development is 0.44613155722618103 vs default 0.44027379155158997 at `P/diagnostics/m52/control/training_result.json:30480-30489`; mixed development is 0.44668149948120117 vs default 0.44027379155158997 at `P/diagnostics/m52/mixed/training_result.json:30480-30489`. This does not imply recursive improvement.

2. **“M52 trained heads load and reproduce recorded classifier choices.”**  
   Qualified to the checked static scopes. The affinity diagnostic asserts CPU classifier equality to recorded GPU classifier choices at `P/diagnostics/m52/inspect_affinity_readout.py:40-43`, and the JSON records true parity across all six arm/scope combinations at `P/diagnostics/m52/affinity_readout_diagnostic.json:8`, `P/diagnostics/m52/affinity_readout_diagnostic.json:12147`, `P/diagnostics/m52/affinity_readout_diagnostic.json:24286`, `P/diagnostics/m52/affinity_readout_diagnostic.json:29062`, `P/diagnostics/m52/affinity_readout_diagnostic.json:41201`, and `P/diagnostics/m52/affinity_readout_diagnostic.json:53340`.

3. **“The direct affinity readout diagnostic argues against using the present auxiliary affinity output directly.”**  
   Qualified to cached static events and the exact readout protocol. The causal previous-choice readout is below default on the 590-event development set for both control and mixed at `P/diagnostics/m52/affinity_readout_diagnostic.json:24296-24305` and `P/diagnostics/m52/affinity_readout_diagnostic.json:53350-53359`.

4. **“The private recomputation strengthens audit confidence.”**  
   Qualified because `S/review_data_binding.json:2` describes these as private copies for read-only completed-result review. They verify the reviewed artifact in this environment but are not public evidence unless a publishable manifest/data package is provided.

---

## Unsupported claims

These claims are unsupported or contradicted by the reviewed evidence:

1. **“M52 policy-state data improves recursive tracking under the frozen M52 gates.”**  
   Unsupported and contradicted. Mixed mean IoU is 0.6308947202644077, below default 0.6522262631762438 and control 0.6438729826612283; mixed low frames are 8,058, above default 7,397 and control 7,429; mixed gates fail at `P/diagnostics/m52/recursive_result.json:14-26`.

2. **“M52 mixed demonstrates a data-effect gain over paired control.”**  
   Unsupported. Incremental mean-IoU and low-frame gates are false at `P/diagnostics/m52/recursive_result.json:22-26`; the pre-training note requires mixed to pass native and incremental gates before describing state-data benefit at `P/diagnostics/m52/PRE_TRAINING_NOTE.md:5-7`.

3. **“M52 control advances as an extra-training result.”**  
   Unsupported. Control fails mean IoU, low-frame, and episode native gates at `P/diagnostics/m52/recursive_result.json:6-12`, and `extra_training_control_pass` is false at `P/diagnostics/m52/recursive_result.json:27`.

4. **“Any M52 arm should advance to public evaluation under the pre-fixed rule.”**  
   Unsupported. `advancing_arm` is null at `P/diagnostics/m52/recursive_result.json:28`.

5. **“M52 has an official/public benchmark result.”**  
   Unsupported. `public_automatic_launch` is false at `P/diagnostics/m52/spec.json:44`, and the recursive result scope is Train development at `P/diagnostics/m52/recursive_result.json:714`.

6. **“The affinity diagnostic proves a causal tracking mechanism, online identity propagation, KeepTrack reproduction, or deployable oracle behavior.”**  
   Unsupported. The plan and report explicitly restrict the diagnostic at `P/diagnostics/m52/AFFINITY_DIAGNOSTIC_PLAN.md:21-26`, `P/diagnostics/m52/AFFINITY_REPORT.md:39-50`, and the JSON footer at `P/diagnostics/m52/affinity_readout_diagnostic.json:58118`.

---

## Remaining evidence limitations

1. **No public/official result is available in the reviewed files.**  
   The M52 recursive evidence is limited to repeated Train development, with `public_automatic_launch: false` at `P/diagnostics/m52/spec.json:44` and scope fixed at `P/diagnostics/m52/recursive_result.json:714`.

2. **Only one seed is evidenced.**  
   Seed 2026 is fixed at `P/diagnostics/m52/spec.json:25` and asserted by training code at `P/overlay/tools/train_sttrack_m52.py:196-198`. I found no multi-seed evidence in the requested artifacts.

3. **The private recomputation copies are not publishable evidence by themselves.**  
   `S/review_data_binding.json:2` scopes them as private read-only review copies. I verified them in this environment, but they are not a public audit package.

4. **This review is advisory and GPT-family only.**  
   It should not be described as a cross-family acquittal or independent external certification.

---

## Prioritized action items

1. **Record M52 as a completed negative recursive result.**  
   The correct result statement is that neither mixed state-data training nor control extra training passed the fixed advancement gates; `advancing_arm` is null.

2. **Do not claim M52 state-data benefit, extra-training advancement, or public benchmark improvement.**  
   The recursive evidence directly blocks those claims under the pre-fixed rules.

3. **If reporting static results, label them as static candidate-set proxy results.**  
   Static development gains are real-GT but do not transfer to recursive performance in this run.

4. **If using the affinity diagnostic, state the exact limit.**  
   It supports only that the present cached direct affinity readout does not improve the static development readout protocol; it does not support online identity propagation or causal tracking claims.

5. **If a public audit trail is needed, publish a non-private manifest or artifact bundle.**  
   The private staged copies let me recompute the result here, but they cannot substitute for publishable evidence if external readers need to reproduce the arithmetic.