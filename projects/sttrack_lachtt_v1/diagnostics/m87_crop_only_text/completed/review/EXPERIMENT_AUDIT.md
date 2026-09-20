# M87 completed training and four-content integrity audit

Date: 2026-09-21 (Asia/Shanghai). Auditor: fresh native Codex reviewer, requested route `gpt-6-astra`, reasoning `max`; `review_independence: same-family`, `acceptance_status: provisional`. This records the requested routing, not independently observed backend model telemetry. Reviewer task: `/root/m87_completed_integrity`.

**Overall integrity verdict: WARN. Deterministic verification: PASS, 1095/1095 checks. Frozen scientific acceptance: FAIL, 2/18 gates passed.** No numerical mismatch, fabricated evaluation target, missing claimed completed result, or unresolved implementation defect was found in the audited completed-run evidence. WARN concerns the limits of historical provenance/operation-order reconstruction and inactive inherited evaluation code; it does not turn the failed scientific result into a success.

The supplied completion narrative is suitable for reporting this failed frozen development experiment with its existing scope qualifiers. Its reviewed SHA256 is `d1af44810868dd6fb7bc218abe15c29ecc7b983d73813c40179fc95a80cbfc59`. No further GPU work, new training seed, new caption, native rerun, or formal evaluation is required to report this result.

## Evidence roots and method

- `E` = `D:\Program Files\UserCache\gb\codex\tmp\sttrack_m87_crop_only_text_20260921\completion_collection\evidence`.
- `M84` = `D:\Program Files\UserCache\gb\codex\tmp\sttrack_m84_centered_20260920\completed`.
- Completion narrative = `D:\Program Files\UserCache\gb\codex\tmp\sttrack_m87_crop_only_text_20260921\completion_collection\COMPLETED_REPORT.md`.
- Historical published claims = `C:\Users\gb\.codex_track_publish_m29_20260902\docs\RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md`, lines 22021–22127, sections 5.179–5.184.

The reviewer read the active training/inference/metric pipeline and architecture sources directly, parsed all 88 current raw prediction files and all 22 supplied GT files, and independently calculated scalar IoU, maximal low-IoU runs, aggregates, all frozen gates and 66 leave-one-sequence-out values. The verifier does not import the experiment's metric function. It also recalculates historical native and M84 Category metrics from locally preserved raw trajectory files. M82 numbers are only reaggregated from hash-bound historical per-sequence result values.

All 352 manifest entries and their sizes/hashes match; the completion archive is 16,079,293 bytes with SHA256 `5af9d14d6af98b653800f0a96a18f4241569681bb71c8574a5792d1f00da6b7b`. The manifest itself is the additional archive member, not one of the 352 listed evidence files. All 161 integration source hashes match. Reproducible commands and full results are in `verify_completed.py`, `deterministic_verification.json` and `audited_input_hashes.json`. This review performed zero optimization steps and zero tracker inference calls.

## A. Ground-truth provenance — WARN

The performance evaluator uses dataset `groundtruth.txt` files from the frozen DepthTrack Train dataset path. The 22 supplied GT files match all frozen `recursive_spec.json` hashes, match each initialization box and frame count, and also match the GT hashes in the older native reference's frozen source specification. No ground-truth target is constructed from the current tracker predictions. This part passes. Evidence: `E/training_spec.json:8`, `E/recursive_spec.json:14–256`, `E/run_recursive.py:102–105`; verification checks `GT_hash:*`, `GT_length_init:*`, `GT_identical_to_prior_frozen_dataset_identity`.

The active inference function receives only the initialized box and fixed text bank, then reads RGB/depth frame paths and calls `tracker.track(frame(i))`. It contains no subsequent GT loading or caption generation. The analyzer verifies all four complete receipts and all 88 prediction hashes before its first subsequent-GT read. The controller waits for both first-wave runs, then both second-wave runs, then invokes analysis. Evidence: `E/run_recursive.py:53–73,81–105`, `E/run_m87.sh:17–27`, `E/collection/collect_completed.py:60–80`.

The historical assertion that no process accessed later GT before every family was sealed cannot be proven solely by current code, hashes, normal exits, self-reported receipt booleans, or wall-clock fields. There is no independent immutable file-access transcript in this package. Accordingly this is a verified code/receipt contract with a provenance limitation, not a proven operating-system history. No evidence contradicting the contract was found.

Training does load each complete fitting GT file before iterating its frames, then passes the current GT to supervision only after `tracker.step` commits the predicted state. Thus `gt_after_prediction_for_loss_only` should mean **use in the loss after prediction**, not literal file access only after prediction. The fitting GT never appears in the `step` interface. Evidence: `E/train_causal.py:99–107,124–129`; `E/causal_training.py:21–69`. This distinction prevents overstating the causal implementation contract.

## B. Score normalization — PASS

Performance is continuous axis-aligned xywh IoU, intersection divided by union. Initialization is excluded. Non-finite GT and non-positive GT width/height are excluded. A low frame is valid IoU ≤0.1; H10 counts each maximal uninterrupted run of at least ten such frame indices once. Invalid GT interrupts a run; the implementation does not compact invalid frames away before H10. Each arm has 33,130 stored frames = 22 initialization + 4,211 invalid subsequent GT + 28,897 valid scored frames. Macro IoU gives each of the 22 sequences equal weight. No metric is divided by the model's own maximum or confidence. Evidence: `E/recursive_metric.py:14–30`; `E/run_recursive.py:106–112`; independent `verify_completed.py:51–85`.

The native-preservation loss does normalize native and student raw response maps by their respective spatial sums before KL. This is a training probability distribution, with a detached native teacher, not a reported tracking performance score or replacement GT. Eligibility uses real fitting GT and same-state native IoU ≥0.5. The hard-negative competition probability is likewise a training diagnostic. Evidence: `E/native_preservation.py:6–13,23–44`; `E/window_competition.py:11–42`. Neither is presented as official accuracy.

## C. Result existence, identity and numbers — PASS

The final artifact is `E/training/category/final.pth`, SHA256 `dbf86cdf5fc5f9b30863725ca0e95cb0cbbde7e65977a5bfb02387d89fbc756b`, 3,495,824 bytes. Its metadata independently decodes to one Category head, `semantic_spatial_centered_v1`, seed 2027, complete status, 130 sequences, 186,694 track calls, 5,798 optimizer steps and the frozen training-spec hash. The independent CPU decoder permits only OrderedDict, Float/Bool storage, and tensor rebuild metadata; it does not load Torch or execute pickle-defined globals. All decoded model/optimizer storages are finite; the 23 optimizer state step values are all 5798. Model parameters excluding the 768-element Empty buffer total 289,154. `checkpoint_verification.json` records every stored tensor shape and all optimizer step values.

All 130 sequence-log rows follow the frozen fitting order. Track-call sums, inside/outside/invalid label totals (157,324 / 17,155 / 12,215), terminal competition/native-preservation counters and optimizer budget agree with the final result. All 3,917 sampled trace rows match the exact first-frame/every-50/final-frame schedule and initialize from the frozen first box; observed template writes match frame%50==0 and score>0.75. Evidence: `E/training/category/result.json:1–37`; `E/training/category/sequence_log.jsonl:1–130`; `E/train_causal.py:86–91,147–195`; independent `training_log_verification.json`.

The four complete inference receipts bind the same final checkpoint, training result and recursive spec; their 22 sequence records match raw hashes and stdout records. The controller, training, four inference and analysis exits are all zero. Every raw row has the expected frame index, finite positive-size box and a finite noninitial score. The analysis is demonstrably numerically reproducible from these saved inputs. Existing published section 5.184's training-only numbers agree; its statements about incomplete development were explicitly historical observations, not claims that override the now-completed receipts. The new completion narrative's aggregate, example, gate, interval and write-count numbers match the recomputation.

M84 comparison: the locally available M84 training spec hashes to the frozen parent identity. Seed, architecture, exact 130-sequence order, loss definitions, optimizer settings, state/update protocol, initial-checkpoint hash and all 161 integration source bytes are unchanged. `spec_comparison.json` preserves all actual differences, including intended text protocol/banks, ancestry, artifact identities and four-content acceptance definition. Current artifacts support a matched **protocol** comparison, not a single-factor causal separation of prompt changes from removal of full-frame context.

## D. Dead-code and actual call-path check — WARN

The actual metric path is `run_m87.sh` → `run_recursive.py --analyze` → imported `recursive_metric.statistics` → `episodes`. The current aggregate/gate/LOO result values are independently reproduced and stdout agrees with receipts. These performance functions are live. Evidence: `E/run_m87.sh:27`, `E/run_recursive.py:76–139`, `E/recursive_metric.py:14–30`.

`recursive_metric.py:33–87` also contains an inherited standalone `main()` for a different experiment layout (`spatial`/`pooled`, old hardcoded paths). M87 never calls that main. It is an inactive archived entry point, not evidence that M87 evaluated those old arms. Similarly `support_loss.support_supervision` is inactive in this frozen weight-zero experiment; the active path is `native_preservation.supervision` → `window_competition.supervision` → `causal_training.base_supervision`. Evidence: `E/train_causal.py:57,127–129`; `E/causal_training.py:103–113`; `E/support_loss.py:3–32`. Preserve the sealed source; documentation should identify the actual entry point rather than encourage direct execution of the inherited main.

The runtime architecture genuinely applies the shared subtraction before the frozen Center Head. Evidence: `E/code/lib/models/sttrack/centered_semantic_adapter.py:12–20`; `E/code/lib/models/sttrack/sttrack.py:144–166`. Training freezes the entire network before enabling only adapter gradients and runs native extraction under no-grad; same-state native teacher boxes are read before the student public state is committed. Query state is native and detached, the crop uses the previous predicted box, and template updates use the student's clipped current prediction. Evidence: `E/causal_training.py:13–69`; `E/code/lib/test/tracker/sttrack.py:96–139`; `E/code/lib/test/tracker/sttrack_semantic.py:20–30`.

## E. Scope assessment — PASS with the stated ceiling

The user explicitly selected seed2027; no multi-seed requirement is added. Evidence supports one complete fitting pass over 130 sequences and four full content recursions of one trained final head over 22 **repeatedly used DepthTrack Train development** sequences. Four text conditions are not four independently trained models. Current fitting/development sequence sets are disjoint, but prior repeated development use means this is not an untouched external test or formal VOT/DepthTrack Test/CDTB result. The narrative respects this scope. Evidence: `E/EXPERIMENT_PLAN.md:3–25`, `E/run_recursive.py:135–136`, `E/recursive_result.json:1452–1454`, completion narrative lines 3,17,33–39.

Generated categories and pre-result assistant screening statuses are not semantic truth. Swapped deliberately uses a different string, which need not be contradictory or wrong. The tighter-crop protocol also changed the requested output style, so results do not isolate removal of the full-frame view. Neither isolated positive sequences nor posthoc screening subgroups establish correct-word causality. The negative gates justify declining promotion of this frozen candidate; they do not refute every language-tracking method.

## F. Evaluation type — PASS

- Tracking performance: `real_gt`, custom continuous-IoU development analysis. It does not run or report an official benchmark evaluator.
- Native spatial preservation during fitting: `self_supervised_proxy` teacher distribution, with real-GT eligibility; a training regularizer, not performance ground truth.
- Caption generation and assistant semantic screening: automatic/proxy information, not independently validated human labels or a real-GT caption-accuracy evaluation.
- Empty equality: deterministic functional/reference-consistency check, not a semantic or external-generalization metric.

## Independently recomputed completed results

| Condition | Frame mean IoU | Macro sequence IoU | Low-IoU frames | H10 |
|---|---:|---:|---:|---:|
| Native preserved raw reference | 0.652226263176 | 0.684336416360 | 7397 | 75 |
| M84 protocol control, raw predictions | 0.708521427828 | 0.700454335697 | 5541 | 72 |
| M82 historical values, reaggregated only | 0.729520895605 | 0.744777274298 | 4952 | 67 |
| M87 new Category | 0.643113940569 | 0.662655027769 | 7694 | 75 |
| M87 same-weight Empty | 0.652226263176 | 0.684336416360 | 7397 | 75 |
| M87 same-weight Old | 0.703120605620 | 0.709197245845 | 5731 | 67 |
| M87 same-weight Swapped | 0.698047290952 | 0.705661978370 | 6001 | 73 |

Scalar summation agrees with saved values within 1e-9 absolute numerical tolerance; integer counts and gate booleans agree exactly. Every per-sequence row is retained in `recomputed_metrics.json` and `per_sequence_recomputed.csv`.

| Frozen gate | Result |
|---|---|
| 1. M84: strictly higher frame mean | FAIL |
| 2. M84: macro mean not lower | FAIL |
| 3. M84: low-IoU frames not higher | FAIL |
| 4. M84: H10 not higher | FAIL |
| 5. Native: strictly higher frame mean | FAIL |
| 6. Native: macro mean not lower | FAIL |
| 7. Native: low-IoU frames not higher | FAIL |
| 8. Native: H10 not higher | PASS, 75=75 |
| 9. Native: protect every native zero-H10 sequence | FAIL, mobilephone02_indoor |
| 10. Same-weight Old: strictly higher frame mean | FAIL |
| 11. Same-weight Old: macro mean not lower | FAIL |
| 12. Same-weight Old: low-IoU frames not higher | FAIL |
| 13. Same-weight Old: H10 not higher | FAIL |
| 14. Same-weight Swapped: strictly higher frame mean | FAIL |
| 15. Same-weight Swapped: macro mean not lower | FAIL |
| 16. Same-weight Swapped: low-IoU frames not higher | FAIL |
| 17. Same-weight Swapped: H10 not higher | FAIL |
| 18. Every sequence Empty/native metric parity | PASS |

The failing protection sequence has native mean 0.851274441112, low-IoU=1, H10=0, versus Category mean 0.560629810183, low-IoU=208, H10=1. All 66 content LOO deltas were recomputed: Category–Empty is positive only after removing `notebook02_indoor` (1/22; range −0.0335857888954 to +0.0210150088059). Category–Old is negative for all 22 removals (range −0.0879746505087 to −0.0192999133020); Category–Swapped is negative for all 22 (range −0.0694042290158 to −0.0327488082738). These are descriptive sensitivity checks, not additional frozen acceptance gates.

## Exact Empty and posthoc checks

M87 Empty is exactly equal to the supplied M84 Empty reference in all 33,130 boxes and all 33,108 noninitial scores; the 22 null initialization scores also match. The reviewer additionally checked M84's locally preserved native raw excerpts: all 33,130 boxes and 33,130 stored score fields match those files too. Their per-file manifest and source-spec hashes pass. This is independent arithmetic over historical files, not a new native inference or proof of the full original shard-extraction history. Evidence: `E/collection/collect_completed.py:85–101`; `M84/native_reference/extract_native_reference.py:11–28`; `empty_exact_reference_verification.json`.

Posthoc strict damage intervals (Category IoU≤0.1 while reference IoU≥0.5 on each of ≥10 consecutive valid frame indices) reproduce exactly: Empty 28/2015 segments/frames, Old 24/2991, Swapped 20/2549. Reverse improvement intervals are 22/1768, 15/1068 and 14/859. They are not total H10 counts. Reconstructed template-write totals are Category/Empty/Old/Swapped 227/292/273/268; severe-overlap writes are 4/27/18/15 and IoU≥0.5 writes are 219/256/250/242. These counts follow the frozen conditional update code and saved scores, not a newly observed template log. Diverged recursive histories preclude causal attribution to write counts. Assistant-proxy subgroup arithmetic also matches; its labels remain a proxy. Evidence: completion analysis script lines 20–81,93–115; `posthoc_verification.json`, `strict_intervals_recomputed.csv`, `template_writes_recomputed.csv`.

## Historical limits and exact uncollected artifacts

The following are not failures of the supplied numerical results, but they limit stronger historical claims:

1. The original base file `/root/autodl-tmp/sttrack_checkpoints/STTrack_Vot22.pth.tar` and the initialization artifact `/root/autodl-tmp/sttrack_m87_crop_only_text_20260921/native_parity/category_zero.pth` are not included in the local completed package; no local M87 initial checkpoint was found. Their identities are checked through frozen metadata and unchanged parent-spec hashes, not by independently reloading those files in this audit. The requested local counterpart `E/native_parity/category_zero.pth` is absent. This does not assert that the remote originals are absent.
2. Training checks base parameters/buffers by hashing before and after and asserts no base gradients (`E/train_causal.py:71,179–180`), then records the result. The package contains the before-hash and completion boolean, not complete before/after base snapshots; the reviewer did not independently replay the base computation or compare full saved base tensors.
3. Only sampled causal states are saved. `E/train_causal.py:147–151` incrementally hashes the full visited state stream, but never serializes the complete stream to a file. Therefore `full_visited_state_stream_sha256` cannot be independently recomputed from 3917 samples. Complete per-sequence totals and final checkpoint budget can be verified; all 186,694 historical state transitions cannot.
4. Original native source shards `/root/autodl-tmp/sttrack_innovation_v1/risk_recovery_full152_v1/shard0.json` and `shard1.json` are referenced by `M84/native_reference/reference_provenance.json` but not supplied locally for replaying extraction. The preserved extracted 22 trajectories are present, hashed and independently compared. Full fitting images and 130 fitting GT originals under `/root/autodl-tmp/depthtrack/train/sequences/<fit_sequence>/` are likewise not part of this completion package; no full fitting-data/runtime replay is claimed.
5. No M82 raw trajectory family is included in `E/references`; its historical per-sequence values in `E/references/M84_recursive_result.json` are reaggregated, not independently reconstructed from M82 boxes. M82 is not part of the 18-gate required primary comparison.

## Claim impact and required reporting actions

- Supported: complete fixed seed2027 training and final saved budget; all four complete content recursions; exact reproduction of the failed 18-gate development result; exact Empty equality to supplied historical references.
- Needs the existing qualifiers: unchanged base and initialization history; all-four-before-GT historical ordering; original native extraction provenance; semantic screening/subgroups; reconstructed template writes.
- Unsupported: M87 acceptance/promotion; correctness of generated categories as ground truth; choosing Old posthoc as the preregistered successful model; correct-word causal mechanism; new official or external benchmark improvement; general safety or generalization from Empty equality.

Retain and report all failed gates and raw negative rows. Keep the single-seed scope selected by the user. Identify `run_recursive.py --analyze` as the active metric entry point. Preserve sealed evidence and existing historical attribution; do not edit the unused archived metric main merely to clean it up. The reviewed completion narrative already preserves these essential boundaries. No unresolved concrete defect requiring code or data changes was identified.

## Review and verification records

`EXPERIMENT_AUDIT.json` carries the provisional semantic verdict and claim impacts. `deterministic_verification.json` contains every individual check, including file bindings, checkpoint storage verification, raw metrics, all gates and posthoc arithmetic. `audited_input_hashes.json` records all read artifact identities, including the reviewed narrative and historical published claim snapshot. The full reviewer request/response record is under `.aris/traces/experiment-audit/2026-09-21_completed/` within this review directory.
