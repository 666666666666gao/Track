# M122 metrics-resume source and integrity review

Date: 2026-10-08T07:29:44.5520840+00:00
Verdict: **PASS — 0 blocking findings; 3 nonblocking observations.**
Requested reviewer: `gpt-6-astra`, reasoning `max`. Fresh review context, **same-family / provisional**. Actual backend identity is **independently unverified**.

PASS for the proposed single-owner source/launch path, with 0 blocking findings. The mplt switch addresses the observed import failure without replaying OPE inference; it does not establish completed metrics. Four OPE analyses and both full VOT evaluations remain to run, and completed outputs require a separate result audit. Any performance report must disclose initialization text reviewed with multiple frames and the single seed, 2027.

This verdict covers one launch of the reviewed source with a fixed new control directory. It is not an audit of completed benchmark results. No SSH, network, GPU, training, prediction inference, or experiment execution was performed; no experiment source was edited.

## Primary evidence and exact source pins

The controller and plan were read directly, as were the original suite, prepare/runtime/OPE/VOT/seal/collect entries, locked PR scorer, frozen TraX bridge and failure-count helper interfaces. All four actual plan/case/receipt sets were inspected in full by deterministic JSON checks without printing their arrays.

- `resume_m122_evaluation_metrics.py`: `6a364b7eb81a02cd4673a7d9209d837087f6aca912b6c0c217977104eef33280`
- `M122_METRICS_RESUME_PLAN.md`: `1761b5e6a33088830c1def5a2a9a0069e80b13033a79ca721a924ecddafff443`

In references below, `S/` means `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/`, `A/` means `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/`, and `H/` means `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1511/`. The JSON report records SHA-256 for all 48 primary/reused evidence files.

The two originally requested `H/evaluation/precision*/bundle.json` paths were absent. They remain absent in the untouched snapshot. The subsequently supplied `A/static_bundle_provenance_1511/precision0_bundle.json` and `precision1_bundle.json` were read and hashed independently: they exactly match selection's `e514bc1721be562b3ee982444b78f09abcaea1584e314ca0971154e539919924` and `ab8f90cc1202dc8cac14dd9138b9e0db83e08e5d51795146df50d48d0d66a8cc`. Both 52-entry source maps equal the original gate, and all 52 digests have exact local source copies. Source-byte checks do not attest current remote bytes.

## Failure, interpreter and remaining stage coverage

Both original failure logs, lines 1–10, stop at `S/run_m122_official.py:54` while importing `vot.region` from the locked scorer. Both actual `.exit` files contain 1. `H/evaluation_control/launches.json:3,23,47,65,83,101,119,137` contains exactly eight records: encode and prepare succeeded, all four OPE prediction jobs succeeded, then both DepthTrack analyses failed. The original queue's fail-stop assertion at `S/run_m122_evaluation_suite.py:37` prevents CDTB analysis and VOT from being reached.

The raw witness at `A/metric_import_probe_raw_stdout.txt:11,22,28` records sttrack exit 1, mplt exit 0 and unchanged-entry help exit 0. `A/prepare_metrics_import_probe.py:11,17,22` shows exactly what was tested: metric import, one identical-rectangle overlap and `--help`; it does not evaluate any benchmark. The saved terminal receipt records actual exit 0. The source switches only analysis invocation to `/root/miniconda3/envs/mplt/bin/python`; no environment is rebuilt.

`S/resume_m122_evaluation_metrics.py:65` correctly schedules all four OPE analyses. Lines 71–84 then schedule both full VOT runs, each official analysis, failure/metric sealing and collection. OPE inference, encoding and training are absent from this controller. The remaining VOT schedule stays four shards in two waves, 441/441/441/442 anchors, GPUs 0/1/0/1, independently for each final.

## Pins, retention and duplicate execution

All four plan and case hashes equal selection/plan pins. Every case's sequence, frame count, initial box and GT digest equals the locked native input/reference. All receipt per-sequence entries match cases, with 50/76,373 and 80/101,956 for each model: 260 receipt entries and 356,658 OPE frames in total. Receipt final/bundle/bank pins agree with the two actual bundles and selection. Both training-result files match bundle hashes and record seed 2027, 3 passes, 456 runs and 659,406 track calls, with saved-roundtrip/frozen/buffer checks true. This review does not independently re-execute training or inspect checkpoint tensors.

`S/resume_m122_evaluation_metrics.py:16–36` checks the original gate, absent original PID, recorded failure shape, bundle/final hashes, prediction receipts and absent metrics/VOT launch. `S/run_m122_official.py:43–53` will rehash the actual raw prediction files and dataset GT before scoring. `S/m122_official_runtime.py:7–20` rechecks the original runtime/final/bank/binding pins. `S/run_m122_vot_shards.py:23–28,50` checks frozen execution inputs before and after tracking. Both actual VOT manifests preserve the exact frozen sequence and trajectory orders, 127 unique sequences and 1,765 unique trajectories.

The two original error logs, eight original launch records and exit files are read only. New logs and exits go to the fresh `--control` directory (`S/resume_m122_evaluation_metrics.py:37,53,61`). A failed phase stops at line 63 without automatic retry. The original absent evaluation success sentinel is not converted into success.

A fixed new control directory is claimed using `mkdir(exist_ok=False)` before any subprocess starts. Existing metrics or VOT launch are rejected; VOT additionally rejects existing controller logs and shard result directories (`S/run_m122_vot_shards.py:29,33`). No duplicate is evidenced. This is not a global lock across arbitrarily different `--control` paths: issue one invocation and do not use a second directory as an automatic retry. Live remote process state was not independently inspected by this reviewer.

## Interface review

No blocking source/interface defect was found in the inspected path.

- `S/full_dense_tracker.py:117–120` returns `target_bbox` and `best_score`, matching OPE at `S/run_m122_official.py:27` and TraX at line 75.
- `A/official_analysis_sources/depthtrack_pr.py:143–154` returns `sequences`, `frames`, fractional P/R/F and `precision_percent/recall_percent/f_score_percent`. These match `S/run_m122_official.py:55–58` and `S/collect_m122_official_results.py:24–25`.
- `A/official_analysis_sources/finalize_vot_transaction_low22.py:193–258` accepts the 1,765-anchor override and returns exactly `outcomes, failures, per_sequence, settings`, matching `S/analyze_m122_vot.py:29`.
- VOT EAO/ACC/ROB extraction at `S/analyze_m122_vot.py:22–24` agrees with the reused locked parser at `A/official_analysis_sources/finalize_vot_full127.py:381–388`. Failures are computed separately, not substituted for ROB.
- Collection binds each dataset row to one final and the unchanged selection, and explicitly leaves `independent_completed_audit=false` (`S/collect_m122_official_results.py:17–47`).

## Integrity checklist A–F

**A — Ground-truth provenance: PASS (source).** OPE opens the dataset's `groundtruth.txt` after prediction sealing and digest verification (`S/run_m122_official.py:48–55`; locked scorer line 97). All 260 case GT pins match the native reference. VOT uses the frozen official dataset/workspace and its GT (`finalize_vot_transaction_low22.py:213–233`). No model-generated GT is used in these paths. Raw remote GT bytes were not opened here.

**B — Score normalization: PASS.** The locked scorer uses overlap precision, visible-GT recall and macro averaging, then selects the standard best-F threshold (`depthtrack_pr.py:117–142`). Confidence-derived thresholds are not a self-normalizing score denominator. Raw fractions and percentages are both returned. VOT fractions are multiplied by 100; no model-output maximum rescales them.

**C — Result existence: WARN / pending.** Actual failure evidence and prediction metadata exist and match. There are no local current metric JSONs, prediction TXT bytes, VOT results or completed nine-metric rows to audit. `H/observation.json:14,48,168,182,196,210` also records VOT unlaunched and metrics absent. A prediction receipt or import witness cannot support a score/completion/target claim.

**D — Reachable metric code: PASS (source).** All four OPE scoring calls, both official VOT analyses, failure counting and collection are wired and their inspected interfaces match. The observed failed analysis never reached the scorer call; this is reachability, not evidence that the metrics already ran.

**E — Scope: WARN / qualification required.** Two configurations, one seed, two fixed third-pass finals, with full Test50/CDTB80/VOT127 intended per model. Only OPE prediction completion is evidenced so far. Human input review used multiple frames/video; this must remain disclosed. No multi-seed robustness or unqualified standard first-frame-only comparison is established.

**F — Evaluation class: PASS, `real_gt`.** Dataset GT supplies the metric reference. Human-confirmed text is an input condition, not human scoring; the evaluation is neither synthetic proxy nor human-evaluation scoring.

## Nonblocking observations

1. **N1 — INFO: completed metrics remain unproven.** Evidence: `H/evaluation/precision0/depthtrack_test/predictions/receipt.json:2`, the other three receipts, `H/observation.json:155`, `A/prepare_metrics_import_probe.py:11,17`. Claim impact: no current P/R/F, EAO/ACC/ROB, full-completion or joint-pass claim.
2. **N2 — WARN: disclose input and seed scope.** Evidence: `S/M122_OFFICIAL_EVALUATION_PLAN.md:5`; both actual bundles line 72; `S/m122_official_runtime.py:16`. Claim impact: report dataset-GT performance under human-confirmed text reviewed with multiple frames, single seed 2027, and both separate final-specific rows.
3. **N3 — LOW: “four analyses are retried” is inaccurate wording.** Evidence: `S/M122_METRICS_RESUME_PLAN.md:9` versus original launch list lines 119 and 137. Only the two DepthTrack analyses are retries; the two CDTB analyses were not yet started. The controller's four-job coverage is correct.

No change to the frozen original scorer, GT, final choice or evaluator sources is required by this source review. Final checkpoint files, PT bank, raw prediction TXT files and remote GT/image bytes were not available in the local snapshot; the reviewed runtime must pass its existing hash checks against them. Completed outputs require a later result audit.
