# M122 posthoc OPE source review

**Verdict: PASS (source readiness only). Blocking findings: 0. Nonblocking warnings: 2.**

PASS for source readiness only: no blocking source, formula, provenance-chain, schema or Python 3.8 syntax defect found. New posthoc computation has not run.

Requested reviewer: `gpt-6-astra` / `max`, native Codex. Review independence: **same-family**; acceptance: **provisional**. Backend/model/effort are not independently attested. No server contact, new analysis execution, inference, dependency installation or source/input mutation occurred.

Source SHA256: `sha256:673b5f516c93611386aca46fa839600e49fc4e3aeb91d3260f909ca5124a8626`.

Trace: `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/traces/experiment-audit/2026-10-08_m122_ope_posthoc_source`.

## A-F checks

### A. Ground-truth provenance and legal posthoc inputs: PASS

Ground truth is loaded from plan.dataset_root/sequence/groundtruth.txt and pinned to the GT SHA in the pinned cases file. It is never derived from predictions. Saved boxes and confidences are separate receipt-pinned inputs.

The locked scorer maps finite positive-size GT rectangles to VOT rectangles and invalid/unknown GT to Special(0), using image-bounded VOT rectangle overlap. The new script checks finite [0,1] overlaps and at least one valid GT frame per sequence.

The original runner seals all OPE prediction files before ope_analyze opens subsequent annotations. The new program contains no tracker, optimizer or neural invocation; reading evaluation GT here is posthoc only.

Human multiframe review is explicitly asserted and retained in the new result metadata. No strict first-frame-only semantic input claim is justified. Raw remote GT/TXT/RGB bytes were not available for this local review; A passes source provenance, not byte-level replay.

Evidence:

- `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:81` — Separate receipt-pinned boxes and confidence loading; GT is independently loaded at line 83.
- `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:89` — Calls the locked _vot_overlaps implementation; finite bounded overlap/visible-GT checks follow at line 90.
- `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/official_analysis_sources/depthtrack_pr.py:40` — Prediction validation, dataset GT Rectangle/Special mapping, bounded calculate_overlaps and valid-GT mask.
- `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m122_official.py:37` — Prediction receipt is completed before the separate analyzer; lines 51-55 hash GT and invoke scoring.
- `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision0/depthtrack_test/plan.json:11` — Dataset root is /root/autodl-tmp/depthtrack/test/sequences.
- `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision0/cdtb/plan.json:11` — Dataset root is /root/autodl-tmp/CDTB/sequences.
- `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:48` — Multiframe-aided human input flag is required; retained at line 129.

### B. Score normalization and aggregation: PASS

For sequence s, selected frames satisfy confidence >= the existing model/dataset threshold. P_s is selected overlap mean (1 when no frames selected); R_s is selected overlap sum divided by the count of valid GT frames. Multiplication by 100 is only percentage conversion. No score is divided by its own prediction maximum, minimum or mean.

All-box R_s uses every saved overlap divided by the same valid-GT count. With the locked nonnegative overlap convention, removing filtering gives an upper bound on recall of these fixed boxes. It is not a new tracking score or an upper bound over alternative trajectories. The filtering cost is this cap minus original recall.

Dataset P and R are sequence-macro means. Dataset F is their harmonic mean, not mean per-sequence F. Reconstructed P/R/F must each agree with the corresponding already pinned aggregate file within 1e-8 percentage points.

The original scorer used model confidences to choose PR threshold locations and maximum aggregate F; that is threshold selection, not a prediction-statistics normalization denominator. The new source reuses the four saved finite thresholds and does not select new ones.

Evidence:

- `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:74` — Loads the already saved threshold and requires a finite scalar.
- `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:91` — Selection, precision, valid-GT recall and all-box recall formulas.
- `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:104` — Macro P/R, harmonic aggregate F and exact 1e-8 agreement assertions.
- `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/official_analysis_sources/depthtrack_pr.py:25` — Original threshold locations use confidence ranking.
- `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/official_analysis_sources/depthtrack_pr.py:123` — Original selected-overlap formulas and macro aggregation at lines 133-142.

### C. Existing results, identifier chains and planned outputs: WARN

All 25 requested artifacts exist and were read directly; their final hashes equal the supplied pre-review input manifest. All four metric files have complete status and exactly match EXPECTED_METRICS in the new source.

Independently checked selection -> bundle -> plan -> cases/receipt/metrics and final/source/bank/binding identifiers, all referenced JSON keys, exact constructed remote OPE paths, sequence order, unique names, per-sequence frame counts and total counts. No current schema, key or path mismatch was found.

There are exactly 50/76,373 and 80/101,956 sequence/frame records per model: 260 sequence/model rows and 356,658 frame/model records overall. Both models use byte-identical cases per dataset. Four prior CPU analyzer logs equal the metric numeric objects and each corresponding exit file is 0.

Local prediction directories contain zero TXT, GT and RGB files. Remote final/bank/binding contents and actual image/GT/box bytes were not rehashed here; the new script will check its pinned remote inputs during execution. The first RGB file has a run-time recorded hash but no historical expected image hash in this script; it is used only for dimensions, matching the old scorer.

result.json and per_sequence.csv in the new dedicated remote output directory remain planned. This audit did not execute the new script or observe remote output existence; no all-box cap, H10 or signed-contribution number is asserted. The previous result audit was read as an artifact, not adopted as this verdict.

Evidence:

- `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:11` — Four metric artifact SHA256 constants anchor the input chain.
- `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:38` — Selection and bundle/plan/receipt/final/hash checks continue through line 70.
- `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:77` — Complete sequence identity and row-count checks before metric computation.
- `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:85` — First RGB is freshly hashed and decoded for dimensions, without an expected historical hash.
- `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:133` — New result.json and per_sequence.csv writes occur only after all input rehashes.
- `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/M122_POSTHOC_OPE_PLAN.md:11` — Plan states there are no new per-sequence or all-box recall values yet.
- `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision0/depthtrack_test/predictions/metrics.json:2` — Complete existing OPE metric object, counts at lines 11-12 and provenance IDs at lines 14-18.
- `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision0/cdtb/predictions/metrics.json:2` — Complete existing OPE metric object with the expected source/final/plan/receipt IDs.
- `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision1/depthtrack_test/predictions/metrics.json:2` — Complete existing OPE metric object with finite threshold 0.075227.
- `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision1/cdtb/predictions/metrics.json:2` — Complete existing OPE metric object with finite threshold 0.165786.
- `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation_resume_control/launches.json:8` — Existing mplt CPU analyzer invocation, exit 0 at line 17; other three exits at lines 35,53,71.

### D. Live call paths and mutation boundary: PASS

__main__ calls main; main calls read/pinned, imports the exact hash-locked scorer, invokes _load_rows/_vot_overlaps and low_segments, computes every per-sequence/aggregate/contrast field and writes them. Every new helper and computation is reachable; none is dead code presented as a result.

The scorer main evaluation function is called by the existing runner. The new script deliberately uses only its lower-level loader/overlap functions because it must keep the original threshold. The scorer format_metrics helper is not called by these two entry points, but no claim depends on it.

All explicit persistent writes target the separate OUTPUT/result.json and OUTPUT/per_sequence.csv. OUTPUT must not exist; no parents or input paths are created. All recorded input hashes are checked again before output creation. There is no call to training, tracking, CUDA, a checkpoint serializer, a shell, networking or a controller.

The metric module top level contains imports and function definitions only. Python import machinery can create bytecode caches: use the existing interpreter with -B when the execution contract includes avoiding those incidental writes. This is a launch flag, not a source repair. Normal non-optimized execution is required because integrity gates are assert statements.

The JSON result is written before the CSV; completion should be accepted only after successful process exit and both outputs are present. This source review does not label either write as having happened.

Evidence:

- `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:19` — pinned reads bytes and records a digest; read parses only JSON.
- `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:37` — Rejects an already existing output directory.
- `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:71` — Hash-pinned source import; helper calls at lines 81-89 and 96.
- `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:120` — All signed per-sequence contributions are constructed, then summed and checked.
- `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:125` — Rehashes all recorded input bytes before creating/writing output at lines 133-141.
- `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:145` — Actual __main__ call.
- `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/official_analysis_sources/depthtrack_pr.py:8` — Module imports cv2, numpy and VOT region helpers; no tracker entry point.
- `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m122_official.py:55` — Existing scorer evaluate_depthtrack_results call and metrics write at line 57.

### E. Scope, fixed-box cap, H10 and attribution limits: PASS

The source is restricted to precision0 and precision1, two datasets and four complete OPE inputs; it retains every one of the 260 sequence/model rows. Directly read training records both specify seed 2027 and are byte-bound to the corresponding bundles. This is one seed, not two independent seed replications.

Per-sequence P/R/F use that model/dataset original global threshold; per-sequence F is descriptive and cannot be averaged into official dataset F. The aggregate assertions enforce the correct distinction.

H10 is each consecutive run of valid-GT frames with IoU <= 0.1 for at least 10 frames. False sentinels and diff locate zero-based half-open [start,end) intervals; invalid GT interrupts a run. Longest_H10_frames is zero when no qualifying run exists. These counts are neither physical-absence labels nor VOT ROB.

Each signed contribution is (P1 all-box recall - P0 all-box recall)/dataset sequence count; the sum must equal the difference of the two macro caps. No negative sequence is filtered out. This is arithmetic between two entire recursive trajectories and does not isolate semantic content, loss, template or module effects.

The targets 64.9 and 75.6 are only recall reference levels. The >= Boolean means the fixed-box cap can reach at least that recall level; it does not establish strict joint P/R/F acceptance or a deployable improvement. Neither exact crop/query/template reconstruction from rounded TXT nor full VOT/full-goal completion is claimed.

Evidence:

- `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:40` — Exactly two named models; line 50 fixes Test50/76,373 and CDTB80/101,956.
- `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/train_precision0/result.json:5` — Seed 2027; training record SHA matches precision0 bundle training_result_sha256.
- `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/train_precision1/result.json:5` — Same seed 2027; training record SHA matches precision1 bundle training_result_sha256.
- `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:31` — False-padded transitions and length >= 10 define half-open H10 intervals.
- `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:95` — Severe mask requires valid GT and IoU <= 0.1.
- `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:110` — Fixed saved-box macro recall cap and recall-only target reachability.
- `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:120` — Signed complete ledger and exact macro contribution identity.
- `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:127` — Scope flags reject crop reconstruction, VOT and text causal increment claims.
- `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/M122_POSTHOC_OPE_PLAN.md:9` — Full 260 rows, fixed-trajectory cap, signed arithmetic and invalid-GT limits are stated.
- `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/M122_POSTHOC_OPE_PLAN.md:13` — Full two-model/three-dataset acceptance remains outside this diagnosis.

### F. Evaluation classification and Python 3.8 launch readiness: WARN

Classification is real_gt: dataset-provided annotations are used for descriptive diagnosis of sealed predictions. The H10 and fixed-box recall diagnostics are auxiliary real-GT analyses, not synthetic or self-supervised reference metrics.

All three reviewed Python sources parse successfully under Python 3.8 grammar using CPython 3.12.12 ast.parse(feature_version=(3,8)); the new AST also compiles without import or execution. Manual inspection finds no newer syntax, evaluated built-in-generic annotation, union annotation or newer standard-library API requirement.

Local original launch records name /root/miniconda3/envs/mplt/bin/python for all four successful CPU scorer runs, corroborated by log/metric equality and zero exit files. The new source uses the same cv2/numpy/VOT overlap dependency route and introduces no new package dependency.

The actual new script has not been imported or executed under remote Python 3.8 by this reviewer. Interpreter/package versions, current remote files, output-path absence and run-time assertions remain execution evidence to collect; this is a nonblocking source-review limit, not a defect or a completed-result claim.

The reviewed source is ready for the planned CPU launch without source changes, using that existing mplt interpreter in normal mode with -B. Retain a successful exit plus both output files and then perform a separate result audit before reporting new diagnostics.

Evidence:

- `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:2` — Imports use Python 3.8-compatible standard modules plus existing cv2/numpy.
- `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:71` — Dynamic import path reuses the locked VOT-based scorer.
- `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:126` — datetime.now(timezone.utc), dict/list operations and output methods are Python 3.8 compatible.
- `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation_resume_control/launches.json:8` — Concrete existing mplt interpreter path; CPU mode gpu=null at line 6 and exit 0 at line 17.
- `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/M122_POSTHOC_OPE_PLAN.md:11` — Execution is planned in existing mplt without dependency installation; current new results absent.

## Existing OPE artifacts, not new posthoc results

| Model | Dataset | Sequences | Frames | P (%) | R (%) | F (%) | Saved threshold |
|---|---|---:|---:|---:|---:|---:|---:|
| precision0 | depthtrack_test | 50 | 76373 | 66.4238873979 | 63.2448915400 | 64.7954208955 | 0.075045 |
| precision0 | cdtb | 80 | 101956 | 72.0446692972 | 70.1745980406 | 71.0973386875 | 0.138744 |
| precision1 | depthtrack_test | 50 | 76373 | 66.2088822305 | 63.7778947382 | 64.9706565561 | 0.075227 |
| precision1 | cdtb | 80 | 101956 | 74.4790626277 | 69.7965033244 | 72.0617951902 | 0.165786 |

These values were read from existing metric JSONs and checked against their corresponding original CPU analyzer logs. No new cap, H10 or per-sequence number was computed. The 260 rows / 356,658 frame records describe the planned reuse scope.

## Actions and limits

- **L1 (WARN; nonblocking):** Raw remote prediction/GT/RGB bytes and new numerical posthoc outputs are unavailable to this local source review. During the planned run, require all existing byte/shape/metric-agreement assertions to pass; archive both outputs and the real exit before a result audit.
- **L2 (WARN; nonblocking):** Python 3.8 grammar compatibility is verified, but a new remote runtime execution is not. Use the already evidenced mplt interpreter; do not treat this source verdict as execution success.
- **N1 (INFO; nonblocking):** Dynamic source imports can generate Python bytecode caches even when analysis input files are only read. Launch with -B and without -O/-OO; no source change is requested.

No source repair is required. Run only the reviewed source with the existing mplt interpreter in normal mode; `-B -u` avoids incidental Python bytecode writes and retains assertions. Require successful exit plus both new files, then audit actual results before making numerical claims. Do not replace pending VOT results or infer causal module/text benefit from this analysis.

## Verification record

51 local source/metadata/hash/schema/arithmetic checks passed. All 25 requested files still match the pre-review hash inventory. Two additional training-record SHA/final pairs establish the single-seed scope; four original log/metric/exit pairs corroborate the existing scorer route. Three source files passed Python 3.8 grammar parsing under local CPython 3.12.12. The new analysis and its imports were never executed.

Two reviewer display-helper errors were retained in the machine report: null attribution display fields and an assumed training-record `sequences` key. Neither touched source or inputs; corrected read-only inspection completed. They are not production failures.

## Input hashes

Every explicitly read input is included below and in `audited_input_hashes`. Governance/continuity files are included only for read accounting; their contents do not support the experiment verdict. Generated reports are excluded as outputs.

| Input | SHA256 |
|---|---|
| `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py` | `673b5f516c93611386aca46fa839600e49fc4e3aeb91d3260f909ca5124a8626` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/M122_POSTHOC_OPE_PLAN.md` | `9e6cfad3bb0a55f797887a1e607cbdbcad3402e5763a43cba81f987fef77c4e0` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/official_analysis_sources/depthtrack_pr.py` | `05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/complete_evaluation_source_gate.json` | `81ad7a35bc04c3c8eee770ab5eaf97eea6e63ac423e1ade44f1e4a569617314a` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m122_official.py` | `d0eeacba480a19ccb717b36b7cf0d7d3fe3cb5395e7bf19d19b37868dd3354f9` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/M122_FIRST_OPE_RESULT_AUDIT_20261008.json` | `9f5d01b3606581f35c819799bc76031f11fe6a13dba74d0793214cd58a8ae702` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/selection.json` | `2ff90eb580c4f09638b79e0ec665898982d113d904ac42a93ae18166c2b45975` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision0/bundle.json` | `e514bc1721be562b3ee982444b78f09abcaea1584e314ca0971154e539919924` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision0/depthtrack_test/plan.json` | `c2d7f9773446ef215dc8a25258ed4f7407087169f7bd6051c503e75796fd6b7c` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision0/depthtrack_test/cases.json` | `158fe61d7726af5e2c1f30c76175a97a4cba56e30e2f599ec7762be034417b44` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision0/depthtrack_test/predictions/receipt.json` | `747f9c72d3a7dca18b8ff1f100c6d8226a4c4b6ce5dab144b30a1d58a518d5cb` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision0/depthtrack_test/predictions/metrics.json` | `04edc9994cc488e8cf7496f2243a58aed18b0777ebbbd8a991b936694154047d` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision0/cdtb/plan.json` | `48d34542a7d19a45895f1ac011f7ea0bcf457a0f9a495d85f799230c969699f0` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision0/cdtb/cases.json` | `b3170466a090fa1a8949e0297566c9e3ab744e789ca0c7f1bcc71633ccfae887` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision0/cdtb/predictions/receipt.json` | `c4dab6e9979ae4606c24479cfefa9b78973882fba2004604134b823606532f82` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision0/cdtb/predictions/metrics.json` | `cbfb344f6e05077dbf061669b74fb8d25d618ce425834e2923d9d8d472ee29d7` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision1/bundle.json` | `ab8f90cc1202dc8cac14dd9138b9e0db83e08e5d51795146df50d48d0d66a8cc` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision1/depthtrack_test/plan.json` | `3b355b65701b783513809ed172422ae11396d7c60ef210419a08e4101c17951e` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision1/depthtrack_test/cases.json` | `158fe61d7726af5e2c1f30c76175a97a4cba56e30e2f599ec7762be034417b44` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision1/depthtrack_test/predictions/receipt.json` | `1a7d4b5bb7464674801d4514f1dd0e586f18113ca21bb4468d12c82ec65dc1c9` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision1/depthtrack_test/predictions/metrics.json` | `88002b20f3db413ad09512b29119e5c3121f3c288693c6c78294523f1fb1bce4` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision1/cdtb/plan.json` | `fa58067626fab31db9e3b3160feb2bb03203359ca7841af5b218b75e353e3a0a` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision1/cdtb/cases.json` | `b3170466a090fa1a8949e0297566c9e3ab744e789ca0c7f1bcc71633ccfae887` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision1/cdtb/predictions/receipt.json` | `754c024821036eb425cf1ccd240ec5edbfb2a816fd2426d3d0834a4705f03977` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision1/cdtb/predictions/metrics.json` | `fd4c545f30963277fef3e6eb111a7f6e67052912a370bfa7c615d08d90aba7a6` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/train_precision0/result.json` | `4a8a44f91bea42b6d4da4d4ba467ceda335c469e354bfdf5af7b7c0f7fc0712f` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/train_precision1/result.json` | `765cb793b932898de33fc6bbfc00866cc0b598bab1dfb6aa5537bcbf196b7ed6` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation_resume_control/launches.json` | `5f629ab926deb054e8991fd0876226e07d975232783dd1746ae709d3bda35350` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation_resume_control/resume_inputs.json` | `dcb053616d28d593972615c9f0f3b118680c5ec5578cd7b084a969b33de5ffd2` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation_resume_control/precision0_depthtrack_test_ope_analyze.log` | `36e98584fcf54e3044361a11fc395f06526400d18ae4459b2a1a8222598da753` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation_resume_control/precision0_depthtrack_test_ope_analyze.exit` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation_resume_control/precision0_cdtb_ope_analyze.log` | `4458801afa3ec6a07db1185ed96c05d845961a8091578263cd5f71fd42038094` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation_resume_control/precision0_cdtb_ope_analyze.exit` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation_resume_control/precision1_depthtrack_test_ope_analyze.log` | `ec47b17587b7e8ff1f133bee813120232e21bc00e0a02203740d893204f9401c` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation_resume_control/precision1_depthtrack_test_ope_analyze.exit` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation_resume_control/precision1_cdtb_ope_analyze.log` | `7de54baa84fb817f5ac87b284e695036d001fd283c128b3368153cb6c6c71377` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation_resume_control/precision1_cdtb_ope_analyze.exit` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/traces/experiment-audit/2026-10-08_m122_ope_posthoc_source/001-source.request.json` | `fc9a4fb2e31b74c3d7224d0a220fb29968e89e5321fdd507980b0109d8132ab7` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/traces/experiment-audit/2026-10-08_m122_ope_posthoc_source/input_hashes_before.json` | `2713dde3f53b650e713407bc07f3861a59928bd9663d54914729008e4ee73a0f` |
| `C:/Users/gb/.codex/skills/experiment-audit/SKILL.md` | `0ab77f219f554cbe42f22b79d780528e61c5e58169af4c2f56acf59596ab8db7` |
| `C:/Users/gb/.codex/skills/shared-references/local-codex-policy.md` | `b0249bfccd48ff79a5976afa7a54c64a3cfb65a5496a2105c5bb21069d197783` |
| `C:/Users/gb/.codex/skills/shared-references/reviewer-independence.md` | `1aa1756a99dc47009072f0ee7bf570d797bf7e2cc84439401e2a9312d73659b0` |
| `C:/Users/gb/.codex/skills/shared-references/experiment-integrity.md` | `22150fbdc5867e7790c88f968bccb367079c2db4c190ddce980709aea7e76f60` |
| `C:/Users/gb/.codex/skills/shared-references/review-tracing.md` | `7f556fb9ef8922cfd0d047e5a2d857b0c7f9408492b8c8dcec00207e76405395` |
| `C:/Users/gb/SOUL.md` | `95c467f710183b2c0397c29ec4ef886bb723671a39ee767dc4d59165f723b68f` |
| `C:/Users/gb/USER.md` | `70ec164015065a98c2b7bf6ec8cd7d7d71156e0b05b85b4644b79b2a7e7b15cd` |
| `C:/Users/gb/memory/2026-10-08.md` | `f3d230b643b2dcab753916f56bdfd1cd9b595adf7c86356437a3e2132cdb8823` |
| `C:/Users/gb/memory/2026-10-07.md` | `0e15c61ebbc09941afb7e59148c943085341ce98c671a73f746de33e89b7617f` |
