# M122 sealed OPE posthoc result audit

**Stage: result. Verdict: WARN. Blocking findings: 0. Nonblocking findings: 1.**

The completed diagnostic has no detected blocking integrity defect. All local archive/source/provenance checks and the full 260-row arithmetic audit pass. The warning is a verification limit: original remote TXT/GT/RGB/weight/bank bytes were not available locally, so this review does not claim an independent raw-overlap replay.

Date: 2026-10-08T10:18:34.045100+00:00. Reviewer: fresh native Codex agent `/root/m122_ope_posthoc_result_review`; requested `gpt-6-astra` with `max` reasoning. Routing/model/effort are not independently attested (`backend_independently_attested=false`). Semantic judgment is **same-family / provisional**. Deterministic local checks are independently reproducible. No previous source/result audit verdict was used.

Trace: `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\traces\experiment-audit\2026-10-08_m122_ope_posthoc_result`.

## A–F checks

### A. Ground truth provenance: WARN

The inspected executed source loads dataset groundtruth.txt, pins its digest to cases.json, and applies the locked VOT bounded-rectangle overlap routine. It does not derive GT from predictions. All 260 GT ledger references resolve to the same 130 dataset GT digests across variants; first-RGB digests and legal initial boxes also match the independently hashed initialization binding. The original OPE runner opens subsequent GT only in the analysis path after sealed prediction files. However, the 130 remote GT files and RGB images themselves are absent from the supplied local evidence, so provenance is verified through source and linked records, not a fresh raw-byte/pixel replay. This is the sole nonblocking warning.

Evidence:

- [analyze_m122_ope_trajectories.py:83](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:83): Dataset GT pinned against case gt_sha256; boxes and scores come from separate receipt-pinned prediction files at lines 81-82.
- [depthtrack_pr.py:40](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/official_analysis_sources/depthtrack_pr.py:40): Finite positive-area dataset rectangles define visible GT; invalid/unknown boxes become Special(0), and bounded overlaps use calculate_overlaps.
- [run_m122_official.py:48](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m122_official.py:48): Original analysis verifies sealed prediction hashes, then opens GT; locked evaluator is called at line 55.
- [plan.json:11](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision0/depthtrack_test/plan.json:11): DepthTrack dataset root is /root/autodl-tmp/depthtrack/test/sequences; CDTB plan line 11 gives /root/autodl-tmp/CDTB/sequences.
- [cases.json:11](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision0/depthtrack_test/cases.json:11): First dataset GT digest; all four full case tables were read and checked.
- [official_initialization_binding.json:23](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/official_initialization_binding.json:23): Bound first-RGB path/digest and initial box; all 130 OPE binding entries checked.
- [result.json:7812](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/ope_posthoc_result_20261008/result.json:7812): 804-entry remote input SHA256 ledger; raw files are recorded, not included in the local result archive.

### B. Score normalization and numerical arithmetic: PASS

P is the overlap mean on frames selected by each original finite dataset threshold; R divides selected overlap sum by the count of valid dataset GT frames. Macro P/R average sequences equally, and official F is their harmonic mean. There is no normalization by prediction max/min/mean. Every CSV numeric field equals its JSON counterpart, all 260 sequence F/cost/count relations are consistent, and four independently recomputed macro P/R/F values match the original metrics within 2.842170943040401e-14 percentage points. The all-box R cap is the macro recall after removing confidence filtering while keeping every saved box and the valid-GT denominator fixed. All four caps remain below their R targets. All 130 signed contrast entries and both signed aggregate sums close.

Evidence:

- [analyze_m122_ope_trajectories.py:74](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:74): Reuse original finite threshold; P/R, all-box recall, and costs are computed at lines 91-105.
- [analyze_m122_ope_trajectories.py:106](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:106): Aggregate F is harmonic of macro P and R, followed by original metric assertions.
- [depthtrack_pr.py:117](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/official_analysis_sources/depthtrack_pr.py:117): Original threshold scan uses a common confidence grid and selects maximum macro F; this posthoc analysis does not rescan thresholds.
- [depthtrack_pr.py:124](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/official_analysis_sources/depthtrack_pr.py:124): Standard selected-overlap precision and visible-GT recall denominators.
- [per_sequence.csv:1](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/ope_posthoc_result_20261008/per_sequence.csv:1): All 260 data rows and all 15 columns were checked, with 260 unique ordered model/dataset/sequence keys.
- [result.json:24](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/ope_posthoc_result_20261008/result.json:24): P0 DepthTrack fixed-box R cap; other caps at lines 1789, 3518, 5168.
- [result.json:6891](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/ope_posthoc_result_20261008/result.json:6891): Signed whole-trajectory contrast, with every sequence retained; CDTB contrast starts at line 7246.

### C. Actual result existence, archive integrity and linkage: PASS

All 32 requested primary artifacts exist and were read in full. The original 92,670-byte result ZIP hashes to be46899a9690200909f767c9118d6ab93790ddee05060c18acc2f1119fb593ef, matching both terminal receipt and extraction record. All six archive members equal local extracted bytes; all five manifest sizes/digests match. analysis.exit contains 0; analysis.log is exactly the four JSON summaries. Native SSH/deploy/download receipts and execution.json record successful completion. All local selection/bundle/plan/cases/receipt/metrics/source links match; both variants keep the same own final across their two datasets, and the exact four original metric hashes match the analysis source constants. The 804-entry remote ledger is fully accounted for, with 21 entries independently rehashed locally and 783 remote-only entries explicitly bounded. No prior audit verdict was used as evidence of correctness.

Evidence:

- [ope_posthoc_CPU_execution_native_receipt.json:3](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/ope_posthoc_CPU_execution_native_receipt.json:3): Recorded actual SSH exit 0; complete native terminal summary at line 20.
- [ope_posthoc_results_extraction.json:3](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/ope_posthoc_results_extraction.json:3): Original archive size/hash, transfer exit and source linkage.
- [files.json:2](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/ope_posthoc_result_20261008/files.json:2): Five original member size/SHA256 records; verified directly against both ZIP and extracted bytes.
- [analysis.exit:1](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/ope_posthoc_result_20261008/analysis.exit:1): Archived analysis exit status is 0.
- [analysis.log:2](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/ope_posthoc_result_20261008/analysis.log:2): Actual log carries the completed diagnostic status and four summary rows.
- [execution.json:4](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/ope_posthoc_result_20261008/execution.json:4): Executed source hash, exit 0, interpreter/flags and 52-source before/after record.
- [selection.json:7](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/selection.json:7): P0 final/bundle/plan hashes; P1 final/bundle begins at line 29.
- [metrics.json:14](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision0/depthtrack_test/predictions/metrics.json:14): Final, bundle, plan, receipt and metric-source hash linkage; equivalent keys in all four original metrics.
- [result.json:8618](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/ope_posthoc_result_20261008/result.json:8618): Result records the same executed analysis source hash.

### D. Executed source and live numerical call paths: PASS

The compressed launcher embedded in the deployment command was decoded without execution and equals the inspected launcher text. The deployment ZIP hash equals the launcher pin; its analysis source and plan are byte-identical to the reviewed current local files. The launcher sets CUDA_VISIBLE_DEVICES empty, PYTHONOPTIMIZE=0 and PYTHONDONTWRITEBYTECODE=1, invokes the mplt Python with -B -u, and records actual exit 0. Assertions therefore remain enabled in the intended executed path. main -> pinned/read -> locked _load_rows/_vot_overlaps -> low_segments -> macro/contrast checks -> JSON/CSV/log is connected and outputs every reported field. Original run_m122_official.py invokes evaluate_depthtrack_results, whose threshold and macro functions feed the sealed metrics. format_metrics is a formatting utility, not an unexecuted claimed metric. Input hashes are checked again before outputs; launcher checks the original 52-source gate before and after. The archived receipt supports that remote guard; this audit did not independently rehash all 52 files on the server. No tracker/neural training or prediction path is called by this diagnostic.

Evidence:

- [ope_posthoc_deployment_command.txt:1](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/ope_posthoc_deployment_command.txt:1): Encoded launcher was decoded and compared with the local launcher, not run.
- [ope_posthoc_deployment_launcher.py:4](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/ope_posthoc_deployment_launcher.py:4): Parent launcher asserts __debug__; deployment/source/gate pins at lines 6, 15, 17.
- [ope_posthoc_deployment_launcher.py:19](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/ope_posthoc_deployment_launcher.py:19): Original locked sources checked before execution; rechecked at line 32.
- [ope_posthoc_deployment_launcher.py:20](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/ope_posthoc_deployment_launcher.py:20): CPU-only environment and assertions enabled; exact subprocess call at line 23.
- [analyze_m122_ope_trajectories.py:71](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:71): Pinned metric dynamically loaded and helpers actually called at lines 81-89.
- [analyze_m122_ope_trajectories.py:96](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:96): low_segments called on the real per-frame severe mask; H10 reaches JSON rows.
- [analyze_m122_ope_trajectories.py:125](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:125): Every pinned input reread before output writes; JSON/CSV/log emission follows at lines 134-142.
- [analyze_m122_ope_trajectories.py:145](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:145): main is actually the script entry point.
- [run_m122_official.py:55](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m122_official.py:55): Original locked evaluation function is called; mode dispatch at line 81.

### E. Scope, H10 meaning and attribution limits: PASS

The completed scope is exactly two frozen final variants x two OPE datasets: 50 sequences/76,373 frames and 80/101,956 per variant, hence 260 sequence-runs and 356,658 frame-runs. This repeats 130 unique sequences and 178,329 underlying frames across variants; it is not 260 distinct videos. There is one frozen final per configuration and no replicated-seed estimate in this evidence. All positive and negative sequences are retained (DepthTrack 27 positive/23 negative; CDTB 42/38). Every one of 625 recorded H10 intervals has valid integer, ordered, nonoverlapping zero-based half-open endpoints, length >=10, and count/longest/covered-frame consistency. Their actual framewise masks were not independently replayed. H10 is contiguous valid-GT IoU <=0.1; invalid GT interrupts runs and is not established physical absence. Human review used multiframe aids. Fixed saved boxes do not recover exact crops, query/template state or causal module effects. No VOT result, full nine-metric result, robustness, first-frame-only annotation, or full-goal completion is established.

Evidence:

- [M122_POSTHOC_OPE_PLAN.md:9](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/M122_POSTHOC_OPE_PLAN.md:9): Fixed-trajectory cap, signed arithmetic ledger and H10 limitations declared.
- [M122_POSTHOC_OPE_PLAN.md:11](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/M122_POSTHOC_OPE_PLAN.md:11): Six-decimal boxes cannot reconstruct exact crop/query/template state; plan is a preparation record, not a current results claim.
- [M122_POSTHOC_OPE_PLAN.md:13](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/M122_POSTHOC_OPE_PLAN.md:13): VOT and full goal are explicitly separate from this OPE diagnostic.
- [analyze_m122_ope_trajectories.py:31](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:31): H10 endpoint construction; valid-GT severe mask is defined at line 95.
- [analyze_m122_ope_trajectories.py:120](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:120): Each signed contribution is (P1-P0)/dataset sequence count; no causal attribution.
- [result.json:4](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/ope_posthoc_result_20261008/result.json:4): Scope and explicit false full_VOT/exact_crop/text-causal flags at lines 8-13.
- [bundle.json:72](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision0/bundle.json:72): Human review used multiframe aids; P1 bundle has the same qualifier.
- [collect_m122_official_results.py:40](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/collect_m122_official_results.py:40): R target values match the original nine-metric acceptance thresholds, without satisfying its VOT requirements.

### F. Evaluation type and completed-diagnostic classification: PASS

real_gt, posthoc diagnostic on sealed saved OPE trajectories. GT is dataset-provided in the inspected code, rather than model-generated reference. Completion is supported for the diagnostic outputs and local arithmetic/hash checks only. This is not a neural rerun, deployable new score, independent replay from remote raw files, VOT completion, or full-goal acceptance. The semantic review uses a fresh native Codex agent context but remains same-family/provisional; requested gpt-6-astra/max routing is not independently attested.

Evidence:

- [analyze_m122_ope_trajectories.py:83](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py:83): Ground truth comes from dataset groundtruth.txt.
- [result.json:2](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/ope_posthoc_result_20261008/result.json:2): Completed diagnostic status, with neural_calls=0 and optimizer_steps=0 at lines 5-6.
- [result.json:9](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/ope_posthoc_result_20261008/result.json:9): No full VOT claim.
- [result.json:7812](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/ope_posthoc_result_20261008/result.json:7812): Remote provenance ledger, distinguished from independently available local bytes.

## Independently recomputed numbers

Percent units; cost and shortfall are percentage points. These are arithmetic recomputations from all archived sequence rows, not raw-box overlap recomputations.

| Model | Dataset | P | R | Harmonic macro F | All-box R cap | Filtering cost | Target R | Cap shortfall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| precision0 | depthtrack_test | 66.423887397936 | 63.244891539991 | 64.795420895543 | 64.045965131219 | 0.801073591229 | 64.9 | 0.854034868781 |
| precision0 | cdtb | 72.044669297194 | 70.174598040621 | 71.097338687467 | 71.055235478658 | 0.880637438037 | 75.6 | 4.544764521342 |
| precision1 | depthtrack_test | 66.208882230491 | 63.777894738243 | 64.970656556069 | 64.398416776093 | 0.620522037849 | 64.9 | 0.501583223907 |
| precision1 | cdtb | 74.479062627712 | 69.796503324373 | 72.061795190153 | 70.449022430382 | 0.652519106009 | 75.6 | 5.150977569618 |

All four `fixed_saved_boxes_can_reach_target_R` flags correctly remain false. The original thresholds are 0.075045, 0.138744, 0.075227 and 0.165786 in table order. Mean sequence F is respectively 64.115317763811, 70.473096756170, 64.379562967509 and 70.608685658786; none was substituted for harmonic macro F.

| Dataset | Positive / negative / tie sequences | Positive sum (pp) | Negative sum (pp) | P1−P0 macro all-box R (pp) |
|---|---:|---:|---:|---:|
| depthtrack_test | 27 / 23 / 0 | 3.367721557846 | -3.015269912972 | 0.352451644873 |
| cdtb | 42 / 38 / 0 | 3.304719225388 | -3.910932273664 | -0.606213048276 |

| Model | Dataset | Valid GT frames | Selected frames | Severe frames | H10 intervals | H10 covered frames | Longest H10 |
|---|---|---:|---:|---:|---:|---:|---:|
| precision0 | depthtrack_test | 73389 | 69425 | 18376 | 228 | 17392 | 883 |
| precision0 | cdtb | 91300 | 90621 | 10222 | 99 | 9910 | 948 |
| precision1 | depthtrack_test | 73389 | 70967 | 16914 | 200 | 16069 | 588 |
| precision1 | cdtb | 91300 | 87532 | 10600 | 98 | 10294 | 696 |

The stdlib-only verifier passed 24,036 assertions. Maximum scalar arithmetic residual was 2.842170943040401e-14; signed contrast closure residual was 9.992007221626409e-15. Totals are 356,658 frame-runs, 329,378 valid-GT frame-runs, 318,545 selected frames, 56,112 severe frames and 53,665 frames in 625 H10 intervals. These structural H10 checks cannot prove the unobserved raw framewise mask.

## Provenance boundary and allowed claims

All 804 input-ledger entries were read and accounted for: 21 were rehashed from independent local copies; 783 remain remote-only (260 box TXT + 260 confidence TXT + 130 GT TXT + 130 first RGB + 2 final weights + 1 text bank). Every GT/box/confidence digest agrees with the original case/receipt record, and every first-RGB digest agrees with the pinned legal initialization binding. This establishes linked recorded provenance, not a fresh verification of those remote bytes.

The 52-entry source gate was read in full; inspected source and archived execution record show before/after remote guards. The audit independently checked the local diagnostic, metric source, original OPE runner, collector, gate and binding bytes. It did not contact the server or claim a new server-side 52-file verification.

Allowed statements:

- The CPU posthoc analysis of the four sealed OPE rows completed, and the archived local result/CSV/log/exit bytes and primary input links are consistent.
- All 260 archived sequence rows reproduce the four original macro P/R and harmonic F values within 2.842170943040401e-14 percentage points, conditional on the archived remote-overlap computations.
- For these fixed saved trajectories, removing confidence reporting filters yields the four recorded recall caps, each below its prescribed recall target; confidence filtering alone cannot make these fixed boxes meet that target.
- The complete signed sequence ledger sums to +0.35245164487331 pp on DepthTrack and -0.60621304827621 pp on CDTB for P1 minus P0 all-box macro recall.
- The reported H10 endpoints/counts are structurally and arithmetically consistent for this archived diagnostic; framewise overlap masks were not independently replayed.

Unsupported statements:

- Independent verification of all original remote prediction/GT/RGB/final-weight/bank bytes or a fresh raw-overlap replay.
- A new deployable tracker score, exact crop/query/template reconstruction, causal effect of an individual loss/text/module, or first actual state-divergence mechanism.
- VOT EAO/ACC/ROB completion or performance, H10 as VOT robustness, full nine-metric acceptance, or full goal completion.
- Replicated-seed robustness, statistical significance, or a first-frame-only human annotation protocol.

Action: preserve the raw-replay qualifier with derived claims. No input/source repair is required by this local audit. Access to original raw bytes would be needed to remove the single warning. This result does not close VOT or the full project goal.

## Full local evidence hash inventory

SHA256 covers every primary and supplemental audit evidence file read, plus this reviewer’s verifier/results/report writer. Session/persona instruction files are not experiment evidence. Archive payload hashes are also retained in the machine JSON; prior-review payloads were byte-checked for deployment integrity without consulting their verdicts.

| Local evidence path | SHA256 |
|---|---|
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/complete_evaluation_source_gate.json` | `81ad7a35bc04c3c8eee770ab5eaf97eea6e63ac423e1ade44f1e4a569617314a` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/extract_ope_posthoc_results.py` | `eaf7a839574136221a6e4596bd9d7599ac95ec19ea6461a134c7f553dcc4ba84` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision0/bundle.json` | `e514bc1721be562b3ee982444b78f09abcaea1584e314ca0971154e539919924` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision0/cdtb/cases.json` | `b3170466a090fa1a8949e0297566c9e3ab744e789ca0c7f1bcc71633ccfae887` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision0/cdtb/plan.json` | `48d34542a7d19a45895f1ac011f7ea0bcf457a0f9a495d85f799230c969699f0` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision0/cdtb/predictions/metrics.json` | `cbfb344f6e05077dbf061669b74fb8d25d618ce425834e2923d9d8d472ee29d7` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision0/cdtb/predictions/receipt.json` | `c4dab6e9979ae4606c24479cfefa9b78973882fba2004604134b823606532f82` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision0/depthtrack_test/cases.json` | `158fe61d7726af5e2c1f30c76175a97a4cba56e30e2f599ec7762be034417b44` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision0/depthtrack_test/plan.json` | `c2d7f9773446ef215dc8a25258ed4f7407087169f7bd6051c503e75796fd6b7c` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision0/depthtrack_test/predictions/metrics.json` | `04edc9994cc488e8cf7496f2243a58aed18b0777ebbbd8a991b936694154047d` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision0/depthtrack_test/predictions/receipt.json` | `747f9c72d3a7dca18b8ff1f100c6d8226a4c4b6ce5dab144b30a1d58a518d5cb` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision1/bundle.json` | `ab8f90cc1202dc8cac14dd9138b9e0db83e08e5d51795146df50d48d0d66a8cc` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision1/cdtb/cases.json` | `b3170466a090fa1a8949e0297566c9e3ab744e789ca0c7f1bcc71633ccfae887` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision1/cdtb/plan.json` | `fa58067626fab31db9e3b3160feb2bb03203359ca7841af5b218b75e353e3a0a` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision1/cdtb/predictions/metrics.json` | `fd4c545f30963277fef3e6eb111a7f6e67052912a370bfa7c615d08d90aba7a6` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision1/cdtb/predictions/receipt.json` | `754c024821036eb425cf1ccd240ec5edbfb2a816fd2426d3d0834a4705f03977` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision1/depthtrack_test/cases.json` | `158fe61d7726af5e2c1f30c76175a97a4cba56e30e2f599ec7762be034417b44` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision1/depthtrack_test/plan.json` | `3b355b65701b783513809ed172422ae11396d7c60ef210419a08e4101c17951e` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision1/depthtrack_test/predictions/metrics.json` | `88002b20f3db413ad09512b29119e5c3121f3c288693c6c78294523f1fb1bce4` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/precision1/depthtrack_test/predictions/receipt.json` | `1a7d4b5bb7464674801d4514f1dd0e586f18113ca21bb4468d12c82ec65dc1c9` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/observations/hour_1713/evaluation/selection.json` | `2ff90eb580c4f09638b79e0ec665898982d113d904ac42a93ae18166c2b45975` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/official_analysis_sources/depthtrack_pr.py` | `05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/official_initialization_binding.json` | `3bf7aa31a3f819ec746b37dc221e55c531a1235726cda3efedc4990c607cb26e` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/ope_posthoc_CPU_execution_native_receipt.json` | `46eb5fe9de60b9a0220b7d89b187411988c621f58f68005a46dea00ba71f1669` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/ope_posthoc_deployment.zip` | `c38b888e110bf7b68acfe7db2d18bc30e14b89ccad27b3308812dafbf423dc45` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/ope_posthoc_deployment_command.txt` | `37fe5b747bf1b4792c3887e25ed672deb7ec0e0dfaf8dfce50132fd250c95e00` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/ope_posthoc_deployment_launcher.py` | `8eb42c711a0626f3eb9923c2ef0c0430c742c94a2baa1025fd25103b046ad152` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/ope_posthoc_result_20261008.zip` | `be46899a9690200909f767c9118d6ab93790ddee05060c18acc2f1119fb593ef` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/ope_posthoc_result_20261008/analysis.exit` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/ope_posthoc_result_20261008/analysis.log` | `671bddeafa88ffe5906b537976893cde02d05e8b6722132c89d83bad65ad7cd9` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/ope_posthoc_result_20261008/execution.json` | `dad5f9251a5cc805ea56a445e68dfae2c5042be6825513274feab5098741fa88` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/ope_posthoc_result_20261008/files.json` | `fdc29ed6ed96e24b490be2e2ab422dda03ca7b1f1ef78d08a1dfbfedb122177d` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/ope_posthoc_result_20261008/per_sequence.csv` | `e134b01a22d3d245e4d603c6e7bbc0ba7a54e93a1203d02e86bbfdc7f7fc12c0` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/ope_posthoc_result_20261008/result.json` | `7fb754ada26f3e6d3fd673a02b8b9f9402f9dd0590787fda4fa11c2e44f445e8` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/ope_posthoc_results_extraction.json` | `43658015104ec1ed2344ad2d973853552ff3153647e036a35da6de75d5e49f11` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/prepare_ope_posthoc_deployment.py` | `ef8a6fa2529b1f6a43cbfa9bf251ae10f9ef0b1c9e0132d7dd8fc2fa851587e6` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/traces/experiment-audit/2026-10-08_m122_ope_posthoc_result/001-result-artifacts.request.json` | `9d95f36b6549de53aa5fdf9f0ccf2b157577d74bd828b791f638ca703173e208` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/traces/experiment-audit/2026-10-08_m122_ope_posthoc_result/002-independent-local-verification.json` | `b20e66c19b4aa8db777ede664fdffe5992d1675987e9c91b238aa7b3dcf91420` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/traces/experiment-audit/2026-10-08_m122_ope_posthoc_result/002-independent-local-verifier.py` | `b8bb240a7304fb71a66c70d8f316304a7d127ff46e5e2b55ad8b5b5a6f536d11` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/traces/experiment-audit/2026-10-08_m122_ope_posthoc_result/003-write-result-audit.py` | `19e172c8ee7cb5f331dde62eb57fffafca8a412d3504bc032027c8e80fed4dc9` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/M122_POSTHOC_OPE_PLAN.md` | `9e6cfad3bb0a55f797887a1e607cbdbcad3402e5763a43cba81f987fef77c4e0` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m122_ope_trajectories.py` | `673b5f516c93611386aca46fa839600e49fc4e3aeb91d3260f909ca5124a8626` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/collect_m122_official_results.py` | `5bcb7f0316f5fcab0205c0ca061152f7a47e3cc6bb57433b50bf7eca5ce555c6` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m122_official.py` | `d0eeacba480a19ccb717b36b7cf0d7d3fe3cb5395e7bf19d19b37868dd3354f9` |
