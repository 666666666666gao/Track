# M124 epoch01 CDTB CPU diagnostic: R2 actual-capture audit

Generated: 2026-10-12T04:13:30.210112+00:00

**Status: WARN_ACTUAL_CAPTURE; blocking issues: 0. Capture consistency: PASS.** The captured diagnostic, source/final/metric bindings, archived native terminal, all 80 GT pairs, all CSV cells and comparison arithmetic are consistent. R1 and all 18 of its inspected inputs remain unchanged.

**Attribution:** requested native `gpt-6-astra` / `max`; actual model and effort **UNATTESTED**. Same reviewer follow-up, OpenAI same-family, provisional. No backend proof, external model or subagent. The parent retains the actual request/response trace.

**Boundary:** this reviewer made no SSH/GPU/NN/training queries and executed none of the producer, launcher, intake or comparison scripts. Computation was restricted to local hashing, AST/payload inspection and independent arithmetic over captured artifacts. The audit is not a raw-GT re-evaluation, NN completion audit, or full-three/full-nine result.

## Captured results and independent recomputation

| Quantity | CurrentC | M124 epoch01 | Difference (pp) |
| --- | ---: | ---: | ---: |
| P (%) | 74.593624625266 | 74.046123602844 | -0.547501022422 |
| R (%) | 69.886238661877 | 71.273842688551 | +1.387604026674 |
| F (%) | 72.163244546477 | 72.633539630522 | +0.470295084046 |
| All saved boxes R (%) | 70.554278387234 | 72.750059560746 | +2.195781173511 |
| Confidence screening R loss (pp) | 0.668039725357 | 1.476216872195 | +0.808177146838 |

These are accepted as consistent **captured CPU diagnostic values**. Their per-sequence macro aggregation was independently recomputed; the reviewer did not independently compute frame overlaps from raw trajectory/GT files. Evidence: [M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json:1](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json:1), [CurrentC_CDTB_recall_diagnostic_original_stdout_20261011.json:1](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/CurrentC_CDTB_recall_diagnostic_original_stdout_20261011.json:1), [summary.json:5](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_saved_comparison_20261012/summary.json:5) and [M124_epoch01_actual_cdtb_metrics_20261012.json:3](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_actual_cdtb_metrics_20261012.json:3).

- Exactly **80 unique sequences, 101,956 frames and 91,300 valid-GT frames** occur in both captured comparisons. All 80 paired GT hashes, names/order, frame counts, valid-GT counts and GT-validity reappearance frame locations match.
- All **325 input-hash entries** have the expected structure: 5 metadata files plus 80 boxes, 80 confidence files, 80 GT files and 80 first RGB images. Every per-sequence prediction/confidence/GT digest matches its captured input mapping.
- Independent `math.fsum` macro means match the aggregate within **1.4210854715202004e-14**. The saved formal P/R/F and **0.180816** threshold match the archived sealed metric JSON.
- The CSV has **80 rows x 28 columns = 2,240 checked cells**. Text/integer cells are exact; parsed numeric difference is **0**. Both largest-eight lists, 29 summary fields and contribution/decomposition arithmetic match independently rebuilt values.
- The all-box-R improvement is **2.195781173511449 pp**, while screening loss increases **0.8081771468375081 pp**, giving the formal R change **1.387604026673941 pp**. **41 sequences improve, 39 worsen, none are unchanged** in all-box R. The fixed-trajectory all-box bound remains **2.8499404392541834 pp** below the comparator target of 75.6%.
- Captured geometry statistics agree: **8,466** serious-low-IoU frames; **4,982 outside / 3,484 inside** nominal reconstructed crops; **91** long segments totaling **8,105** frames; **54 inside / 37 outside** segment onsets; **25** inside-onset segments later have outside frames; **159** GT-validity reappearance events, **60** center-outside. These are descriptive geometry, not causal attribution.

Evidence: [M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json:1](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json:1), [all80_sequence_diagnostics.csv:1](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_saved_comparison_20261012/all80_sequence_diagnostics.csv:1) (all data rows 2-81), [summary.json:25](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_saved_comparison_20261012/summary.json:25), [summary.json:516](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_saved_comparison_20261012/summary.json:516).

## Native and source binding

- Native archive: invocation chunk `4204fd`, auth chunk `0112a5`, shared session **18420**, terminal chunk **0dd4cc**, terminal **exit_code=0**. Completion object omits a session ID. The supplied archive groups the terminal with this invocation; the reviewer did not make a new host query or independently authenticate a tool backend. Evidence: [M124_epoch01_CDTB_CPU_diagnostic_native_receipts_20261012.json:2](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_CPU_diagnostic_native_receipts_20261012.json:2) and [M124_epoch01_CDTB_CPU_diagnostic_native_receipts_20261012.json:16](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_CPU_diagnostic_native_receipts_20261012.json:16).
- The launcher returns `subprocess.run(args, stdout=stream).returncode` and writes the exact exclusive stdout capture. Its compressed payload was decoded as data; source bytes equal the current pinned producer. It clears CUDA visibility, limits CPU thread variables, and invokes the metric-only producer with `-B -u`, without `-O`. Evidence: [M124_epoch01_CDTB_CPU_diagnostic_native_20261012.py:3](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_CPU_diagnostic_native_20261012.py:3) and [prepare_m124_epoch01_cdtb_CPU_diagnostic_native_20261012.py:10](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/prepare_m124_epoch01_cdtb_CPU_diagnostic_native_20261012.py:10).

| Binding | SHA256 |
| --- | --- |
| Actual captured stdout | `238b30bab4d573e3a4bde0ee00e63a235aa6c26fbc9cdc1127ed80dbea47b4f7` |
| Producer / captured analysis source | `508802fa97a762aa8e9f6309ac38c10290f44ce1bdc97dcfe2ad573199454143` |
| Native launcher | `3f7ef34bd10241b6517323702e38b9329b8d53d6df6dc6266f9e3c95e077b8a7` |
| Embedded bootstrap | `1809a4d9fedc2c1472674addf8ab02236553c18d31c41fa8b4c497f5912b7441` |
| Final reference | `75aee7ac912b799f9160baadf95ccc6ff406b6b338760a239da6249bfce505db` |
| Sealed metric JSON | `68973ac268c758a46e60d2493bf397b45ddd7b58126cfdf83ecf28a30217e853` |
| Runtime metric source / captured pin | `05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc` |
| Captured plan | `f30435b5ade9bf0f2b58eedf07f1968959b1c8c3f3e7a7de9681482f4cb9d4a5` |
| Captured cases | `b3170466a090fa1a8949e0297566c9e3ab744e789ca0c7f1bcc71633ccfae887` |
| Captured prediction receipt | `bc36f31e1c4d747bf5f6d8cbb25685a49142a13b4ce4ea13cf80915df43d880a` |

The **three receipt types remain separate**: the original `cdtb_track` stage receipt reports the NN tracking command exit; the new native receipt records the CPU diagnostic tool invocation/terminal; `predictions/receipt.json` is the sealed per-sequence raw prediction manifest whose digest is recorded in metrics and captured inputs. The reviewer directly inspected the first two local artifacts but did not reopen the remote manifest.

## Checks

### source_and_native_binding: PASS

All 18 R1 inputs and both original R1 report files remain byte-identical. The native wrapper compressed payload equals the pinned producer bytes. Captured producer, saved-metric, final and metric digests match locally computed or independently inspected archived pins. The captured raw prediction receipt digest matches formal metrics, distinct from the stage receipt.

Evidence: [M124_epoch01_CDTB_CPU_diagnostic_native_20261012.py:3](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_CPU_diagnostic_native_20261012.py:3), [analyze_m124_epoch01_cdtb_saved_trajectory.py:7](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epoch01_cdtb_saved_trajectory.py:7), [analyze_m124_epoch01_cdtb_saved_trajectory.py:23](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epoch01_cdtb_saved_trajectory.py:23), [analyze_m124_epoch01_cdtb_saved_trajectory.py:77](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epoch01_cdtb_saved_trajectory.py:77), [M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json:1](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json:1), [M124_epoch01_actual_cdtb_metrics_20261012.json:14](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_actual_cdtb_metrics_20261012.json:14).

### native_terminal: PASS

The supplied archived native records contain invocation/auth session 18420 and terminal chunk 0dd4cc with exit_code 0. The pinned wrapper returns subprocess.run(...).returncode and redirects stdout to the exact captured file. This verifies the supplied capture lineage; it is not a new host or backend attestation.

Evidence: [M124_epoch01_CDTB_CPU_diagnostic_native_receipts_20261012.json:2](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_CPU_diagnostic_native_receipts_20261012.json:2), [M124_epoch01_CDTB_CPU_diagnostic_native_20261012.py:4](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_CPU_diagnostic_native_20261012.py:4).

### coverage_and_GT_pairing: PASS

Captured data has exactly 80 unique M124_epoch01 rows, 101956 frames and 91300 valid-GT frames. Names/order, per-sequence frame counts, valid-GT counts, all 80 GT hashes, and every GT-validity reappearance frame agree with CurrentC. Exactly 325 input-map entries equal 5 metadata entries plus 80 boxes, 80 scores, 80 GT files and 80 first RGB images. Every row prediction/score hash matches its captured input map.

Evidence: [M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json:1](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json:1), [M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json:1](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json:1), [CurrentC_CDTB_recall_diagnostic_original_stdout_20261011.json:1](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/CurrentC_CDTB_recall_diagnostic_original_stdout_20261011.json:1), [CurrentC_CDTB_recall_diagnostic_original_stdout_20261011.json:1](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/CurrentC_CDTB_recall_diagnostic_original_stdout_20261011.json:1), [all80_sequence_diagnostics.csv:1](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_saved_comparison_20261012/all80_sequence_diagnostics.csv:1).

### aggregate_recomputation: PASS

Independent math.fsum macro means and harmonic F reproduce captured P/R/F, all-box R, serious-low counts, outside counts and segment counts; maximum aggregate roundoff is 1.4210854715202004e-14. Formal P/R/F and threshold match the sealed metrics JSON. Counts, segment ranges/non-overlap, and reappearance bookkeeping are internally consistent.

Evidence: [M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json:1](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json:1), [M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json:1](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json:1), [M124_epoch01_actual_cdtb_metrics_20261012.json:3](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_actual_cdtb_metrics_20261012.json:3).

### comparison_and_CSV: PASS

All 80 CSV rows, 28 columns and 2240 cells match independently constructed paired rows with zero parsed numeric difference. All 29 summary fields (including source-declared metadata literals) match the captures; both top-8 orderings and all per-sequence contribution sums agree. No target comparison code was executed by the reviewer.

Evidence: [all80_sequence_diagnostics.csv:1](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_saved_comparison_20261012/all80_sequence_diagnostics.csv:1), [summary.json:5](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_saved_comparison_20261012/summary.json:5), [summary.json:32](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_saved_comparison_20261012/summary.json:32), [summary.json:516](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_saved_comparison_20261012/summary.json:516), [analyze_m124_epoch01_CDTB_saved_comparison_20261012.py:14](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/analyze_m124_epoch01_CDTB_saved_comparison_20261012.py:14).

### intake_crosscheck: WARN

Eleven derived intake fields were crosschecked directly against local files/captures/native receipt rather than accepting intake assertions. Other fields such as server_CPU_analysis_invocations=1, training_source_changed=False and training_restarted=False are unconditionally written source literals, not independent remote observations.

Evidence: [intake_m124_epoch01_CDTB_CPU_diagnostic_20261012.py:9](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/intake_m124_epoch01_CDTB_CPU_diagnostic_20261012.py:9), [M124_epoch01_CDTB_CPU_diagnostic_actual_intake_20261012.json:3](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_CPU_diagnostic_actual_intake_20261012.json:3).

### A_to_F_integrity_scope: PASS

Real dataset GT remains separate from model predictions; no self-normalizing metric is introduced. Source call path is reachable and generated capture exists. Accepted scope is the captured real_gt CPU trajectory diagnosis and its arithmetic comparison, not a raw-dataset re-evaluation, causal intervention, actual Python-version attestation, NN inference, or full-three/full-nine completion.

Evidence: [analyze_m124_epoch01_cdtb_saved_trajectory.py:46](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epoch01_cdtb_saved_trajectory.py:46), [analyze_m124_epoch01_CDTB_saved_comparison_20261012.py:38](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/analyze_m124_epoch01_CDTB_saved_comparison_20261012.py:38), [summary.json:534](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_saved_comparison_20261012/summary.json:534).

## Warnings and exact claim limits

### W01: Actual-capture audit is not an independent raw-dataset or host audit

This review directly checked the captured JSON, CSV, source, archived terminal and their hashes. It did not reopen the remote plan/cases/raw prediction receipt/boxes/scores/GT/images or the final checkpoint. The 325 recorded hashes and successful pinned-source run bind the capture; they do not equal 325 independent remote rehashes by this reviewer. The terminal archive has no session_id in its completion object and no command/time attestation beyond the supplied archival grouping. Its wall_time_seconds is a tool-call wait duration, not the diagnostic runtime.

Evidence: [M124_epoch01_CDTB_CPU_diagnostic_native_receipts_20261012.json:2](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_CPU_diagnostic_native_receipts_20261012.json:2), [M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json:1](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json:1), [M124_epoch01_actual_cdtb_metrics_20261012.json:14](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_actual_cdtb_metrics_20261012.json:14).

### W02: Intake bookkeeping is not observational telemetry

intake source line 25 hardcodes NN/query counters, one analysis invocation, training_source_changed=False and training_restarted=False. The producer also declares zero counters. The no-NN/query conclusion is supported by the inspected pinned call graph, not runtime instrumentation. Local source pins are unchanged; the review does not certify global remote training activity or the absence of unrelated restarts.

Evidence: [intake_m124_epoch01_CDTB_CPU_diagnostic_20261012.py:25](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/intake_m124_epoch01_CDTB_CPU_diagnostic_20261012.py:25), [M124_epoch01_CDTB_CPU_diagnostic_actual_intake_20261012.json:28](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_CPU_diagnostic_actual_intake_20261012.json:28), [analyze_m124_epoch01_cdtb_saved_trajectory.py:77](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epoch01_cdtb_saved_trajectory.py:77), [M124_epoch01_CDTB_CPU_diagnostic_native_20261012.py:3](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_CPU_diagnostic_native_20261012.py:3).

### W03: One harmless floating-point sign residue is preserved

bag_outside has confidence_R_loss_pp=-1.1102230246251565e-14, while its reported R and all-box R both equal 85.07310467499526. The producer computes the loss before multiplying/rounding the two reported percentages. This is machine-precision roundoff, not a real negative screening loss. An initial exact nonnegative check flagged it; a documented 1e-10 arithmetic tolerance passes. No captured data was clamped or changed.

Evidence: [M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json:1](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json:1), [all80_sequence_diagnostics.csv:6](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_saved_comparison_20261012/all80_sequence_diagnostics.csv:6), [analyze_m124_epoch01_cdtb_saved_trajectory.py:61](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epoch01_cdtb_saved_trajectory.py:61), [analyze_m124_epoch01_cdtb_saved_trajectory.py:72](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epoch01_cdtb_saved_trajectory.py:72).

### W04: Crop and causal limits remain

Crops are reconstructed from previous %.6f saved boxes with native factor-4 geometry, not replayed internal tensors. The 54 inside / 37 outside long-segment onsets and 60 outside GT-validity reappearances are descriptive; invalid GT does not prove physical absence. CurrentC and M124 use different trained weights/own histories and different best-F thresholds. The suite discloses external Test50/CDTB checkpoint selection. No module/text causality, deployable recovery or untouched-holdout claim follows.

Evidence: [run_m124_epochwise_official.py:33](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m124_epochwise_official.py:33), [analyze_m124_epoch01_cdtb_saved_trajectory.py:63](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epoch01_cdtb_saved_trajectory.py:63), [analyze_m124_epoch01_CDTB_saved_comparison_20261012.py:38](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/analyze_m124_epoch01_CDTB_saved_comparison_20261012.py:38), [summary.json:519](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_saved_comparison_20261012/summary.json:519), [run_m124_epochwise_suite.py:58](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m124_epochwise_suite.py:58), [run_m124_epochwise_suite.py:99](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m124_epochwise_suite.py:99).

### W05: Runtime version and metric archive qualifications remain

The capture supports successful execution of the pinned producer under the configured mplt environment; no Python version/package inventory was collected. R1 Python 3.8 grammar checks and the new intake grammar check do not attest an actual Python 3.8 runtime. The local metric source raw CRLF hash remains 446416a7..., while its LF-normalized hash exactly equals the runtime/captured 05879f2e... pin. These two byte identities remain explicitly distinct.

Evidence: [M124_epoch01_CDTB_CPU_diagnostic_native_20261012.py:3](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_CPU_diagnostic_native_20261012.py:3), [prepare_m124_epoch01_cdtb_CPU_diagnostic_native_20261012.py:14](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/prepare_m124_epoch01_cdtb_CPU_diagnostic_native_20261012.py:14), [analyze_m124_epoch01_cdtb_saved_trajectory.py:23](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epoch01_cdtb_saved_trajectory.py:23), [M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json:1](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json:1), [M124_epoch01_CDTB_CPU_diagnostic_source_review_R1_20261012.json:1](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_CPU_diagnostic_source_review_R1_20261012.json:1).

No blocking change is required for use of this capture as a descriptive CPU saved-trajectory diagnostic. Retain all qualifiers. In particular, do not convert intake literals into proof of global training state, and do not claim a module mechanism, a recovered performance guarantee, actual Python 3.8 attestation, epoch02/VOT/full-nine completion, or independent remote rehashing.

## R1 preserved

- R1 Markdown: `0d925afb1442dcd91364cb94fac366c76516269b20c9941b43c694246cffec0d`.
- R1 JSON: `da3fb1398c631f9083ca3b07b6e197c5f828604ae0d3e1135cd098b6234c0597`.
- All 18 original input raw hashes match R1. The source-only report has not been edited to retroactively claim actual results. Its missing-capture limitation is partially resolved by the new capture and terminal evidence; raw remote verification, geometric and scientific limitations remain.

## All inspected local input hashes

| File relative to repository | Raw SHA256 |
| --- | --- |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epoch01_cdtb_saved_trajectory.py` | `508802fa97a762aa8e9f6309ac38c10290f44ce1bdc97dcfe2ad573199454143` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m124_epochwise_official.py` | `e8ef46a5c1f1e1debecfe1c0b315ba7c766f49907982c4cc20360a0bcedf916a` |
| `.aris/m122_full_causal_20261007/CurrentC_CDTB_locked_metric_source_20261011.py` | `446416a741bdc0d35a359ffdf28280b743b430e8680ee365ec018bea0a652048` |
| `.aris/m122_full_causal_20261007/M124_epoch01_actual_cdtb_metrics_20261012.json` | `68973ac268c758a46e60d2493bf397b45ddd7b58126cfdf83ecf28a30217e853` |
| `.aris/m122_full_causal_20261007/M124_epoch01_actual_cdtb_track_receipt_20261012.json` | `0909fa00b58eccda00508c8f85201ee79a7d69c8307bc450182923b299bde3ea` |
| `.aris/m122_full_causal_20261007/CurrentC_CDTB_recall_diagnostic_remote_20261011.py` | `14ac1fcb71b1b0473dee4c01fdda36bdba9ff6fdc126527ad5992cc555932962` |
| `.aris/m122_full_causal_20261007/prepare_m124_epoch01_cdtb_CPU_diagnostic_native_20261012.py` | `568fd3e76b39d2fce4edaf7e7b4230213f78782d34b27c371f29380290afcdae` |
| `.aris/m122_full_causal_20261007/M124_epoch01_CDTB_CPU_diagnostic_native_20261012.py` | `3f7ef34bd10241b6517323702e38b9329b8d53d6df6dc6266f9e3c95e077b8a7` |
| `.aris/m122_full_causal_20261007/analyze_m124_epoch01_CDTB_saved_comparison_20261012.py` | `5fc6460d0894513dfe3373449418ecb55d1e3d7504d1839059bf5656909e8a12` |
| `.aris/m122_full_causal_20261007/CurrentC_CDTB_recall_diagnostic_original_stdout_20261011.json` | `bcdc6791cf67d71b77e08cacad8b91cefed1c7a89260f98ade04183dfe64de72` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/full_dense_tracker.py` | `0aee7de2932d7294976e2ed512218cd58d281e24bf3fb34b6267167470aa1778` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/trusted_template_tracker.py` | `1dd4873192b05f6921cdf729a525ed684ed96a18bfdfc48673cda14d50352e0d` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m124_epochwise_suite.py` | `63924cc9f7f5e38ab605cb8f40dddf083321bee12248e71a7e8516f5a2d17dfc` |
| `lib/train/data/processing_utils.py` | `3f1b7a11b5559f7766e589501597bb96493f63f6a2a836557638922144c73dfe` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/m124_epochwise_official_runtime.py` | `60de3f27538c76bcba1178d5b24f9064c587bda9ba564a60aae179cbcc023e48` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/m122_official_runtime.py` | `25aa67000cf581c454b160c524c8dcf8f55ec30d227b387990cc03ed18be3157` |
| `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/prepare_m124_epochwise_evaluation.py` | `ad05b5029b6e27e5c163c8ffa190d1db83312f011331f5d8dd426e89192ee8cb` |
| `.aris/m122_full_causal_20261007/CurrentC_CDTB_recall_diagnostic_native_20261011.py` | `c46520cdd2f55b878bc0869a09f8d556c7274ecbeeb9b79fd5b38268d28efcb9` |
| `.aris/m122_full_causal_20261007/M124_epoch01_CDTB_CPU_diagnostic_source_review_R1_20261012.json` | `da3fb1398c631f9083ca3b07b6e197c5f828604ae0d3e1135cd098b6234c0597` |
| `.aris/m122_full_causal_20261007/M124_epoch01_CDTB_CPU_diagnostic_source_review_R1_20261012.md` | `0d925afb1442dcd91364cb94fac366c76516269b20c9941b43c694246cffec0d` |
| `.aris/m122_full_causal_20261007/M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json` | `238b30bab4d573e3a4bde0ee00e63a235aa6c26fbc9cdc1127ed80dbea47b4f7` |
| `.aris/m122_full_causal_20261007/M124_epoch01_CDTB_CPU_diagnostic_native_receipts_20261012.json` | `20a290ec5cb93a77b2a0fa6f9e4fe56a9a2d86ad8f0f38a7ceff146aacd62aba` |
| `.aris/m122_full_causal_20261007/M124_epoch01_CDTB_CPU_diagnostic_actual_intake_20261012.json` | `48b39bc2683868e421096f8bdfc44c3a2d63786ba3a98d10a77e10e504d4aaa1` |
| `.aris/m122_full_causal_20261007/intake_m124_epoch01_CDTB_CPU_diagnostic_20261012.py` | `83ec285ad06fb7681c72520d91186e5db3574a25997e8a537f999fbe0d03daf4` |
| `.aris/m122_full_causal_20261007/M124_epoch01_CDTB_saved_comparison_20261012/summary.json` | `f64a532b53a2cd749b7d583facfdeef207a1c0842b5fe4d7720901c106f1b3fe` |
| `.aris/m122_full_causal_20261007/M124_epoch01_CDTB_saved_comparison_20261012/all80_sequence_diagnostics.csv` | `262c136d6980709707943f64046d6777266824eba78137d70f5585383a331259` |

The JSON sidecar records all 26 artifact identities, 16 Python source hashes, deterministic checks, native lineage limits and all 80 per-sequence GT/prediction/confidence bindings.
