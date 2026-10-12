# M124 epoch01 CDTB CPU diagnostic: source review R1

Generated: 2026-10-12T04:02:42.748437+00:00

**Status: WARN_SOURCE_ONLY. Blocking issues: 0.** No source blocker was found for the inspected diagnostic and its exact generated native launcher. This is a source gate only: the reviewer did not deploy, run the diagnostic, run the comparison, recompute raw-trajectory metrics, query SSH/GPU/training, or verify M124 VOT/full-three completion.

**Attribution:** requested native reviewer `gpt-6-astra` / `max`; actual model and effort `UNATTESTED` because no backend proof was supplied. Independent review task, OpenAI same-family, provisional. No subagents or external review models were used. The parent owns the actual request/response trace.

## A-F integrity checks

### A_gt_provenance: PASS

The diagnostic opens dataset_root/sequence/groundtruth.txt, verifies each dataset GT hash from the plan-bound case manifest, and computes VOT overlap with real dataset GT. It never derives reference boxes from predictions. Invalid GT is only a validity label, not proof of physical absence. No GT is fed to tracking by this diagnostic.

Evidence:
- [analyze_m124_epoch01_cdtb_saved_trajectory.py:26-29](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epoch01_cdtb_saved_trajectory.py:26)
- [analyze_m124_epoch01_cdtb_saved_trajectory.py:46-58](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epoch01_cdtb_saved_trajectory.py:46)
- [CurrentC_CDTB_locked_metric_source_20261011.py:40-60](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/CurrentC_CDTB_locked_metric_source_20261011.py:40)
- [run_m124_epochwise_official.py:51-58](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m124_epochwise_official.py:51)

### B_score_normalization: PASS

Precision is selected overlap mean; recall is selected overlap sum / valid-GT count; dataset metrics are macro means. Confidence-derived thresholds select predictions rather than divide or rescale scores. The saved threshold is 0.180816. The diagnostic asserts exact saved P/R/F parity to 1e-8. All-box recall is a fixed-trajectory bound only.

Evidence:
- [analyze_m124_epoch01_cdtb_saved_trajectory.py:44](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epoch01_cdtb_saved_trajectory.py:44)
- [analyze_m124_epoch01_cdtb_saved_trajectory.py:59-76](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epoch01_cdtb_saved_trajectory.py:59)
- [CurrentC_CDTB_locked_metric_source_20261011.py:25-37](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/CurrentC_CDTB_locked_metric_source_20261011.py:25)
- [CurrentC_CDTB_locked_metric_source_20261011.py:117-151](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/CurrentC_CDTB_locked_metric_source_20261011.py:117)
- [M124_epoch01_actual_cdtb_metrics_20261012.json:4-12](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_actual_cdtb_metrics_20261012.json:4)

### C_result_binding_and_existence: WARN

The supplied formal metric JSON exists and has the exact SAVED hash, and its P/R/F arithmetic is internally consistent. Source binds its final, plan, bundle, raw prediction receipt and per-sequence prediction/confidence/GT hashes. The supplied track receipt is only a stage-command exit record; it is NOT predictions/receipt.json. Remote plan, cases, raw prediction receipt, predictions, GT and images were not inspected. New diagnostic output and comparison output do not yet exist locally, so no diagnostic result is accepted by this review.

Evidence:
- [analyze_m124_epoch01_cdtb_saved_trajectory.py:7-10](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epoch01_cdtb_saved_trajectory.py:7)
- [analyze_m124_epoch01_cdtb_saved_trajectory.py:20-55](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epoch01_cdtb_saved_trajectory.py:20)
- [M124_epoch01_actual_cdtb_metrics_20261012.json:14-18](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_actual_cdtb_metrics_20261012.json:14)
- [M124_epoch01_actual_cdtb_track_receipt_20261012.json:2-16](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_actual_cdtb_track_receipt_20261012.json:2)
- [M124_epoch01_CDTB_CPU_diagnostic_native_20261012.py:4-6](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_CPU_diagnostic_native_20261012.py:4)
- [analyze_m124_epoch01_CDTB_saved_comparison_20261012.py:5-22](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/analyze_m124_epoch01_CDTB_saved_comparison_20261012.py:5)

### D_dead_code: PASS

Native bootstrap calls runpy.run_path(..., run_name="__main__"), reaching main. main calls runs and metric._load_rows/_vot_overlaps and prints the computed JSON. Official ope_analyze calls evaluate_depthtrack_results, which calls threshold and overlap helpers. The comparison is a separately invoked top-level consumer. format_metrics is only an unused presentation helper in the supplied call graph, not an uncalled performance computation.

Evidence:
- [prepare_m124_epoch01_cdtb_CPU_diagnostic_native_20261012.py:10](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/prepare_m124_epoch01_cdtb_CPU_diagnostic_native_20261012.py:10)
- [analyze_m124_epoch01_cdtb_saved_trajectory.py:52-78](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epoch01_cdtb_saved_trajectory.py:52)
- [analyze_m124_epoch01_cdtb_saved_trajectory.py:80](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epoch01_cdtb_saved_trajectory.py:80)
- [run_m124_epochwise_official.py:56-62](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m124_epochwise_official.py:56)
- [CurrentC_CDTB_locked_metric_source_20261011.py:97-117](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/CurrentC_CDTB_locked_metric_source_20261011.py:97)
- [CurrentC_CDTB_locked_metric_source_20261011.py:157-158](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/CurrentC_CDTB_locked_metric_source_20261011.py:157)
- [analyze_m124_epoch01_CDTB_saved_comparison_20261012.py:7-44](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/analyze_m124_epoch01_CDTB_saved_comparison_20261012.py:7)

### E_scope: PASS

The primary diagnostic fixes epoch-01 and requires 80 unique cases / 101956 frames, matching ordered receipt rows and every file length. Comparison selects only CurrentC from the historical CurrentC/P1 record and matches all 80 names, frame counts, valid-GT counts and GT hashes. It disclaims paired causal inference, epoch02 and M124 VOT. The suite separately discloses Test50/CDTB80 checkpoint selection; this diagnostic does not alter that controller.

Evidence:
- [analyze_m124_epoch01_cdtb_saved_trajectory.py:7](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epoch01_cdtb_saved_trajectory.py:7)
- [analyze_m124_epoch01_cdtb_saved_trajectory.py:28-29](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epoch01_cdtb_saved_trajectory.py:28)
- [analyze_m124_epoch01_cdtb_saved_trajectory.py:41-53](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epoch01_cdtb_saved_trajectory.py:41)
- [analyze_m124_epoch01_cdtb_saved_trajectory.py:77](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epoch01_cdtb_saved_trajectory.py:77)
- [analyze_m124_epoch01_CDTB_saved_comparison_20261012.py:14-22](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/analyze_m124_epoch01_CDTB_saved_comparison_20261012.py:14)
- [analyze_m124_epoch01_CDTB_saved_comparison_20261012.py:38](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/analyze_m124_epoch01_CDTB_saved_comparison_20261012.py:38)
- [run_m124_epochwise_suite.py:58-74](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m124_epochwise_suite.py:58)
- [run_m124_epochwise_suite.py:99-102](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m124_epochwise_suite.py:99)

### F_evaluation_type: PASS

Intended offline real_gt trajectory diagnostic and saved-output comparison. Crop membership and low-IoU segments are descriptive geometric measurements. This is not NN replay, a module/text intervention, a fresh benchmark execution, or a full-three/full-nine completion audit.

Evidence:
- [analyze_m124_epoch01_cdtb_saved_trajectory.py:49-77](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epoch01_cdtb_saved_trajectory.py:49)
- [analyze_m124_epoch01_CDTB_saved_comparison_20261012.py:38](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/analyze_m124_epoch01_CDTB_saved_comparison_20261012.py:38)

## Deterministic evidence and exact bindings

- The captured M124 metric file SHA is `68973ac268c758a46e60d2493bf397b45ddd7b58126cfdf83ecf28a30217e853`, exactly the diagnostic SAVED pin. Its recorded P/R/F are **74.0461236028442 / 71.27384268855087 / 72.6335396305224 percent**, threshold **0.180816**, **80 sequences / 101,956 frames**. These values were read from the supplied JSON; only percentage and harmonic-mean arithmetic were checked locally. They were not recomputed from predictions.
- The code binds final `75aee7ac912b799f9160baadf95ccc6ff406b6b338760a239da6249bfce505db`, plan `f30435b5ade9bf0f2b58eedf07f1968959b1c8c3f3e7a7de9681482f4cb9d4a5`, bundle `8c7bbb059c3e8cf085edbf26a8eac403c865ec39bce6e46674de476a25ec69f9`, and raw prediction receipt `bc36f31e1c4d747bf5f6d8cbb25685a49142a13b4ce4ea13cf80915df43d880a` through the exact saved metrics file. Final weights themselves are not opened by the diagnostic or this review.
- The native payload was decoded as data, without executing it. Its embedded source SHA is `508802fa97a762aa8e9f6309ac38c10290f44ce1bdc97dcfe2ad573199454143`, exactly equal to the current diagnostic bytes. Decoded bootstrap SHA: `1809a4d9fedc2c1472674addf8ab02236553c18d31c41fa8b4c497f5912b7441`. Its shell command clears CUDA visibility, limits OMP/OpenBLAS/MKL to one thread, uses `-B -u`, and calls `runpy.run_path(..., run_name="__main__")` on a newly created diagnostic file. There is no `-O` disabling the assertions.
- All **15 Python files** plus the decoded bootstrap passed Python 3.8 grammar parsing and local AST compilation. The actual verifier was `E:\python.exe`, CPython 3.12.6. Target source modules were never imported/executed.
- The historical diagnostic JSON raw SHA matches the comparator pin `bcdc6791cf67d71b77e08cacad8b91cefed1c7a89260f98ade04183dfe64de72`. Its CurrentC and P1 sections each contain 80 unique rows, 101,956 frames and 91,300 valid-GT frames. Aggregate arithmetic agrees with the saved per-sequence values to at most 1.5e-14, and low-IoU, segment, inside/outside and reappearance count bookkeeping is consistent. This only validates the saved JSON internally; it does not reverify historical raw trajectories.

## Invocation, geometry, and state-write boundary

The actual sample import is [`full_dense_tracker.py:27`](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/full_dense_tracker.py:27): `lib.train.data.processing_utils.sample_target`, installed into the actor at line 34. Its repository comes from the base bundle in [`m124_epochwise_official_runtime.py:68`](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/m124_epochwise_official_runtime.py:68). The inspected local module is [`lib/train/data/processing_utils.py:12`](C:/Users/gb/.codex_track_publish_m29_20260902/lib/train/data/processing_utils.py:12). Its lines 30-39 match factor-4 `ceil(sqrt(w*h)*4)` and rounded origin used by the diagnostic. The remote bundle/source identity remains unverified in this source-only review.

The new analyzer reads only the epoch01 plan-bound files and 80 first RGB images for image dimensions, then prints JSON ([`analyze_m124_epoch01_cdtb_saved_trajectory.py:20`](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epoch01_cdtb_saved_trajectory.py:20), [`:78`](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epoch01_cdtb_saved_trajectory.py:78)). It does not import `run_m124_epochwise_official`, `checked_plan`, a tracker, PyTorch or the training controller; it does not load model weights. Its dynamic import is the pinned metric source, whose top-level code contains only imports and definitions.

The launcher is intentionally not entirely write-free: the preparation source writes a new local launcher, the native wrapper creates an exclusive local stdout file, and its bootstrap creates one new remote diagnostic `.py` outside the experiment root ([`prepare_m124_epoch01_cdtb_CPU_diagnostic_native_20261012.py:10`](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/prepare_m124_epoch01_cdtb_CPU_diagnostic_native_20261012.py:10), [`:18-19`](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/prepare_m124_epoch01_cdtb_CPU_diagnostic_native_20261012.py:18), [`M124_epoch01_CDTB_CPU_diagnostic_native_20261012.py:4`](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_CPU_diagnostic_native_20261012.py:4)). The comparison creates only its new local result directory/summary/CSV ([`analyze_m124_epoch01_CDTB_saved_comparison_20261012.py:39`](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/analyze_m124_epoch01_CDTB_saved_comparison_20261012.py:39)). No inspected diagnostic path writes model/checkpoint/optimizer/controller state. The existing official runner and suite do contain NN and training-state operations; they are context sources, not part of this diagnostic invocation.

The controller selects checkpoints and writes `epoch_history.json`, `best.pt`, `best.json`, and `current_training_state.json` from its official evaluation path ([`run_m124_epochwise_suite.py:52`](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m124_epochwise_suite.py:52), [`:58-74`](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m124_epochwise_suite.py:58)); it never reads these new diagnostic outputs. The training source files were read and hashed only.

## Nonblocking warnings and post-execution boundary

### W01: Source-only boundary and receipt distinction

The recorded stage exit_code=0 does not independently prove raw predictions, case coverage, the remote prediction receipt or later diagnostic execution. metrics.receipt_sha256 is bc36f31e1c4d747bf5f6d8cbb25685a49142a13b4ce4ea13cf80915df43d880a, whereas the supplied stage receipt has raw SHA 0909fa00b58eccda00508c8f85201ee79a7d69c8307bc450182923b299bde3ea. These are intentionally different artifact types. No remote or result-dependent execution occurred.

Required interpretation/follow-up: After authorized execution, require the native process exit and inspect its captured JSON; verify emitted analysis_source_sha256, saved_metrics_sha256, final_sha256 and input hashes before accepting the comparison. The standalone comparator checks NEW status/count/metric/zero-query fields, while final/source identity relies on this exact native producer chain.

Evidence:
- [M124_epoch01_actual_cdtb_track_receipt_20261012.json:2-16](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_actual_cdtb_track_receipt_20261012.json:2)
- [M124_epoch01_actual_cdtb_metrics_20261012.json:17](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_actual_cdtb_metrics_20261012.json:17)
- [analyze_m124_epoch01_cdtb_saved_trajectory.py:34-50](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epoch01_cdtb_saved_trajectory.py:34)
- [analyze_m124_epoch01_cdtb_saved_trajectory.py:77](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epoch01_cdtb_saved_trajectory.py:77)
- [analyze_m124_epoch01_CDTB_saved_comparison_20261012.py:7-13](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/analyze_m124_epoch01_CDTB_saved_comparison_20261012.py:7)

### W02: Python 3.8 compatibility is static, not a remote runtime attestation

All 15 inspected Python files and the decoded bootstrap parse under the Python 3.8 grammar and compile locally on CPython 3.12.6. No post-3.8 language or standard-library requirement was found in the new diagnostic path. Actual remote Python version and NumPy/OpenCV/VOT dependencies were not imported or checked. The default local python launcher failed with No pyvenv.cfg file; the static verifier explicitly used E:\python.exe.

Required interpretation/follow-up: Do not describe the static grammar check as an actual Python 3.8 diagnostic run.

Evidence:
- [analyze_m124_epoch01_cdtb_saved_trajectory.py:2-5](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epoch01_cdtb_saved_trajectory.py:2)
- [CurrentC_CDTB_locked_metric_source_20261011.py:8-10](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/CurrentC_CDTB_locked_metric_source_20261011.py:8)
- [prepare_m124_epoch01_cdtb_CPU_diagnostic_native_20261012.py:14](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/prepare_m124_epoch01_cdtb_CPU_diagnostic_native_20261012.py:14)
- [M124_epoch01_CDTB_CPU_diagnostic_native_20261012.py:3](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_CDTB_CPU_diagnostic_native_20261012.py:3)

### W03: Archived metric source has CRLF bytes

The inspected metric archive raw SHA is 446416a741bdc0d35a359ffdf28280b743b430e8680ee365ec018bea0a652048. Replacing its 158 CRLF endings with LF, in memory only, yields exactly the required 05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc. This explains the byte mismatch and establishes the reviewed source text; it does not attest the unvisited remote file. The runtime raw-byte assert must remain.

Required interpretation/follow-up: Preserve raw and LF-normalized hashes separately; do not report the local raw archive as byte-identical to the runtime source.

Evidence:
- [analyze_m124_epoch01_cdtb_saved_trajectory.py:9](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epoch01_cdtb_saved_trajectory.py:9)
- [analyze_m124_epoch01_cdtb_saved_trajectory.py:23-25](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epoch01_cdtb_saved_trajectory.py:23)
- [run_m124_epochwise_official.py:56](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m124_epochwise_official.py:56)
- [CurrentC_CDTB_locked_metric_source_20261011.py:1-158](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/CurrentC_CDTB_locked_metric_source_20261011.py:1)

### W04: Crop reconstruction is geometric and quantized

The diagnostic reproduces native factor-4 ceil/sqrt and round origin geometry. However, prior boxes were saved with %.6f while the tracker keeps unrounded double state. It reconstructs the nominal padded crop rectangle, not the actual saved crop tensor or valid-pixel mask, and frame 0 is assigned GT validity. Counts cannot prove precise internal crop/candidate visibility or a causal divergence.

Required interpretation/follow-up: Retain the saved-box reconstruction qualifier; do not call these exact internal-state replay results.

Evidence:
- [run_m124_epochwise_official.py:31-35](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m124_epochwise_official.py:31)
- [analyze_m124_epoch01_cdtb_saved_trajectory.py:63-70](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epoch01_cdtb_saved_trajectory.py:63)
- [full_dense_tracker.py:27-34](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/full_dense_tracker.py:27)
- [full_dense_tracker.py:54](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/full_dense_tracker.py:54)
- [trusted_template_tracker.py:30-45](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/trusted_template_tracker.py:30)
- [processing_utils.py:25-69](C:/Users/gb/.codex_track_publish_m29_20260902/lib/train/data/processing_utils.py:25)

### W05: Scientific comparison and selection scope

CurrentC and M124 epoch01 have different weights/training histories and their own dataset-wide best-F thresholds (archived CurrentC 0.166124; archived M124 0.180816). The all-box-R delta removes only confidence reporting selection, not changed trajectories. Test50/CDTB80 participate in checkpoint selection in the controller. Existing source explicitly discloses these limits.

Required interpretation/follow-up: Limit claims to descriptive saved-trajectory comparison; no module efficacy, text causality, recovery guarantee, untouched holdout, epoch02, M124 VOT, or full-three/full-nine completion claim.

Evidence:
- [CurrentC_CDTB_recall_diagnostic_original_stdout_20261011.json:1](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/CurrentC_CDTB_recall_diagnostic_original_stdout_20261011.json:1)
- [M124_epoch01_actual_cdtb_metrics_20261012.json:10](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epoch01_actual_cdtb_metrics_20261012.json:10)
- [analyze_m124_epoch01_CDTB_saved_comparison_20261012.py:30-38](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/analyze_m124_epoch01_CDTB_saved_comparison_20261012.py:30)
- [run_m124_epochwise_suite.py:52-74](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m124_epochwise_suite.py:52)
- [run_m124_epochwise_suite.py:99-102](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m124_epochwise_suite.py:99)

## Inspected inputs: raw SHA256

All hashes below were recomputed unchanged immediately before report creation. Source pins are local evidence; they are not a statement that remote runtime artifacts were inspected.

| File (relative to repository) | Raw SHA256 |
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

The JSON sidecar additionally records byte lengths, physical line counts, LF-normalized source hashes, evidence ranges, execution limits, checks and attribution. Neither report authorizes a scientific claim beyond its source-only scope.
