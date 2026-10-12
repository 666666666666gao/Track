# M124 epoch02 CDTB CPU diagnostic — R1 source integrity review

Overall: **WARN**. Scoped source gate: **PASS**, with **0 blocking issues**. No source change is required by this review. The remaining warnings limit execution and scientific claims; they are not a finding of fabricated GT or score inflation.

Requested reviewer: **gpt-6-astra / max**. Actual runtime model and effort: **UNATTESTED** (no host attestation supplied). Review independence: **same-family**. Acceptance: **provisional**. Date: 2026-10-12. Reviewer task: `/root/m124_epoch02_cdtb_cpu_integrity_20261012`.

This reviewer read all supplied Python sources line by line (344 lines), parsed the local captured evidence, decoded the native payload without executing it, and performed local arithmetic/hash checks. There were no SSH/network calls, neural executions, GPU/training-progress queries, or changes to experiment sources. Only this report and its JSON companion were written. The full metric sweep and raw remote evaluation were not executed.

The expected epoch02 diagnostic stdout file already existed during inspection. R1 deliberately did not open it; the separate captured-result review must establish its content and completion. Thus this report does not equate a prepared wrapper with successful execution, and does not assert that execution never happened.

## A–F findings

### A. Ground truth provenance: PASS

Scope: source-level dataset-GT provenance; raw remote authenticity not attested.

- The analyzers read groundtruth.txt from plan.dataset_root / sequence, separately from saved prediction and confidence files, then require the GT hash to match the case manifest. No ground truth is generated from predictions.
- The locked metric treats finite positive-width/height GT boxes as valid and invalid GT boxes as Special(0). This is annotation validity, not proof of physical absence.
- The epoch01 capture names /root/autodl-tmp/CDTB/sequences as the GT root and contains 80 GT hashes. Its embedded hashes are internally consistent, but this reviewer did not read those remote GT bytes.
- The local metric is a project evaluator using vot.region.calculate_overlaps. Hash equality establishes identity with the pinned evaluator, not independent official-toolkit conformance or dataset-release authenticity.

Evidence:

- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\analyze_m124_epoch02_cdtb_saved_trajectory.py:26-29`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\analyze_m124_epoch02_cdtb_saved_trajectory.py:46-58`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\analyze_m124_epoch01_cdtb_saved_trajectory.py:46-58`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\native_ope\source_snapshots\depthtrack_pr.py:40-60`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\native_ope\source_snapshots\depthtrack_pr.py:95-111`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json:1` JSON pointer `/inputs_sha256`

### B. Score normalization: PASS

Scope: all supplied metric arithmetic read; saved scalar arithmetic independently checked.

- At the saved threshold the analyzers compute per-sequence P as mean selected overlap and R as selected-overlap sum divided by valid-GT count. P and R are macro-averaged over 80 sequences and F is their harmonic mean. Percentage units are a factor of 100.
- The full evaluator samples thresholds from confidence values, builds macro PR curves, and chooses the highest F. This is a threshold sweep, not division by the model own maximum/minimum/mean score. The selected-frame count in precision is its metric denominator, not score self-normalization.
- The diagnosis reuses threshold 0.31897 from sealed epoch02 metrics; it does not rerun the threshold sweep. Epoch02 fractional values times 100 and F recomputed from P/R match exactly in Python arithmetic.
- All-saved-boxes R is the overlap sum over the unchanged trajectory divided by valid-GT count. It upper-bounds R obtainable by retaining any subset of these same boxes, but does not imply attainable P/F gains or a changed inference trajectory.

Evidence:

- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\analyze_m124_epoch02_cdtb_saved_trajectory.py:44`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\analyze_m124_epoch02_cdtb_saved_trajectory.py:59-61`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\analyze_m124_epoch02_cdtb_saved_trajectory.py:73-77`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\analyze_m124_epoch01_cdtb_saved_trajectory.py:59-77`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\native_ope\source_snapshots\depthtrack_pr.py:25-37`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\native_ope\source_snapshots\depthtrack_pr.py:117-153`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_actual_cdtb_metrics_20261012.json:4-12`

### C. Result existence and claimed numbers: WARN

Scope: existing local captures and exact values checked; epoch02 diagnosis runtime deferred.

- All nine supplied files exist. Epoch02 CDTB metrics are status complete and contain P=74.40346155295778%, R=68.72944147468972%, F=71.45398784136414%, threshold=0.31897, 80 sequences and 101956 frames; these exactly match README and the paired CSV.
- The epoch02 CDTB track-stage receipt records ope_track, the correct epoch02 plan, and exit_code=0. It is a track-stage process receipt, not predictions/receipt.json; its SHA must not be substituted for the metrics receipt_sha256 58eb7dd2....
- Additional captured DepthTrack metrics support README P/R/F=68.19866074231768/62.074999101829654/64.9929050798145%, 50 sequences/76373 frames, with the same final SHA. The copied CDTB metrics and track receipt are byte-identical to supplied copies.
- Independent arithmetic on the epoch01 captured per-sequence summaries reproduces P=74.04612360284422, R=71.27384268855087, F=72.6335396305224, all-box R=72.75005956074581, 8466 serious-low frames, 3484 reconstructed inside and 4982 outside, and 91 long segments. The 80 distinct rows sum to 101956 frames and 91300 valid-GT frames; the 325 input hashes comprise 80 each of GT/bbox/confidence/image plus 5 metadata files. Row/hash linkage, bounds, partition totals, segment and reappearance-event consistency checks found no mismatch. This recomputes captured JSON aggregates, not raw overlaps.
- CDTB epoch02-minus-epoch01 deltas are +0.3573379501135747/-2.54440121386115/-1.1795517891582676 pp, agreeing with README rounding. Paired mean F is 68.5415903449273 then 68.22344646058932; best-of-two is epoch01 and epoch02 is not a strict decline on both F scores.
- The preparation script correctly labels its action prepared_CPU_diagnostic_not_executed. The expected epoch02 stdout path already existed during this source-stage inspection; file existence alone is not success. Its contents and a terminal execution receipt were not reviewed in R1. Therefore R1 neither claims the epoch02 CPU diagnosis executed successfully nor claims it has never executed.

Evidence:

- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_actual_cdtb_metrics_20261012.json:2-18`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_actual_cdtb_track_receipt_20261012.json:2-16`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\refine-logs\m122_trusted_memory\M124_EPOCH02_PAIRED_OPE_COMPLETE_20261012\depthtrack_test_metrics.json:2-18`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\refine-logs\m122_trusted_memory\M124_EPOCH02_PAIRED_OPE_COMPLETE_20261012\depthtrack_test_track_receipt.json:2-16`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\refine-logs\m122_trusted_memory\M124_EPOCH02_PAIRED_OPE_COMPLETE_20261012\epoch_metrics.csv:1-4`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\refine-logs\m122_trusted_memory\M124_EPOCH02_PAIRED_OPE_COMPLETE_20261012\README.md:3-16`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json:1` JSON pointer `/aggregate`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json:1` JSON pointer `/per_sequence`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json:1` JSON pointer `/inputs_sha256`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\prepare_m124_epoch02_CDTB_CPU_diagnostic_20261012.py:16-20`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_CPU_diagnostic_native_20261012.py:4-6`

### D. Called and dead code: WARN

Scope: static call path plus epoch01 output linkage; no epoch02 runtime invocation attested.

- The supplied epoch02 and epoch01 analyzers are 80 lines each and were read line by line. sha and runs are called by main, and the __main__ guard calls main. _load_rows and _vot_overlaps are invoked directly from the pinned metric module. All newly computed summaries are inserted into the JSON printed at completion.
- The entire 158-line metric source, 20-line preparation source and 6-line native wrapper were read. The wrapper payload was decoded without execution: its embedded 7016-byte analyzer is byte-identical to the supplied epoch02 source, SHA256 9ff1df469a6f58b17cba2de34410be0031af552146576addbff8019e2d51f32c. The decoded bootstrap hashes the payload, writes only a new standalone diagnostic path guarded by nonexistence, checks its hash, then runpy executes that path.
- evaluate_depthtrack_results is not called by the diagnosis; _determine_thresholds is only called inside that unused full-evaluator branch, and format_metrics has no caller in the supplied chain. This is an intentional fixed-threshold decomposition, so these are not fabricated results; they do mean R1 cannot claim a newly executed full PR sweep or prove the original scoring entrypoint was executed.
- The prepared wrapper invokes SSH transport with CUDA_VISIBLE_DEVICES empty and single-thread numerical environment settings. The decoded analyzer and locked metric contain no neural model load/forward, GPU query or training-progress query. Zero-counter fields in its future JSON are source declarations, not an independent system-wide telemetry audit. The wrapper does write a new remote diagnostic file and a new local stdout file if executed; the reviewer did neither.

Evidence:

- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\analyze_m124_epoch02_cdtb_saved_trajectory.py:12-17`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\analyze_m124_epoch02_cdtb_saved_trajectory.py:19-25`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\analyze_m124_epoch02_cdtb_saved_trajectory.py:50-58`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\analyze_m124_epoch02_cdtb_saved_trajectory.py:70-80`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\analyze_m124_epoch01_cdtb_saved_trajectory.py:70-80`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\native_ope\source_snapshots\depthtrack_pr.py:25-37`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\native_ope\source_snapshots\depthtrack_pr.py:63-158`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\prepare_m124_epoch02_CDTB_CPU_diagnostic_20261012.py:6-20`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_CPU_diagnostic_native_20261012.py:1-6`

### E. Scope assessment: PASS

Scope: qualified CDTB trajectory diagnosis; no broader result acceptance.

- The source hard-requires 80 unique cases and 101956 frames, matches ordered cases to 80 receipt entries, verifies each prediction/score/GT hash and frame count, and anchors epoch02 metrics/final/evaluator hashes. The prior epoch01 capture satisfies the 80-sequence/101956-frame structure. Epoch02 runtime completion is outside R1.
- The evidence concerns one epoch02 checkpoint on CDTB, compared with one epoch01 checkpoint; README states seed2027. No multi-seed robustness, causal module attribution, semantic-text intervention, candidate replay or complete nine-metric claim is established.
- Crop membership reconstructs a square from the previous saved bbox with ceil(sqrt(w*h)*4), Python rounding and current GT center. This is an estimate conditional on saved-state/native-geometry agreement; the running tracker crop implementation/state is not supplied or audited. Center-inside does not mean the whole object was available or a good candidate existed. Invalid-GT-to-valid-GT transitions are annotation-validity events. Segment starts are zero-based, ends exclusive, and low-IoU runs are interrupted by invalid GT.
- The supplied source and README disclose these limits, explicitly avoid causal conclusions and full-nine claims, and disclose Test50/CDTB80 use in checkpoint selection. Those datasets cannot be described here as untouched final holdouts.

Evidence:

- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\analyze_m124_epoch02_cdtb_saved_trajectory.py:28-43`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\analyze_m124_epoch02_cdtb_saved_trajectory.py:46-58`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\analyze_m124_epoch02_cdtb_saved_trajectory.py:62-77`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json:1` JSON pointer `/sequences`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\refine-logs\m122_trusted_memory\M124_EPOCH02_PAIRED_OPE_COMPLETE_20261012\README.md:10-20`

### F. Evaluation type classification: PASS

Scope: source-defined and captured metadata classification; raw authenticity not independently attested.

- The saved-trajectory diagnosis compares saved tracker boxes to dataset-provided groundtruth.txt using VOT bounded overlap. It is real_gt evaluation, not synthetic_proxy, self_supervised_proxy, simulation_only or human_eval.
- The crop reconstruction is a derived geometric diagnostic within this real-GT analysis; it is not a neural replay or causal experiment. The source-stage review itself is static and performs no evaluation against raw remote data.

Evidence:

- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\analyze_m124_epoch02_cdtb_saved_trajectory.py:48-58`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\analyze_m124_epoch02_cdtb_saved_trajectory.py:77`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\analyze_m124_epoch01_cdtb_saved_trajectory.py:77`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\native_ope\source_snapshots\depthtrack_pr.py:40-60`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json:1` JSON pointer `/evaluation_type`

## Warning register and claim limits

- **W1**: Raw remote plan/cases/predictions/GT/images/weights and installed VOT implementation were not independently read. Hash claims in local captures do not substitute for those bytes.
- **W2**: Epoch02 diagnostic capture exists but was not opened in R1; runtime completion, recomputed aggregates and exit status require the separately requested captured-result review.
- **W3**: The diagnosis calls overlap/loading helpers and reuses the sealed threshold; it does not execute evaluate_depthtrack_results or its threshold sweep. format_metrics is unused in this chain.
- **W4**: Crop-center membership, annotation-validity reappearance and segment onset cannot establish the causal neural failure or actual whole-object visibility; frame indices are zero-based and segment end indices are exclusive.
- **W5**: Single-seed checkpoint comparisons with externally selected Test50/CDTB80 scores do not support multi-seed robustness, untouched-test, language/module causality, or full-nine success claims.
- **W6**: Requested reviewer is gpt-6-astra/max. The host supplied no actual runtime model/effort attestation; both are UNATTESTED. This is same-family, provisional review.

The only established performance numbers here are values present in local captured files and arithmetic consistency between those values. The previous epoch01 JSON aggregate checks do not recompute its raw IoU. Classification `real_gt` describes the code/data path; it does not attest the authenticity of an unseen server dataset. A maximum-F reporting threshold and all-box recall upper bound cannot be advertised as an inference-policy intervention.

## Claims

- **C1 — supported_at_source_stage**: Prepared epoch02 source computes a sealed, dataset-GT, fixed-threshold CPU trajectory decomposition.
- **C2 — supported_as_local_captured_metadata_not_raw_server_verification**: Existing captured epoch02 CDTB P/R/F and README deltas are numerically consistent.
- **C3 — not_reviewed_in_R1**: Epoch02 CPU diagnostic has completed and produced verified decomposition numbers.
- **C4 — unsupported**: This review independently audited raw server data or reran the neural tracker.
- **C5 — unsupported**: A confidence-only change can improve deployable P/F or explains the neural cause of failure.
- **C6 — unsupported**: Epoch02 has achieved all nine metrics or demonstrates multi-seed/module/text causality.

## Audited input SHA256

All hashes were checked again before writing this report and matched the bytes initially reviewed. The nine supplied files are listed first; remaining files were read to verify README claims or hashed to establish byte-identical captured copies. The epoch01 capture is a single physical JSON line; exact JSON pointers disambiguate citations to line 1.

- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\analyze_m124_epoch02_cdtb_saved_trajectory.py`
  - SHA256: `9ff1df469a6f58b17cba2de34410be0031af552146576addbff8019e2d51f32c`; 7016 bytes; 80 lines; supplied.
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\analyze_m124_epoch01_cdtb_saved_trajectory.py`
  - SHA256: `508802fa97a762aa8e9f6309ac38c10290f44ce1bdc97dcfe2ad573199454143`; 7016 bytes; 80 lines; supplied.
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\native_ope\source_snapshots\depthtrack_pr.py`
  - SHA256: `05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc`; 7143 bytes; 158 lines; supplied.
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\prepare_m124_epoch02_CDTB_CPU_diagnostic_20261012.py`
  - SHA256: `94bf69e4a7be7105ece24034f98803bd2539f4b29c7702e2400b12508ca2b582`; 2095 bytes; 20 lines; supplied.
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_CPU_diagnostic_native_20261012.py`
  - SHA256: `9bf638d96954707ab478a51d3e1a2b835ef0252667f02be3ee972c6126218061`; 5109 bytes; 6 lines; supplied.
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_actual_cdtb_metrics_20261012.json`
  - SHA256: `129a2af0c7f6586c1bee42ae5d27a6e7a9e69f005341bec192d009d8f5d5f07a`; 788 bytes; 19 lines; supplied.
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_actual_cdtb_track_receipt_20261012.json`
  - SHA256: `830f9c1e6a9b719517087eb93164e2a0fb1e474fce20b4b18786a536fdc8eb67`; 485 bytes; 17 lines; supplied.
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json`
  - SHA256: `238b30bab4d573e3a4bde0ee00e63a235aa6c26fbc9cdc1127ed80dbea47b4f7`; 139211 bytes; 1 lines; supplied.
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\refine-logs\m122_trusted_memory\M124_EPOCH02_PAIRED_OPE_COMPLETE_20261012\README.md`
  - SHA256: `65aeece5676e1f2f5c157e09ebedbb7cce98bfd558c2314cfc29d9b30a40be4d`; 2981 bytes; 20 lines; supplied.
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\refine-logs\m122_trusted_memory\M124_EPOCH02_PAIRED_OPE_COMPLETE_20261012\depthtrack_test_metrics.json`
  - SHA256: `b78dbfe1a696beac2545a72db20351a5c9648d0cb85132fe740f98ed1ac636e0`; 788 bytes; 19 lines; supplementary captured evidence.
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\refine-logs\m122_trusted_memory\M124_EPOCH02_PAIRED_OPE_COMPLETE_20261012\depthtrack_test_track_receipt.json`
  - SHA256: `81220780abd6f4e5f432d43c79ff03e477323806a37f86d223bbceab7287a6ba`; 507 bytes; 17 lines; supplementary captured evidence.
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\refine-logs\m122_trusted_memory\M124_EPOCH02_PAIRED_OPE_COMPLETE_20261012\epoch_metrics.csv`
  - SHA256: `00bf88aee75686ef9c16fdd2864e279f53241a4fc9d095dc010394e656b19272`; 908 bytes; 4 lines; supplementary captured evidence.
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\refine-logs\m122_trusted_memory\M124_EPOCH02_PAIRED_OPE_COMPLETE_20261012\cdtb_metrics.json`
  - SHA256: `129a2af0c7f6586c1bee42ae5d27a6e7a9e69f005341bec192d009d8f5d5f07a`; 788 bytes; 19 lines; supplementary captured evidence.
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\refine-logs\m122_trusted_memory\M124_EPOCH02_PAIRED_OPE_COMPLETE_20261012\cdtb_track_receipt.json`
  - SHA256: `830f9c1e6a9b719517087eb93164e2a0fb1e474fce20b4b18786a536fdc8eb67`; 485 bytes; 17 lines; supplementary captured evidence.

Decoded native bootstrap SHA256: `494246bc4e5e8cfcb05ab92535158b55ccdd3b3244221e4eba4faa1339378f12`. Its embedded analyzer is exactly the 7016-byte epoch02 source listed above. The seven source-line changes from epoch01 are confined to the epoch label/root, final hash, metrics hash, module name, row model label and output status/limits epoch label; arithmetic and geometry are unchanged.

## Next captured-result review

- Read the captured epoch02 diagnostic JSON and its actual terminal execution receipt.
- Verify execution/source identities, status, scope, per-sequence identities and hashes, all aggregate arithmetic, exact parity to sealed epoch02 P/R/F, and epoch01 comparison identities.
- Preserve the distinction between captured server output and independent access to raw server bytes; do not upgrade to NN or causal audit.

The source-stage report remains provisional. It must not be cited as an independent raw-server, neural-network, full-nine-metric, or causal-module audit.
