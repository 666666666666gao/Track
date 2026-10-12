# M124 epoch02 CDTB CPU diagnostic — R2 captured-result integrity review

Overall: **WARN**. Captured consistency: **PASS**, with **0 blocking issues**. All **2,240 CSV data cells**, **28 header cells**, and **448 fields in the 16 ranked rows** match the primary captures. Aggregate arithmetic agrees to at most **1.4211e-14 pp**. No experiment source change or rerun is required by this review.

Requested reviewer: **gpt-6-astra / max**. Actual runtime model/effort: **UNATTESTED**. Independence: **same-family**. Acceptance: **provisional**. Date: 2026-10-12. Reviewer task: `/root/m124_epoch02_cdtb_cpu_integrity_20261012`.

R1 remains a source-stage record. R2 directly reads the actual epoch02 capture, native terminal metadata, complete comparison source, summary and CSV. It confirms existing captured output and local arithmetic, not independent access to server predictions/GT or a neural rerun. There were zero reviewer SSH/network calls, GPU/training-progress queries, NN executions, or experiment-source modifications. Only the requested R2 Markdown/JSON reports were written.

## Verified result

| Quantity | Epoch01 | Epoch02 |
|---|---:|---:|
| P (%) | 74.0461236028442 | 74.40346155295776 |
| R (%) | 71.27384268855087 | 68.72944147468972 |
| F (%) | 72.6335396305224 | 71.45398784136412 |
| All saved boxes R (%) | 72.75005956074581 | 70.21499771001007 |
| Report threshold | 0.180816 | 0.31897 |
| Serious-low IoU frames | 8466 | 9795 |
| Reconstructed outside low frames | 4982 | 7897 |
| Long low-IoU segments | 91 | 97 |

Values in this table are the captured CPU aggregates. The formal saved epoch02 P/F differ only by 1.42e-14 pp due to floating reduction order. Both epochs contain 80 sequences, 101956 frames and 91300 valid-GT frames.

The recall identity is **-2.54440121386115 = -2.535061850735744 - 0.009339363125405953 pp**. The first term is the all-box trajectory change; the last is the increase in confidence-filter loss. Positive and negative per-sequence filtering changes largely cancel (+0.4941638879548937 and -0.4848245248294969 macro pp). This is descriptive accounting, not a causal network diagnosis.

## A–F findings

### A. Ground truth provenance: PASS

Scope: captured dataset-GT identity and source-level provenance; remote raw bytes not independently accessed.

- The audited analyzer reads dataset groundtruth.txt separately from prediction/score files and checks each against the sealed case/prediction receipts. Epoch02 captured output is real_gt and records the same 80 sequence identities, GT hashes, first-image hashes, image dimensions, per-sequence frame/valid-GT counts and GT-validity transition frames as epoch01.
- Both captures contain 80 unique sequence rows, 101956 frames and 91300 valid-GT frames. Each has 325 input hashes: 80 GT, 80 bbox, 80 confidence, 80 first-image and 5 metadata inputs. Every bbox/confidence/GT row hash maps to the corresponding captured input hash. Both captures use the same cases manifest SHA b3170466a090fa1a8949e0297566c9e3ab744e789ca0c7f1bcc71633ccfae887 and the same VOT-based metric SHA.
- The epoch02 capture links its plan hash 539aecae..., prediction-receipt hash 58eb7dd2..., saved-metrics hash 129a2af0..., evaluator hash 05879f2e..., and final 1d44ec68... to the supplied formal metrics metadata. The track-stage receipt is a different artifact from predictions/receipt.json.
- These are source and captured-hash consistency checks. No raw server GT, prediction, image, weight, plan/cases/receipt bytes or installed VOT package were independently retrieved. This does not establish an independent official-dataset or official-toolkit audit.

Evidence:

- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\analyze_m124_epoch02_cdtb_saved_trajectory.py:20-58`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\native_ope\source_snapshots\depthtrack_pr.py:40-60`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json:1` JSON pointer `/per_sequence`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json:1` JSON pointer `/inputs_sha256`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_CPU_diagnostic_original_stdout_20261012.json:1` JSON pointer `/per_sequence`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_CPU_diagnostic_original_stdout_20261012.json:1` JSON pointer `/inputs_sha256`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_actual_cdtb_metrics_20261012.json:14-18`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_actual_cdtb_track_receipt_20261012.json:2-16`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\compare_m124_cdtb_epoch01_epoch02_saved_trajectory.py:19-32`

### B. Score normalization and comparison arithmetic: PASS

Scope: supplied source formulas and independently recomputed captured arithmetic.

- No metric is scaled by the model own maximum/minimum/mean score. P is mean selected IoU, R is selected IoU sum divided by valid-GT count, and both are macro-averaged; F is the harmonic mean. The /80 comparison contribution is exactly equal-sequence macro weighting.
- The two formal thresholds are 0.180816 and 0.31897, each epoch own dataset-wide best-F reporting threshold. The CPU diagnosis verifies that saved-threshold P/R/F agrees with formal metrics, without re-executing the full threshold sweep.
- Independently using math.fsum for the 80 row means and a reciprocal-form harmonic mean reproduces both captures and summary within 1.4210854715202004e-14 percentage points. Epoch02 capture minus formal sealed metrics is P=-1.4210854715202004e-14, R=0, F=-1.4210854715202004e-14 pp; threshold difference is zero.
- The reported decomposition is arithmetically correct: -2.54440121386115 pp formal R change = -2.535061850735744 pp all-box R change minus +0.009339363125405953 pp confidence-filter-loss change. Confidence R loss changes from 1.4762168721949394 to 1.4855562353203453 pp.
- The small net filtering term includes cancellation: independent sums of positive and negative per-sequence macro contributions are +0.4941638879548937 and -0.4848245248294969 pp. It therefore does not establish that confidence filtering is unimportant on each sequence.
- Epoch02 all-box R is 70.21499771001007%, leaving 5.385002289989927 pp to the comparison source configured target 75.6%. This is a fixed-trajectory recall upper-bound calculation, not achieved performance after an intervention. The target value is hard-coded in the comparison source; its external authorization/provenance is not established by the supplied primary files.
- The epoch02 jug row has R exceeding all-box R by only 7.105427357601002e-15 pp and confidence loss -5.551115123125783e-15 pp. A strict zero-tolerance bound check initially flagged it; the directly inspected values resolve this as floating-point roundoff. All rows satisfy the mathematical bounds within 1e-12 pp. No source change or clamp is indicated.

Evidence:

- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\analyze_m124_epoch02_cdtb_saved_trajectory.py:44-77`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\native_ope\source_snapshots\depthtrack_pr.py:117-153`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\compare_m124_cdtb_epoch01_epoch02_saved_trajectory.py:27-67`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_saved_comparison_20261012\summary.json:5-31`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_saved_comparison_20261012\summary.json:349-351`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_saved_comparison_20261012\summary.json:439-441`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_saved_comparison_20261012\all80_sequence_diagnostics.csv:1-81`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_saved_comparison_20261012\all80_sequence_diagnostics.csv:53`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_CPU_diagnostic_original_stdout_20261012.json:1` JSON pointer `/per_sequence/51`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_actual_cdtb_metrics_20261012.json:4-12`

### C. Result existence, execution receipts and exact values: WARN

Scope: captured result and terminal metadata are consistent; process binding is not independently attested.

- The epoch02 JSON capture is nonempty, parses as one complete JSON object, has status actual_complete_M124_epoch02_CDTB_saved_trajectory_diagnosis, timestamp 2026-10-12T08:51:49.172571+00:00, and SHA aa54318793d5666fa8c5aa7cc831ea6916a0dbed37082a9e2dfdea21b47a90b9. Its analysis_source_sha256 exactly matches the audited 7016-byte source embedded in the native wrapper.
- The supplied native receipt records initial session 61301 (chunk 33f79d) and terminal chunk b9a489 with exit_code 0. The auth and terminal objects are byte-for-value identical records of the same result; they are not two independent successful runs.
- The receipt JSON does not include the invocation command, wrapper hash or capture hash, and the terminal object does not retain the initial session_id. The source self-hash, capture identity, wrapper payload and supplied terminal metadata are mutually consistent, but R2 cannot independently attest a cryptographic/process-level binding between those records. No extra execution is requested or justified by this observation.
- All 80 CSV rows and all 28 columns were checked independently against the primary captures: 2240 data cells match exactly as serialized strings, all 28 header cells match, no missing/duplicate/extra sequence exists, and row order matches epoch01. The two eight-row ranking lists were independently selected from all 80 rows and all 448 ranked row fields match exactly.
- All summary scalars, both embedded epoch aggregates, input hashes, labels, declared counters and the literal limits text were verified against primary captures or source. Epoch02 totals are 9795 serious-low frames = 1898 reconstructed inside + 7897 outside, 97 long segments containing 9486 frames, 35 inside onsets and 62 outside onsets; 17 inside onsets later contain an outside frame and 18 remain inside. There are 20 all-box-R improvements, 60 regressions and zero unchanged sequences.
- Both captures have 159 annotation-validity reappearance events at identical frame indices; reconstructed outside events change 60 to 84. All 80 bbox hashes and all 80 score hashes differ between the two epochs while the captured GT/image identities remain identical.
- R2 advances the state from prepared-source review to verified consistency of existing captured output and its local comparison. It does not upgrade that evidence to independent raw-server reproduction or NN execution.

Evidence:

- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_CPU_diagnostic_original_stdout_20261012.json:1` JSON pointer `/status`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_CPU_diagnostic_original_stdout_20261012.json:1` JSON pointer `/analysis_source_sha256`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_CPU_diagnostic_native_receipts_20261012.json:2-22`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_CPU_diagnostic_native_20261012.py:3-6`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\compare_m124_cdtb_epoch01_epoch02_saved_trajectory.py:14-25`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\compare_m124_cdtb_epoch01_epoch02_saved_trajectory.py:33-79`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_saved_comparison_20261012\summary.json:5-31`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_saved_comparison_20261012\summary.json:32-515`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_saved_comparison_20261012\summary.json:516-534`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_saved_comparison_20261012\all80_sequence_diagnostics.csv:1-81`

### D. Called and dead code: WARN

Scope: complete source read and captured output linkage; full PR sweep not rerun.

- The complete 83-line comparison source was read directly. Its main is reached through __main__, reads the two hash-pinned captures, joins all 80 sequences, computes the CSV deltas and summary, writes both artifacts, then prints a short summary. The resulting files match the expected arithmetic; the reviewer did not import or run this comparison source.
- The analyzer calls sha, runs, _load_rows and _vot_overlaps, computes all returned diagnostic fields and prints its completed record. Its matching captured source hash and populated rows support that intended data path at the captured-consistency level.
- evaluate_depthtrack_results, its _determine_thresholds branch, and format_metrics are not called by the CPU diagnosis/comparison. This is an intentional fixed-threshold trajectory decomposition; a new full PR-threshold search must not be claimed from this execution.
- The supplied source uses no neural model loading/forward, GPU query or training-progress query. The native wrapper uses SSH to run the CPU diagnostic and write a new remote diagnostic source. The comparison summary server_queries=0 refers to local comparison only, not the earlier SSH execution. Counter fields are source declarations rather than an independent machine-wide telemetry trace.

Evidence:

- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\compare_m124_cdtb_epoch01_epoch02_saved_trajectory.py:1-83`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\analyze_m124_epoch02_cdtb_saved_trajectory.py:12-17`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\analyze_m124_epoch02_cdtb_saved_trajectory.py:19-25`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\analyze_m124_epoch02_cdtb_saved_trajectory.py:50-80`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\native_ope\source_snapshots\depthtrack_pr.py:25-37`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\native_ope\source_snapshots\depthtrack_pr.py:63-158`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\prepare_m124_epoch02_CDTB_CPU_diagnostic_20261012.py:9-20`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_CPU_diagnostic_native_20261012.py:3-6`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_saved_comparison_20261012\summary.json:528-534`

### E. Scope assessment: PASS

Scope: 80-sequence two-epoch descriptive analysis with stated limitations.

- The comparison covers exactly the two own-history saved trajectories for one CDTB80 run per checkpoint, with 101956 frames each and unchanged captured GT/first-image metadata. This is not a same-state candidate intervention, module ablation, text intervention, training replay, additional seed or recovery trial.
- The largest-loss list is ranked by per-sequence all-box-R delta, not by formal R or causal importance. Its largest two contributions are humans_corridor_occ_1 (-0.723091389158222 macro pp) and robot_corridor_occ_1 (-0.6382112150540831 macro pp). The largest gains are boxes_backpack_room_occ_1 (+0.6833711581707134 macro pp) and humans_shirts_room_occ_1_A (+0.5846639763768083 macro pp). Rankings and every retained field are supported by the full CSV.
- Crop-center membership is reconstructed from the previous saved box with factor4 geometry; it does not prove actual tracker hidden state, whole-object visibility, candidate availability, or the first cause of divergence. Low-IoU segments are zero-based [start,end) runs of valid-GT frames with IoU<=0.1 and length>=10. Invalid GT does not prove physical absence.
- The source/summary explicitly disclaim neural/module/text causality, deployable P/F gains and M124 VOT/full-nine completion. README discloses seed2027 and Test50/CDTB80 checkpoint selection. These results cannot establish multi-seed robustness or untouched-test performance.
- The decomposition supports saying the net saved-trajectory all-box R term accounts for almost all of the net formal R decrease in these captures. It does not establish that the network cause is geometry rather than confidence or language, or that a threshold-only inference change would reproduce the all-box bound.

Evidence:

- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\compare_m124_cdtb_epoch01_epoch02_saved_trajectory.py:15-32`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\compare_m124_cdtb_epoch01_epoch02_saved_trajectory.py:49-67`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_saved_comparison_20261012\summary.json:32-92`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_saved_comparison_20261012\summary.json:274-334`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_saved_comparison_20261012\summary.json:516-534`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_saved_comparison_20261012\all80_sequence_diagnostics.csv:35-36`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_saved_comparison_20261012\all80_sequence_diagnostics.csv:47`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_saved_comparison_20261012\all80_sequence_diagnostics.csv:59`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\analyze_m124_epoch02_cdtb_saved_trajectory.py:62-77`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\refine-logs\m122_trusted_memory\M124_EPOCH02_PAIRED_OPE_COMPLETE_20261012\README.md:12-20`

### F. Evaluation type classification: PASS

Scope: saved trajectories versus dataset GT; descriptive comparison only.

- The evaluation type is real_gt: dataset annotations determine valid frames and overlaps. Ground truth is not created from model output. Both captures and the comparison summary explicitly use this label.
- The capture comparison performs arithmetic on real-GT-derived diagnostics; it is neither a new NN benchmark nor a causal experiment. The local review validates consistency and attribution limits, not independently observed server raw data.

Evidence:

- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\analyze_m124_epoch02_cdtb_saved_trajectory.py:48-61`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\analyze_m124_epoch02_cdtb_saved_trajectory.py:77`
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\native_ope\source_snapshots\depthtrack_pr.py:40-60`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json:1` JSON pointer `/evaluation_type`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_CPU_diagnostic_original_stdout_20261012.json:1` JSON pointer `/evaluation_type`
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_saved_comparison_20261012\summary.json:529-534`

## Ranking verification

Both lists were independently selected from all 80 primary-capture rows. Every ranked field matches; the table reports macro all-box-R contribution in percentage points, with equal 1/80 sequence weight.

| Rank | Largest loss | Macro delta (pp) | Largest gain | Macro delta (pp) |
|---:|---|---:|---|---:|
| 1 | humans_corridor_occ_1 | -0.723091389158222 | boxes_backpack_room_occ_1 | 0.6833711581707134 |
| 2 | robot_corridor_occ_1 | -0.6382112150540831 | humans_shirts_room_occ_1_A | 0.5846639763768083 |
| 3 | case | -0.5319064580375625 | thermos_office_occ_1 | 0.3133387068755 |
| 4 | paperpunch | -0.40418340441443457 | boxes_humans_room_occ_1 | 0.30128053060222026 |
| 5 | trashcans_room_occ_1_B | -0.3548981732418216 | cartman | 0.2533922664356722 |
| 6 | humans_corridor_occ_2_B | -0.2902092692511117 | humans_longcorridor_staricase_occ_1 | 0.1972230222348937 |
| 7 | humans_corridor_occ_2_A | -0.23886652032109906 | box_humans_room_occ_1 | 0.11370702329789957 |
| 8 | trashcans_room_occ_1_A | -0.2373477833442177 | box_darkroom_noocc_7 | 0.06799588697205347 |

## Verification coverage and numerical note

The JSON companion includes all 80 sequence names with their exact CSV line numbers (2–81), 28 checked cells per row, the checked column names, ranking coverage, numerical residuals and audited input hashes. Checks also covered GT/image equality, all input-hash links, segment ranges and lengths, reappearance-event counts and frame lists, row partitions, scope and summary labels, and the summary source-limit string. No material mismatch remains.

A strict nonnegative-loss check initially flagged epoch02 jug. Direct inspection showed only a 7.1054e-15 pp R/all-R inversion and a -5.5511e-15 pp loss value. Bounds pass at 1e-12 pp; the source is unchanged. This roundoff note preserves the original diagnostic result rather than silently altering it.

## Warning register and claim limits

- **W1**: Primary server raw bytes and installed evaluation dependencies were not independently accessed. Captured hashes and arithmetic cannot certify remote dataset/weight authenticity or raw-NN reproduction.
- **W2**: Native terminal metadata is exit0 and consistent with the capture, but its JSON lacks the invocation command, wrapper/capture hashes and terminal session_id; auth/terminal duplicate the same chunk. Runtime binding remains captured, not independently attested.
- **W3**: The CPU diagnosis verifies fixed saved thresholds; the full evaluate_depthtrack_results threshold sweep and format_metrics are not executed in this chain.
- **W4**: The net confidence-filter loss change is small but masks opposing per-sequence changes. Crop and ranking diagnostics are descriptive and cannot isolate neural, module or language causes.
- **W5**: The 75.6% target is a comparison-source literal; the verified gap is arithmetic relative to that parameter. Fixed-trajectory all-box R is not a deployable P/F result.
- **W6**: Single-seed, successive-epoch own-history results with disclosed checkpoint selection do not support untouched-test, multi-seed, recovery-intervention or full-nine completion claims.
- **W7**: Requested reviewer gpt-6-astra/max has no host runtime attestation. Actual model/effort remain UNATTESTED; review is same-family and provisional.

## Claim assessment

- **C1 — supported_as_captured_evidence_with_runtime_binding_warning**: Captured epoch02 CPU diagnosis and supplied terminal exit0 metadata are consistent with audited source and formal score identities.
- **C2 — supported; 2240 exact CSV cells and 448 exact ranked field cells**: All 80-row CSV values, summary arithmetic and ranking lists agree with the primary captures.
- **C3 — supported_as_descriptive_arithmetic**: The net formal R loss decomposes into all-box trajectory R loss and a small net filtering-loss increase.
- **C4 — unsupported**: Confidence filtering is irrelevant to every sequence or a specific network/module/text mechanism caused the regression.
- **C5 — unsupported**: These results independently reproduce raw server/NN evaluation or establish complete nine-metric success.

## Audited input SHA256

All audited inputs were rehashed before report writing and are unchanged. Single-line capture citations include JSON pointers. Sources first read completely in R1 were reused without re-executing them; the new 83-line comparison source was read completely in R2.

- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\analyze_m124_epoch02_cdtb_saved_trajectory.py`
  - SHA256: `9ff1df469a6f58b17cba2de34410be0031af552146576addbff8019e2d51f32c`; 7016 bytes; 80 lines; first read R1.
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\analyze_m124_epoch01_cdtb_saved_trajectory.py`
  - SHA256: `508802fa97a762aa8e9f6309ac38c10290f44ce1bdc97dcfe2ad573199454143`; 7016 bytes; 80 lines; first read R1.
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\native_ope\source_snapshots\depthtrack_pr.py`
  - SHA256: `05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc`; 7143 bytes; 158 lines; first read R1.
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\prepare_m124_epoch02_CDTB_CPU_diagnostic_20261012.py`
  - SHA256: `94bf69e4a7be7105ece24034f98803bd2539f4b29c7702e2400b12508ca2b582`; 2095 bytes; 20 lines; first read R1.
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_CPU_diagnostic_native_20261012.py`
  - SHA256: `9bf638d96954707ab478a51d3e1a2b835ef0252667f02be3ee972c6126218061`; 5109 bytes; 6 lines; first read R1.
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_actual_cdtb_metrics_20261012.json`
  - SHA256: `129a2af0c7f6586c1bee42ae5d27a6e7a9e69f005341bec192d009d8f5d5f07a`; 788 bytes; 19 lines; first read R1.
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_actual_cdtb_track_receipt_20261012.json`
  - SHA256: `830f9c1e6a9b719517087eb93164e2a0fb1e474fce20b4b18786a536fdc8eb67`; 485 bytes; 17 lines; first read R1.
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json`
  - SHA256: `238b30bab4d573e3a4bde0ee00e63a235aa6c26fbc9cdc1127ed80dbea47b4f7`; 139211 bytes; 1 lines; first read R1.
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\refine-logs\m122_trusted_memory\M124_EPOCH02_PAIRED_OPE_COMPLETE_20261012\README.md`
  - SHA256: `65aeece5676e1f2f5c157e09ebedbb7cce98bfd558c2314cfc29d9b30a40be4d`; 2981 bytes; 20 lines; first read R1.
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_CPU_diagnostic_original_stdout_20261012.json`
  - SHA256: `aa54318793d5666fa8c5aa7cc831ea6916a0dbed37082a9e2dfdea21b47a90b9`; 139542 bytes; 1 lines; first read R2.
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_CPU_diagnostic_native_receipts_20261012.json`
  - SHA256: `05c3a7cb14258ae24f4f0f996d5c08dab6e6203e4c1208ec9e1b4a321516db98`; 806 bytes; 23 lines; first read R2.
- `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\compare_m124_cdtb_epoch01_epoch02_saved_trajectory.py`
  - SHA256: `5598855adcb58f6b62a2947ad1ce2b3d821d6878ee433d2777a994c5420131a3`; 6407 bytes; 83 lines; first read R2.
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_saved_comparison_20261012\summary.json`
  - SHA256: `a88e6f055ede87d0a98c73201c5136a85f5ec5316dbd929e81d9b7b1ea7a02f7`; 26031 bytes; 535 lines; first read R2.
- `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_epoch02_CDTB_saved_comparison_20261012\all80_sequence_diagnostics.csv`
  - SHA256: `fe01aa0cc9aa66bf52a9ba68ed874b00b32325c99d87be1b9409bab2e25488fb`; 29427 bytes; 81 lines; first read R2.

This review is complete for the supplied captured artifacts. The reports do not authorize stronger raw-server, neural, causal, untouched-test or full-nine claims.
