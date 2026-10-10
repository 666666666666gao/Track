# CurrentC completed OPE experiment audit

**Verdict: WARN. Blocking findings: 0 for reporting the two observed OPE results as completed, receipt-supported results.** They have not been independently recomputed from their full prediction/GT bytes in this audit. This verdict does not certify complete ABC joint training, a complete VOT result, M123 execution, or overall goal attainment.

Date: 2026-10-11. Observation cutoff: **2026-10-11 01:30:58.058870 +08:00** (the existing captured observation, not a new live query).
Reviewer: fresh native Codex context, canonical agent `/root/current_c_actual_ope_integrity_20261011`. Requested model/reasoning: **gpt-6-astra / max**. Actual backend, model, and reasoning identity: **UNATTESTED**. `review_independence=same-family`; `acceptance_status=provisional`. No cross-family acceptance is claimed.

This is a bounded local read-only source/receipt audit using `experiment-audit`. No network/SSH/SCP, GPU query, Torch import, model construction, dataset loader, neural call, training, prediction, or official metric execution was performed. Only this new Markdown report and its companion JSON are written.

## Evidence paths and scope

In file:line references below:

- `R` = `C:/Users/gb/.codex_track_publish_m29_20260902`
- `A` = `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007`
- `D` = `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928`
- `F` = `A/CurrentC_third_hour_actual_files_20261011`
- `B` = `D/m122_native_observation_20261010_051718/evaluation/precision1`
- `T` = `A/R3_terminal_runtime_integrity_audit_20261010`
- `M` = `R/projects/sttrack_lachtt_v1/diagnostics/native_ope/source_snapshots/depthtrack_pr.py`

All **28 collected files** were read. Their exact bytes, SHA-256 hashes, and decoded text match all 28 embedded files in `A/CurrentC_third_hour_original_observer_stdout_20261011.txt:1`. All 23 current source files match the gate pins; the actual launch's deployment ZIP hash matches the locally retained ZIP, whose 26 members contain exactly those 23 source files, the exact gate, and the two exact audit JSONs. Source audit v2 was read as a historical source-only artifact, not substituted for this actual-result check.

The JSON retains the full review request and report response, an embedded reproducible standard-library verifier, its **81 passing deterministic checks**, primary captured artifact snapshots, exact input hashes, and the failed read-only auditor probes. Tracing is embedded in the two authorized reports rather than creating additional trace files. The input hash table distinguishes evidence bytes from runtime claims; a hash string for an absent checkpoint or prediction file is not a local byte verification of that file.

## Observed completed results

| Dataset | Sequences | Frames including initialization | P (%) | R (%) | F (%) |
|---|---:|---:|---:|---:|---:|
| DepthTrack Test | 50 | 76,373 | 67.3650693502503 | 64.86510340917104 | 66.09145399091419 |
| CDTB | 80 | 101,956 | 74.59362462526585 | 69.88623866187693 | 72.1632445464765 |

Values and exact metric keys exist at `F/evaluation/depthtrack_test/predictions/metrics.json:2-18` and `F/evaluation/cdtb/predictions/metrics.json:2-18`. Each metric stdout is fully captured in the observer and equals the corresponding metrics object. Every prediction receipt row equals its captured tracking log row: 50/50 and 80/80, unique sequence names, positive frame counts, and sums exactly 76,373 and 101,956. The full ordered sequence/frame lists equal the hash-verified inherited cases, not merely the same totals.

The four OPE track/analyze terminal receipts and four exit files are all zero. DepthTrack tracking finished at 2026-10-10 16:25:38.482019 UTC; its metric stage finished at 16:46:00.174138 UTC. CDTB tracking finished at 16:45:55.478538 UTC; its metric stage finished at 16:46:07.638218 UTC. Tracking completion precedes analysis. Evidence: each `F/{depthtrack_test,cdtb}_{track,analyze}.receipt.json:14-16` and corresponding `.exit:1`.

These are completed results for the **existing Current-quality C composite**. Use the exact qualifier “completed, receipt-supported OPE result under the locked inherited evaluator; not independently recomputed in this bounded audit.”

## A. Ground-truth provenance — PASS within source/receipt scope

The OPE tracking path uses the fixed initial box and RGB/depth frames. It saves all box/confidence rows before creating the complete prediction receipt; it has no subsequent-GT input in its track loop. The analyzer verifies prediction hashes and each dataset `groundtruth.txt` hash, then invokes the locked evaluator. That evaluator loads dataset GT separately from predictions and checks equal frame lengths. Evidence: `D/run_m122_current_C_official.py:12-41,44-60`; `M:93-111`; `D/m122_current_C_official_runtime.py:9-35`.

The inherited cases point to real dataset roots `/root/autodl-tmp/depthtrack/test/sequences` and `/root/autodl-tmp/CDTB/sequences`, with 50 and 80 per-sequence GT SHA records. Both case files match their inherited plan hashes. The current plans were reconstructed **in memory** using the inspected preparation transformation; their resulting exact SHA values match selection, prediction receipts and metric files. This ties these observed outputs to the inherited real-GT plan. It does not substitute for reading all runtime GT bytes locally.

Training labels are also GT-derived: teacher replay finishes W/K rollouts before loading dataset GT and calculating current-box IoU (`D/run_template_write_teacher.py:129-153`); `D/train_m122_full_causal.py:14-22` checks the GT file hash and row alignment, including the explicit existing toy07 row contract. `D/train_template_write_C.py:39-99` loads the sealed teacher's current IoU and cached features. This is a GT-derived regression target on model states, not a model-generated “ground truth” reference. The current trainer selects that `current` target at `D/train_m122_current_C_full152.py:36-40`.

**Limit:** no fresh GT/image bytes or neural feature origin were independently regenerated. Historical recomputation outputs were checked for their exact linkage and all 1,297 label rows were parsed; those historical checks are not represented as rerun now.

## B. Score normalization — PASS

The locked evaluator file is exactly SHA-256 `05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc`, both in the older native OPE snapshot and in `A/official_analysis_sources/depthtrack_pr.py`. The latter's capture receipt identifies the actual remote source as `/home/SRTrack_RGBD_L/lib/test/analysis/depthtrack_pr.py` (`A/official_analysis_sources/receipt.json:12-15`).

Its derivation is explicit:

1. Convert predictions and valid GT boxes to VOT rectangles; absent/unknown GT becomes a special region; call VOT bounded rectangle overlap with image width/height (`M:40-60`).
2. Form the fixed resolution-100 global confidence-threshold grid, including infinity endpoints (`M:25-37,117`).
3. At each threshold, compute each sequence's precision as mean selected IoU and recall as selected-IoU sum divided by the number of visible GT frames. No selected prediction gives precision 1 and recall 0 (`M:120-131`).
4. Macro-average sequence curves; choose the maximum harmonic F; return the P/R at that same threshold and multiply by 100 for percentages (`M:133-154`).

This divides recall by dataset visibility and precision by selected detection count, not by a model-output maximum/mean to inflate a score. The prediction-score threshold grid and best-F selection are part of the fixed PR definition; they are not checkpoint selection. The reported thresholds, 0.089707 and 0.166124, are PR thresholds, separate from the fixed C write gate 0.5. All six percentage conversions and both harmonic-F identities were checked exactly against the stored floating values. No value is silently rounded for gate comparisons.

The inspected file is the **locked inherited project evaluator using VOT overlap and long-term macro PR**, not evidence that this review ran an upstream official evaluator. Its installed VOT dependency/version/binary and bitwise equivalence to an upstream official implementation were not freshly attested. The metric imports `vot.region`; the successful receipt names the existing `mplt` analysis Python.

## C. Result existence and numeric fidelity — WARN (reportable with limits)

The two result files, metric keys, exact numbers, complete prediction receipts, source hash, and successful metric exits exist. Metric `receipt_sha256` equals the actual local receipt bytes. The receipt count/unique-name/frame checks all pass. Evidence: both `F/evaluation/*/predictions/metrics.json:2-18`; DepthTrack `receipt.json:2-7,359-365`; CDTB `receipt.json:2-7,569-575`.

The common final is `cc71e0b4f42c2e489325627d0a16dcdfc4fa6bcb941908c30bc20c8c3c5828c0`; common bundle is `eb8a540f4b3ced87c1c25217ff4eb4c2d950f4a4d15e103f9af0eaa39439d991`; bank is `78cbae28b4b63ecab768d747a889ca120c62e1368ce80fa7f798ee6508bc29a9`; initialization binding is `3bf7aa31a3f819ec746b37dc221e55c531a1235726cda3efedc4990c607cb26e`. Selection has exactly one model and explicitly records zero external optimizer steps and no external metric checkpoint selection (`F/evaluation/selection.json:2-46`).

The current bundle was independently reconstructed as JSON from the actual C fit result, source gate and captured inherited P1 bundle; its bytes hash exactly to `eb8a540f…`. The same procedure gives current plans `79e0ee6530c9a63d4b09879fafb28acc102918d5f088ec60c1f7d8ac2f1c2dee` and `ccc4eff25249feeebde8575d1389088305279deead03c033ac9579edf1a797ea`. These are deterministic reconstruction checks, explicitly **not captured original current plan/bundle files**.

The tracker remains stale relative to this observation: its QF152 row says FIT_COMPLETE/OPE_RUNNING and its latest note describes the earlier 00:30 snapshot (`D/refine-logs/m122_trusted_memory/EXPERIMENT_TRACKER.md:52,58,66`). Therefore tracker DONE is not available; terminal receipts support the two individual completions. The original R4/R5 rows must not be relabeled as completed by these CurrentC results.

**Limit:** the bounded collected directory contains **0/130 raw bbox files and 0/130 confidence files**. Their hashes are present in receipts, not their bytes. No fresh PR threshold curve, per-frame overlap, or GT-content comparison was recomputed. Current final.pt and bank tensors were not opened or independently rehashed here.

## D. Call path, frozen bindings, and dead code — PASS within recorded execution scope

The current controller really wires track on the existing STTrack Python and analyze on the existing mplt Python, with blocking child waits and exit checks (`D/run_m122_current_C_full_suite.py:14-29,33-60`). Actual stage command arrays select `run_m122_current_C_official.py --mode ope_track/ope_analyze`, not the old resume script. `D/resume_m122_evaluation_metrics.py:15-42,65-70` is the historical P0/P1 missing-VOT analysis recovery path; it is not used to assert current completion.

The analyzer calls `evaluate_depthtrack_results(..., resolution=100, sequence_names=...)` at `D/run_m122_current_C_official.py:55-60`, and its returned values appear in both actual metric outputs. `_load_rows`, `_vot_overlaps`, and `_determine_thresholds` are used by that function. The unused `format_metrics` helper at `M:157-158` contributes no claimed number. No phantom metric or unreachable-result claim was found.

Current runtime verifies base/current source and checkpoint hashes, exact A_B key/tensor equality to P1 at load, fixed C state, same bank/binding, and inherited cases/dataset/metric definitions (`D/m122_current_C_official_runtime.py:9-35,47-68`). Same-index box/score/feature/quality assertions remain in `D/trusted_template_tracker.py:32-46`; C only gates native-qualified template writes, interval 50 / confidence >0.75 and C >0.5 (`D/trusted_template_tracker.py:47-67`; `D/template_write_C.py:21-24`). No GT is supplied to these calls. The frame score and box saved by OPE come from that same selected record.

The source compares native/CLIP and C state digests before/after tracking and records no optimizer or online text changes. The A_B decoder does **not** have a separate before/after digest in this OPE receipt: its freeze is supported by load equality, eval/no_grad and absence of an optimizer, not by a newly exported A_B digest (`D/full_dense_tracker.py:39-40`; `D/run_m122_current_C_official.py:17-19,38-41`). Complete internal tensor identity and live imports were not independently observed.

Text binding bytes were read and hash-verified: 1,895 unique legal initialization keys, counts 50/80/1765, all rows human-confirmed and containing 1–5 phrases. Each OPE sequence/frame/init-box entry matches its binding row. Runtime uses fixed initialized bank words (`D/full_dense_tracker.py:75-85`; `D/encode_m122_official_human_text.py:12-33`). The original binding explicitly discloses **multiframe human review aids** (`A/official_initialization_binding.json:54578-54589`); do not describe the annotation process as blind first-frame-only or freshly re-reviewed.

## E. Scope and training provenance — WARN

The completed evidence is **one fixed composite, one seed (2027), two full OPE datasets**, not a pilot subset. It is not multi-seed robustness, causal module effectiveness, or all-three-dataset completion.

CurrentC fitting is specifically:

- 1,297 original teacher events; 1,257 common eligible events from **142 contributing sequences**, with 35 current-GT-unknown and 5 future-unavailable exclusions.
- The original 993 fit plus 264 development events are merged. Former development32 is now inside C training; those sequences cannot remain an unseen validation set.
- C architecture 519→128→32→1, **70,721 trainable C parameters**, seed 2027, ten epochs, batch 32, lr 3e-5, weight decay 0.01, **400 optimizer updates**, fixed epoch-10 final.
- Previously trained P1 A+B is copied unchanged into the composite; no backbone instantiation or fresh own-C-history teacher recollection occurs in this fit.

Evidence: `D/train_m122_current_C_full152.py:36-90`; `F/fit/result.json:2-40,41-92`; `D/train_template_write_C.py:39-99`. All 1,297 inherited label rows were parsed again; eligibility counts, 142 sequences and exclusions match. Dataset digest `5894dd48e03ca466cddd56311e8efbf31a9c4d979897b35d588ae10bc3ce5dc9` links the current result to the earlier raw-audit output. The local training spec has 152 unique sequences, 219,802 noninitial calls per pass, and no DepthTrack Test sequence-name overlap; this confirms manifest scope, not new full-network execution.

The captured P1 training result `D/m122_native_observation_20261010_051718/train_precision1/result.json:2-21` matches its inherited bundle hash and records 380,167 parameters, three passes/456 sequence runs/659,406 calls. This history belongs to **A+B before the cached C fit**. It does not turn the later 400-update cached fit into the user's requested simultaneous complete Train152 ABC training. The tracker explicitly preserves this distinction and records M123 NOT_LAUNCHED (`.../EXPERIMENT_TRACKER.md:60-66`). This audit supplies no M123 execution evidence.

The historical initialization reproduction limitation remains: recorded current seed-state digest `d83b325d…` differs from the earlier local CPU reconstruction `94c24193…`; original initial tensors were not exported. `T/FIT_PREFLIGHT_RECOMPUTED.json:26-30` states this limitation. It was not retried, explained away, or reclassified as an exact independent initialization check.

The existing collector's unrounded target comparisons yield **3/6 available OPE conditions true**: DepthTrack P/F and CDTB P pass; DepthTrack R = 64.86510340917104 is below 64.9, and CDTB R/F are below 75.6/74.2. Evidence: `D/collect_m122_current_C_results.py:41-44`. Rounding DepthTrack recall to 64.9 must not be used to claim passage.

At the captured observation, VOT has only **69 nonempty trajectory triples (34 + 35 + 0 + 0) of 1,765**. There is a launch, no collected VOT terminal receipt, merge, official analysis, complete result, or full-suite terminal receipt. Nonempty triples are progress evidence, not completed official evaluation. `F/vot_track.launch.json:2-12`; `F/evaluation/vot/run/launch.json:1-32`; observer `:1`. The remaining VOT/collection source is wired (`D/run_m122_current_C_full_suite.py:61-79`; `D/analyze_m122_current_C_vot.py:17-35`), but wiring is not execution.

## F. Evaluation type — PASS classification

- Observed DepthTrack Test and CDTB OPE: **real_gt**, with the receipt/source limitations above.
- C fitting: real-GT-derived current-IoU supervised surrogate on cached original P1 states; not a separate formal tracking-performance metric.
- This review's checks: deterministic byte, JSON, count, arithmetic and static-call-path verification; no new performance evaluation.
- Human phrase review is initialization annotation, not human rating of tracking outputs. Existing source fixtures and previous development outputs are not new CurrentC formal results.
- VOT completion and M123 performance remain **unproven**.

## Action items and claim ceiling

The two OPE rows may be reported now with their exact method identity and receipt-supported qualifier. Update the current tracker in a separate authorized publication step, citing these actual terminal artifacts; this reviewer has not changed it.

To claim independently recomputed CurrentC metrics, obtain the same 130 bbox/confidence files plus hash-bound GT/image-dimension inputs and independently rerun the locked metric/official-equivalence checks. To claim a complete VOT or full nine-metric row, wait for its actual complete artifacts and audit them. To satisfy full Train152 ABC joint training, use evidence from an actual jointly updated complete training run; these cached C outputs do not satisfy that requirement.

No detected integrity contradiction blocks the **limited reporting decision**. Missing stronger evidence remains missing; the report is not a blanket assurance of runtime correctness or a scientific promotion.

