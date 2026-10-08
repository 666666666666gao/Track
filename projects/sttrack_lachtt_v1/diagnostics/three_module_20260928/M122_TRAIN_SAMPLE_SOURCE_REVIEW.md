# M122 Train152 source experiment-integrity audit

Date: 2026-10-08  
Overall verdict: **PASS — source audit only**  
Checks A–F: **PASS / PASS / PASS / PASS / PASS / PASS**  
Blocking findings: **0**. Nonblocking defects: **0**. Warnings: **0**.  
Deterministic evidence checks: **6,536 passed; 0 failed**.

The pinned CPU diagnostic is suitable for execution within its stated scope. This is not a result audit: I did not import or execute the planned analyzer, run a neural model, compute new diagnostic IoU/calibration/error figures, use network/SSH/GPU, install packages, or modify an input. New numerical diagnostic claims remain pending execution and review of the actual outputs.

Requested reviewer route: `gpt-6-astra`, effort `max`, fresh `fork_turns: none`, Codex backend. Actual backend/model/effort are not independently attested. Reviewer family: `openai`; `review_independence: same-family`; `acceptance_status: provisional`. Reviewer identifier is the canonical task path `/root/m122_train_sample_source_review`, not an independently attested backend model ID.

## Evidence scope and counting

I read all 23 explicitly listed artifacts, including the eight Python sources in full, and directly read and parsed every row of all 152 listed GT files. I also read the supplied local export ZIP and compared every member with the extracted files. The machine-readable companion contains SHA256 values for all **180** audit inputs/references: 175 requested files, the ZIP, the request, and three relevant audit-policy references. It also contains a separate 152-file GT inventory with byte hashes, raw/used row counts, validity counts, and update counts.

The complete GT files contain **219,993 raw rows**. The existing, explicit `toy07_indoor_320` exception excludes its final 39 unmatched rows, leaving **219,954 image-aligned GT rows**. Excluding the 152 initialization frames gives **219,802 tracking calls per pass**, with **203,376 valid supervision frames** and **16,426 invalid-GT frames**. Of those invalid rows, 16,304 contain nonfinite coordinates and 122 have finite but nonpositive dimensions. The original validity rule handles both.

The 6,536 checks comprise 154 manifest byte/hash checks, 168 plan-pin checks, eight source AST checks, 912 GT checks across six checks per file, 912 compound ledger-row checks, 3,996 compound sampled-state checks, 155 archive-member comparisons, 176 final input-stability checks, and 55 other identity/schema/scope checks. A compound row check tests several conditions; this is an audit assertion count, not a statistical sample-size claim. Every final input-stability check passed.

For compact file:line citations below, source filenames resolve under:

- `D = C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928`
- `P = C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007`

## A. Ground-truth provenance — PASS

The reference boxes come from the dataset directory `/root/autodl-tmp/depthtrack/train/sequences`, through the original training manifest. The exporter reads each sequence's `groundtruth.txt`, verifies its original manifest digest, and copies the bytes. The trainer independently uses that same manifest-bound GT. No GT is generated from predictions. Evidence: `P/train_diagnostic_inputs_20261008/training_spec.json:6`; `P/export_train_diagnostic_inputs.py:5`, `:16`, `:18`, `:20`; `D/train_m122_full_causal.py:14`, `:15`, `:16`.

All 152 local GT byte hashes match both the exported manifest and original sequence manifest. All raw row counts, four-column schemas, first boxes, RGB/depth frame counts, and declared initial-image paths match. The only length adjustment is the existing `toy07_indoor_320` rule: 1,406 GT rows, 1,367 RGB/depth frames, and a prefix of 1,367 rows in both trainer and diagnostic. Evidence: `D/analyze_m122_train_samples.py:45`, `:47`, `:48`, `:50`, `:53`, `:54`; `D/train_m122_full_causal.py:17`; `P/train_diagnostic_inputs_20261008/training_spec.json:1187`; `P/train_diagnostic_inputs_20261008/groundtruth/toy07_indoor_320.txt:1367`, `:1368`, `:1406`.

Indexing is consistent: saved `frame=f` is zero-based; Python GT index `f` corresponds to text line `f+1` and RGB filename `%08d.jpg % (f+1)`. Frame 0 supplies initialization; tracking starts at frame 1. Evidence: `D/train_m122_full_causal.py:84`, `:85`, `:88`; `P/export_train_diagnostic_inputs.py:31`; `D/analyze_m122_train_samples.py:69`.

Invalid GT is tested for finite coordinates and strictly positive width/height, exactly as in training. It does not become an absence label. An actual invalid input is `P/train_diagnostic_inputs_20261008/groundtruth/cube04_indoor.txt:567` (`nan,nan,nan,nan`). Evidence: `D/analyze_m122_train_samples.py:12`, `:70`, `:78`, `:129`; `D/train_m122_full_causal.py:90`.

Provenance is established to the sealed original manifest/export chain. I did not independently authenticate the dataset publisher or access the remote dataset.

## B. Score normalization and label arithmetic — PASS

The proposed metrics do not divide scores by the model's own maximum, minimum, or mean. IoU uses the ordinary box-area union. Sample mean IoU, quality absolute error, observation Brier score, and bin means use valid retained-sample counts. Those denominators are nonzero for all six model/pass groups in the supplied inputs. Evidence: `D/analyze_m122_train_samples.py:15`, `:20`, `:107`, `:108`, `:109`, `:110`.

The two IoUs have distinct purposes. The runtime constructs and commits Float64 boxes; the training objective casts those boxes to Float32 before forming the quality target. The diagnostic retains the recorded Float64 box for geometric IoU and separately rounds inputs and arithmetic to Float32 for the loss-label reconstruction. Evidence: `D/full_dense_tracker.py:46`, `:94`, `:101`; `D/train_m122_full_causal.py:26`, `:92`; `D/train_m121_dense_target.py:19`, `:51`; `D/analyze_m122_train_samples.py:9`, `:17`, `:18`, `:20`, `:80`, `:81`.

The observation label reproduces the geometric intersection of the GT box, recorded crop square, and image bounds ending at `width-1`/`height-1`. It requires positive extent in both coordinates and matches the historical loss definition. Evidence: `D/m121_dense_inputs.py:97`, `:98`, `:99`, `:100`, `:101`; `D/analyze_m122_train_samples.py:23`, `:26`, `:27`.

Prediction-normalized ranking gaps do exist in the historical preservation **training loss** (`D/train_m121_dense_target.py:59`, `:61`). They are not reported as performance metrics by the new analyzer. No P/R/F, EAO, ROB, or substituted benchmark score is calculated. Float32 CPU arithmetic is explicitly a diagnostic reimplementation, not a GPU bitwise-replay assertion (`D/analyze_m122_train_samples.py:130`; `D/M122_TRAIN_SAMPLE_DIAGNOSIS_PLAN.md:11`, `:13`).

## C. Input existence, SHA binding and archived alignment — PASS

Every one of the 168 pinned local inputs exists and matches its digest. The new analyzer's digest matches `analysis_source_sha256`. All 154 entries in the extracted input manifest match their sizes and SHA256 values. The local ZIP is **1,036,355 bytes**, SHA256 `998b5d8afae7c14f8dd4c15104f05e2b1f74deceb6bd0a0da0307c2944b8c565`; its 155 members are exactly the 154 listed inputs plus `files.json`, and every member is byte-identical to the corresponding extracted file. Evidence: `P/train_samples_plan.json:3`, `:15`; `P/train_diagnostic_input_export_receipt.json:6`, `:7`; `P/extract_train_diagnostic_inputs.py:5`, `:6`, `:13`, `:16`; `P/train_diagnostic_inputs_20261008/files.json:763`, `:768`.

For each arm, all 456 sequence-log records equal the full result's `sequence_records` array. Training identity, seed, arm weight, spec digest, trainer/runtime digests, counters, and final ledger rows align. All six result/log/trace files match the bytes and SHA256 values in the archived observation. Relevant local trainer/runtime/loss/geometry source digests also match the five supplied entries checked against the historical source gate. This does not certify all remote sources or current model-weight bytes. Evidence: `P/observations/hour_2017/train_precision0/result.json:2`, `:4`, `:18`, `:47`; corresponding `train_precision1/result.json:2`, `:4`, `:18`, `:47`; `P/observations/hour_2017/observation.json:340`, `:345`, `:350`, `:365`, `:370`, `:375`; `P/complete_evaluation_source_gate.json:18`, `:20`, `:25`, `:35`, `:36`.

Image metadata has exactly 666 unique sequence/frame keys, equal to the expected schedule. All dimensions and image-byte counts are positive, image hashes have valid SHA256 form, and recorded shapes are 320×640 or 360×640. The exporter uses the actual sampled frame filenames and CPU OpenCV decoding. **The raw RGB image bytes are not present locally**, so this audit verifies the sealed metadata and export code, not an independent re-decoding or image-content certification. Evidence: `P/export_train_diagnostic_inputs.py:23`, `:31`, `:32`, `:33`, `:34`; `P/train_diagnostic_inputs_20261008/image_dimensions.json:3`; `D/analyze_m122_train_samples.py:41`, `:43`, `:132`.

The reused `training_spec.json` identifies M82 and retains its old learning-rate/selection metadata. It is used here as the dataset/order/GT manifest; actual M122 settings are defined in its own trainer and sealed results. In particular, the M122 learning rate is 3e-5, weight decay is 0.01, and there are three passes. Evidence: `P/train_diagnostic_inputs_20261008/training_spec.json:2`, `:4`, `:2901`, `:2919`; `D/train_m122_full_causal.py:77`, `:79`; both training results `:6`, `:12`, `:13`.

The diagnosis plan is explicitly pending source review and contains no new diagnostic results (`D/M122_TRAIN_SAMPLE_DIAGNOSIS_PLAN.md:3`, `:15`). The archived observation records completed training but unfinished overall evaluation (`P/observations/hour_2017/observation.json:6`, `:7`, `:11`). I did not convert that historical snapshot into a current-progress or final-VOT claim.

## D. Live code paths and side effects — PASS

The new analyzer has six functions: `sha`, `f32`, `valid`, `overlap`, `intersects`, and `main`. Each has a call site; the CLI guard invokes `main`. Source paths connect GT validation, both overlap variants, observation labels, summary construction, per-sequence rows, and serialization. Evidence: `D/analyze_m122_train_samples.py:29`, `:35`, `:70`, `:80`, `:82`, `:96`, `:114`, `:142`.

Imports are limited to `argparse`, `csv`, `hashlib`, `json`, `math`, `struct`, `collections`, and `pathlib`. The new CLI does not import any historical PyTorch/model/optimizer module and contains no SSH, network, GPU, or current-progress-query call. Historical neural source files were read as evidence only. Evidence: `D/analyze_m122_train_samples.py:2`, `:3`, `:4`.

The plan and local inputs are read without mutation. Input hashes are checked before processing and again before output. A new output directory is required; the three intended output artifacts are `result.json`, `sample_rows.json`, and `per_sequence.csv`. Six summaries, 912 sequence/pass rows, and 3,996 state rows are connected to those writers. Evidence: `D/analyze_m122_train_samples.py:37`, `:123`, `:124`, `:134`, `:135`, `:136`, `:137`, `:139`.

This is static reachability and serialization review. Runtime completion of those writes remains untested because executing the diagnosis was outside this audit.

## E. Full versus sparse scope — PASS

The complete archived scope, verified per model, is:

| Quantity | Per model |
|---|---:|
| Dataset sequences | 152 |
| Seed | 2027 |
| Training passes | 3 |
| Sequence runs | 456 |
| Tracking calls | 659,406 |
| Valid supervision frames | 610,128 |
| Optimizer updates | 20,421 |
| Saved sparse states | 1,998 |

The GT validity stream, partitioned into the trainer's 32-call accumulation blocks with a sequence-end remainder, independently reproduces **6,807 updates per pass** and every ledger's cumulative calls/updates. Every ledger record's supervision count matches the actual GT file. Evidence: `D/train_m122_full_causal.py:86`, `:88`, `:90`, `:94`, `:99`, `:100`, `:108`, `:109`; both archived results `:7`, `:8`, `:9`, `:10`; both sequence logs `:1`, `:456`.

Sampling is precisely frame 1, each 500th frame, and the final frame, with duplicate endpoints removed. The complete trace key sets and order match that schedule for both models and all three passes. Together, the data contain **912 complete ledger rows** and **3,996 saved states** at **666 distinct sequence/frame pairs**. Every sampled GT-validity flag, crop origin, resize factor, probability range, selected-cell index, valid tracked box, and native same-selected-cell write decision satisfies the checked source rule. Evidence: `D/train_m122_full_causal.py:97`, `:98`; `D/analyze_m122_train_samples.py:64`, `:65`, `:66`, `:70`, `:71`, `:73`, `:74`, `:75`, `:76`; `D/full_dense_tracker.py:104`, `:105`, `:110`.

Full call/supervision/window-intersection/write counters come from complete logs. Localization, quality, observation calibration, and write-quality diagnosis use only saved states. Sample means and bins equally weight retained valid samples; they are neither sequence-macro averages nor unbiased all-frame estimates. Endpoints and short sequences receive different relative weights, and every 500th-frame sample falls on a write-eligible 50-frame interval. Missing states prevent an all-write error-rate conclusion. Stratification can describe retained samples; reweighting alone cannot recover unrecorded states. Evidence: `D/analyze_m122_train_samples.py:96`, `:97`, `:98`, `:104`, `:105`, `:107`, `:110`, `:128`; `D/M122_TRAIN_SAMPLE_DIAGNOSIS_PLAN.md:9`, `:13`.

These are two configurations with one seed each, not replicated independent seeds. Three passes and repeated frames are dependent training observations. Weights change during these histories; the means do not describe a fixed final checkpoint or held-out performance. The source and plan state those limits (`D/analyze_m122_train_samples.py:127`; `D/M122_TRAIN_SAMPLE_DIAGNOSIS_PLAN.md:13`).

## F. Evaluation type and claim boundaries — PASS

Classification: **`real_gt`**, subtype **`real_gt_training_diagnostic`**. It is an original-dataset-GT historical diagnostic, not an official benchmark evaluation or synthetic reference evaluation. Evidence: `D/analyze_m122_train_samples.py:124`; `D/M122_TRAIN_SAMPLE_DIAGNOSIS_PLAN.md:11`, `:13`.

The source correctly distinguishes:

- GT-box/window nonintersection from GT-center exclusion. A box can intersect the observed window while its center lies outside the crop square. The center flag tests the crop square; it is not a physical visibility test (`D/analyze_m122_train_samples.py:82`, `:83`, `:84`, `:101`, `:102`).
- Invalid GT from a valid geometric negative. Invalid rows are unknown and remain outside geometric/calibration labels (`:78`, `:106`, `:129`).
- Selected-cell quality diagnostic errors from the all-candidate BCE used in training (`:108`; `D/train_m121_dense_target.py:51`).
- CPU arithmetic reconstruction from bitwise GPU replay (`:130`).
- Logged association from a text/module/template causal effect. There is no counterfactual intervention or template-content audit here (`:133`; `D/M122_TRAIN_SAMPLE_DIAGNOSIS_PLAN.md:13`).

No official P/R/F/EAO/ROB, robustness, held-out improvement, best-checkpoint selection, physical absence, identity-negative label, or new C-module efficacy claim is supported by these future diagnostic outputs alone.

## Findings, claim impact and next step

There are **no source changes required by this audit**. The disclosed provenance, sparse-sampling, changing-weight, arithmetic-replay, and causal-attribution limits are claim ceilings, not newly discovered source defects.

Existing local archived evidence supports the stated complete training scope and the input/source binding. The proposed arithmetic and output scope pass static review. **All new numerical diagnostic claims remain pending**: run only the pinned CPU diagnosis under the existing authorization, preserve these limits, and review the actual outputs before reporting new diagnostic numbers.

The complete input hash inventory, all 152 GT per-file records, assertion-group counts, exact evidence references, requested-route attribution, and execution restrictions are in `M122_TRAIN_SAMPLE_SOURCE_REVIEW.json`. This report and that JSON are the only files written by this reviewer.
