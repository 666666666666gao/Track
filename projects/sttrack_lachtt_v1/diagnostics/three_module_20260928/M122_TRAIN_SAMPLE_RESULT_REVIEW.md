# M122 Train152 result experiment-integrity audit

Date: 2026-10-08  
Overall verdict: **PASS within the historical sparse-training scope**  
Checks A–F: **PASS / PASS / PASS / PASS / PASS / PASS**  
Blocking findings: **0**. Nonblocking defects: **0**. Warnings: **0**.

The actual CPU diagnostic outputs agree with an independent, stdlib-only reconstruction from the original dataset GT and sealed state traces. All 3,996 output states, 912 CSV ledgers, six summaries, and 56 populated quality bins were checked. Every valid sampled Float64 IoU, Float32 loss-label IoU, summary mean/error, and bin mean matched exactly as parsed Python numbers; maximum observed absolute difference was 0. The audit did not import or execute the analyzer or a model.

Requested route: Codex, `gpt-6-astra`, reasoning effort `max`, fresh `fork_turns: none`. Actual backend/model/effort are **not independently attested**. Reviewer: `/root/m122_train_sample_result_review` (canonical task path). `review_independence: same-family`; `acceptance_status: provisional`. This is not cross-family acceptance.

For exact file:line references below:

- `D = C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928`
- `P = C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007`
- `H = P/observations/hour_2017`
- `I = P/train_diagnostic_inputs_20261008`
- `R = P/train_samples_result_20261008`

## A. Original GT, indexing and image provenance — PASS

All 152 GT files were directly read and parsed. Their byte hashes match both the original sequence manifest and the exported file manifest. The original manifest SHA256 is `3569d9c42b8db332281ad70bec719650b876e8b9003c48ba73b17d1451e4b425`. The dataset path is `/root/autodl-tmp/depthtrack/train/sequences`; GT is not derived from model output. The exporter verifies each source GT hash before copying its bytes. Evidence: `I/training_spec.json:6`; `P/export_train_diagnostic_inputs.py:5`, `:16`, `:18`, `:20`; `D/train_m122_full_causal.py:14`; `D/analyze_m122_train_samples.py:45`.

The complete files contain 219,993 raw GT rows. Only the existing `toy07_indoor_320` exception trims 1,406 rows to the 1,367 paired image frames, excluding the final 39 unmatched rows. That leaves 219,954 image-aligned rows. Removing 152 initialization frames gives 219,802 tracking frames per pass: 203,376 valid and 16,426 invalid. The invalid tracking rows comprise 16,304 with nonfinite coordinates and 122 finite rows with nonpositive dimensions. Evidence: `D/train_m122_full_causal.py:17`, `:19`, `:90`; `D/analyze_m122_train_samples.py:12`, `:50`, `:53`; `I/groundtruth/toy07_indoor_320.txt:1367`, `:1368`, `:1406`.

Saved frame `f` is the zero-based GT index, text line `f+1`, and image filename `%08d.jpg % (f+1)`. Frame 0 initializes; tracking begins at 1. All saved validity flags agree with the corresponding raw GT. Invalid GT produces no geometric or calibration label. For example, `cat02_indoor`, frame 750 has `nan,nan,nan,nan` at `I/groundtruth/cat02_indoor.txt:751`; its output at `R/sample_rows.json:668` omits all four derived geometric fields. Evidence: `D/train_m122_full_causal.py:84`, `:85`, `:88`; `P/export_train_diagnostic_inputs.py:31`; `D/analyze_m122_train_samples.py:69`, `:78`.

Image metadata covers exactly the 666 unique sampled sequence/frame pairs: 653 shapes of 360×640 and 13 of 320×640. The metadata records positive image sizes and SHA256 values; the exporter reads and decodes those sampled RGB files on the remote CPU. The local ZIP contains the text GT and metadata, not the RGB image bytes. I verified the sealed metadata/export chain, **not independent image re-decoding or dataset-publisher authentication**. Evidence: `P/export_train_diagnostic_inputs.py:29`, `:31`, `:32`, `:33`, `:34`; `I/image_dimensions.json:3`; `R/result.json:666`.

## B. Independent arithmetic and denominators — PASS

I reconstructed ordinary Float64 box IoU for all 3,690 valid sampled states, then separately rounded inputs and each arithmetic operation to IEEE Float32 for the historical quality-loss target. The recorded prediction is the committed Float64 box; training casts it to Float32 before computing the loss. Both reconstructions agree exactly with every output value. No prediction maximum, minimum, or mean normalizes these reported metrics. Evidence: `D/full_dense_tracker.py:46`, `:94`, `:101`; `D/train_m122_full_causal.py:26`, `:92`; `D/train_m121_dense_target.py:19`, `:51`; `D/analyze_m122_train_samples.py:15`, `:80`, `:81`.

The observation label is a positive-area intersection of the GT box, recorded crop square, and historical image bounds ending at `width-1`/`height-1`. It is distinct from the separate center-in-crop test. All 3,690 labels of each kind match. Sixty retained valid states have a center outside the crop but a box intersecting the observed window. Example: `toiletpaper04_indoor`, precision0/pass 1/frame 1000, GT `228,285,57,73`, crop `180,317,74`: box intersection true, center-in-crop false. Evidence: `D/m121_dense_inputs.py:97`, `:99`, `:100`, `:101`; `D/analyze_m122_train_samples.py:23`, `:82`, `:84`; `I/groundtruth/toiletpaper04_indoor.txt:1001`; `H/train_precision0/sampled_state_trace.jsonl:247`; `R/sample_rows.json:8561`.

For each summary, mean IoU, quality MAE, and observation Brier score use exactly 615 valid retained states. All 56 populated quality bins, their counts, and both means per bin were independently checked; their counts sum to 615 per summary. Reaggregation used `math.fsum` and independently assigned decile bins. All reported values match exactly. Evidence: `D/analyze_m122_train_samples.py:91`, `:95`, `:107`, `:108`, `:109`, `:110`; `R/result.json:200`, `:278`, `:356`, `:434`, `:512`, `:596`.

The historical preservation training loss contains response-based ranking ratios; these are not substituted for diagnostic performance metrics. No benchmark P/R/F, EAO, ROB, or official VOT score is computed. CPU Float32 reconstruction establishes agreement with the written arithmetic, not GPU bitwise replay. Evidence: `D/train_m121_dense_target.py:59`, `:61`; `D/M122_TRAIN_SAMPLE_DIAGNOSIS_PLAN.md:11`, `:13`; `R/result.json:664`.

## C. Execution, complete outputs and SHA linkage — PASS

The receipt records the exact analyzer/plan/output arguments and actual exit code 0, from 2026-10-08 21:09:31.028164 to 21:09:31.819898 +08:00. The wrapper waits for the subprocess and records its return code. The one-line execution log contains the completion status, 3,996 states, 912 ledgers, and six summaries, exactly matching the output JSON. The source review predates execution and its hash matches the receipt and wrapper gate. Evidence: `P/execute_reviewed_train_samples.py:8`, `:9`, `:18`, `:21`, `:22`, `:24`; `P/train_samples_execution_receipt.json:3`, `:13`, `:14`; `P/train_samples_execution.log:1`; `P/M122_TRAIN_SAMPLE_SOURCE_REVIEW.json:6`.

Analyzer SHA256 `cf7ba331aaf034a79da3aaa422cca682a7112742080c3d385faf12564908add6` agrees with the plan and result. Plan SHA256 `899a40b784711304590a56a94a5f52e2435ad9ba10fd0b7026b62035fe50b518` agrees with the result. All 168 plan-pinned inputs match, and all 154 manifest entries match both hashes and byte sizes. The 1,036,355-byte input ZIP matches receipt SHA256 `998b5d8afae7c14f8dd4c15104f05e2b1f74deceb6bd0a0da0307c2944b8c565`; all 155 members match the extracted bytes. All 178 currently read inputs also present in the prior source audit retain its recorded hashes. Evidence: `P/train_samples_plan.json:3`, `:15`; `R/result.json:6`, `:7`, `:8`; `P/train_diagnostic_input_export_receipt.json:6`, `:7`.

Both archived training runs have exit 0, complete 456-record ledgers, and stdout records identical to their result arrays. Twelve relevant archived launch/exit/log/result/trace files match the observation manifest. GT validity partitioned into 32-call accumulation blocks independently reproduces every ledger's cumulative optimizer count: 6,807 updates per pass, 20,421 per configuration. Every per-sequence call and supervision count is recomputed from original GT. Evidence: `H/train_launch.json:37`, `:75`; `H/train_precision0.exit:1`; `H/train_precision1.exit:1`; both `H/train_precision*/sequence_log.jsonl:1`, `:456`; `D/train_m122_full_causal.py:99`, `:100`, `:108`.

All 3,996 output rows copy their corresponding original states, in order, without extra or missing rows. All valid geometries were recomputed; the other 306 states retain unknown-GT status. All 912 CSV data rows agree field-by-field with GT, original full logs, and independently derived sampled states. All six summaries and all 56 populated bins agree. Evidence: `R/sample_rows.json:3`; `R/per_sequence.csv:1`, `:2`, `:913`; `R/result.json:180`, `:258`, `:336`, `:414`, `:492`, `:576`; both `H/train_precision*/sampled_state_trace.jsonl:1`, `:1998`.

**Full window-intersection and write counts are verified sums of complete archived ledgers.** The sparse package cannot independently reconstruct those decisions on every unsaved frame. Neural execution itself is supported by sealed launch/exit/log artifacts; this review did not replay it or rehash remote weights.

## D. Live serialization and input immutability — PASS

The analyzer's reachable CLI connects GT loading, both IoUs, observation labels, summaries, and all three output writers. Its imports are stdlib-only; it does not import the trainer, model, optimizer, Torch, or a network client. Actual output existence, receipt/stdout linkage, and independent reproduction establish execution beyond static reachability. Evidence: `D/analyze_m122_train_samples.py:2`, `:29`, `:80`, `:96`, `:114`, `:134`, `:135`, `:136`, `:137`, `:142`.

The executor wrapper archives source-review trace metadata and writes the diagnostic receipt/log; it does not modify model inputs or parameters. The analyzer hashes inputs before processing and again before writing. The reviewed trainer/runtime/loss/geometry source hashes match five entries of the archived source gate. All 193 audit input/reference files retain their review-start hashes and pass a final on-disk hash recheck. Evidence: `P/execute_reviewed_train_samples.py:13`, `:16`, `:17`, `:20`, `:23`; `D/analyze_m122_train_samples.py:37`, `:123`; `P/complete_evaluation_source_gate.json:18`, `:20`, `:25`, `:35`, `:36`.

This reviewer performed no analyzer execution/import, neural execution, network/SSH access, GPU access, installation, current NN-progress query, input mutation, optimizer/threshold/checkpoint change, or new training. Only the requested Markdown and JSON result-review files were written. No claim is made about unexamined remote model bytes.

## E. Historical training and sparse-sampling scope — PASS

The verified scope is 152 Train sequences, two configurations, seed 2027 once per configuration, and three passes per configuration. Each configuration has 659,406 tracking calls, 610,128 supervised frames, 20,421 updates, 456 ledgers, and 1,998 sampled states. Combined: 1,318,812 calls, 1,220,256 supervised frames, 98,556 invalid-GT tracking frames, 40,842 updates, 912 ledgers, and 3,996 sampled states. The 3,996 states repeat 666 distinct sequence/frame locations; 3,690 have valid GT and 306 do not.

The exact trace order and schedule match frame 1, each 500th frame, and the last tracking frame, with duplicate endpoints removed. Each of the six summaries contains 666 samples: 615 valid and 51 invalid. The complete write ledgers contain 7,571 writes, while the retained states contain 632 writes: 595 with valid GT and 37 with unknown GT. Ninety-six of the 595 valid-GT sampled writes have geometric IoU ≤ 0.1. These are retained-sample counts, not an all-write failure rate. Evidence: `D/train_m122_full_causal.py:97`, `:98`; `D/analyze_m122_train_samples.py:64`, `:104`, `:105`, `:106`; `D/full_dense_tracker.py:104`, `:105`; `R/result.json:662`.

Every 500th-frame sample is also eligible for the 50-frame write interval. There are 2,274 interval-eligible sampled states, versus 25,944 eligible tracking states across the complete histories. Endpoints and short sequences receive different relative representation. These deterministic samples do not identify unbiased all-frame localization, all-frame calibration, or all-write error rates. The means are retained-valid-state averages, not sequence-macro averages. Reweighting cannot recover unsaved outcomes.

Weights change during each pass; pass 1 already follows the common warm-trained model. Three passes and repeated frames are dependent observations, not independent seeds or frozen-final evaluations. The reused M82 manifest supplies dataset/order/GT metadata; its old training settings are not M122 settings. M122 uses three passes, learning rate 3e-5 and weight decay 0.01. Evidence: `D/train_m122_full_causal.py:48`, `:57`, `:77`, `:79`, `:108`; both `H/train_precision*/result.json:6`, `:12`, `:13`; `R/result.json:660`.

The original diagnosis-plan header still reflects its pre-execution source-review-pending stage. Actual diagnostic completion comes from the later receipt and reproduced outputs. The historical hour_2017 snapshot records completed training but unfinished evaluation; it is not evidence of current or final VOT completion. Evidence: `D/M122_TRAIN_SAMPLE_DIAGNOSIS_PLAN.md:3`; `H/observation.json:6`, `:7`, `:11`.

## F. Classification and exact claim ceiling — PASS

Classification: **`real_gt_training_diagnostic`**, a subtype of real-dataset-GT evidence. The following claims are supported:

- Exact archived training scope and full-log counter sums, with full counters distinguished from independently reconstructed sampled geometry.
- The stated sparse-sample localization, quality-error, observation-calibration, and write-associated error counts on each configuration's own historical training states.
- Descriptive differences between these two one-seed histories, without causal or final-performance interpretation.

These artifacts do **not** establish final-checkpoint or held-out improvement; unbiased all-frame/all-write performance; seed robustness; official benchmark P/R/F or VOT scores/completion; physical absence from invalid GT or window nonintersection; causal benefit/harm of text, a module, or template writing; template-content contamination; or new C-module efficacy. A write concurrent with poor localization does not show that the write caused that error or later errors. Selected-cell quality MAE is not the all-candidate BCE training loss. Evidence: `D/train_m121_dense_target.py:51`; `D/analyze_m122_train_samples.py:124`, `:127`, `:128`, `:129`, `:130`, `:132`, `:133`; `R/result.json:3`, `:660`, `:661`; `D/M122_TRAIN_SAMPLE_DIAGNOSIS_PLAN.md:13`.

## Independently verified numerical results

All six rows use 219,802 full calls, 203,376 full valid-GT frames, 16,426 full invalid-GT frames, and 615 valid retained states out of 666. IoU, MAE, and Brier below are raw 0–1 values rounded for display; exact verified values and all 56 bins are in the companion JSON.

| Configuration/pass | Full window intersections | Full writes | Sample mean IoU | Quality MAE | Observation Brier |
|---|---:|---:|---:|---:|---:|
| precision0 / 1 | 176,481 | 1,286 | 0.749311474 | 0.271382341 | 0.049040026 |
| precision0 / 2 | 175,137 | 1,185 | 0.756257825 | 0.187546993 | 0.041412427 |
| precision0 / 3 | 178,757 | 1,247 | 0.755849737 | 0.170892378 | 0.044749083 |
| precision1 / 1 | 173,860 | 1,276 | 0.749203671 | 0.259513708 | 0.043409598 |
| precision1 / 2 | 178,174 | 1,246 | 0.759819468 | 0.159466274 | 0.028869089 |
| precision1 / 3 | 181,725 | 1,331 | 0.780056608 | 0.142684356 | 0.037642427 |

| Configuration/pass | Severe samples | Center out | Window nonintersection | Nonintersection predicted ≥ 0.5 | Sampled valid-GT writes | Severe sampled writes | Unknown-GT sampled writes |
|---|---:|---:|---:|---:|---:|---:|---:|
| precision0 / 1 | 97 | 86 | 74 | 13 | 105 | 21 | 7 |
| precision0 / 2 | 94 | 84 | 74 | 13 | 88 | 14 | 4 |
| precision0 / 3 | 94 | 84 | 70 | 14 | 97 | 21 | 6 |
| precision1 / 1 | 99 | 89 | 82 | 21 | 104 | 14 | 6 |
| precision1 / 2 | 90 | 80 | 74 | 9 | 89 | 8 | 7 |
| precision1 / 3 | 80 | 67 | 56 | 12 | 112 | 18 | 7 |

“Severe” means selected geometric IoU ≤ 0.1 with valid GT. For a concrete retained event, precision0/pass 1/`cube04_indoor`/frame 1000 has IoU 0, native same-selected-cell response 0.9099556803703308 and a recorded write; this matches the unchanged interval-50 / >0.75 rule. Evidence: `I/groundtruth/cube04_indoor.txt:1001`; `H/train_precision0/sampled_state_trace.jsonl:3`; `R/sample_rows.json:73`. It is an association at a sampled training state.

## Deterministic checks, findings and disposition

**30,855 deterministic assertions passed; 0 failed.** This is an assertion count, not a sample-size claim. It includes 30,661 content/arithmetic assertions and 193 final per-file comparisons and one aggregate start-snapshot comparison across 193 audit inputs/references. Compound state and ledger assertions check several related fields. The JSON records every assertion-group count, all input SHA256 values, separate session-context hashes, all 152 GT inventories, exact recomputed summaries/bins, and the independent checker source.

No correction is required to the reviewed diagnostic outputs. Reporting may use these verified historical Train152 diagnostic numbers **only with the scope and claim ceilings above**. The declared provenance, sparse-sampling, changing-weight, missing-image-byte, CPU-replay and causal limits are limits of the evidence, not undisclosed defects. This review adds no benchmark or VOT result.
