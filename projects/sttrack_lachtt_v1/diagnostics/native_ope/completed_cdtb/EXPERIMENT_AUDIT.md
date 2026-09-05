Overall verdict: **PASS for completed CDTB full80 artifact integrity and arithmetic; WARN for scope/claim strength**. I found **no concrete aggregate or per-sequence export defect**. The completed CDTB native reference is evidence-backed as a local/download-data audit of the published package and private raw bundle, but I did **not** independently access or verify the live server state.

I recomputed CDTB from raw boxes, confidence files, GT, and actual first images using `vot.region` bounded rectangle overlaps and an independently implemented 100-threshold macro-PR calculation. I did not invoke `evaluate_depthtrack_results`, did not run tracking, did not use GPU, did not contact the server, and did not edit project or raw files.

Key independent comparisons:

| Item | Published / expected | Independent recomputation | Status |
|---|---:|---:|---|
| Result files in raw bundle | 160 bbox/confidence files | 160 | PASS |
| Raw dataset dirs | 80 | 80 | PASS |
| Sequences | 80 | 80 | PASS |
| Total frames | 101,956 | 101,956 | PASS |
| Visible GT frames | not separately aggregated in metrics | 91,300 | PASS for per-row recomputation |
| Selected frames at best threshold | from per-sequence export | 91,650 | PASS |
| Threshold grid size | 100 | 100 | PASS |
| Finite thresholds | 98 | 98 | PASS |
| Best threshold index | 89 | 89 | PASS |
| Best threshold | 0.580139 | 0.580139 | PASS |
| Precision | 69.93311256529843% | 69.93311256529843% | PASS |
| Recall | 68.0677435212022% | 68.0677435212022% | PASS |
| F-score | 68.98782086902926% | 68.98782086902926% | PASS |
| Aggregate mismatches | 0 expected | 0 | PASS |
| JSON per-sequence mismatches | 0 expected | 0 | PASS |
| CSV per-sequence mismatches | 0 expected | 0 | PASS |
| CSV/JSON mismatches | 0 expected | 0 | PASS |
| Initial confidence bad rows | 0 expected | 0 | PASS |
| Initial box max delta vs input init box | 0 | 0.0 | PASS |

Hash/seal checks passed:

- Published `metrics_cdtb.json`, `cdtb_receipt.json`, and `download_binding.json` matched the same files in the private raw bundle byte-for-byte.
- All 80 bbox hashes, 80 confidence hashes, 80 GT hashes, and 80 first-image hashes matched `download_binding.json`.
- Receipt-vs-binding sequence frames and bbox/confidence hashes matched for all 80 sequences.
- Metrics GT hashes matched binding GT hashes for all 80 sequences.
- First-image dimensions used for bounded overlaps were readable for every sequence: 31 at `640x360`, 23 at `768x432`, 23 at `960x540`, and 3 at `1920x1080`.

Exact file evidence:

- Published CDTB aggregate is marked complete and reports the expected native-reference metrics at `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/native_ope/completed_cdtb/metrics_cdtb.json:1-18`: precision `0.6993311256529843`, recall `0.6806774352120221`, F `0.6898782086902927`, threshold `0.580139`, 80 sequences, 101,956 frames.

- The published metric file records this as an unchanged native STTrack reference at `metrics_cdtb.json:3-8`, and records `new_trained_module: false` and `training_gate_promotion: false` at `metrics_cdtb.json:102-103`.

- The download binding is complete, names CDTB, and binds metric/receipt hashes at `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/native_ope/completed_cdtb/download_binding.json:1-8`. It closes with `frames: 101956`, `all_result_gt_hashes_verified: true`, `gt_and_images_private: true`, and `tracking_analysis_controller_exits_zero: true` at `download_binding.json:731-734`.

- The receipt is complete and binds the same spec/checkpoint at `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/native_ope/completed_cdtb/cdtb_receipt.json:1-5`. It closes with 101,956 frames, 1,614 template updates, `subsequent_gt_opened: false`, `labels_used_for_inference: false`, and `optimizer_steps: 0` at `cdtb_receipt.json:808-812`.

- The three terminal exit files are line-1 `0`: `tracking_cdtb.exit:1`, `analysis_cdtb.exit:1`, and `cdtb.controller.exit:1`. I also computed the same SHA-256 for each single-line exit file, `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`.

- The per-sequence JSON binds the metric hash, download-binding hash, exporter hash, metric-source hash, threshold `0.580139`, and threshold index `89` at `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/native_ope/completed_cdtb/per_sequence.json:1-9`.

- The per-sequence JSON starts at `per_sequence.json:10`, first row `XMG_outside` is at `per_sequence.json:12-18`, and the last row `two_tennis_balls` is at `per_sequence.json:723-729`. The scope note correctly states that rows use the dataset-wide maximum-F threshold and are not per-sequence maximum-F values at `per_sequence.json:732`.

- The CSV schema and first rows are at `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/native_ope/completed_cdtb/per_sequence.csv:1-5`; the tail rows are at `per_sequence.csv:77-81`. It has 80 data rows plus the header.

- The lowest per-sequence F row I recomputed is `trashcan_room_occ_1`: 1,305 frames, 979 visible, 1,211 selected, P/R/F `14.064172732888416 / 17.39705125590181 / 15.554075963039153%`. The exported JSON has that row at `per_sequence.json:642-648`, and the CSV has it at `per_sequence.csv:72`.

- The highest per-sequence F row I recomputed is `box_room_noocc_2`: 406 frames, 406 visible, 406 selected, P/R/F all `89.36713045781389%`. The exported JSON has that row at `per_sequence.json:219-225`, and the CSV has it at `per_sequence.csv:25`.

Metric/protocol evidence:

- The metric source defines the 100-point global threshold construction with `+inf` and `-inf` endpoints at `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/native_ope/source_snapshots/depthtrack_pr.py:25-37`.

- Bounded VOT rectangle overlap and invalid/absent GT handling are at `depthtrack_pr.py:40-60`.

- The evaluator computes per-sequence precision/recall curves, macro-averages sequence P/R, then selects the best global F operating point at `depthtrack_pr.py:117-139`; it returns precision, recall, F, threshold, sequence count, and frame count at `depthtrack_pr.py:140-153`.

- The per-sequence exporter follows the same sealed-bundle logic: it requires complete binding and verified hashes at `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/native_ope/export_per_sequence.py:22-30`, checks each bbox/confidence/GT/first-image hash at `export_per_sequence.py:35-44`, loads prediction/confidence/GT and validates row counts and confidence initialization at `export_per_sequence.py:45-49`, reads the actual first image dimensions and computes overlaps at `export_per_sequence.py:50-53`, computes the threshold grid and selected counts at `export_per_sequence.py:54-65`, selects the best global F point at `export_per_sequence.py:66-70`, asserts aggregate agreement at `export_per_sequence.py:71-75`, and writes JSON/CSV at `export_per_sequence.py:84-90`.

Input/no-training/no-GT-during-inference scope:

- The native reference spec binds the unchanged checkpoint at `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/native_ope/spec.json:4-8`, the Python runner hash at `spec.json:30`, the metric source/hash at `spec.json:38-39`, CDTB’s expected 80 sequences and 101,956 frames at `spec.json:48-51`, and `language: false`, `optimizer_steps: 0`, `learned_head: null`, `pr_resolution: 100`, and `output_decimals: 6` at `spec.json:54-58`.

- The tracker runner verifies source/checkpoint/input/metric/plan hashes before execution at `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/overlay/tools/run_sttrack_native_ope.py:18-25`.

- The tracking path uses `case['init_bbox']` for the first box and initializes the tracker from that box at `run_sttrack_native_ope.py:54-66`; subsequent frames call `tracker.track(image)` and save predicted boxes/scores at `run_sttrack_native_ope.py:67-85`. This supports first-box-only inference inputs in the inspected runner.

- The analysis path requires the tracking exit and sealed receipt before evaluation at `run_sttrack_native_ope.py:107-119`, and the code comment states that the unchanged evaluator opens subsequent GT only after output seals at `run_sttrack_native_ope.py:120`. It writes GT hashes and explicitly records `new_trained_module: false` and `training_gate_promotion: false` at `run_sttrack_native_ope.py:127-131`.

- The shell launcher invokes the Python runner for tracking and, only after exit code 0, invokes analysis at `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/native_ope/run_dataset.sh:6-14`.

Source naming clarification:

- The Python runner is `overlay/tools/run_sttrack_native_ope.py`, bound in the spec as `11748e8e005e9cf0706a1ff24aa59bf9830dcaf23160db5274fa702720a04ec4` at `spec.json:30`. I verified this hash only against the local published/Git file.

- The shell launcher is `diagnostics/native_ope/run_dataset.sh`, and its local hash is `9947aa8dec69267e3bbd0d51b20dcfad5a2bbb9e4e0cfe23a9b8b868d6bddbf7`. That is the value recorded as `runner_sha256` in `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/native_ope/execution_binding.json:5`. This is a naming ambiguity in the binding label, not an arithmetic or result-integrity defect, because `run_dataset.sh` is the executed launcher and it calls the Python runner at `run_dataset.sh:6` and `run_dataset.sh:10`.

A-F status:

- **A. Ground-truth provenance: PASS.** CDTB GT comes from the private raw dataset bundle and is hash-bound in `download_binding.json`; analysis occurs after sealed tracking output. Evidence: `download_binding.json:731-734`, `cdtb_receipt.json:808-812`, `run_sttrack_native_ope.py:107-131`.

- **B. Score normalization: PASS.** The metric uses overlap means/sums and visible-GT denominators, macro-averaged across sequences. I found no self-referential normalization by prediction maxima. Evidence: `depthtrack_pr.py:117-139`.

- **C. Result existence and numeric agreement: PASS.** Published aggregate and all 80 JSON/CSV per-sequence rows exist and match independent recomputation exactly. Evidence: `metrics_cdtb.json:9-18`, `per_sequence.json:1-9`, `per_sequence.csv:1-81`.

- **D. Execution/dead-code distinction: PASS.** The completed artifact has all terminal exits at 0, a complete receipt, and sealed outputs. Exporter code is wired to write both JSON and CSV from the sealed bundle. Evidence: `run_dataset.sh:6-14`, exit files line 1, `export_per_sequence.py:84-90`.

- **E. Scope: WARN.** This supports only the completed native STTrack CDTB full80 OPE reference over local/downloaded artifacts. It does not establish a live-server state, CDTB VOT protocol equivalence beyond this OPE metric, M54 learned-head performance, or any promoted learned module. Evidence: native-reference role and no learned head in `spec.json:4-8` and `spec.json:54-58`; metric file says `new_trained_module: false` at `metrics_cdtb.json:102-103`.

- **F. Evaluation type: real_gt.** This is dataset-GT evaluation with post-seal GT use and VOT bounded overlap arithmetic. Evidence: `depthtrack_pr.py:40-60`, `depthtrack_pr.py:117-153`, `download_binding.json:731-734`.

Claim status:

- **Supported:** “Completed native STTrack CDTB full80 OPE reference produced P/R/F = `69.93311256529843 / 68.0677435212022 / 68.98782086902926%` at threshold `0.580139` over 80 sequences and 101,956 frames.” This is supported by `metrics_cdtb.json:9-18` and my independent recomputation.

- **Supported:** “All80 per-sequence JSON/CSV rows are at the dataset-wide threshold index 89 / threshold 0.580139, not per-sequence-max F.” This is supported by `per_sequence.json:7-9`, `per_sequence.json:732`, and exact row-wise recomputation.

- **Supported with scope qualifier:** “The CDTB package is sealed against the local/private raw download bundle.” I verified local/downloaded bytes and hashes; I did not verify the live remote server.

- **Unsupported / contradicted by result:** “The native CDTB reference meets current targets P/R/F 72.9/75.6/74.2.” It misses by `-2.966887434701576`, `-7.532256478797791`, and `-5.212179130970739` percentage points respectively.

- **Unsupported:** “This is a new trained M54 result or learned-head promotion.” The spec and metrics explicitly record unchanged native reference, no learned head, zero optimizer steps, and no training-gate promotion.

Concrete defects found: **none** in CDTB aggregate arithmetic, all80 per-sequence export arithmetic, hash seals, row counts, selected/visible/frame counts, or CSV/JSON consistency.