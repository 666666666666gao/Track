Scoped verdict: **PASS for the completed DepthTrack full50 per-sequence export**. I found **0 JSON-vs-independent mismatches**, **0 CSV-vs-independent mismatches**, and **0 CSV-vs-JSON mismatches** across all 50 sequences. This is still a same-family GPT Type-A advisory review, not a cross-family acquittal.

I recomputed the per-sequence arrays from the private raw bundle under `C:/Users/gb/.codex_remote_staging/native_ope_depthtrack_completed_20260906`, using the raw boxes, confidences, GT, and first-image dimensions with VOT bounded overlaps. I did not rerun tracking, contact the server, use GPU, edit files, or call the executor’s aggregate evaluator as the arithmetic source.

Evidence and checks:

- The exporter is correctly scoped to sealed downloaded evidence. It reads `download_binding.json`, requires `status == complete`, `all_result_gt_hashes_verified`, and zero tracking/analysis/controller exits at `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/native_ope/export_per_sequence.py:22-27`. It checks the metric source hash at `export_per_sequence.py:29-33`, then for every bound sequence checks bbox, confidence, GT, and first-image hashes before loading data at `export_per_sequence.py:35-49`.

- The exporter computes the same dataset-wide threshold grid and selected index used by the public metric: it builds thresholds from all confidences at `export_per_sequence.py:54-56`, computes per-sequence precision/recall/selected counts over the threshold grid at `export_per_sequence.py:57-65`, macro-averages P/R and selects `argmax(f_curve)` at `export_per_sequence.py:66-70`, then asserts agreement with the published aggregate metrics at `export_per_sequence.py:71-75`.

- The exported JSON declares the expected global threshold and index: `threshold = 0.387226`, `threshold_index = 96`, and `global_metrics_match = true` at `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/native_ope/completed_depthtrack/per_sequence.json:7-9`.

- The exported JSON sequence list begins at `per_sequence.json:10`, first row `adapter01_indoor` is at `per_sequence.json:12-18`, last row `yogurt_indoor` is at `per_sequence.json:453-459`, and the sequence list closes at `per_sequence.json:461`. The scope string correctly says these are all sequences at the dataset-wide maximum-F threshold and that sequence F is not individually maximized at `per_sequence.json:462`.

- The exported CSV has the matching schema at `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/native_ope/completed_depthtrack/per_sequence.csv:1`, 50 data rows at `per_sequence.csv:2-51`, with the same first row at `per_sequence.csv:2` and same last row at `per_sequence.csv:51`.

Independent recomputation results:

| Check | Independent result | Export comparison |
|---|---:|---|
| Sequence rows | 50 | JSON 50, CSV 50 |
| Frame sum | 76,373 | matches published aggregate |
| Visible GT frame sum | 73,389 | matches recomputed per-sequence visible counts |
| Selected frame sum at threshold 0.387226 | 74,052 | matches exported selected counts |
| Threshold grid size | 100 | matches protocol |
| Finite thresholds | 98 | matches protocol endpoints with `inf` and `-inf` |
| Best threshold index | 96 | matches JSON |
| Best threshold | 0.387226 | matches JSON and aggregate metrics |
| JSON row mismatches | 0 | PASS |
| CSV row mismatches | 0 | PASS |
| CSV/JSON mismatches | 0 | PASS |

The global aggregate at index 96 also matches the published full50 metrics: precision `62.41533634819073%`, recall `62.682003798172694%`, F `62.54838584839805%`. The published aggregate file carries the same values at `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/native_ope/completed_depthtrack/metrics_depthtrack.json:9-18`.

The lowest and highest per-sequence F values from my recomputation were:

| Sequence | Frames | Visible | Selected | P% | R% | F% |
|---|---:|---:|---:|---:|---:|---:|
| `cube05_indoor` | 2,975 | 2,856 | 2,974 | 2.2232327958838582 | 2.3150890528566506 | 2.2682313327473738 |
| `lock02_indoor` | 521 | 521 | 521 | 88.738925935566 | 88.738925935566 | 88.738925935566 |

These match the exported rows: `cube05_indoor` is in `per_sequence.csv:22` and `lock02_indoor` is in `per_sequence.csv:36`; the corresponding JSON entries are at `per_sequence.json:192-198` and `per_sequence.json:318-324`.

Provenance naming clarification:

- The Python runner hash for `overlay/tools/run_sttrack_native_ope.py` is `11748e8e005e9cf0706a1ff24aa59bf9830dcaf23160db5274fa702720a04ec4`. The spec binds that exact hash under `tools/run_sttrack_native_ope.py` at `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/native_ope/spec.json:30`. I also verified the live/local/Git file hash directly.

- The hash `9947aa8dec69267e3bbd0d51b20dcfad5a2bbb9e4e0cfe23a9b8b868d6bddbf7` in `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/native_ope/execution_binding.json:5` is the shell launcher `diagnostics/native_ope/run_dataset.sh`, not the Python runner. I verified the local hash directly. The launch prose also calls this value “运行入口” at `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/native_ope/LAUNCH_REPORT.md:25-28`, which is understandable historically but ambiguous unless labelled as the shell launcher.

- `run_dataset.sh` invokes the Python runner for tracking at `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/native_ope/run_dataset.sh:6` and analysis at `run_dataset.sh:10`, so preserving the original `execution_binding.runner_sha256` while explicitly labelling it as `run_dataset.sh` is consistent with the evidence. No metric result changes are implied by this naming clarification.

Updated A-F status for this follow-up:

- **A. Ground-truth provenance: PASS.** The per-sequence export uses the completed download binding and checks GT hashes before reading raw GT; see `export_per_sequence.py:22-49`. This remains real dataset GT, with private GT/images marked in the completed binding at `completed_depthtrack/download_binding.json:461-464`.

- **B. Score normalization: PASS.** Per-sequence precision is mean overlap over selected frames, recall is selected-overlap sum divided by visible GT frames, and F is the harmonic mean of those P/R values; see `export_per_sequence.py:60-79`. I found no self-normalization against prediction maxima.

- **C. Result existence and number matching: PASS.** `per_sequence.json` and `per_sequence.csv` exist, contain 50 rows, and match my independent recomputation exactly. Published aggregate metrics at `metrics_depthtrack.json:9-18` also match the recomputed index-96 aggregate.

- **D. Dead code / execution wiring: PASS for the export artifact.** The exporter writes both JSON and CSV at `export_per_sequence.py:84-90`, and the written source hash in `per_sequence.json:5` matches the current exporter file hash I computed.

- **E. Scope: WARN, unchanged.** This export covers DepthTrack Test full50 only, at one dataset-wide threshold. It does not establish CDTB, VOT/full127, M54 learned-head performance, or a promoted learned module. The export itself states the per-sequence values are not per-sequence maximum-F values at `per_sequence.json:462`.

- **F. Evaluation type: real_gt.** The per-sequence rows use dataset GT and VOT bounded overlaps. The metric source defines the 100-point threshold grid at `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/native_ope/source_snapshots/depthtrack_pr.py:25-37`, bounded overlaps and invalid-GT handling at `depthtrack_pr.py:40-60`, and macro P/R with best global F selection at `depthtrack_pr.py:117-139`.

Concrete defects found: **none in the per-sequence export arithmetic or CSV/JSON serialization**.

Remaining limits: this follow-up verifies only the exported all50 DepthTrack per-sequence table and the provenance-name distinction. It does not rerun inference, verify new CDTB/VOT results, promote M54, or change the earlier conclusion that the native DepthTrack full50 reference fails the supplied target P/R/F of 65.2/64.9/65.1.