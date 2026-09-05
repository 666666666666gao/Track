Scoped verdict: **PASS for M39 per-sequence export arithmetic and CSV/JSON consistency; WARN for interpretation/scope**. I found **no concrete defect** in the new export. The warning is that the per-sequence EAO values are **singleton fixed-interval diagnostics derived from exported VOT dependencies**, not fresh per-sequence overlap recomputations and not additive contributions to global EAO.

I used only local files under `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/m39/per_sequence_20260906` and the local original result `../m39_result.json`. I did not access the live server, rerun tracking, rerun the exporter, use GPU, or edit project/raw files.

Independent arithmetic results:

| Check | default | no_update | Status |
|---|---:|---:|---|
| Rows | 22 | 22 | PASS |
| CSV/JSON row mismatches | 0 | 0 | PASS |
| Dependency sequences | 22 | 22 | PASS |
| Anchors | 303 | 303 | PASS |
| Confirmed failures | 124 | 155 | PASS |
| Accuracy progress weight | 149,727 | 134,640 | PASS |
| Robustness frame weight | 13,944 | 13,944 | PASS |
| Singleton EAO mismatches from dependency curves | 0 | 0 | PASS |
| EAO active-bin mismatches | 0 | 0 | PASS |
| Failure/progress/frame mismatches vs original outcomes | 0 | 0 | PASS |
| Recombined EAO | 0.5713599285619029 | 0.5468557242611695 | PASS |
| Recombined ACC | 0.7571962197910564 | 0.7556417563380551 | PASS |
| Recombined ROB | 0.7302240116819271 | 0.6662821793720225 | PASS |

The frozen original M39 result hash was verified locally as `cf953c0d3c69609bcd83c11cb24ba57f37e30b38d3b3bcad32860b3a9ba9c1b5`. The exporter hard-codes and asserts that hash at `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/m39/per_sequence_20260906/export.py:18` and `export.py:30-34`; the exported JSON also records it at `per_sequence.json:4`.

The VOT dependency source check passed. The exported JSON records toolkit `0.7.1`, multistart source SHA `5a09065e2315387405f4cb8f96c0b8fd32d7428996a34eedd92ac4e2a4deeb02`, and interval `[115, 755]` at `per_sequence.json:6-10`. I independently located the installed VOT 0.7.1 `vot/analysis/multistart.py` source in the uv environment and computed the same SHA. The relevant VOT source initializes unsupported EAO bins to zero and uses weighted curve aggregation at `multistart.py:394-402`, then computes EAO as `mean(x[0][self.low:self.high + 1])` at `multistart.py:442`. With dependency arrays of length 755, the toolkit’s `115:756` slice covers 640 actual bins, matching the exported maximum active-bin count of 640.

The copied analysis/config evidence matches the fixed interval and analysis types. Both arm analysis files use `EAOScore` low `115`, high `755`, bounded `true`, threshold `0.1`, and `EAOCurve` high `755` at `default_analysis.json:141-161` and `no_update_analysis.json:141-161`. The AR analysis is `AverageAccuracyRobustness` at `default_analysis.json:163-170` and `no_update_analysis.json:163-170`. The copied analysis files carry the same final AR weights/values as the recomputation: default at `default_analysis.json:944-953`, no_update at `no_update_analysis.json:944-953`.

The copied merge manifests are present and internally complete: default has `anchor_count: 303`, `result_file_count: 909`, and status complete at `default_merge.json:2-4` and `default_merge.json:919`; no_update has the same counts/status at `no_update_merge.json:2-4` and `no_update_merge.json:919`. The exporter code verifies both sealed result sets before loading the dataset/workspace for analysis, including analysis hash, merge hash, merge complete status, 303 anchors, 909 result hashes, and every workspace result-file hash at `export.py:38-57`. Since raw VOT prediction/GT files are not included locally, I cannot independently rehash those 909 workspace result files; this is exporter-code plus `export.exit:1` evidence, not a fresh raw-file seal audit.

The exported aggregate rows match the original M39 global metrics. Original default metrics are `acc/eao/rob = 0.7571962197910564 / 0.5713599285619029 / 0.7302240116819271` at `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/m39/m39_result.json:2743-2751`; exported default recomputation records the same values and weights at `per_sequence.json:21-28`. Original no_update metrics are `acc/eao/rob = 0.7556417563380551 / 0.5468557242611695 / 0.6662821793720225` at `m39_result.json:5585-5593`; exported no_update recomputation records the same values and weights at `per_sequence.json:325-332`.

The CSV and JSON contain the same 44 rows. The CSV schema and first rows are at `per_sequence.csv:1-8`; its final rows are at `per_sequence.csv:39-45`. The JSON default arm begins at `per_sequence.json:14-29`, no_update begins at `per_sequence.json:318-333`, and the no_update row list closes at `per_sequence.json:620-623`. My row-by-row comparison found 0 mismatches for arm, sequence, frames, anchors, confirmed failures, EAO/ACC/ROB percent, progress weight, robustness weight, and EAO active-bin count.

The cube05 caveat is real and should be reflected in labels. Default `cube05_indoor_2` has 4 anchors, 0 confirmed failures, `eao_percent = 4.339054267799119`, `rob_percent = 100.0`, and only 31 active EAO bins in the fixed score slice at `per_sequence.json:225-236` and `per_sequence.csv:17`. No_update `cube05_indoor_2` similarly has `eao_percent = 4.344879500415875`, `rob_percent = 100.0`, and 31 active bins at `per_sequence.json:529-540` and `per_sequence.csv:39`. The dependency files contain that sequence’s `partial_curve`, `active_weight`, and `sequence_weight` entries at `default_eao_dependencies.json:22757-24272` and `no_update_eao_dependencies.json:22757-24272`. This confirms per-sequence EAO here is a singleton fixed-interval curve diagnostic; it should not be described as a failure rate, a direct robustness proxy, or an additive contribution to global EAO.

A-F status:

- **A. Ground-truth provenance: WARN.** Original M39 VOT analyses are real-GT based, but this audit did not have raw VOT predictions/GT locally. The per-sequence export is dependency-derived from copied VOT analysis/workspace outputs. Evidence: export scope at `per_sequence.json:12`; exporter pre-analysis seal checks at `export.py:38-57`.

- **B. Score normalization: PASS.** EAO is computed from VOT dependency curves over the fixed low/high slice, with unsupported bins zero; ACC/ROB recombination uses VOT AR weights. Evidence: `multistart.py:394-402`, `multistart.py:442`, and AR weighted aggregation at `multistart.py:175-188`.

- **C. Result existence/number matching: PASS.** `per_sequence.json`, `per_sequence.csv`, arm analyses, arm merge manifests, dependency files, log, and exit exist. Export exit is 0. All aggregate and per-row numbers matched independent dependency arithmetic.

- **D. Dead code/execution: PASS.** Exporter writes dependencies plus JSON/CSV at `export.py:129-155`; `export.log:70` reports default completion and `export.log:140` reports no_update completion. `export.exit:1` is `0`.

- **E. Scope: WARN.** This is a 22-sequence-per-arm, 44-row diagnostic over sealed M39 low22 outputs. It is not a new experiment, not a full127 promotion, not fresh tracking, and not independent raw-overlap recomputation. The exported scope says “No tracking, training, checkpoint change, or benchmark promotion” at `per_sequence.json:12`.

- **F. Evaluation type: real_gt-derived diagnostic.** The original VOT analyses are real-GT benchmark analyses, but this audit validates exported dependencies and aggregate consistency rather than recomputing from raw GT/trajectories.

Claim status:

- **Supported:** “The new M39 per-sequence export has 44 rows: 22 default and 22 no_update, and all CSV/JSON rows are internally consistent.”
- **Supported:** “Singleton EAO values can be derived exactly from exported partial_curve/active_weight/sequence_weight using the VOT 0.7.1 fixed interval.”
- **Supported:** “All22 recombinations reproduce the original global EAO/ACC/ROB for both arms exactly.”
- **Supported with qualifier:** “ROB/progress/failure counts match the original sealed anchor outcomes.” This is supported from the frozen `m39_result.json` outcomes and exported row weights, but not from raw local trajectory/GT files.
- **Needs mandatory label:** “Per-sequence EAO” should be labelled **singleton fixed-interval EAO diagnostic**.
- **Unsupported:** Any claim that this export is a fresh raw-overlap/GT recomputation, live-server verification, new tracking run, training result, full127 result, or benchmark promotion.

Concrete defects found: **none**.