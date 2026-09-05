Overall verdict: **PASS for the completed native STTrack DepthTrack Test full50 result integrity; WARN for scope and target outcome**. The artifacts are complete for **DepthTrack Test full50 only**, the raw private bundle supports the published metrics exactly, and I found no concrete arithmetic, provenance, hash, row-count, serialization, or denominator defect. The result is a negative/insufficient native-reference number relative to the supplied target P/R/F 65.2/64.9/65.1, and it does **not** establish CDTB, VOT/full127, M54 training, or full-goal completion.

This is a same-family GPT Type-A advisory review, not a cross-family acquittal. I did not edit files, contact the server/network, use GPU, or rerun inference. I recomputed metrics from the private raw bundle with a standalone CPU script using `vot.region.calculate_overlaps`; I did **not** invoke `evaluate_depthtrack_results`.

### A. Ground-truth provenance: PASS

The inference runner uses only the first-frame initialization box during tracking. It initializes `boxes = [init_bbox]` and `scores = [1.]` at `overlay/tools/run_sttrack_native_ope.py:55-57`, initializes the tracker on frame 0 at `overlay/tools/run_sttrack_native_ope.py:60-66`, and only then tracks subsequent frames from images at `overlay/tools/run_sttrack_native_ope.py:67-71`. The preparation file explicitly says the only GT-derived value used before inference is the first-frame initialization box, with no subsequent GT use: `diagnostics/native_ope/preparation.json:23-29`.

The receipt records no subsequent GT opening, no labels used for inference, and no optimizer steps: `diagnostics/native_ope/completed_depthtrack/depthtrack_receipt.json:508-514`. Analysis verifies sealed tracking outputs before loading/evaluating GT: `overlay/tools/run_sttrack_native_ope.py:107-120` checks tracking exit, receipt status, spec/checkpoint binding, sequence order, frame count, inference flags, and result-file hashes; GT hashes are only recorded in the analysis result at `overlay/tools/run_sttrack_native_ope.py:127-131`.

The metric GT is dataset GT, not prediction-derived. The evaluator loads `groundtruth.txt`, prediction boxes, and confidence files at `diagnostics/native_ope/source_snapshots/depthtrack_pr.py:93-103`. Invalid/absent GT is converted to VOT `Special(0)` at `diagnostics/native_ope/source_snapshots/depthtrack_pr.py:49-55`, with visibility tracked separately at `diagnostics/native_ope/source_snapshots/depthtrack_pr.py:55-60`.

### B. Score normalization and denominators: PASS

No published metric is normalized by the tracker’s own maximum/minimum/mean score. The confidence values are used to define the VOT long-term threshold grid, not as a denominator. The threshold rule sorts finite scores and builds a 100-point grid with `inf`, 98 sampled finite thresholds, and `-inf`: `diagnostics/native_ope/source_snapshots/depthtrack_pr.py:25-37`; the aggregate uses that grid at `diagnostics/native_ope/source_snapshots/depthtrack_pr.py:117-151`.

Precision/recall are computed per sequence and then macro-averaged. For each threshold, precision is the mean bounded overlap among selected predictions, and recall is overlap sum divided by visible GT frames for that sequence: `diagnostics/native_ope/source_snapshots/depthtrack_pr.py:120-131`. The macro precision and recall curves are then averaged across sequences at `diagnostics/native_ope/source_snapshots/depthtrack_pr.py:133-134`, with F-score computed from the macro curves at `diagnostics/native_ope/source_snapshots/depthtrack_pr.py:135-142`.

My independent recomputation used the same denominator semantics and found:

| Field | Recomputed | Published | Difference |
|---|---:|---:|---:|
| precision | 0.6241533634819073 | 0.6241533634819073 | 0 |
| recall | 0.6268200379817269 | 0.6268200379817269 | 0 |
| f_score | 0.6254838584839805 | 0.6254838584839805 | 0 |
| precision_percent | 62.41533634819073 | 62.41533634819073 | 0 |
| recall_percent | 62.682003798172694 | 62.682003798172694 | 0 |
| f_score_percent | 62.54838584839805 | 62.54838584839805 | 0 |
| threshold | 0.387226 | 0.387226 | 0 |
| sequences | 50 | 50 | 0 |
| frames | 76373 | 76373 | 0 |

The published values are at `diagnostics/native_ope/completed_depthtrack/metrics_depthtrack.json:9-19` and in the analysis log at `diagnostics/native_ope/completed_depthtrack/analysis_depthtrack.log:1-10`.

Additional recomputation details:

- Threshold grid length: 100.
- Finite thresholds: 98.
- First finite threshold: 0.957008.
- Last finite threshold: 0.232105.
- Best grid index: 96.
- Best threshold: 0.387226.
- Total rows/frames evaluated: 76,373.
- Total visible GT frames across the 50 sequences: 73,389.
- All 50 first images were readable and had actual dimensions 640×360, which were used as VOT bounded-overlap image bounds.

### C. Result existence, hashes, and row counts: PASS for DepthTrack full50

The completed published DepthTrack artifacts exist and the three completion exits are zero:

- `diagnostics/native_ope/completed_depthtrack/tracking_depthtrack.exit:1` is `0`.
- `diagnostics/native_ope/completed_depthtrack/analysis_depthtrack.exit:1` is `0`.
- `diagnostics/native_ope/completed_depthtrack/depthtrack.controller.exit:1` is `0`.

The receipt is complete for the DepthTrack dataset and binds the spec/checkpoint hashes at `diagnostics/native_ope/completed_depthtrack/depthtrack_receipt.json:1-5`. It contains 50 per-sequence entries from `diagnostics/native_ope/completed_depthtrack/depthtrack_receipt.json:6` through `diagnostics/native_ope/completed_depthtrack/depthtrack_receipt.json:507`, with the final aggregate reporting 76,373 frames, 817 template updates, no subsequent GT opened, no labels used for inference, and 0 optimizer steps at `diagnostics/native_ope/completed_depthtrack/depthtrack_receipt.json:508-514`.

The download binding records the completed dataset, spec/checkpoint/metrics/receipt hashes at `diagnostics/native_ope/completed_depthtrack/download_binding.json:1-8`, per-sequence bbox/confidence/GT/first-image hashes beginning at `diagnostics/native_ope/completed_depthtrack/download_binding.json:9-18`, and final aggregate flags of 76,373 frames, result/GT hashes verified, GT/images private, and tracking/analysis/controller exits zero at `diagnostics/native_ope/completed_depthtrack/download_binding.json:460-465`.

I independently verified from `C:/Users/gb/.codex_remote_staging/native_ope_depthtrack_completed_20260906`:

- Raw top-level `depthtrack_receipt.json`, `metrics_depthtrack.json`, and `download_binding.json` match the published copies byte-for-byte by SHA-256.
- Raw `spec.json` SHA-256 is `e623ef63de89da423fc12b877f58d56219f88ef9f1232f20de9e42fb6c8665e1`, matching the published native OPE spec.
- Raw `inputs.json` SHA-256 is `61541e35f7b9e3c40427df79067fc0be20b8622cf275e93025e4a1547bf68601`, matching `diagnostics/native_ope/spec.json:36`.
- Raw result file count is 100: 50 bbox files plus 50 `_all_scores.txt` files.
- Raw dataset sequence directory count is 50.
- All 50 bbox hashes matched the receipt/download binding.
- All 50 confidence hashes matched the receipt/download binding.
- All 50 GT hashes matched the download binding and `metrics_depthtrack.json:20-70`.
- All 50 first-image hashes matched the download binding.
- All per-sequence bbox, confidence, and GT row counts matched the corresponding input/receipt/download frame counts.
- Total frame count was exactly 76,373.
- Initial confidence was exactly `1.000000` / numeric 1.0 for all 50 sequences.
- Initial prediction box matched the input initialization box within six-decimal serialization tolerance for all 50 sequences.
- Initial GT row matched the input initialization box exactly for all 50 sequences.
- No non-finite predictions/confidences and no negative prediction dimensions were found.

### D. Dead code and execution wiring: PASS

The protocol source shows the native runner writes exactly the result files that the analyzer later verifies. Tracking writes per-sequence bbox and confidence files at `overlay/tools/run_sttrack_native_ope.py:82-85`, reloads them to verify six-decimal round trip and first confidence `1.0` at `overlay/tools/run_sttrack_native_ope.py:86-89`, and records each file hash in the receipt at `overlay/tools/run_sttrack_native_ope.py:90-103`.

The analyzer then reads that receipt, checks sequence order, frame totals, inference flags, bbox hashes, and confidence hashes before invoking the metric code: `overlay/tools/run_sttrack_native_ope.py:107-125`. The metric source path and hash are bound in the spec at `diagnostics/native_ope/spec.json:38-39`. The current runner hash also matches the spec binding: `diagnostics/native_ope/spec.json:30` declares `tools/run_sttrack_native_ope.py` hash `11748e8e005e9cf0706a1ff24aa59bf9830dcaf23160db5274fa702720a04ec4`, which I recomputed from the local runner. The evaluator snapshot hash recomputed to `05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc`, matching `diagnostics/native_ope/spec.json:39`.

I found no dead metric function relevant to this completed DepthTrack result. `_determine_thresholds`, `_vot_overlaps`, and the per-sequence macro PR loop are all part of the evaluator path at `diagnostics/native_ope/source_snapshots/depthtrack_pr.py:25-37`, `diagnostics/native_ope/source_snapshots/depthtrack_pr.py:40-60`, and `diagnostics/native_ope/source_snapshots/depthtrack_pr.py:117-154`.

### E. Scope and claim strength: WARN

The completed evidence supports exactly one result family: **unchanged native STTrack reference on DepthTrack Test full50**. The spec declares the native reference role at `diagnostics/native_ope/spec.json:2-8`, the DepthTrack test scope as 50 sequences / 76,373 frames at `diagnostics/native_ope/spec.json:42-47`, no language, 0 optimizer steps, no learned head, 100 PR resolution, and six-decimal output serialization at `diagnostics/native_ope/spec.json:54-58`.

The metrics file correctly labels the role as “Unchanged native STTrack reference” at `diagnostics/native_ope/completed_depthtrack/metrics_depthtrack.json:2-8`, and explicitly records `new_trained_module: false` and `training_gate_promotion: false` at `diagnostics/native_ope/completed_depthtrack/metrics_depthtrack.json:72-73`.

This result **fails** the supplied target P/R/F 65.2/64.9/65.1:

| Metric | Verified percent | Target percent | Gap |
|---|---:|---:|---:|
| Precision | 62.41533634819073 | 65.2 | -2.78466365180927 pp |
| Recall | 62.682003798172694 | 64.9 | -2.217996201827307 pp |
| F-score | 62.54838584839805 | 65.1 | -2.551614151601947 pp |

This is not a defect in the computation; it is the verified outcome. It must not be promoted as a successful learned-head result or as an M54 result.

Unsupported by this artifact set:

- CDTB completion.
- VOT/full127 completion.
- Any M54 trained-reader result.
- Any recursive M54 development improvement.
- Any full-goal completion.
- Any claim that the native reference passes target P/R/F 65.2/64.9/65.1.

### F. Evaluation type classification: PASS

Classification: **real_gt**.

This completed DepthTrack evaluation uses dataset-provided `groundtruth.txt` and VOT bounded rectangle overlaps. The evaluator loads GT from each sequence at `diagnostics/native_ope/source_snapshots/depthtrack_pr.py:93-103`, reads image dimensions from the first color image at `diagnostics/native_ope/source_snapshots/depthtrack_pr.py:106-111`, computes bounded VOT overlaps at `diagnostics/native_ope/source_snapshots/depthtrack_pr.py:40-60`, and macro-averages per-sequence PR curves at `diagnostics/native_ope/source_snapshots/depthtrack_pr.py:120-154`.

It is **not** synthetic_proxy, self_supervised_proxy, simulation_only, or human_eval. It is also **not newly trained M54**: the metric artifact says unchanged native reference at `diagnostics/native_ope/completed_depthtrack/metrics_depthtrack.json:4`, and records no trained module / no training-gate promotion at `diagnostics/native_ope/completed_depthtrack/metrics_depthtrack.json:72-73`.

### Concrete issues found

None in the completed DepthTrack full50 result integrity.

The only limitations are scope limits:

- The result is DepthTrack Test full50 only.
- The raw GT/images are private, not published; I verified them locally from the private raw bundle.
- CDTB, VOT/full127, and M54 trained-reader results remain unverified here.
- The verified native reference misses the stated target P/R/F thresholds and should be treated as a failed/insufficient baseline result, not as a successful advancement.