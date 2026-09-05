# Independent prepared native OPE review

Same-family GPT Type-A advisory review, not cross-family acquittal.
The full response below is preserved verbatim. Only preparation and code
were reviewed; no completed contract or benchmark result existed.

---

Read-only native STTrack OPE reference review completed. I did not edit files, launch inference, contact the server, run GPU jobs, or open experiment GT. This is a same-family GPT Type-A advisory review only.

Overall verdict: **WARN**. I found **no concrete implementation defect** in the prepared native OPE runner, manifest, spec, preparation record, or PR evaluator snapshot. The warning is evidence-scope only: no OPE inference, contract receipt, tracking exits, receipts, or metric results exist yet, so no performance number can be audited or claimed.

### A. Ground-truth provenance — PASS for implementation, unverified for execution

The prepared interface uses first-frame initialization boxes only for inference. The preparation record says GT values used are only `first_frame_initialization_box`, `subsequent_gt_values_used` is false, GT row counts were checked, and RGB/depth pairs were checked at `diagnostics/native_ope/preparation.json:21-29`. The input manifest entries contain only `sequence`, `root`, `frames`, and `init_bbox`; examples are `diagnostics/native_ope/inputs.json:2-13` and the Train contract entries at `diagnostics/native_ope/inputs.json:1435-1459`.

The runner’s tracking path reads frames and initializes from `init_bbox`, then tracks without loading GT at `overlay/tools/run_sttrack_native_ope.py:54-85`. It records `subsequent_gt_opened=False`, `labels_used_for_inference=False`, and `optimizer_steps=0` in receipts at `overlay/tools/run_sttrack_native_ope.py:98-103`.

Post-seal GT evaluation is correctly ordered in code. `analyze()` first requires the tracking exit, receipt status, checkpoint/spec match, sequence order, frame coverage, no inference GT use, and per-sequence output hashes at `overlay/tools/run_sttrack_native_ope.py:107-120`. Only then does it import and run the unchanged evaluator, which opens `groundtruth.txt` internally at `overlay/tools/run_sttrack_native_ope.py:120-126` and `diagnostics/native_ope/source_snapshots/depthtrack_pr.py:93-111`.

### B. Score normalization — PASS

No self-normalized score is present in the prepared implementation. The evaluator builds a global confidence threshold grid from finite tracker scores at `diagnostics/native_ope/source_snapshots/depthtrack_pr.py:25-37` and computes bounded VOT rectangle overlaps at `diagnostics/native_ope/source_snapshots/depthtrack_pr.py:40-60`. It computes per-sequence precision/recall curves, macro-averages them, and returns the maximum F-score at `diagnostics/native_ope/source_snapshots/depthtrack_pr.py:117-154`. This matches the plan’s requirement to preserve the existing macro PR/F-score evaluator without calibration or post-result threshold changes at `diagnostics/native_ope/EXPERIMENT_PLAN.md:29-35`.

### C. Result existence — WARN

This entry is prepared but not executed. The preparation file explicitly says `inference_launched: false` at `diagnostics/native_ope/preparation.json:29`. I also found no `contract_receipt.json`, `tracking_depthtrack.exit`, `tracking_cdtb.exit`, `depthtrack_receipt.json`, `cdtb_receipt.json`, `metrics_depthtrack.json`, or `metrics_cdtb.json` in `diagnostics/native_ope/`.

This is not an implementation defect. It means no native OPE performance claim exists yet.

### D. Execution wiring / dead-code check — PASS for prepared code

The runner checks native bundle identity before each mode: all bound source hashes, checkpoint hash, input manifest hash, metric-source hash, and plan hash at `overlay/tools/run_sttrack_native_ope.py:18-25`. The spec binds the unchanged checkpoint, YAML, native runner, tracker/model/helper sources, input manifest, and metric source at `diagnostics/native_ope/spec.json:4-39`.

The runtime setup uses the native `STTrack` class, the bound YAML, and the bound base checkpoint at `overlay/tools/run_sttrack_native_ope.py:28-38`. It asserts two templates, update interval 50, and update threshold 0.75 at `overlay/tools/run_sttrack_native_ope.py:39`, matching the frozen plan at `diagnostics/native_ope/EXPERIMENT_PLAN.md:15-20`.

The Train-only interface contract is wired separately. Contract mode uses `inputs['contract']` at `overlay/tools/run_sttrack_native_ope.py:137-147`; those entries are `chair01_indoor` and `cube04_indoor` under `/root/autodl-tmp/depthtrack/train/sequences` at `diagnostics/native_ope/inputs.json:1435-1459`. During contract tracking, the runner checks native expected bbox/score tolerances against the sealed native inputs at `overlay/tools/run_sttrack_native_ope.py:40-44` and `overlay/tools/run_sttrack_native_ope.py:72-78`.

### E. Scope and claim strength — WARN

The intended full OPE scope is complete and fixed in the manifest/spec, but not yet executed. The spec declares DepthTrack Test as 50 sequences / 76,373 frames and CDTB as 80 sequences / 101,956 frames at `diagnostics/native_ope/spec.json:42-52`. The preparation record repeats those counts at `diagnostics/native_ope/preparation.json:5-15`. The runner asserts the case count and total frames against the spec before tracking or analysis at `overlay/tools/run_sttrack_native_ope.py:149-156`.

The claim must remain “prepared unchanged native reference.” The plan states this evaluation does not promote any failed association head or claim a new trained model at `diagnostics/native_ope/EXPERIMENT_PLAN.md:5-8`. The spec records `role: Unchanged native reference`, `language: false`, `optimizer_steps: 0`, and `learned_head: null` at `diagnostics/native_ope/spec.json:4` and `diagnostics/native_ope/spec.json:54-58`. The future metrics file would also mark `new_trained_module=False` and `training_gate_promotion=False` at `overlay/tools/run_sttrack_native_ope.py:127-131`.

### F. Evaluation classification — WARN because no result exists yet

- Prepared OPE tracking: **not an evaluation result yet**. It is an inference plan/runner with no completed outputs.
- Train-only contract: **self-supervised/interface contract**, using sealed native predictions and serialization checks, not test performance. Evidence: `diagnostics/native_ope/EXPERIMENT_PLAN.md:37-42`, `overlay/tools/run_sttrack_native_ope.py:72-89`.
- Planned DepthTrack/CDTB OPE metrics: **real_gt**, because the evaluator opens dataset `groundtruth.txt` only after sealed output checks. Evidence: `overlay/tools/run_sttrack_native_ope.py:107-126`, `diagnostics/native_ope/source_snapshots/depthtrack_pr.py:93-111`.
- Learned-head promotion: **not applicable / unsupported**. Evidence: `diagnostics/native_ope/spec.json:54-58`, `overlay/tools/run_sttrack_native_ope.py:127-131`.

### Specific requested checks

Native confidence and six-decimal serialization are implemented. The runner stores initialization confidence `1.0` at `overlay/tools/run_sttrack_native_ope.py:56-57`, appends native `best_score` for later frames at `overlay/tools/run_sttrack_native_ope.py:68-70`, writes boxes and scores with `%.6f` at `overlay/tools/run_sttrack_native_ope.py:82-85`, reloads them, and asserts six-decimal round-trip error plus first score exactly `1.0` at `overlay/tools/run_sttrack_native_ope.py:86-89`.

The runner keeps unrounded tracker state during inference. It tracks through all frames before writing serialized output, and only serializes after the sequence loop completes at `overlay/tools/run_sttrack_native_ope.py:60-85`; no rounded saved value is fed back into the tracker.

Reuse of the existing macro PR evaluator is implemented. The spec points to `/home/SRTrack_RGBD_L/lib/test/analysis/depthtrack_pr.py` and binds its SHA at `diagnostics/native_ope/spec.json:38-39`; the runner loads that exact source by path and calls `evaluate_depthtrack_results()` at `overlay/tools/run_sttrack_native_ope.py:120-126`. The snapshot shows the per-sequence curve/macro-average/max-F implementation at `diagnostics/native_ope/source_snapshots/depthtrack_pr.py:63-154`.

Concrete defects found: **none**.

Remaining unverified: contract execution/receipt, DepthTrack tracking exit/receipt, CDTB tracking exit/receipt, output file hashes, post-seal GT metric execution, `metrics_depthtrack.json`, `metrics_cdtb.json`, final PR/F-score values, and consistency with the separate existing full127 VOT run.