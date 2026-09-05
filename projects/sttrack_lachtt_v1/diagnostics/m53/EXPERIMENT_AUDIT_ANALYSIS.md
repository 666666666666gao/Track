# Independent M53 analyzer implementation review

Same-family GPT Type-A advisory review, not cross-family acquittal.
The complete response below is preserved verbatim. The reviewer did not run
experiment GT analysis. Their bare-Python synthetic-check attempt failed at
the local interpreter shim; the separate server checker passed, with evidence
in `analysis_contract.json` and `analysis_contract.exit`. No full capacity
result is claimed here.

---

Read-only M53 analyzer implementation review completed. I did not edit files, read experiment GT, launch GPU jobs, or contact the server. I attempted the synthetic checker only; it failed immediately in the local shell with `No pyvenv.cfg file` from the broken `python.exe` shim before analyzer code executed.

Overall verdict: **WARN**. I found **no concrete analyzer implementation defect** against the frozen M53 plan/spec/collector, but the full collection is still running and no `capacity_result.json`/capacity metrics are available to audit. This is a same-family GPT Type-A advisory review only.

### A. Ground-truth provenance — PASS for implementation; result unverified

The analyzer enforces output sealing before GT. It first requires `collection.exit` and `controller.exit` to be `0`, checks spec/source/execution bindings, source hashes, collection receipt status, receipt SHA/size for every event file, replay exactness, event ordering, template-write ordering, past-shadow counts, and baseline-vs-sealed-native equality at `overlay/tools/analyze_sttrack_m53.py:81-150`. Only after that does it mark “GT is first opened below” and load `groundtruth.txt` at `overlay/tools/analyze_sttrack_m53.py:151-160`.

This matches the plan: seal outputs and source hashes before GT at `diagnostics/m53/EXPERIMENT_PLAN.md:45-51`; collection itself reads no later GT/labels at `diagnostics/m53/EXPERIMENT_PLAN.md:22-27`. The collector also records `labels_opened=False` and `optimizer_steps=0` in its receipt output at `overlay/tools/collect_sttrack_m53.py:173-180`.

Limit: the actual post-seal GT analysis result is not present, so only the implementation order is verified.

### B. Score normalization and denominators — PASS for implementation

The analyzer uses direct IoU, not self-normalization. `overlap()` computes intersection over union from candidate box and GT box at `overlay/tools/analyze_sttrack_m53.py:16-20`. Invalid GT is defined as non-finite or non-positive size at `overlay/tools/analyze_sttrack_m53.py:23-24`.

Invalid GT rows are excluded from overlap means and capacity counts: `summarize()` filters `rows` to `valid_gt`, reports `valid_events` and `invalid_gt_events`, and returns `mean_oracle_top1_iou=None` if no valid values exist at `overlay/tools/analyze_sttrack_m53.py:52-74`. The final limitations explicitly state invalid GT events are excluded and reported separately at `overlay/tools/analyze_sttrack_m53.py:205-212`.

### C. Result existence / execution status — WARN

The analyzer is prepared to consume a completed collection, but the task states collection is still running and no full capacity result is available. The prior launch observation showed a running collector with `completed_sequence_files: 0` and `terminal_files: []` at `diagnostics/m53/launch_observation.json:2-7`. The analyzer itself requires terminal files and `collection_receipt.json` before analysis at `overlay/tools/analyze_sttrack_m53.py:81-116`.

Therefore: no capacity outcome, screen pass/fail, per-sequence capacity table, or GT arithmetic can be claimed yet. That is an evidence limit, not an implementation defect.

### D. Dead code / execution wiring — PASS for implementation

The analyzer wires the requested checks into the main path. It verifies every collected event file SHA and byte size against the receipt at `overlay/tools/analyze_sttrack_m53.py:117-121`, checks sequence/fold/split and fit-fold membership at `overlay/tools/analyze_sttrack_m53.py:121-124`, validates event frames and template writes at `overlay/tools/analyze_sttrack_m53.py:125-131`, checks current-template replay and public-state preservation flags per event at `overlay/tools/analyze_sttrack_m53.py:132-137`, validates candidate box shape/finite/positive size and candidate-zero retention at `overlay/tools/analyze_sttrack_m53.py:138-141`, and checks baseline bbox/score against sealed native expected rows at `overlay/tools/analyze_sttrack_m53.py:142-144`.

Native-path preservation is consistent with the collector. The collector captures pre-forward search/templates/query state at `overlay/tools/collect_sttrack_m53.py:60-69`, decodes the baseline and requires it equal the live tracker result at `overlay/tools/collect_sttrack_m53.py:130-133`, hashes tracker state before shadows at `overlay/tools/collect_sttrack_m53.py:134`, and asserts unchanged state after current-template replay and alternatives at `overlay/tools/collect_sttrack_m53.py:143-152`.

### E. Scope and predeclared screen — WARN

The analyzer implements the predeclared screen without promoting a tracker. The spec defines the capacity screen as severe default IoU `<=0.1`, past top1 IoU `>=0.5`, at least 10 events, at least 3 sequences, and “privileged capacity only” at `diagnostics/m53/spec.json:88-94`. The analyzer asserts the frozen numeric screen at `overlay/tools/analyze_sttrack_m53.py:149-150` and computes pass only from `summary['all_past']['severe_events_recovered']` and recovered sequence count at `overlay/tools/analyze_sttrack_m53.py:194-195`.

The plan limits the claim: passing only justifies designing a reader experiment and does not demonstrate a causal reader or authorize training/public evaluation at `diagnostics/m53/EXPERIMENT_PLAN.md:54-60`. The analyzer repeats that limitation in result text at `overlay/tools/analyze_sttrack_m53.py:205-212`.

### F. Evaluation type — WARN because only implementation exists

The planned capacity analysis is **real_gt offline capacity diagnostic** once run: it loads dataset `groundtruth.txt` after seal at `overlay/tools/analyze_sttrack_m53.py:151-160`. The historical-write filtering and best-view choice are **privileged diagnostics**, not deployable inputs: write quality is computed from GT at the write frame at `overlay/tools/analyze_sttrack_m53.py:161-168`, and the result limitation states best-view selection and GT filtering are privileged at `overlay/tools/analyze_sttrack_m53.py:205-207`.

### Specific requested checks

Strictly past template use is preserved. The collector asserts the latest archived template frame is before the current event, uses `archive[:-1]` for alternatives, and appends frame-`t` writes only after frame-`t` counterfactuals at `overlay/tools/collect_sttrack_m53.py:147-162`. The analyzer independently checks `active_template_frame == available[-1]` and alternatives equal `available[:-1]` at `overlay/tools/analyze_sttrack_m53.py:135-137`.

Current-control retention is correct. `event_capacity()` always prepends the current control to all modes at `overlay/tools/analyze_sttrack_m53.py:27-38`; `valid_past` filters only alternatives while retaining evaluated current control at `overlay/tools/analyze_sttrack_m53.py:34-36`.

Top1/top10 choices are separate. The analyzer selects `best_one` by `top1` and `best_ten` by `top10`, and records distinct template frames at `overlay/tools/analyze_sttrack_m53.py:38-43`.

Healthy-read harms are counted separately. Harmful reads on healthy current events are counted per event at `overlay/tools/analyze_sttrack_m53.py:44-49` and aggregated, including healthy-event counts with harmful past reads, at `overlay/tools/analyze_sttrack_m53.py:187-193`.

Rank/new-candidate decomposition is implemented. `current_candidate_rank_improved` counts cases where current top10 already contains a valid candidate but current top1 is wrong; `new_candidate_events` counts cases where only union top10 reaches validity at `overlay/tools/analyze_sttrack_m53.py:65-69`.

Concrete defects found: **none** in `analyze_sttrack_m53.py` or `check_analysis.py`.

Remaining unverified: the completed 63-sequence collection receipt, sealed event-file hashes, actual GT capacity arithmetic, screen pass/fail, generated `capacity_result.json`, generated `per_sequence.csv`, any learned reader, and any recursive/public benchmark result.