Read-only completed M53 capacity audit completed. I recomputed the capacity arithmetic from the staged raw event JSON files and staged GT text copies under `C:/Users/gb/.codex_remote_staging/m53_completed_review`, not from executor summaries or flags. I used `uv run --no-project --offline --with numpy python`, so the recomputation did not contact the network. I did not edit files, run GPU work, contact the server, or launch experiments.

This is a same-family GPT Type-A advisory review, not a cross-family acquittal.

Overall verdict: **PASS for internal arithmetic/integrity of the completed offline capacity result; WARN for claim scope.** I found **no concrete disagreement** between the raw recomputation and the published M53 capacity artifacts. The result supports an offline privileged historical-template capacity screen, not causal reader performance, recursive tracking improvement, or public benchmark improvement.

## A. Ground-truth provenance — PASS

The execution ordering is correctly represented and verified. The analyzer checks exits, bindings, source hashes, collection receipt, event-file hashes/sizes, event ordering, template-write ordering, current-template replay, state preservation, and sealed-native baseline equality before GT is opened at `overlay/tools/analyze_sttrack_m53.py:81-150`. The first GT access is after the explicit comment “GT is first opened below” at `overlay/tools/analyze_sttrack_m53.py:151-160`.

The completed collection receipt reports `status: complete` at `diagnostics/m53/collection_receipt.json:1-2`, 1,511 events, 93,362 frames, 14,349 past-template shadows, 1,131 native template updates, exact current-template replay, unchanged public state, `labels_opened: false`, and `optimizer_steps: 0` at `diagnostics/m53/collection_receipt.json:761-769`.

The private review binding states the event seals were checked before GT download at `C:/Users/gb/.codex_remote_staging/m53_completed_review/review_data_binding.json:4-7`, then binds 63 event files and 63 GT files at `review_data_binding.json:8-133`. I independently verified those staged hashes: 63/63 event files and 63/63 GT files matched their binding hashes, and event file hashes matched the collection receipt.

## B. Score normalization and denominators — PASS

The analyzer uses raw box/GT IoU, not self-normalization. `overlap()` computes intersection-over-union directly at `overlay/tools/analyze_sttrack_m53.py:16-20`. Invalid GT is finite/positive-size checked at `overlay/tools/analyze_sttrack_m53.py:23-24`.

The denominator is valid GT events only. `summarize()` filters `rows` by `valid_gt`, reports total/valid/invalid event counts, accumulates top1 IoU over valid rows, and returns `None` for mean IoU when there are no valid rows at `overlay/tools/analyze_sttrack_m53.py:52-74`.

Independent recomputation matched the published denominators exactly:

- Total events: 1,511.
- Valid GT events: 1,300.
- Invalid GT events: 211.
- Strata: severe 325, partial 213, correct 762, invalid_gt 211.

Published evidence: default summary has 1,511 events, 1,300 valid, 211 invalid at `diagnostics/m53/capacity_result.json:83-96`; invalid-GT stratum has 211 events, 0 valid, 211 invalid, and null means at `diagnostics/m53/capacity_result.json:300-338`.

## C. Result existence and completion — PASS

The relevant completed files exist and the terminal exits are zero:

- `diagnostics/m53/collection.exit:1` is `0`.
- `diagnostics/m53/controller.exit:1` is `0`.
- `diagnostics/m53/analysis.exit:1` is `0`.

The published capacity result reports `status: complete` and scope “Privileged single-frame historical-template capacity on fixed fitting events” at `diagnostics/m53/capacity_result.json:1-3`. It binds the spec, analysis binding, analyzer source, and collection receipt at `diagnostics/m53/capacity_result.json:4-7`. The analysis binding records the bound spec/source/execution/native-training hashes and says experiment GT was not opened before binding, collector was not modified, and screen was not modified at `diagnostics/m53/analysis_binding.json:1-12`.

The per-sequence CSV exists with the expected schema at `diagnostics/m53/per_sequence.csv:1`; I counted 189 data rows, matching 63 sequences × 3 modes. Its head and tail rows are consistent with the recomputed JSON values at `diagnostics/m53/per_sequence.csv:2-12` and `diagnostics/m53/per_sequence.csv:180-190`.

## D. Execution wiring and arithmetic — PASS

The analyzer preserves current-control inclusion. `event_capacity()` prepends the active current-template baseline to every mode at `overlay/tools/analyze_sttrack_m53.py:27-38`; `valid_past` filters only historical alternatives while retaining current control at `overlay/tools/analyze_sttrack_m53.py:34-36`. My recomputation found **0 control-inclusion violations**: neither `all_past` nor `valid_past` ever dropped below default top1/top10 capacity, as expected when current control is included.

The collector and analyzer agree on strictly past historical templates. The collector replays the current dynamic template as control, requires identical maps, excludes the active template from alternatives via `archive[:-1]`, and appends frame-`t` writes only after frame-`t` counterfactuals at `overlay/tools/collect_sttrack_m53.py:143-162`. The analyzer rechecks `active_template_frame == available[-1]` and alternatives equal `available[:-1]` at `overlay/tools/analyze_sttrack_m53.py:132-137`.

Separate top1/top10 view choices are implemented. The analyzer chooses `best_one` by top1 and `best_ten` by top10, then records separate template-frame IDs at `overlay/tools/analyze_sttrack_m53.py:38-43`. My recomputation found different best-top1 and best-top10 frames in 234 valid `all_past` events and 226 valid `valid_past` events, confirming this distinction is active, not dead metadata.

Historical-write filtering is implemented and matched. Write quality is computed from GT at each historical write frame at `overlay/tools/analyze_sttrack_m53.py:161-168`, and valid-past filtering uses only historical writes with write IoU ≥ 0.5 at `overlay/tools/analyze_sttrack_m53.py:34-36`. Published archived-write counts are 1,194 total writes including initialization, 1,168 valid-GT writes, and 1,004 writes with IoU ≥ 0.5 at `diagnostics/m53/capacity_result.json:353-357`; my recomputation matched exactly.

Healthy-read harms are implemented and matched. Event-level harm counts are defined at `overlay/tools/analyze_sttrack_m53.py:44-49`, aggregated at `overlay/tools/analyze_sttrack_m53.py:187-193`, and published at `diagnostics/m53/capacity_result.json:342-351`. My recomputation matched all harm counts exactly:

- past_reads: 12,158.
- valid_past_reads: 10,363.
- harmful_past_reads: 18.
- harmful_valid_past_reads: 1.
- healthy_past_reads: 6,108.
- healthy_valid_past_reads: 5,731.
- invalid_event_gt_past_reads: 2,191.
- healthy_events_with_harmful_past_read: 4.
- healthy_events_with_harmful_valid_past_read: 1.

The invariant `past_reads + invalid_event_gt_past_reads == receipt past_template_shadows` holds: 12,158 + 2,191 = 14,349, matching `diagnostics/m53/collection_receipt.json:763`.

## E. Scope and predeclared screen — PASS for screen arithmetic; WARN for claim strength

The predeclared screen is raw and fixed in the result: severe default IoU ≤ 0.1, past top1 IoU ≥ 0.5, at least 10 events, at least 3 sequences, and privileged capacity only at `diagnostics/m53/capacity_result.json:75-82`. The analyzer computes pass from `summary['all_past']['severe_events_recovered']` and recovered sequence count at `overlay/tools/analyze_sttrack_m53.py:194-195`.

Independent recomputation matched the published pass result:

- `capacity_screen_pass: true`.
- `all_past severe_events_recovered: 26`.
- recovered sequences: 12.
- threshold: ≥10 events and ≥3 sequences.

Published evidence: `capacity_screen_pass` at `diagnostics/m53/capacity_result.json:73-82`; all-past summary at `diagnostics/m53/capacity_result.json:97-121`; severe-stratum all-past recovery at `diagnostics/m53/capacity_result.json:165-189`.

Ranking/new-candidate decomposition matched exactly. The analyzer defines ranking capacity as default top1 wrong, default top10 available, and oracle top1 correct; it defines new-candidate events as default top10 unavailable but union top10 available at `overlay/tools/analyze_sttrack_m53.py:65-68`. Published totals are 61 rank-improved and 44 new-candidate events in `all_past` at `diagnostics/m53/capacity_result.json:103-105`; severe stratum contributes 16 rank-improved and 13 new-candidate events at `diagnostics/m53/capacity_result.json:169-175`; partial stratum contributes 45 and 31 at `diagnostics/m53/capacity_result.json:232-243`. My recomputation matched.

The claim-strength warning remains material. The result scope is explicitly privileged single-frame capacity at `diagnostics/m53/capacity_result.json:2-3`, with `optimizer_steps: 0` and `public_evaluation: false` at `diagnostics/m53/capacity_result.json:73-75`. The limitations state that best-view selection and GT-based filtering are privileged, no causal reader was evaluated, current control is present in every oracle mode, all past writes are archived, invalid GT is excluded/reported, fixed events are not a deployable trigger, and screen success permits reader design only, not recursive or benchmark promotion at `diagnostics/m53/capacity_result.json:132229-132236`.

## F. Evaluation classification — PASS/WARN

- Completed M53 collection: **self_supervised_proxy / sealed prediction artifact**, GT-free by receipt. Evidence: `diagnostics/m53/collection_receipt.json:761-772`.
- Completed M53 capacity analysis: **real_gt offline privileged capacity diagnostic**. It uses dataset GT after seals to evaluate whether historical template views could have produced better single-frame candidates. Evidence: GT loading at `overlay/tools/analyze_sttrack_m53.py:151-160`, privileged limitations at `diagnostics/m53/capacity_result.json:132229-132236`.
- It is **not** causal reader evaluation, recursive tracking evaluation, training, public benchmark evaluation, or human evaluation. Evidence: `diagnostics/m53/capacity_result.json:73-75` and `diagnostics/m53/capacity_result.json:132229-132236`.

## Verified top-level arithmetic

Independent recomputation matched every checked published block with zero discrepancies:

| block | recomputed result |
|---|---|
| staged hash checks | 63 event files + 63 GT files matched |
| summary modes | all fields matched `default`, `all_past`, `valid_past` |
| strata | all fields matched `severe`, `partial`, `correct`, `invalid_gt` |
| harms | all fields matched |
| archived write quality | all fields matched |
| capacity screen | pass matched |
| per-sequence JSON | all 63 × 3 mode records matched |
| per-sequence CSV | 189 rows matched JSON/recompute |

Published summary values matched exactly:

- Default: top1 correct 762, top10 available 872, mean oracle top1 IoU 0.5685636535115736 at `diagnostics/m53/capacity_result.json:83-96`.
- All-past: top1 correct 851, top10 available 916, severe recovered 26, rank-improved 61, new-candidate 44, mean 0.6085715908803444 at `diagnostics/m53/capacity_result.json:97-121`.
- Valid-past: top1 correct 851, top10 available 916, severe recovered 26, rank-improved 61, new-candidate 44, mean 0.6083797861588973 at `diagnostics/m53/capacity_result.json:123-148`.

## Concrete defects found

None. I found no concrete disagreement in hashes, valid/invalid denominators, current-control inclusion, top1/top10 view separation, historical-write filtering, severe rescue counts, recovered sequence count, rank/new decomposition, healthy-read harms, per-sequence JSON, or per-sequence CSV.

## Remaining limits

This result supports only the statement that, on the fixed 63 DepthTrack Train fitting sequences and 1,511 fixed events, a privileged offline oracle over archived native historical template views finds capacity to recover 26 severe-current events across 12 sequences. It does **not** show that a deployable reader can identify those views, does **not** show recursive tracking improvement, does **not** advance M52/M53 to public benchmarks, and does **not** validate a bounded memory bank.