Read-only audit completed for the post-hoc M53 read-budget exploration. I read the source and data directly, and I recomputed the exploration CPU-only from the staged raw event JSON and staged GT text files using:

`uv run --no-project --offline --with numpy python`

I did not edit files, contact the server/network, use GPU, launch experiments, or rely on the executor’s supplied summaries as evidence. This is a GPT-family Type-A advisory review, not a cross-family acquittal.

Overall verdict: **PASS for arithmetic and provenance of the post-hoc read-budget artifact; WARN for interpretation scope.** I found **no concrete defect** in the read-budget JSON against an independent raw-box/GT recomputation. The output is a post-hoc fitting-set anatomy over privileged oracle selections and one untrained native-score comparator; it is not a causal reader result, recursive evaluation, runtime policy, promotion condition, or public benchmark result.

## Files inspected

Published files:

- `diagnostics/m53/inspect_read_budget.py`
- `diagnostics/m53/read_budget_exploration.json`
- `diagnostics/m53/capacity_result.json`
- `diagnostics/m53/collection_receipt.json`
- `diagnostics/m53/EXPERIMENT_PLAN.md`

Raw review copies:

- `C:/Users/gb/.codex_remote_staging/m53_completed_review/events/*.json`
- `C:/Users/gb/.codex_remote_staging/m53_completed_review/groundtruth/*.txt`
- `C:/Users/gb/.codex_remote_staging/m53_completed_review/review_data_binding.json`

Verified hashes/provenance:

- `read_budget_exploration.json` SHA-256: `a24b3a7e39c789f8dbca3d3241ebe912e3d0e1ae6b13f9538b42b3b012149623`
- `inspect_read_budget.py` SHA-256: `84027554859464d1caa833a1471025122b770fae9b0aea5b9cd4006cd1e1b2c3`
- `capacity_result.json` SHA-256: `60e40d6e7a4efe1301f3f4153986f1d03825ab01e33bc0127b7f62f47c0cbe6a`
- `collection_receipt.json` SHA-256: `3bd4eb79a1ad5a96bac5d2766a4e2967c7c3f980bbfc7d81e706e56e8d42be45`
- Raw staged copy binding reports `status: complete`, binds the capacity and collection receipt hashes, and states event seals were checked before GT download at `C:/Users/gb/.codex_remote_staging/m53_completed_review/review_data_binding.json:1-7`. It binds the 63 event JSONs and 63 GT text files at `review_data_binding.json:8-133`.

## Raw provenance and GT ordering

The read-budget script itself is a derivative post-hoc analysis over `capacity_result.json` plus raw event JSONs. It does **not** independently open GT: it reads `capacity_result.json` at `diagnostics/m53/inspect_read_budget.py:15-18`, reads raw event JSONs and verifies their SHA against `collection_receipt.json` at `diagnostics/m53/inspect_read_budget.py:19-25`, then iterates `capacity['events']` at `diagnostics/m53/inspect_read_budget.py:27-42`.

That makes the provenance clear: read-budget top1 IoUs come from the already completed M53 capacity result, while `max_native_score` uses raw event scores from the raw event JSON. This is not a defect because the artifact is scoped as “Post-hoc fitting-set anatomy; original capacity screen unchanged” at `diagnostics/m53/read_budget_exploration.json:1-4`, and its notes explicitly mark the privileged modes and non-deployment status at `diagnostics/m53/read_budget_exploration.json:45619-45623`.

For the underlying GT-derived capacity result, output sealing before GT was already enforced by the analyzer: it checks collection exits, bindings, collection receipt, event hashes/sizes, event ordering, template-write ordering, current-template replay, public-state preservation, and sealed-native baseline equality before GT access at `overlay/tools/analyze_sttrack_m53.py:81-150`; GT is first opened below that point at `overlay/tools/analyze_sttrack_m53.py:151-160`. The plan required this ordering at `diagnostics/m53/EXPERIMENT_PLAN.md:45-51`.

The completed collection receipt supports the sealed/no-training provenance: 1,511 events, 93,362 frames, 14,349 past-template shadows, 1,131 native template updates, exact current-template replay, unchanged public state, `labels_opened: false`, `optimizer_steps: 0`, checkpoint hash, spec hash, and `source_unchanged: true` at `diagnostics/m53/collection_receipt.json:761-772`.

## Independent recomputation result

I recomputed every valid event from raw event boxes and staged GT, then compared against `read_budget_exploration.json` row-level selections and summary values.

Verified with **zero mismatches**:

| Check | Result |
|---|---:|
| staged event files | 63/63 hash-matched |
| staged GT files | 63/63 hash-matched |
| event receipt hashes | 63/63 matched collection receipt |
| strict-past ordering violations | 0 |
| active-control inclusion violations | 0 |
| per-event selected frame/IoU mismatches | 0 |
| published-vs-recomputed summary mismatches | 0 |
| valid read-budget rows | 1,300 |
| published read-budget rows | 1,300 |

The read-budget output rows begin at `diagnostics/m53/read_budget_exploration.json:117` and end at `diagnostics/m53/read_budget_exploration.json:45618`. The first row example shows all option selections for `bag04_indoor@10` at `diagnostics/m53/read_budget_exploration.json:117-151`; the final row example shows `trophy_indoor@611` at `diagnostics/m53/read_budget_exploration.json:45583-45618`.

## Denominators and invalid GT

The read-budget script excludes invalid GT events by skipping `event['valid_gt'] == false` at `diagnostics/m53/inspect_read_budget.py:27-30`. Its denominator is therefore valid capacity events only. The summary builder uses `len(rows)` after that skip and computes mean IoU over those rows at `diagnostics/m53/inspect_read_budget.py:43-51`.

Independent recomputation confirmed:

- Total M53 capacity events: 1,511.
- Valid GT events used by read-budget: 1,300.
- Invalid GT events excluded from read-budget: 211.

The capacity result records the same top-level denominator: default summary has 1,511 events, 1,300 valid, and 211 invalid at `diagnostics/m53/capacity_result.json:83-96`. The invalid-GT stratum explicitly reports 211 events, 0 valid events, 211 invalid events, and null means at `diagnostics/m53/capacity_result.json:300-338`.

The read-budget JSON itself reports `valid_events: 1300` for every option at `diagnostics/m53/read_budget_exploration.json:6-115`. It does not repeat the invalid-GT count; that is acceptable for this derived anatomy artifact because the underlying capacity result records it.

## Option sets and active-control inclusion

The source defines exactly these option sets at `diagnostics/m53/inspect_read_budget.py:31-35`:

- `current`
- `initial_or_current`
- `recent1_or_current`
- `initial_recent1_current`
- `initial_recent3_current`
- `all_past`

It then adds `max_native_score` as an untrained saved-state comparator using raw event scores at `diagnostics/m53/inspect_read_budget.py:37-42`.

All options include the active current-template baseline. This is explicit in the source because each option starts from `[baseline]`, and `all_past=views` where `views[0]` is the baseline at `diagnostics/m53/inspect_read_budget.py:30-35`. The raw recomputation found **0 active-control inclusion violations**.

The current-control inclusion matters: the oracle modes are protected against choosing below current when selecting by top1 IoU. That explains why severe harms are zero in the privileged top1 modes. `max_native_score` is not IoU-protected, but the raw recomputation still found zero severe harms for it.

Published option summaries matched exactly:

| Option | Valid events | Correct | Severe rescued | Rescue sequences | Severe harms | Mean IoU |
|---|---:|---:|---:|---:|---:|---:|
| current | 1300 | 762 | 0 | 0 | 0 | 0.5685636535115736 |
| initial_or_current | 1300 | 811 | 16 | 7 | 0 | 0.5911285691495738 |
| recent1_or_current | 1300 | 792 | 8 | 4 | 0 | 0.5797726878584293 |
| initial_recent1_current | 1300 | 818 | 16 | 7 | 0 | 0.5935467957168007 |
| initial_recent3_current | 1300 | 831 | 19 | 10 | 0 | 0.5989301867388636 |
| all_past | 1300 | 851 | 26 | 12 | 0 | 0.6085715908803444 |
| max_native_score | 1300 | 780 | 2 | 1 | 0 | 0.5699338967134121 |

Evidence: `diagnostics/m53/read_budget_exploration.json:5-115`.

## Strictly-past write ordering

Strictly-past ordering is preserved.

The collector makes current-template replay with the active current dynamic template, then evaluates only `archive[:-1]` as historical alternatives, with `archive[-1]['frame'] < frame` asserted before alternatives are used at `overlay/tools/collect_sttrack_m53.py:143-152`. It appends a frame-`t` template write only after all frame-`t` counterfactuals are complete at `overlay/tools/collect_sttrack_m53.py:158-162`.

The capacity analyzer independently rechecks this ordering: available writes are those with frame `< event['frame']`; the active frame must be the last available write; alternatives must equal all earlier available writes at `overlay/tools/analyze_sttrack_m53.py:132-137`.

My raw recomputation found **0 strict-past ordering violations** across the 1,511 raw events.

## Initial-template substitution versus adding a view

The read-budget option `initial_or_current` uses `template_frame == 0` from the historical alternatives at `diagnostics/m53/inspect_read_budget.py:31-35`. This should be interpreted as substituting the dynamic second template input with the initialization template, not adding a third template view.

The collector’s shadow call always uses a two-template interface: the first template is the captured initialization template, and the second template is the tested dynamic template at `overlay/tools/collect_sttrack_m53.py:136-141`. Therefore, when a historical alternative has `template_frame == 0`, the first template remains initialization and the dynamic slot is also filled with the initialization template. The read-budget note states this directly: “Recent memories are actual native writes; initialization may also be a dynamic input” at `diagnostics/m53/read_budget_exploration.json:45620-45623`.

This is not a defect. It is an important interpretation limit for `initial_or_current`, `initial_recent1_current`, and `initial_recent3_current`.

## Per-event view/score mapping and ties

Per-event view mapping matched exactly. For every valid event, I recomputed raw IoUs for the baseline and all historical alternatives from raw boxes and GT, reconstructed all option sets, applied the same selection rule, and compared selected `frame` and `iou` against `read_budget_exploration.json`. Result: **0 per-event mismatches**.

The option-selection source uses Python `max(..., key=lambda v: v['top1'])` for the oracle options at `diagnostics/m53/inspect_read_budget.py:36`, and uses max raw native score for `max_native_score` at `diagnostics/m53/inspect_read_budget.py:37-42`.

Tie behavior is deterministic by list order and was verified by row comparison. I found top1 tie events in the privileged IoU-selected modes:

| Option | Events with tied best top1 IoU |
|---|---:|
| current | 0 |
| initial_or_current | 223 |
| recent1_or_current | 225 |
| initial_recent1_current | 223 |
| initial_recent3_current | 221 |
| all_past | 217 |
| max_native_score | 0 native-score ties |

There were no row mismatches despite these ties, so the published selected frame/IoU values are consistent with stable first-in-option-order selection. The artifact does not report tie counts; that is a reporting limitation, not a numeric defect.

## Historical-write filtering

The read-budget exploration does **not** implement the GT-quality `valid_past` filter as an option set. Its options are defined at `diagnostics/m53/inspect_read_budget.py:31-35`, and none is `valid_past`. It uses `all_past=views`, which corresponds to all evaluated historical alternatives plus current control, not the GT-filtered subset.

This is not a defect because `read_budget_exploration.json` does not claim these budget options are GT-write-quality filtered. The underlying M53 capacity artifact separately reports `valid_past`, and its limitations state that filtering historical writes by GT is privileged at `diagnostics/m53/capacity_result.json:132229-132236`.

If this exploration is cited, it should be described as a post-hoc budget anatomy over unfiltered actual native writes, except where the option name explicitly restricts by recency or initialization. It should not be described as a valid-write-only budget result.

## Rescue, harm, and sequence-count arithmetic

The summary logic defines:

- `correct`: selected IoU ≥ 0.5.
- `severe_rescued`: selected IoU ≥ 0.5 and default IoU ≤ 0.1.
- `severe_harms`: default IoU ≥ 0.5 and selected IoU ≤ 0.1.
- rescue/harm sequence sets from the corresponding rows.
- `mean_iou`: arithmetic mean over valid read-budget rows.

Evidence: `diagnostics/m53/inspect_read_budget.py:43-51`.

I recomputed all of these from raw boxes/GT and matched every published field at `diagnostics/m53/read_budget_exploration.json:5-115`.

The `all_past` option reproduces the frozen capacity screen’s severe rescue count: the read-budget script asserts `summary['all_past']['severe_rescued'] == capacity['summary']['all_past']['severe_events_recovered']` at `diagnostics/m53/inspect_read_budget.py:52`, and the independent recomputation confirms both are 26. The capacity result’s all-past summary gives 26 severe events recovered across 12 sequences at `diagnostics/m53/capacity_result.json:97-121`.

The broader capacity result’s rank/new-candidate decomposition remains distinct from read-budget’s top1-only budget anatomy. Capacity `all_past` reports `current_candidate_rank_improved: 61` and `new_candidate_events: 44` at `diagnostics/m53/capacity_result.json:103-105`. The read-budget artifact does not recompute or report rank/new-candidate decomposition for the budget options; it reports only top1-selected correctness, severe rescues, severe harms, sequence lists, and mean IoU. That is a scope distinction, not a defect.

## Summary arithmetic

I independently recomputed and matched all summary values:

| Option | Correct | Severe rescued | Severe harms | Mean IoU |
|---|---:|---:|---:|---:|
| current | 762 | 0 | 0 | 0.5685636535115736 |
| initial_or_current | 811 | 16 | 0 | 0.5911285691495738 |
| recent1_or_current | 792 | 8 | 0 | 0.5797726878584293 |
| initial_recent1_current | 818 | 16 | 0 | 0.5935467957168007 |
| initial_recent3_current | 831 | 19 | 0 | 0.5989301867388636 |
| all_past | 851 | 26 | 0 | 0.6085715908803444 |
| max_native_score | 780 | 2 | 0 | 0.5699338967134121 |

Evidence for the published values: `diagnostics/m53/read_budget_exploration.json:5-115`.

Additional recomputed selection counts, useful for interpretation:

| Option | Selected active current frame | Selected frame 0 |
|---|---:|---:|
| current | 1300 | 182 |
| initial_or_current | 876 | 606 |
| recent1_or_current | 877 | 208 |
| initial_recent1_current | 745 | 511 |
| initial_recent3_current | 647 | 394 |
| all_past | 578 | 311 |
| max_native_score | 563 | 271 |

The `selected frame 0` count includes cases where the active current dynamic template itself is frame 0, plus cases where frame 0 is used as the substituted dynamic input. It does not mean a new third template was added.

## Post-hoc oracle versus causal/runtime scope

The scope is correctly limited in the artifact and source.

The read-budget result says its scope is “Post-hoc fitting-set anatomy; original capacity screen unchanged” at `diagnostics/m53/read_budget_exploration.json:1-4`. Its notes say all modes except `current` and `max_native_score` use privileged best-view IoU selection, `max_native_score` is untrained and not recursively deployed, initialization may be used as a dynamic input, and the exploration adds no promotion condition at `diagnostics/m53/read_budget_exploration.json:45619-45623`.

The plan requires this distinction: best-view choice and GT-based write filtering are privileged diagnostics, never deployment inputs, at `diagnostics/m53/EXPERIMENT_PLAN.md:45-51`; passing the capacity screen only justifies reader design and does not show a causal reader can identify the views, with no automatic training or public evaluation, at `diagnostics/m53/EXPERIMENT_PLAN.md:54-60`.

The capacity result repeats those limits: no causal reader was evaluated, current control is included in every oracle mode, all past writes are archived, invalid GT is excluded/reported, fixed training events are not a deployable trigger, and screen success is not recursive or benchmark promotion at `diagnostics/m53/capacity_result.json:132229-132236`.

## Concrete defects found

None.

I found no concrete disagreement in:

- raw staged file hashes,
- collection receipt event seals,
- GT hash provenance,
- valid/invalid denominators,
- strict-past write ordering,
- active-control inclusion,
- per-event selected view/IoU mapping,
- max-native-score frame mapping,
- deterministic tie handling,
- rescue counts,
- harm counts,
- rescue sequence sets,
- mean IoU arithmetic,
- `all_past` consistency with the frozen capacity result.

## Remaining limits

The read-budget exploration remains a post-hoc fitting-set anatomy. It supports statements such as:

- “Under privileged best-top1 selection over these option sets, the recomputed summaries match the published read-budget artifact.”
- “`all_past` recovers 26 severe-current events across 12 sequences on 1,300 valid fitting events.”
- “`max_native_score` is an untrained saved-state comparator and recovers 2 severe-current events in this static anatomy.”

It does **not** support:

- a deployable causal reader claim,
- a runtime read-policy claim,
- recursive tracking improvement,
- public benchmark improvement,
- a promotion of M52/M53 to public evaluation,
- a claim that a bounded memory bank works,
- a claim that GT-filtered historical writes were used in the read-budget options.

The one reporting limitation I would keep explicit is that top1 ties are not summarized in `read_budget_exploration.json`; I recomputed tie counts and verified the published selected frames/IoUs match deterministic first-in-list selection.