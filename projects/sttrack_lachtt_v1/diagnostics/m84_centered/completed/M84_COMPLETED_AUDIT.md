# M84 completed experiment integrity audit

Date: 2026-09-21. Reviewer: fresh `gpt-6-astra`, reasoning `max`, task `/root/m84_completed_integrity`. Semantic review is `same-family / provisional`; it is not cross-family acceptance. Root for relative evidence references: `D:\Program Files\UserCache\gb\codex\tmp\sttrack_m84_centered_20260920\completed`. Paths beginning `../` refer to the sibling preparation/training evidence. The full machine-readable audit is `M84_COMPLETED_AUDIT.json`.

**Result integrity: PASS within the saved-artifact scope. Deterministic checks: PASS, 4,480 checks, 0 errors. Frozen performance acceptance: FAIL, 8/14 criteria pass. Overall semantic verdict: WARN because this is one seed on a repeatedly used development set and part of the source lineage remains remote-only.** A negative model result is not an integrity failure; the final narrative preserves that distinction.

## Deterministic verification and exact scope

The reviewer wrote and ran `../audit/independent_recompute.py`, using scalar continuous rectangle IoU and explicit consecutive-run counting. It does not import `recursive_metric.statistics`, `run_recursive.analyze`, or the saved-damage diagnostic. Initialization is excluded. GT is valid only when all four coordinates are finite and width/height are positive. An invalid GT frame breaks a low-IoU run. H10 counts maximal contiguous runs with at least 10 valid frames having IoU <= 0.1; it does not count sliding windows and does not join runs across invalid frames.

- Verified all **283 original evidence-manifest files** and **25 supplemental native-extraction manifest files**, including their byte sizes; verified **161 integration-source hashes**. Across the active bindings there are **671 SHA comparisons over 367 distinct local paths**, all matching. `M84_COMPLETED_AUDIT.json` records each comparison and an inventory of declared digests that could not be locally rehashed. This is not a claim that every inherited historical hash was reproduced.
- Recomputed **110 raw sequence files**: 66 M84 files, 22 M82 Category-control files, and 22 extracted independent native files. Each family contains 33,130 positions across 22 sequences: 28,897 valid noninitial frames, 4,211 invalid noninitial frames, and 22 initialization frames. M84 contributes 99,390 stored positions / 86,691 valid metric observations.
- Verified all three M84 receipt identities, chronology, initialization boxes, finite positive-size boxes, finite noninitial scores, and checkpoint/spec/result bindings. The six exit files are `0`; all per-sequence evaluation log receipts match the JSON receipts, and `recursive_analysis.log` matches the saved result.
- Recomputed all 110 per-sequence rows, all five aggregate rows, all 44 leave-one-sequence-out differences, all 14 frozen gates, and the 22 saved-damage rows. CSV coverage is exactly 5 / 110 / 14 / 44 rows. The largest absolute numeric difference from recorded values is **7.275957614183426e-12**, attributable to floating-point summation order; integer counts and gate booleans match exactly.
- Checked 130 training-log rows in frozen sequence order, 186,694 track calls, 5,798 optimizer steps, and all 3,917 sampled-state row positions from the sibling training export. Fit and development sequence IDs are disjoint. Label totals are 156,814 inside-crop, 17,665 outside-crop and 12,215 invalid frames.
- Inspected the final checkpoint directly with a restricted stdlib pickle/ZIP reader, without executing model code: complete status, seed2027, architecture/spec/base binding and counts agree; 24 model tensors include the 768-value `empty_text` buffer, leaving **289,154 stored trainable parameters**. All **868,253 float values in 93 storage entries** (model and optimizer) are finite; all 23 optimizer step tensors are 5,798. This does not reproduce the missing t0/latest/base tensors.

Supporting source and receipt anchors: `run_recursive.py:13-28`, `run_recursive.py:82-105`, `training/category/result.json:2-21`, `category_recursive_receipt.json:2-8`, `category_recursive_receipt.json:143`, `evidence_manifest.json:1086-1105`, `training_saved_verification.json:4-20`, and `../verify_training_complete.py:21-58`.

## A. Ground truth provenance — PASS

The scored references are exported dataset `groundtruth.txt` files, not predictions or model-derived targets. All 22 label files match the M84 frozen `gt_sha256`, the independently available M82 label copies at `../../m82_complete_20260920/dataset_gt/<sequence>/groundtruth.txt`, and the earlier M57 frozen `development_gt_sha256` (`recursive_spec.json:13-255`; `native_reference/source_recursive_spec.json:19-41`). The export copies those exact source files after the prediction receipts are sealed (`export_completed.py:62-69`).

The evaluation inference loop reads RGB-D frames and initializes once from each frozen first-frame box. Its subsequent `tracker.track(frame(i))` call has no GT input (`run_recursive.py:53-64`). The analyzer checks and seals all three families before loading labels at `run_recursive.py:82-105`. The active tracker samples from its preceding predicted state and commits its own bbox/query/template updates (`code/lib/test/tracker/sttrack.py:96-139`). No subsequent-GT-dependent prediction path was found in this active route. GT source identity is supported by these local copies and frozen hashes; the original dataset publisher's distribution was not downloaded again.

Training legitimately uses dataset GT as supervision. `../train_causal.py:124-129` first calls the state-committing step and then exposes the current target to the loss. `../causal_training.py:21-69` contains no GT step argument and commits the predicted state; `../causal_training.py:72-100` masks invalid targets and uses GT-selected cells only for supervision. The native KL teacher is a detached same-state readout, not a native recursive trajectory and not evaluation GT (`../native_preservation.py:23-43`).

## B. Score normalization — PASS

IoU is intersection divided by union. The pooled denominator is the valid GT frame count and the macro denominator is the fixed 22 sequences (`recursive_metric.py:20-30`; `run_recursive.py:107-112`). No reported evaluation score is divided by the model's own maximum, mean or score mass. The separate training-only spatial KL normalizes raw score maps into distributions (`../native_preservation.py:6-13`; `training_spec.json:2564`); this is documented loss construction and does not normalize the reported IoU.

## C. Result existence, identities and final narrative — PASS

Every reported numerical table was checked against independently recomputed raw predictions. The final `NARRATIVE_REPORT.md:3-19` has the correct model/seed/training scope, five metrics rows, 8/14 result, native-protection failure, pooled differences and leave-one-out counts. Frozen metadata still says `prepared_not_frozen` inside the immutable specs; `frozen.json:2-13` is the separate freezing receipt and binds those exact bytes. That inherited status string is not evidence of a missing freeze.

One final M84 checkpoint is shared across all three content evaluations: `c63605ebcb1f702de66f96967255e5301bfdca1e3c40450b6d4b8a952791b1e0` (`recursive_result.json:1048`; receipt line 5 in each family). Training spec: `0f9bb841edd3e2c5171cd78ce9d1030d29a243561006d111d2c98eebfc74abd5`. Recursive spec: `132002a4c0755e4b07c6c9758645d380273e0a0aff41809c2fe6f49d08b1f5d8`. These hashes match the actual local bytes, not just one another.

| Condition | Pooled IoU | Sequence-equal IoU | IoU <= 0.1 frames | H10 |
|---|---:|---:|---:|---:|
| native | 0.652226263176 | 0.684336416360 | 7397 | 75 |
| M82_mask_control | 0.729520895605 | 0.744777274298 | 4952 | 67 |
| category | 0.708521427828 | 0.700454335697 | 5541 | 72 |
| category_empty | 0.652226263176 | 0.684336416360 | 7397 | 75 |
| category_swapped | 0.704142181052 | 0.705933734418 | 5796 | 73 |

The Category-minus-native pooled gain is +5.629516 percentage points. Relative to M82 it is -2.099947 points, with 589 additional low-overlap frames and 5 additional H10 runs. Category-minus-Swapped is +0.437925 pooled points but -0.547940 macro points. These values support the limited descriptive statements in the final narrative, not acceptance of the method.

## D. Invoked metric and sealing order — PASS

The queue waits for Category and Empty, then completes Swapped, and only then calls `run_recursive.py --analyze` (`run_m84.sh:17-28`). The analysis imports `statistics` at `run_recursive.py:80`, calls it at line 105, and writes the result at lines 130-139. `statistics` calls `episodes` (`recursive_metric.py:14-30`). The inherited standalone `recursive_metric.py:33-87` main implements an old M42 CLI and is not invoked by this queue; it is unused code, not a phantom result metric. The accepted values were independently recomputed, so they do not rely on trusting that unused entry point.

## E. Scope and method/control comparability — WARN

Only one Category model was trained, with seed2027 and one chronological pass over 130 fit sequences. Empty and Swapped are content recursions of that same final model, not independent trained controls (`frozen.json:8-13`; `training_spec.json:3`; `training_spec.json:2526`; `recursive_result.json:1049-1051`). Fit IDs are disjoint from the 22 development IDs, but the development set is repeatedly reused; no untouched-test generalization or across-seed uncertainty is established. The final report states this boundary (`NARRATIVE_REPORT.md:5`, `NARRATIVE_REPORT.md:23`).

Seventeen protocol/data/seed/optimizer fields agree exactly with M82. The active integration differs only in adapter construction and the new centered adapter. M82 reuse as a hard-empty control for the specified nonempty Category/Swapped inputs is supported by `../prepare_m84.py:37-54` and `../preparation_receipt.json:3-33`, which check all 130 fit Category slots and 22 + 22 development Category/Swapped slots as nonempty and assert exact t0 trainable-tensor identity. Those original bank/t0 bytes are absent locally; this part is a reviewed and SHA-bound preparation assertion, not an independently replayed tensor check.

The centered implementation calls the same adapter twice with shared weights, subtracts before adding fused features, and keeps both gradient paths (`code/lib/models/sttrack/centered_semantic_adapter.py:12-20`). It therefore costs an extra adapter branch and cancels the final 768-dimensional bias from the real-arithmetic residual. It is not a compute/effective-capacity matched comparison with M82; the narrative explicitly admits this (`NARRATIVE_REPORT.md:3`). Category-versus-Empty may include generic nonempty conditioning. Frozen automatic captions are not semantic GT, and Category-versus-Swapped does not establish correct identity understanding.

## F. Evaluation type — real_gt, custom development protocol

These are real dataset labels under a custom continuous-IoU/H10 development evaluation. They are not official DepthTrack Test, CDTB, or full VOT-RGBD2022 scores (`recursive_metric.py:1`; `NARRATIVE_REPORT.md:5`). No official performance result exists in this audited bundle, and none is inferred from the completed training, preflight, or readiness interfaces. Subtraction novelty is expressly not claimed; the cited prior-art literature was not re-adjudicated in this result audit (`NARRATIVE_REPORT.md:21`).

## Frozen acceptance: all 14 criteria recomputed

All original directions/tolerances were preserved: strict pooled improvement; nondecreasing macro IoU; nonincreasing low-frame/H10 counts; zero-native-H10 protection; and per-sequence Empty/native IoU-sum tolerance 1e-8 plus exact counts (`EXPERIMENT_PLAN.md:17-24`; `run_recursive.py:114-137`; `recursive_result.json:940-968`).

| Group | Criterion | Recomputed outcome |
|---|---|---|
| M82_increment | mean | FAIL |
| M82_increment | macro | FAIL |
| M82_increment | low | FAIL |
| M82_increment | H10 | FAIL |
| native | mean | PASS |
| native | macro | PASS |
| native | low | PASS |
| native | H10 | PASS |
| native | native_zero_H10_protection | FAIL |
| swapped_content | mean | PASS |
| swapped_content | macro | FAIL |
| swapped_content | low | PASS |
| swapped_content | H10 | PASS |
| empty_native_parity | all_sequences | PASS |

`mobilephone02_indoor` is the sole violated native-zero-H10 protection: native mean IoU 0.851274 / H10=0, Category 0.560285 / H10=1, Swapped 0.842371 / H10=0. Its Category H10 interval is **[497,701)**, 204 frames, and Empty/Swapped have no H10 interval. Category worsens mean IoU on 10 of 22 sequences relative to native, so pooled improvement must not be described as uniform sequence benefit.

All 22 Category-minus-Empty leave-one-out pooled differences remain positive (range 0.021132245741 to 0.067568669054). Category-minus-Swapped is positive in 19/22. Removing `colacan01_indoor` yields -0.018629633916; removing `flower02_wild`, -0.002256649202; removing `ghostmask_indoor`, -0.002565009745. These are dependence diagnostics, not independent replications or confidence intervals (`run_recursive.py:124-129`; `recursive_result.json:993-1041`; `leave_one_sequence_out.csv:1`).

## Empty/native parity: aggregate, per-sequence, and saved raw outputs

All 22 frozen per-sequence metric parity checks pass. A stronger supplemental check is now supported: all **33,130 bbox arrays** and **33,108 noninitial scores** in the SHA-bound native extraction are exactly equal to the saved Empty outputs. The native reference metrics were independently recomputed from those raw boxes, not copied from the old aggregates.

The supplemental extraction reads the two M57-frozen baseline shards and exports their independent `public_bbox` / `public_score` rows (`native_reference/extract_native_reference.py:11-26`). It is not the M83 same-state native readout on Category history. The local 25-file manifest, source spec and archive verify; the original result binds the same M57 source-spec SHA and GT hashes (`native_reference/source_recursive_spec.json:15-41`; `native_reference/reference_provenance.json:3-6`). **The original full shard bytes remain remote, so the reviewer independently verifies the extraction bytes and comparisons but relies on the recorded extraction execution for the original-shard hash assertions.** All-frame query/template states were not exported and are not claimed equal by this audit. An empty intervention after a divergent Category history is a different experiment from an independent Empty/native recursion.

## Saved-damage diagnostic

The independent verifier reproduced all 22 sequence rows in `saved_damage_diagnostic.json`, including first bbox differences, H10 intervals and the separate criterion of >=10 consecutive frames with Category IoU <=0.1 while Empty >=0.5 (and its converse). Invalid GT breaks these intervals too (`inspect_saved_damage.py:15-46`; `saved_damage_diagnostic.json:2`). Totals are **18 sustained-damage runs / 1,074 frames** and **26 sustained-rescue runs / 2,924 frames** (`saved_damage_diagnostic.json:1449-1452`).

For `car02_indoor`, sustained damage is [344,550) and [659,675); Category H10 is [215,550) and [644,675), while Empty H10 is [215,344) and [644,658). For `mobilephone02_indoor`, sustained damage is [497,701). Category bbox already differs from both Empty and Swapped at frame 1 in each of these sequences (`saved_damage_diagnostic.json:402`; `saved_damage_diagnostic.json:1387`). The diagnosis localizes differences in saved independent recursions; it does not identify whether text, response ranking, crop drift, query, or template state caused the later damage.

## Remaining evidence limits and actions

1. Retain this as a completed negative result under the frozen acceptance definition; do not promote the method or relabel Empty as the primary model. No recomputed threshold or checkpoint selection is justified by this audit.
2. Preserve the single-seed/reused-development/custom-metric qualifiers and the automatic-caption/compute/effective-capacity caveats when publishing these numbers.
3. Keep local independent checks distinct from recorded remote checks. Native checkpoint, t0 checkpoint, bank tensors, full native shards, latest-checkpoint parity, base immutability and full training visited-state reconstruction were not all independently reproduced locally. `training_saved_verification.json:17-20` itself acknowledges its saved-artifact scope. The full missing-digest inventory is retained in the JSON.
4. Saved bbox/score equality is established for this complete development extraction. It is not proof of general algebraic floating-point identity at arbitrary states, unseen data, or every internal query/template tensor.

No existing result, code, checkpoint, receipt or label file was modified by this reviewer. The audit's deterministic scripts and reports are new artifacts. The initial verifier path assumption for the older M82 GT layout was corrected before the accepted final run; the final run has zero missing-file, hash or numerical errors.

## Key output hashes

Independent verifier script SHA256: `26368aa0a37860cdfc10647a4084610f74d44a9c9b34497b25f39354cc9a8510`.

Independent verifier detailed result SHA256: `e4fe48b10bf97fc7e84b9da3ef29983e67115f5c2f0b811b6b2faa3feee3bb8d`.

Audited final narrative SHA256: `27273e02d4b953122f3daef09d8bf19e7bb364117ffec801dd57686672f4b923`.

Native extraction archive SHA256: `9b62820ccea97a5593768d6e04e085fac51b4e6856d2db651364c73296c5a1ea`.

Machine-readable audit SHA256: `249c36d06426ea29aa706cb6504ea02996a880138af6a0314d3325928c5130dd`.
