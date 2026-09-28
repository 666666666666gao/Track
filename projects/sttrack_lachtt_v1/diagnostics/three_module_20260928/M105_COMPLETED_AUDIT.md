# M105 completed experiment integrity audit

**Overall: WARN — bounded provenance/runtime evidence; no numerical or forward-path blocker found.**

The independent CPU checks pass for all 3,039 ordered rows, 12,156 selected-box IoUs, 140 original M104 summary values, 672 component summary values and 131 retained rescue/break records. Both local checkpoint byte hashes and all downloaded runtime-file identities match. The completed report's numerical claims are correct.

The warnings concern evidence limits and one wording issue: original remote GT files were not rehashed by this reviewer; all-candidate zero-delta and frozen-state checks are completed-runtime assertions rather than a new reviewer GPU execution; and the plan says GT is *read* after decoding although it is loaded beforehand and only consumed by overlap afterward. None is evidence of a numerical error or current-GT leakage.

## Reviewer and acceptance

- Date: 2026-09-28; recorded at 2026-09-28T07:04:30Z.
- Reviewer: `/root/m105_completed_integrity`, fresh-context `gpt-6-astra`, reasoning `max`, `fork_turns=none`. Model/settings attribution comes from the parent assignment and preserved trace request; no separate agent UUID was exposed.
- `review_independence: same-family`; `acceptance_status: provisional`. The semantic verdict is not cross-family acceptance.
- Deterministic artifact identity and CPU arithmetic: **PASS / accepted**, limited to the checked bytes and arithmetic.
- Trace: `.aris/traces/experiment-audit/2026-09-28_m105_completed/`, relative to `C:/Users/gb/.codex_track_publish_m29_20260902`. The executor preserves the prompt/response/meta trace; this reviewer writes only these two audit files.
- Read-only source/primary-artifact review and CPU stdlib checks. No model import or execution, remote access, GPU work, source changes, result changes, or annotation-label changes.

All file:line references below are relative to the diagnostics directory containing this report. The JSON companion includes every audited input hash, exact checked counts, all rescue/break keys, and the executed independent verifier source. Prior reviewer conclusions were not used as proof.

## A. Ground-truth provenance — WARN

The target source is dataset GT: `prepare_train_states.py:69-74` opens each sequence's `groundtruth.txt`, verifies its hash against the source spec and checks validity; `:99-105` copies the current/previous dataset boxes to labels. Native predictions enter state selection/strata, not the target boxes (`:48-56,75-90`). The source spec identifies `/root/autodl-tmp/depthtrack/train/sequences` (`../full152_paired_20260925/M82_training_spec.json:6`).

I rehashed the actual source spec, former split spec and training-label bytes. They equal the preparation's recorded identities (`m105_inputs/preparation.json:3-10`), and both downloaded label files match their byte/hash receipt (`m105_inputs/download_receipt.json:2-8`). All 3,502 label records were parsed; every evaluated key, split, stratum and valid current box matches. The source spec contains 152 individual GT hashes, but those original remote text files were not accessed; this is a provenance chain check, not a new raw-dataset attestation.

The current GT is excluded from the parent batch: `train_ab_visual_control.py:86-94` drops `iou` and omits non-tensor keys/strata, while supplying five Empty slots. `train_local_visual_geometry.py:28-30,70-84` exposes the refiner inputs as encoded visual tokens, validity and native geometry. M105 calls that forward before overlap (`diagnose_learned_geometry_components.py:73-89`); eligibility loaded by `metadata` is unused. Parent selection is model-score argmax (`instance_ab_prototype.py:108-121`) and remains fixed across variants. The source collector obtains proposals from native output and does not load current labels (`collect_train_states.py:110-153`). Normal first-frame GT initialization is part of the source tracking protocol, not current-event label leakage.

**Wording caveat:** `M105_GEOMETRY_COMPONENT_PLAN.md:23` says GT is read only after decoding. Labels are actually loaded earlier by `load_inputs` and `metadata` (`train_ab_visual_control.py:25-27`; `train_local_visual_geometry.py:55-66`). The accurate statement is that current GT does not enter forward/selection and is consumed by overlap after decoding. `m105_completed/RESULT.md:79-80` already uses the accurate interpretation.

## B. Score normalization — PASS

`analyze_train_states.py:20-25` implements ordinary box intersection divided by geometric union. M105 means divide by the number of events; correctness/rescue/break values are raw IoU≥0.5 counts (`diagnose_learned_geometry_components.py:19-30`). There is no performance normalization by prediction-score maximum, minimum or mean. The GT oracle takes the candidate maximum as a separate coverage diagnostic and is explicitly not a deployable selector (`m105_completed/RESULT.md:30-32`).

Native score differences, rank/grid scaling and box/prior geometry appear as **features**, not performance normalization (`collect_ab_interface_panel.py:52-62`). M104 box-size normalization belongs to regression targets (`train_local_visual_geometry.py:49-52`). The high healthy-group accuracy is conditioned on GT/native-IoU strata, not evidence of whole-sequence performance.

## C. Result existence, completion and claims — PASS

Both real child exits and the driver exit equal zero (`m105_completed/driver.json:35,68,73`; the three `.exit` files at line 1). Launch commands, PIDs, split/GPU assignments and controller-log entries agree; fit is GPU0 and development GPU1 (`driver.json:5-35,38-68`; `controller.log:1-2`). The driver took 14.418560199 seconds (`driver.json:71-73`), matching the rounded report.

All **13 runtime downloads / 4,469,888 bytes** exactly match their SHA256/size receipt (`m105_completed/download_receipt.json:4-57`). All 11 deployment entries match local bytes (`M105_DEPLOY_RECEIPT.json:4-14`). The downloaded label/preparation files match both receipt entries. The local M104 checkpoint is SHA256 `546bd218a14906beb28033717bb53cc17721e9bb6b3fb53f36a4bac15dff9f6e`; the local M101 parent is `1d187d78c1dd76ebf59e41fd31af16fa69cecc37b4cb6c74fbdfe3ed5c237293`. They match the actual file bytes and the M104/M101/M105 records (`m104_completed/train/result.json:89,262`; `m101_completed/train_weight1/result.json:5186`; both M105 result tails).

The diagnosis has no optimizer or `torch.save` call; imports do not execute the earlier training `main` functions. It only writes events/result files after checks (`diagnose_learned_geometry_components.py:133-144`). The completed M105 output contains no checkpoint. This does not deny the earlier M104 training's 384 optimizer steps (`m104_completed/train/result.json:9`).

Every value in the report's eight localization rows and four harm rows matches the raw artifacts at the printed precision (`m105_completed/RESULT.md:34-58`). Its marginal counts and rescue-intersection claims also match (`:45-49,67-70`). The next preservation-training proposal at `:84-88` is an untested next experiment, not an established remedy.

## D. Metric execution, replay and raw checks — PASS

Metrics are actually called: overlap at `diagnose_learned_geometry_components.py:88`, marginal and joint summaries at `:112-113`, output serialization at `:143-144`. Checks use primary JSON/JSONL data, not merely acceptance booleans. I did not run `accept_m105_components.py`, because that program would rewrite `acceptance.json`; an independent stdlib verifier reproduced its arithmetic and strengthened replay verification.

| Independent check | Result |
| --- | ---: |
| Ordered M104 full-row equality, including keys/strata/selection/native/parent/full/oracle | 3,039 / 3,039 exact |
| Selected-box IoU computed from coordinates and dataset label using explicit float32 arithmetic | 12,156 / 12,156 exact |
| Candidate IoUs finite, in [0,1], with selected-index equality | 121,560 / 121,560 |
| Original M104 fields, 14 × 5 groups × 2 splits | 140 / 140 exact |
| Eight component fields × four variants × (10 marginal + 11 joint groups) | 672 / 672 exact |
| Complete retained rescue/break records versus raw filters | 131 / 131 exact |

The reconstructed original fields are `valid_gt, native_iou50, selected_iou50, oracle_iou50, rescues, breaks, native_mean_iou, selected_mean_iou, parent_correct, parent_mean_iou, parent_rescues, parent_breaks, original_top10_correct, original_miss_refined_hit`. Complete dictionaries, including the group-key sets, equal M104. Evidence: the two M105 JSONL files at lines 1–2544 / 1–495; the corresponding M104 JSONLs; `m104_completed/train/result.json:91-254`; `diagnose_learned_geometry_components.py:91-97,118-132`. The original incomplete source gate recorded in the historical initial review is not the audited current gate.

**Runtime invariants:** M105 snapshots both complete state dictionaries after loading, sets eval/no-grad, checks exact zero-delta decoding on every batch, and compares every parameter/buffer afterward (`diagnose_learned_geometry_components.py:59-69,83-84,103-104`). Successful output is written only after these checks. The source hash in both results equals the inspected/deployed script; both exits are zero. This supports executed checks of **25,440 fit + 4,950 development = 30,390 candidate boxes** and unchanged states (`fit/result.json:477-487`; `development/result.json:435-445`).

Those are source-bound runtime assertions, **not** a new CPU/GPU tensor replay by this reviewer. Intermediate zero/frozen tensors are not saved. Unselected candidate boxes are also not saved: their reported IoUs were checked for validity and summary arithmetic, but only the four selected boxes per event received independent coordinate/GT overlap recomputation.

## E. Scope — PASS within the stated boundary

There are 130 fit and 22 development sequences, all from DepthTrack **Train**, with 2,544 and 495 valid events. All are exactly the valid-label sets; 368 fit and 95 development sampled records with invalid current GT are excluded. Preparation has 3,502 sampled records, not 3,039 unconditionally valid frames (`m105_inputs/preparation.json:11-26`; `train_ab_visual_control.py:60-61`; `m105_inputs/training_labels.json:1`). Sampling is GT/native-IoU stratified (`prepare_train_states.py:75-90`). This is one completed M104 final, trained with seed 2027, and four same-final readouts—not four independently trained models (`m104_completed/train/result.json:4-9`; `M105_GEOMETRY_COMPONENT_PLAN.md:14-22`).

| Marginal population | Fit | Development |
| --- | ---: | ---: |
| all | 2544 | 495 |
| healthy | 1560 | 264 |
| transition | 452 | 127 |
| late_low | 178 | 41 |
| intermediate | 381 | 69 |

Joint profiles exhaust each split exactly. Fit: healthy 1560; transition 425; late_low 152; intermediate 380; late_low|transition 26; intermediate|transition 1. Development: healthy 264; transition 121; late_low 35; intermediate 69; late_low|transition 6. Marginals overlap and cannot be summed as disjoint populations (`fit/result.json:224-475`; `development/result.json:224-433`; `RESULT.md:48-49`).

The full correction preserves M104's known harms: `pine01_indoor@1276` (fit JSONL line 2436) and `mobilephone02_indoor@65` (development line 462). The separate size-only harms are `basket_indoor@67` (fit line 274) and `ball19_indoor@688` (development line 302). Full's original healthy-protection check remains false (`m104_completed/train/result.json:255-259`). Center-only development has zero breaks, but fit has one; no blanket harmless-component claim follows.

Both report and plan correctly exclude retrained-ablation, semantic/physical-identity, recursive and official benchmark claims. Clipping and threshold interactions make component gains non-additive (`RESULT.md:60-80`; `M105_GEOMETRY_COMPONENT_PLAN.md:20-33`). The verified rescue intersections are one per split; full-only-over-neither-component rescues are 11 fit and 4 development; three fit rescues from each individual component are absent from full's rescue set. These support the report's fixed-state observations, not a claimed cause of complete tracking failure or a proven new training method.

## F. Evaluation type — PASS: real_gt

**real_gt**, specifically a GT-labeled fixed-state localization diagnostic. The target boxes are dataset-derived, not generated by either tested model. Cached features and proposals are model outputs, but are predictions/inputs rather than reference truth. This is not synthetic_proxy, self_supervised_proxy, simulation_only or human_eval. It is also not the official three-dataset evaluation; GT boxes do not establish semantic or physical-instance identity (`prepare_train_states.py:69-105`; `m105_completed/acceptance.json:10-11`; `RESULT.md:76-80`).

## Required qualifications and actions

1. Preserve the raw-dataset provenance boundary. This audit verifies the downloaded labels and recorded source-spec chain; do not claim that it freshly rehashed original GT files. An end-to-end GT attestation would require that separate read-only comparison.
2. Describe zero-delta/frozen-state success as source-bound completed-runtime evidence. Do not call this reviewer pass an independent GPU rerun or a reconstruction of all unselected candidate boxes.
3. When reusing the plan's GT statement, say **used by overlap after decoding, excluded from forward/selection**. Its current literal read-order wording is inaccurate; no leakage path was found.
4. Keep the failed healthy gate and the current claim ceiling. The next preservation-training proposal is a hypothesis requiring its own fit-defined experiment. No M105 rerun or source correction is requested by this audit.
5. Keep `same-family / provisional` on the semantic audit and save the executor-managed response/meta trace. Deterministic evidence may separately remain `accepted`.

## Audited input SHA256 manifest

Paths are relative to this diagnostics directory; `../` entries are adjacent source specifications. Both checkpoint files were hashed as bytes without deserialization.

| Input | SHA256 |
| --- | --- |
| `diagnose_learned_geometry_components.py` | `26ae8344efd95549d6710993f6b18a88f4cf7dce83318b8110dfa4309a465ce5` |
| `run_m105_components.py` | `266cd99afa9415cafbe3b141731746d2c4c47614907d5bd3533f6dfa34cc3e5c` |
| `accept_m105_components.py` | `f2cf9a5277a745ad9a8d14512b06ae09a332d19b4493352ca700f6ce99c4ded9` |
| `train_local_visual_geometry.py` | `bfb63d22a4e3d2d5f86475854f31210000ebea3d14ce26b40b340b4695e562b8` |
| `train_ab_visual_control.py` | `b1f5c1a458c380eae481b5f4ebccdc619524b8eb8a275ceb5d25fc7a24f49323` |
| `instance_ab_prototype.py` | `09db6833f161b6028e64905f2e73a69027e0caf28c611b6c9d81a9fa6862d9fb` |
| `analyze_train_states.py` | `2439a8d0cdb7a7d6c35bbf68aaacd28850bfb8db2211f07eb9e9e7dac66b6def` |
| `prepare_train_states.py` | `cceb53710cdbde8350000c25d6f0ed25b490e4084022b37c76062315ad641c79` |
| `M105_GEOMETRY_COMPONENT_PLAN.md` | `10a21bc50032f1ed8f1bfba0273da401ad1a1f3765f8f928a1b86ff24ccd790a` |
| `M105_GEOMETRY_COMPONENT_REVIEW.md` | `6427c6e4520bba36f9c6b449cedaa15b542f3ca2829ce653dfdbdeb9040ea4d2` |
| `M105_GEOMETRY_COMPONENT_RECHECK.md` | `5eb9e771ac8642bf3b32ea7dfadb81bb8196768486758b498d233adff2fba644` |
| `M105_RUNNER_REVIEW.md` | `16ef49925aba0d70d3230beef36ad110fae1b7dfc6dbfbb9d3cd850e9a0ea11f` |
| `M105_DEPLOY_RECEIPT.json` | `e151059d985aed89969fd17f9de2e6157fc2c3684b6da917653f9f7dafe1a62d` |
| `m105_completed/driver.json` | `386c10938632843b8ae10d8528a2b88db10a885fe1f9d27d9e0b1b35e9c79205` |
| `m105_completed/launch.json` | `213442f240a4a5d025c3c9d215101f24ddc3aadec5c4273fc8b50a8f2aae57de` |
| `m105_completed/launch_receipt.json` | `143cb2cde9655abc606050beb0f5a0cac8be79d459b4bdc952710e0709457f4d` |
| `m105_completed/download_receipt.json` | `d11ceff6a5c9a2d480e035139c08434e4cf3f453e131c1f118ee5c60ecc773bc` |
| `m105_completed/acceptance.json` | `c8623e76fb138a6cf5ecbbf27f33e72d21249a5e9ca69de36a48e72faee840ff` |
| `m105_completed/driver.exit` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `m105_completed/fit.exit` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `m105_completed/development.exit` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `m105_completed/controller.log` | `1d41c7a9b4069eaeff6e7496a18a9b863063a6f051891bd677e76eb4dbd5e5ad` |
| `m105_completed/fit.log` | `d32ad8cf9f47c40c788cc7a2d9cddc31c41b15c1a918f3579994ce2c0ea82da9` |
| `m105_completed/development.log` | `b24526f80544ebac560eb687aebeac207020fdb7ff115f85cfdc100a4fdf54e2` |
| `m105_completed/fit/result.json` | `d1ad44ee36bdf7d835910e4635b98d91c7a055107967c1f8c4c5d951f4a53053` |
| `m105_completed/fit/events.jsonl` | `de10c0fcab2cadafcd75f1108af1bfdec232b250e3d2977f001d8166013874b5` |
| `m105_completed/development/result.json` | `e36a8ac1aef140e1ed5da535d6c44ff937ad9d89cfba991745b012206ad7758a` |
| `m105_completed/development/events.jsonl` | `b9fe5eca45e20b567517aa7239fbc4219d220f81add2583ecb98c7d43b864217` |
| `m104_completed/train/result.json` | `315ff9ae562e71878d36e8fe84b04343a285f5c295e0f62cce3b449e87f14a7f` |
| `m104_completed/train/fit_events.jsonl` | `0cf535db21f0baf6933686936e8acc129a2b7424c5ef08c230fe57b951a6676f` |
| `m104_completed/train/development_events.jsonl` | `c339e6ec301e16a630501ae5f7a07cc62ef6da604967bcb40e3b00b63a5bf136` |
| `m105_inputs/preparation.json` | `59cedd45335583228f59451b818d9a4a5c06c3c67295bf2796c8bdefa18539aa` |
| `m105_inputs/training_labels.json` | `433226372eb045b0dede646ec1fa016506390728724ceaaf194701bb1b213a0f` |
| `m105_inputs/download_receipt.json` | `86f20ffbabf072c6655b1ce5e71d021c35b850d24b4eb6393b805e65e8cf1792` |
| `collect_ab_interface_panel.py` | `361559ef53623dc53665534243a0b4cfb078821dc852e20b34a75d14dca45086` |
| `train_fixed_visual_selector.py` | `a8c644a8ca2201c6c36a67c7146725a08ee8a9624fd9ef111da9ca596a930784` |
| `M105_GEOMETRY_COMPONENT_REVIEW_20260928_135551.md` | `6427c6e4520bba36f9c6b449cedaa15b542f3ca2829ce653dfdbdeb9040ea4d2` |
| `M105_GEOMETRY_COMPONENT_RECHECK_20260928_135901.md` | `5eb9e771ac8642bf3b32ea7dfadb81bb8196768486758b498d233adff2fba644` |
| `m105_completed/RESULT.md` | `9eb550fd92c9b3754619476430f4842ee62c39d1d553a4a876c8ed55b73e917a` |
| `m104_completed/train/final.pt` | `546bd218a14906beb28033717bb53cc17721e9bb6b3fb53f36a4bac15dff9f6e` |
| `m101_completed/train_weight1/final.pt` | `1d187d78c1dd76ebf59e41fd31af16fa69cecc37b4cb6c74fbdfe3ed5c237293` |
| `m101_completed/train_weight1/result.json` | `92d7df716aa369b5ba85f71ac0cc07e943c0a4c5cd7e4788474bde68991024d6` |
| `../full152_paired_20260925/M82_training_spec.json` | `3569d9c42b8db332281ad70bec719650b876e8b9003c48ba73b17d1451e4b425` |
| `../m82_native_preservation/preflight/training_spec.json` | `6163ef5b21f897ca9819c6358b03682c1278bc1b37e5ac93146ff47d332dd700` |
| `collect_train_states.py` | `3961c894e42411bb0cbb2eb245e0162b8765a742b08d5cd72ee9d8dfaa6618f3` |
