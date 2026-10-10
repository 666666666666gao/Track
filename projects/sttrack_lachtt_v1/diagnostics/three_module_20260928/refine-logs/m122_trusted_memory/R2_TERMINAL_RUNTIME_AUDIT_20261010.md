# R2 terminal teacher experiment integrity audit

Date: 2026-10-10. Fresh reviewer: `/root/m122_r2_terminal_runtime_integrity_20261010`. Requested route: `gpt-6-astra`, reasoning `max`; actual backend and actual reasoning effort are **UNATTESTED**. Review independence is **same-family**, acceptance **provisional**, with separately reproducible deterministic checks.

**Overall verdict: WARN. Blocking findings: 0. The scoped R2 terminal raw-artifact audit is complete.**

This verdict covers the completed, original fixed-P1 Train152 teacher and the supplied compact evidence together with its actual CPU readback. It does not certify C training, changed-policy histories, new benchmark performance, a new deployment, or the full research goal. It does not claim that this reviewer inspected full original template/decoder PT tensors or reconstructed model features.

Paths below use `A = ../R2_terminal_actual`, `S = ../../../projects/sttrack_lachtt_v1/diagnostics/three_module_20260928`, both relative to this report's directory; source references also have absolute paths in the machine report. Audit-output filenames are relative to this report's directory.

## Independently verified result

All 227 listed inputs, totaling 319,741,745 bytes, match their recorded SHA256 and size. The actual ZIP has 195 payload entries plus its `files.json`; all 196 archive members match the extracted local bytes. Every listed JSON was parsed, every source was read, every dataset GT file was loaded, and every compact event/reference and native prefix row was processed. One supplemental source, `run_template_write_pilot.py`, was read to inspect the actual imported `context`, GT-validity, and IoU definitions; it matches the launch gate and has its own before/after hash record.

| Quantity | Verified count |
|---|---:|
| Train sequences / seed | 152 / 2027 |
| Noninitialization native-prefix records | 219,802 |
| Native interval50 / response > 0.75 events | 1,297 |
| Collection/replay events, shards 0 / 1 | 701 / 596 |
| Future W/K frame pairs | 40,954 |
| Future branch tracking records | 81,908 |
| Future valid / unknown GT rows | 39,297 / 1,657 |
| Event signed means: positive / negative / zero / unavailable | 607 / 552 / 119 / 19 |
| Common eligible / excluded events | 1,257 / 40 |
| Common fit120 / development32 events | 993 / 264 |
| Short-tail events / short tails still eligible | 27 / 23 |
| Empty future-image tails | 4 |
| Sequences with zero native write events | 10 |

All 1,297 current labels and all 40,954 future W/K label pairs reproduce from actual dataset GT and supplied scalar predictions. The maximum discrepancy against both replay JSON and the CPU result is **0** for current IoU, W IoU, K IoU and signed event means. No positive-only selection is present. The 40 exclusions comprise 35 current-GT-unknown events and 5 additional events with no usable future GT. There are 19 events with no usable future GT overall; 14 overlap the current-GT-unknown group. Unknown GT remains unknown, rather than zero-IoU negative supervision. Evidence: `A/CPU_v3/readback/result.json:25`, `DETERMINISTIC_RESULT.json:3`, `INDEPENDENT_EVENT_RECOMPUTATION.json:1`, and all rows of `INDEPENDENT_FUTURE_GT_RECOMPUTATION.jsonl`.

## A. Ground-truth provenance — PASS

`S/train_m122_full_causal.py:14` reads `groundtruth.txt`, verifies the manifest SHA, checks frame counts and first-frame box, and preserves the explicit `toy07_indoor_320` exception: 1,406 annotation rows, first 1,367 paired image frames used. The independent reader loaded all 152 GT files and verified their individual SHA/size, raw row counts and initialization boxes. The resulting used total is 219,954 rows, including one initialization row per sequence.

`S/run_template_write_teacher.py:37` initializes from the legal first frame/box and selects events using actual native writes. Collection has no current/future GT query. In replay, W/K trajectories are completed at lines 129–138; dataset GT first enters label calculation at line 140. The actual imported validity/IoU implementations are `S/run_template_write_pilot.py:104` and `:108`. They use finite dataset rectangles with positive width/height. Current/future IoUs are genuine dataset-GT metrics; signed future utility is the mean difference of those GT IoUs over the common valid future-frame set.

These training teacher labels use explicit rectangle IoU and are not presented as official DepthTrack, CDTB or VOT benchmark scores. Official benchmark protocol execution is outside the terminal teacher claim.

Human text is initialization evidence, not a substitute tracking ground truth. All 152 token/mask rows, every reference's words/mask/empty vector, and every initial box agree exactly with the fixed local `human_text.pt`, labels, sequence lookup and GT first box. The 10 zero-event sequences also have checked initialization references. The labels' human-confirmation declaration was not recreated. The fixed bank's historical exception is disclosed under N3.

Evidence: `S/full_dense_tracker.py:75`, `S/run_template_write_teacher.py:38`, `:119`, `:140`, `:146`; `A/inputs/human_train_labels.json:2`; `A/CPU_v3/readback/result.json:19520`; `INDEPENDENT_GT_READBACK.json:1`.

## B. Score normalization — PASS

IoU divides geometric intersection by geometric union. No denominator uses the model's own maximum, minimum or mean score. `delta_future` is the unnormalized signed arithmetic mean of `IoU_W - IoU_K` over valid dataset GT frames; negatives and exact zeros are retained. The independent scalar implementation reproduces every recorded value exactly.

The native event threshold applies to the original same-position native response, using interval 50 and threshold 0.75. `selected_quality`, observation-window-intersection output and native response are deployment features, not claimed benchmark accuracy. High equality rates for replay/control witnesses are equality proxies and are not interpreted as near-100% tracking performance.

Evidence: `S/run_template_write_pilot.py:108`; `S/run_template_write_teacher.py:145`, `:149`; `S/full_dense_tracker.py:104`; `S/template_write_features.py:35`; `verify_actual_evidence.py:27`; `DETERMINISTIC_RESULT.json` maximum-error fields.

## C. Result existence and numerical claims — PASS within the stated scope

The aggregate reports the completed original teacher at `A/teacher/result.json:2`, with counts at lines 165–174. Actual controller, collect0/1 and replay0/1 exits are 0. Stage receipts equal the aggregate's pipeline entries and preserve collection-before-replay on each physical GPU. Both shard results, both full prefixes and the aggregate independently hash to the digests below. Collection/replay sequence lists cover each of the 152 indices exactly once and in the specified parity assignment.

The original CPU verification genuinely failed with exit 1. Its traceback ends in `TypeError: Object of type bool_ is not JSON serializable`, at final JSON serialization after computation. The supplied v3 source differs only by wrapping the validity result in Python `bool(...)`; strict JSON, label definitions, tolerance and all checks remain unchanged. The corrected runtime has actual execution/exit/log/result, original stdout and terminal tool receipts. Its exit is 0 and its result SHA is `e4607b1ce6b09a664890f416397a188eabe9c6c55315dd4b703af8a65ca1d5b2`. This audit independently recomputed the critical outputs instead of treating that exit code as the verdict.

Evidence: `../R2_CPU_failure_evidence_actual_20261010/CPU/execution.json:3`; its `CPU_readback.log:4` and `:22`; `S/verify_template_write_teacher_raw_v3.py:43`; `A/CPU_v3/execution.json:3`; `A/CPU_v3/CPU_readback.exit:1`; `A/CPU_v3/CPU_readback.log:1`; `../R2_CPU_v3_actual_join_01_20261010.json:1`; `CHECK_RUN_01.exit:1`.

The gate's complete 77-entry inventory is exactly reproduced in the actual CPU readback, whose source directly rehashes those files at `S/verify_template_write_teacher_raw_v3.py:79`. Of these 77 entries, 19 have matching bytes directly available among this review's inputs and supplemental source; the other 58 are supported by that actual runtime receipt. `GATE_BINDING_SCOPE.json` records the distinction. The reported native, CLIP and decoder state digests agree before/after collection and replay and across both shards; these are persisted runtime digest witnesses, not locally recomputed full-model digests. Evidence: `A/CPU_v3/readback/result.json:21650`; `A/teacher/teacher_shard0/replay/result.json:173183`; shard1 `:147413`.

The immutable plan's original “pending” wording and original source audits remain historical snapshots. The supplied terminal document closeout explicitly kept `R2_terminal_raw_audit_complete=false`, R3 unexecuted and no new formal metrics before this review (`../R2_terminal_CPU_actual_document_closeout_20261010.json:53`). This report adds the fresh, scoped verdict without editing historical records.

## D. Dead metric code / executed-path evidence — PASS

The R2 controller constructs the actual collect/replay argv, requires successful child exits and aggregates their results (`S/m122_R2_teacher_controller_20261010.py:22`, `:61`). The teacher `main` dispatches those phases at `S/run_template_write_teacher.py:186`; replay calls GT loading and IoU after the two branch rollouts at lines 140–155. Their outputs exist for every event and independently reproduce exactly. Thus the terminal metrics are not merely unused helper definitions.

The CPU v3 `main` is actually called at line 334; its result contains all 1,297 event summaries, 152 GT summaries and 2,905 hashed raw-file records. The compact exporter reads the bound original payload/rollout PT files, checks their SHAs, and exports actual values/records at `../R2_CPU_v3_compact_query_v2_raw_evidence_remote_20261010.py:31`. Its real archive and exit-zero packaging receipt are supplied.

The separate C preflight, C fit, recursive development inference and development analyzer are prospective consumers. They were read but neither imported nor executed here. Their existence does not establish completed C training or evaluation. The C consumer's audit gate is at `S/train_template_write_C.py:111`; the common eligible sample definition is at `:65`; the recursive development analyzer's explicitly limited metric scope is at `S/analyze_template_write_C_dev.py:169`. No unused prospective metric is presented as a completed result.

## E. Experimental scope — WARN, nonblocking

The completed teacher is full **DepthTrack Train152 under one fixed P1 and seed 2027**, with every native-qualified opportunity on that original history retained. A+B previously trained on all 152 sequences. The 120/32 split is a C-optimizer holdout only; it is not an unseen whole-model validation set. The reviewer independently reproduced SHA ranking and every sequence's assignment. Evidence: `A/inputs/C_split.json:10`, `:11`, `:12`; immutable `S/refine-logs/m122_trusted_memory/EXPERIMENT_PLAN_20261008_235856.md:26`.

The teacher's 32-frame action utility does not establish a learned C benefit, longer-horizon benefit, identity recognition, crop-out recovery, language contribution or public-benchmark performance. In particular, these labels belong to original teacher states; they must not be copied as labels onto changed C-policy histories. The immutable plan separately requires recursive development, new C-history collection, final full152 C fitting and one final model's three-dataset evaluation at lines 62–72. Those claims remain unsupported by the present runtime.

## F. Evaluation-type classification — PASS

| Evidence | Classification | Claim ceiling |
|---|---|---|
| Current IoU, future W/K IoUs, signed W−K event means | `real_gt` | Dataset-GT label correctness on original teacher states |
| W-prefix equality, frozen digest equality, stored historical K/K witnesses | `synthetic_proxy` (replay/control consistency) | Correctness/equality witnesses; no accuracy claim |
| Human text labels / fixed token bank | Human-provided initialization provenance | No newly conducted `human_eval` of tracking outcomes |
| Prospective C development metrics | Intended `real_gt`, unexecuted here | No new C result |

Evidence: `S/run_template_write_teacher.py:135`, `:140`, `:161`; `S/template_write_forks.py:128`; `S/analyze_template_write_C_dev.py:140`. There is no simulation or self-supervised performance score in the audited terminal labels.

## Raw trajectory and feature checks

All 219,802 native prefix records have the expected sequence/frame order, finite valid boxes and exact own-history `previous_bbox` continuity from the legal initialization box. The native-write Boolean equals `frame % 50 == 0 and native_same_position_response > .75` at every frame. The resulting complete event list equals collection, replay and compact records, without missing, duplicate or additional opportunities.

All 81,908 future branch records follow their own prior predicted box. All 40,954 W records equal the full native prefix record exactly. Both branches have no future template write in their at-most-32-frame window. All 163,816 query hash descriptors have two `(1,4,768)` float32 tensors per branch row; every selected-feature hash descriptor has `(1,128)` shape. W/K future feature and query hashes differ on all 40,954 pairs; this is a descriptive stored-hash observation, not proof of model reconstruction or performance. The full tensors for those descriptors were checked by the actual CPU readback, not loaded locally in this review.

The reviewer verified 4,651 actual-value tensor hashes: 5 reference tensors × 152 sequences plus 3 event tensors × 1,297 events. Every 519-value float32 input has the exact order **current128 + past128 + initial256 + quality3 + motion4**. The first 515 values agree byte-for-byte with the independently reconstructed concatenation. Current/past feature values and initial-reference values have the declared shapes and finite values, and their SHA256 matches their actual float32 bytes. The four motion values recomputed from recorded boxes differ by at most **2.9802322387695312e−8**, within the already prescribed CPU/GPU `rtol=atol=1e−6`; no tolerance was changed.

The bound bank's all152 initialization semantics, sequence mapping, phrase masks, empty slots, empty embedding and legal first boxes are checked, including zero-event sequences. Initial-reference values are verified as stored, not re-derived through the decoder's learned projection. Cached past-feature membership is checked against the cached value and source capture path; unsaved full-prefix feature history cannot be independently reconstructed. Evidence: `S/template_write_features.py:12`, `:29`; `S/run_template_write_teacher.py:47`; `S/full_dense_tracker.py:95`; `A/compact/all_event_scalar_records.jsonl:1`; `A/compact/all_shared_reference_semantics.jsonl:1`; `DETERMINISTIC_RESULT.json`.

## Nonblocking findings and proof limits

**N1 — Compact evidence and original-runtime witnesses.** Full original template/cache/decoder PT files, RGB/depth image arrays and full model checkpoints are not supplied locally. Only the human text bank was loaded as a local PT file. The actual exit-zero CPU v3 run read and checked the 2,746 original payload/rollout/reference PT files, as evidenced by its checked source, result and all 2,905 raw-file SHA records. This reviewer directly checks the exported scalar trajectories, C-input/reference values and GT and joins them to those SHA bindings. Nineteen gate entries also have local bytes; 58 are receipt-only. Prefixes contain no full-frame query/feature tensors, so neither W-query/feature equality with an uninterrupted prefix nor the past-feature's neural origin is independently established. These are declared limits on this **scoped terminal stored-artifact audit**, not missing evidence for the exact scalar/GT checks that passed. Full local PT or model-feature certification would require additional artifacts and a separate scope.

**N2 — Original history and C-only holdout.** The completed teacher is not a completed learned controller or final benchmark evaluation. Keep the explicit C-optimizer-only 32-sequence qualifier and re-collect labels on changed C histories as prescribed. No training, NN calls, GPU queries, SSH/SCP, package installation, timers, or source/input modifications were performed by this reviewer.

**N3 — Fixed human-bank exception and historical spec.** `A/inputs/human_train_labels.json:2186` retains `pine02_wild_320` with five semicolon clauses in one attribute slot at line 2191 and the historical combination `human_status=conflicting`, `human_confirmed=true` at lines 2195–2196. The same bank is actually used. This audit certifies byte/semantic binding, not fresh human adjudication or a uniform one-attribute-per-slot protocol. The reused `training_spec.json` still names M67/M82 at line 2 and the old generated-text protocol at line 2920. The active context binds actual P1/human-bank inputs explicitly (`S/run_template_write_pilot.py:25`). Neither immutable artifact was silently rewritten.

## Claim impact and machine-consumable decision

- Completed original P1 Train152 teacher, all native-qualified opportunities, real-GT labels and stored 519-value C inputs: **supported within the declared artifact scope**.
- Uninterrupted W **record** equality, branch bbox continuity and no intervening write: **supported**.
- Full local template/decoder/model-feature reconstruction or complete prefix query/feature equality: **not certified**.
- Learned C benefit, changed-history utility labels, unseen whole-model validation, new final nine metrics or goal completion: **unsupported by this audit**.

The supplied proof is sufficient for the scoped terminal teacher raw-artifact audit because all critical scalar/GT/input checks are independently reproducible, the complete opportunity accounting matches, and the omitted raw-PT inspections have a concrete, source-bound, successful original CPU readback. No contradiction or missing critical evidence was found within that scope. Therefore the report sets `review_call_status=completed`, `blocking_count=0` and `R2_terminal_raw_audit_complete=true`, while leaving all C/performance/model-reconstruction certifications false. This does not replace a separate source/deployment admission for the next stage.

Verified SHA256 values:

| Artifact | SHA256 |
|---|---|
| Teacher aggregate | `24c8461b4d71c573fbaa4bf4f91c4e359cca1614c7fe81246ac0504a325c836c` |
| Replay shard 0 | `831c988ac9d0ec578072b11d9faac4040c5733bc4ccfda5e32c4150c073a2627` |
| Replay shard 1 | `4286f72adc0d5eef950d134c763cd5399585c4c48c4f94db51033f8f409b5068` |
| Prefix shard 0 | `778590127e7ff48ace7c8c42d752f4e154fbd349e7828e6441fefcda59af1f9a` |
| Prefix shard 1 | `72178d2f15da48b804fee6c512f96138d495e85d4d1e8679c4fa1427f3b970f1` |

Reproduction: run `E:/anaconda/python.exe verify_actual_evidence.py` with assertions enabled. The actual independent run returned **exit 0** (`CHECK_RUN_01.exit`). `AUDITOR_CHECK_NOTES.md` preserves the one unrelated failed plan-decoding probe and its correction. Before/after input inventories, full deterministic outputs, private request/response trace and `AUDIT_SEAL.json` accompany the report. Historical inputs and historical audits are unchanged.
