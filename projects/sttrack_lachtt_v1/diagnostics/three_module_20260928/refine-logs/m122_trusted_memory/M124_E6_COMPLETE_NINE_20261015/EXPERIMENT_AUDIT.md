**Experiment integrity audit — M124-E6**

Date: 2026-10-15. Overall verdict: **WARN**. Integrity status: **warn**. Deterministic captured-evidence checks: **PASS, 35,167 checks, zero failures**. There are **zero confirmed blockers to reporting the qualified captured E6 results**. This does not certify raw-trajectory metric reproduction, untouched-test generalization, or live model execution.

Auditor: fresh delegated Codex reviewer `/root/audit_m124_e6_full_suite`. Requested reviewer: `gpt-6-astra`, reasoning `max`. The tool-visible record does not attest the actual model route or reasoning setting; both are **UNATTESTED**. `review_independence=same-family`; `acceptance_status=provisional`. No NN execution, SSH, installation, additional agents, or modification of supplied inputs was used.

The 14 supplied primary files were read directly. The ten supplied Python scripts were inspected line by line and matched to their captured frozen copies. All 7,288 primary captured entries, totaling 20,780,756 original UTF-8 bytes, passed SHA-256 and byte-count checks. The compact observation contains 62 exact file captures: 27 duplicate primary entries agree exactly, and 35 add distinct captured files. Its four file-count summaries and two reduced training reports are summaries, not complete original payloads. All 142 bundle source seals and all 7,073 execution source seals match the captured bytes. The hash inventory contains 7,338 entries: the 14 supplied primary files, the assignment, and 7,323 distinct captured originals.

Remote paths below identify the original files stored in the primary JSON's `files[path].text`; they do not imply access to live remote files. Evidence from the single-line compact JSON is identified by its JSON member.

| Check | Status | Finding |
|---|---|---|
| A. Ground-truth provenance | WARN | Dataset GT and legal initialization paths are evident; raw OPE inputs and the text-bank payload are incomplete in this capture. |
| B. Score normalization | PASS | Inspected formulas use protocol denominators; no metric is normalized by its own maximum score. |
| C. Result existence and consistency | WARN | E6 values, bindings, extents, and terminal receipts agree; the historical baseline's primary result and raw prediction payloads are absent. |
| D. Metric execution paths | PASS | Every reported metric has a connected call-and-write path and completed-stage evidence. |
| E. Scope and claims | WARN | One seed, test-informed checkpoint selection, multi-frame-assisted text, and the inherited source-label error constrain claims. |
| F. Evaluation classification | PASS | All three evaluations are `real_gt`, under the disclosed protocols. |

**Verified reported results**

| Dataset | Metric order | First | Second | Third |
|---|---|---:|---:|---:|
| DepthTrack Test | P / R / F, percent | 69.646390 | 66.070900 | 67.811546 |
| CDTB | P / R / F, percent | 74.237275 | 69.959378 | 72.034870 |
| VOT-RGBD2022 | EAO / ACC / ROB, percent | 75.756244 | 82.151385 | 92.127839 |

All three rows record the same selected E6 final SHA-256, `4d5c7e411a047fa996835647d1efa3bb94f4184fdc3a592380523be8c04156eb`, the same evaluation bundle `64587cf1169266815bc3e609d1b1be48f6b2d563cb135ca9ee6fdeb9595f692f`, text bank `78cbae28b4b63ecab768d747a889ca120c62e1368ce80fa7f798ee6508bc29a9`, and initialization binding `3bf7aa31a3f819ec746b37dc221e55c531a1235726cda3efedc4990c607cb26e`. The selected-weight digest is also recorded by the terminal collector. Original weight, bank, and binding bytes were not supplied for independent local rehashing or tensor inspection.

The independently checked project predicates give **5/9 passed**: all three DepthTrack metrics, CDTB precision, and VOT accuracy. CDTB recall/F and VOT EAO/robustness fail; the joint target remains unmet. Evidence: `/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/epochs/epoch-06/evaluation/all_results.json:26`, `:68`, and `:103`; `/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/code/collect_m124_epochwise_results.py:40`. The CSV and draft roundings agree with these original values.

**A. Ground-truth provenance — WARN**

DepthTrack/CDTB scoring loads `groundtruth.txt` from the dataset root and the predicted boxes/confidences from separate output files. Invalid/absent GT boxes become VOT special regions; predicted boxes do not generate the reference. Evidence: `/home/SRTrack_RGBD_L/lib/test/analysis/depthtrack_pr.py:40`, `:95`, and `:111`. The captured plans name `/root/autodl-tmp/depthtrack/test/sequences` and `/root/autodl-tmp/CDTB/sequences`, with complete case lists and GT hashes. All 130 case-to-GT-metadata bindings agree. Six original OPE GT payloads, three per dataset, are present and match their hashes, frame counts, and initialization boxes; the remaining OPE GT payloads and image bytes are not supplied.

The OPE inference loop initializes from the recorded first box, then calls `actor.track(image)`. GT file verification and scoring occur after the predictions and receipt have been written. Evidence: `/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/code/run_m124_epochwise_official.py:22`, `:38`, and `:45`. The active tracker uses its previous predicted box and committed state; the reviewed path has no post-initialization GT argument. Evidence: `/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/code/trusted_template_tracker.py:26` and `:44`.

VOT legitimately initializes separately at each official forward/backward anchor, then supplies image frames to runtime updates. GT is used by the evaluator for overlap and failure analysis. Evidence: `/root/miniconda3/envs/mplt/lib/python3.8/site-packages/vot/experiment/multistart.py:99`, `:112`, and `:116`; `/root/miniconda3/envs/mplt/lib/python3.8/site-packages/vot/analysis/multistart.py:100` and `:115`. All 127 original anchor files were verified. The 462 captured shard GT copies cover all 127 sequences and match the collector's corresponding original-GT hashes and lengths. This establishes internal capture consistency, not independent authentication against a newly downloaded dataset distribution.

The input protocol includes a fixed human text bank. Its preparation explicitly records multi-frame/video assistance, and its runtime binding depends on initialization RGB bytes and the legal bbox. Evidence: `/home/SUTrack_RGBD_L/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/prepare_m122_external_human_labels.py:31` and `:43`; `/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/code/bind_m122_official_initializations.py:28` and `:49`; `/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/code/m124_epochwise_official_runtime.py:52` and `:72`. Fixed online text does not establish that text creation used only an initialization frame. Human review correctness and actual encoded bank contents were not independently checked.

**B. Score normalization — PASS**

The locked OPE adapter computes per-sequence precision as the mean overlap among confidence-selected predictions, recall as their overlap sum divided by the number of visible GT frames, and then macro-averages the curves. F is their harmonic mean at the curve's maximum-F operating point. Confidence-derived thresholds select points on the PR curve; they are not a denominator that rescales performance toward 100%. Evidence: `/home/SRTrack_RGBD_L/lib/test/analysis/depthtrack_pr.py:25`, `:117`, and `:133`. The corresponding installed VOT long-term implementation uses the same basic PR denominators and best-F reporting mechanism: `/root/miniconda3/envs/mplt/lib/python3.8/site-packages/vot/analysis/longterm.py:65`, `:73`, and `:370`. OPE is this locked project adapter using VOT overlap primitives, not a direct invocation of the VOT long-term CLI.

VOT ACC uses overlap accumulated before confirmed failure divided by accumulated progress; ROB uses protocol progress/run-length weighting. EAO averages the official overlap curve over the configured slice. Evidence: `/root/miniconda3/envs/mplt/lib/python3.8/site-packages/vot/analysis/multistart.py:120`, `:129`, `:175`, and `:442`. Conversion to percentages multiplies by 100 only. The model's internal confidence calibration at `/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/code/dense_target_decoder.py:111` is not an evaluation-metric normalization.

Independent arithmetic reproduced both OPE F values from the saved P/R values, VOT EAO `0.7575624436259986` from the saved curve, and VOT ROB `0.9212783857800407` from all saved per-anchor progress records with the official sequence weighting. For EAO, the captured toolkit code's slice `[115:756]` contains 640 entries in the saved 755-entry curve. This reproduces that implementation exactly; it is not a new overlap computation. ACC was checked against the official saved scalar, not reconstructed from raw overlaps.

**C. Result existence and consistency — WARN**

The original E6 OPE metrics, VOT analysis/result, selection, merged-output hash inventory, CSV, controller result, and terminal receipts are present. Every supplied local source matches the frozen counterpart. Cross-file plan, bundle, receipt, result, analysis, and manifest hashes checked by the verifier agree. Evidence: `/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/epochs/epoch-06/evaluation/depthtrack_test/predictions/metrics.json:2`; the corresponding CDTB file `:2`; `/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/epochs/epoch-06/evaluation/vot/result.json:2` and `:16419`; `/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/epochs/epoch-06/evaluation/selection.json:3`.

The 53 recorded controller stages have zero exits. All four VOT shards record zero exits and 441/441/441/442 completed anchors, and the merge inventories exactly the expected 5,295 trajectory/confidence/time paths. Official analysis, sealing, collection, and the supervisor all record completion. Evidence: `/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/epochs/epoch-06/evaluation/vot/run/launch.json:16`, `:34`, `:52`, and `:70`; `/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/best_vot_official_analysis.receipt.json:20`; `/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/collect_best.receipt.json:13`; `/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/controller.supervisor.terminal.json:13`. The supervisor finished at **2026-10-15 10:15:22.578496 UTC**. These are captured process receipts, not a new local observation of the original processes.

The 1,765 saved failure outcomes form the exact anchor cover and sum to **231 confirmed failures**; per-sequence counts agree. The raw `.bin`, confidence, time, and OPE prediction files are not in this capture. Their recorded hashes and the original source's checks do not let this reviewer rehash or recompute them. The full E6 training result, full training logs, checkpoint bytes, text-bank/binding bytes, base-plan payloads, and source-gate payload are also absent; references to their hashes remain recorded provenance rather than independently verified payloads. The compact E7/E8 training reports explicitly omit `sequence_records`.

The draft's historical STTrack VOT baseline `77.321654/82.471190/93.669109` has no primary result in the supplied files. Its asserted ordering relative to E6 is arithmetically true if those reference values are accepted, but this audit does not authenticate the baseline or its protocol. Attach that original result before describing the comparison as audited. This absence does not invalidate the internally consistent E6 row.

**D. Metric execution paths — PASS**

The controller calls OPE tracking and analysis for each dataset. `ope_analyze` calls the hash-locked `evaluate_depthtrack_results` function and writes `metrics.json`; the collector reads those same files. Evidence: `/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/code/run_m124_epochwise_suite.py:41` and `:48`; `/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/code/run_m124_epochwise_official.py:56`; `/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/code/collect_m124_epochwise_results.py:15`.

The controller invokes `python -m vot analysis`, followed by `analyze_m124_epochwise_vot.py` and the collector. The sealer extracts EAO/ACC/ROB and calls `collect_confirmed_failure_outcomes`; the saved official analysis explicitly names the multistart EAO and AR classes. Evidence: `/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/code/run_m124_epochwise_suite.py:88`; `/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/code/analyze_m124_epochwise_vot.py:22`; `/home/SUTrack_RGBD_L/tools/finalize_vot_transaction_low22.py:193`; `/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/epochs/epoch-06/evaluation/vot/run/master/analysis/ABC_joint_epoch06_full127.json:666`.

No reported score depends on a metric function that is merely defined and unused. The unused `format_metrics` convenience formatter and other installed toolkit analyses are not presented as executed results. Invocation/receipt evidence is distinct from reviewer execution of those metric functions.

**E. Scope and claims — WARN**

The selected model's complete captured scope is DepthTrack Test **50 sequences / 76,373 frame slots**, CDTB **80 / 101,956**, and VOT-RGBD2022 **127 / 80,741 source frames**, with **1,765 anchors: 867 forward and 898 backward**. The source-derived VOT proxy-frame total is **1,327,004**, including each initialization slot. This is a protocol extent, not a measured NN-forward or latency count. The exact original-anchor/shard cover, shard anchor masks, dataset sequence sets, and merged path inventory all agree.

There is one training seed, **2027**. Eight sequential epoch checkpoints are not eight independent runs. The captured cumulative E8 summary records **1,216 sequence runs, 1,758,416 track calls, and 54,456 optimizer steps applying to A+B and C**; these totals are arithmetically consistent with the source and E7/E8 summaries. A complete raw-log recount is unavailable. Evidence: `/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/code/train_m124_joint_epoch.py:37`, `:199`, and `:214`; `M124_best_VOT_seal_completion_compact_observation_20261015.json:1`, members `training_reports["8"]` and `sequence_log_summaries["8"]`.

The earliest maximum of the mean Test50/CDTB80 F scores selects **E6**. Two consecutive dual declines stop after **E8**. I independently recomputed these decisions for all eight history rows. Evidence: `/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/code/m124_epochwise_selection.py:4`; `/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/code/run_m124_epochwise_suite.py:58`; `/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/best.json:2`; `/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/current_training_state.json:16`. Zero external optimizer steps does not make those datasets untouched tests. The VOT score was produced after this E6 selection, but the multi-frame-assisted human input still precludes an unqualified automatic, initialization-frame-only benchmark comparison.

The manifest's `native_source` says **“Exact M39 default wrapper and checkpoint; no experimental head or text”**. That is a stale inherited metadata label, contradicted by the sealed selected entry and source chain. Evidence: `/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/epochs/epoch-06/evaluation/vot/run/shard_manifest.json:3`; `/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/epochs/epoch-06/evaluation/vot/selected_m124_epochwise_vot.py:2`; `/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/code/m124_epochwise_official_runtime.py:61`; `/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/code/full_dense_tracker.py:75`. The inspected composition loads joint A+B/C weights, a frozen native STTrack observation model and CLIP model, and fixed human text; inference runs through `TrustedTemplateTracker`. This is source and binding evidence, not live-tensor attestation. Preserve the original manifest and annotate the label separately.

The first epoch inherited 37 completed sequences and a checkpoint without saved RNG state; a partial sequence was repeated from its first frame. Evidence: `/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/actual_launch.json:17`; `/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/code/train_m124_joint_epoch.py:79`. Do not claim exact equivalence to an uninterrupted seeded run or that the cumulative committed-call count covers all discarded partial work and prior warm-start cost.

This suite establishes a complete selected-model result under its disclosed protocol. It does not establish robustness across seeds, an isolated language benefit, three independently effective modules, future-memory utility, speed superiority, or gains from proposed region/state changes. C is trained on current selected-frame GT IoU, after the action, with detached features; it is not trained on future memory benefit. Evidence: `/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/code/train_m124_joint_epoch.py:125`, `:140`, and `:230`.

**F. Evaluation classification — PASS**

DepthTrack Test, CDTB, and VOT-RGBD2022 are all **`real_gt`** evaluations in the inspected paths. None uses model output as the metric reference. Test-informed selection and the human-text protocol qualify the interpretation of these real-GT results; they do not turn them into synthetic-proxy evaluations. Training C's selected-frame IoU label is also based on dataset GT and is a training target, not an independent evaluation of future utility.

**Claim impact and required actions**

| Claim | Audit impact |
|---|---|
| The captured E6 suite reports these nine numbers with one recorded final and full listed extents | Supported with the capture/rehashing limitations above. |
| Five of nine project predicates pass; the full target is unmet | Supported. |
| The original controller, four VOT shards, analysis, sealer, collector, and supervisor completed | Supported by captured terminal receipts. |
| Test50/CDTB80 were untouched held-out tests, or text used only initialization-frame observation | Unsupported; contradicted by the declared selection/input protocol. |
| The evaluated model was an unmodified M39/no-text tracker | Unsupported; contradicted by the sealed entry/source chain. |
| Every metric was independently reproduced from raw predictions and GT | Unsupported; only saved-value arithmetic and EAO/ROB reaggregation were performed. |
| The historical STTrack comparison was independently authenticated in this audit | Unsupported pending its primary result. |
| Seed robustness, isolated module causality, future utility, efficiency superiority, or proposed-route gains | Unsupported by this suite. |

Required reporting actions are to retain the checkpoint-selection, multi-frame-text, and legacy-handoff disclosures; describe the E6 values as audited captured results; annotate the inherited `native_source` error outside the immutable inputs; and attach or explicitly mark the historical baseline as unverified in this audit. Preserve `independent_completed_audit=false` in the original results and link this separate audit. Full metric reproduction would require the existing raw prediction/confidence/trajectory payloads and complete relevant GT/image metadata; no new NN run is implied. Claims of locally verified weight/text identity would additionally require those original payload bytes.

The deterministic verifier, detailed checks, complete input hashes, full request/response trace, machine-readable verdict, and final output hashes are saved beside this report as `verify_captured_evidence.py`, `deterministic_checks.json`, `audited_input_hashes.json`, `trace/`, `EXPERIMENT_AUDIT.json`, and `OUTPUT_HASHES.json`. Output hashes use a sidecar so neither report requires an impossible self-referential SHA-256.
