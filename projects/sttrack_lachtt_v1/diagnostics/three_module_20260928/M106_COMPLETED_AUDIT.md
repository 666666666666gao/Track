# M106 completed experiment integrity audit
**Verdict: WARN. Independent local deterministic checks: PASS. Scientific qualification: completed negative fixed-cache comparison; no promotion.**
Reviewer: `gpt-6-astra`, reasoning `max`, fresh `fork_turns: none`; `review_independence: same-family`; `acceptance_status: provisional`. Date: 2026-09-28. Task: `/root/m106_completed_integrity`. No fabricated agent UUID. Trace: `.aris/traces/experiment-audit/2026-09-28_m106_completed/`.
References below are relative to `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/` unless absolute. I read the assigned artifacts directly, all 29 files under `m106_completed/`, the later fitting diagnostic and 152 raw GT files. I wrote only this report and its JSON companion. No credentials, server connection, training, Torch import or GPU replay was used.
## A–F findings
### A. Ground-truth provenance: PASS

All 152 supplied raw localization GT files match receipt byte lengths and SHA256s. All 3,502 cached current labels and 3,502 previous labels exactly match their raw rows under the finite/positive-size validity rule; 3,039 current and 2,852 previous are valid. Preparation reads dataset groundtruth.txt, not model predictions. Original official-distribution identity and live remote invocation remain attestations, not independent remote observations.

Evidence: `prepare_train_states.py:69-78`; `prepare_train_states.py:102-105`; `attest_m106_raw_gt.py:16-41`; `M106_RAW_GT_RECEIPT.json:5-18`; `m105_inputs/preparation.json:3-19`; `m105_inputs/training_labels.json:1`; `m106_gt_inputs/cube04_indoor.txt:10-11`; `m106_gt_inputs/pine01_indoor.txt:1276-1277`; `m106_gt_inputs/mobilephone02_indoor.txt:65-66`.
### B. Score normalization: PASS

Reported metrics are raw dataset IoUs, counts at inclusive IoU >= 0.5, and arithmetic means over valid sampled states. No denominator uses a prediction-score maximum/minimum/mean. GT-best/top10 maxima are explicitly oracle coverage, not a deployment metric or normalization divisor. Regression coordinate normalization and feature normalization are not reported evaluation normalization.

Evidence: `analyze_train_states.py:20-25`; `train_ab_visual_control.py:114-128`; `train_selected_geometry_preservation.py:49-52`; `train_selected_geometry_preservation.py:101-115`; `collect_ab_interface_panel.py:52-62`; `m106_completed/summary.csv:1-21`.
### C. Result existence and numerical agreement: PASS

Read all 29 runtime-tree files, including both checkpoints, every row/log/exit/receipt, RESULT.md, acceptance.json and summary.csv. All 25 download-manifest SHA/size bindings, source bindings, log/result histories, four child exit markers, checkpoint hashes, identities/order, 91,170 unique IoUs, 30,390 saved row scalar fields, marginal/joint summaries, gates and parent/control change lists match. Control M104 bytes, six tensor storages, 3,039 ordered rows and summaries are exact. Local arithmetic acceptance does not itself authenticate remote execution.

Evidence: `m106_completed/download_receipt.json:4-152`; `m106_completed/driver.json:5-151`; `m106_completed/controller.log:1-3`; `m106_completed/train_weight0.log:1-13`; `m106_completed/train_weight1.log:1-13`; `m106_completed/train_weight0/result.json:117-288`; `m106_completed/train_weight1/result.json:117-288`; `m106_completed/verification/result.json:1-20`; `m106_completed/acceptance.json:1-12`; `m106_completed/RESULT.md:7-27`.
### D. Metric invocation and source/evaluation integrity: PASS

Active evaluate calls use overlaps, summarize and the four recorded gates; save/reload and replay source invoke the same paths, and serialized outputs agree independently. The added loss is invoked only for weight1, with positive recorded epoch losses. Historical/reference-only modes are not used to manufacture reported metrics. Source inspection and consistent invocation receipts do not replace an auditor GPU replay or seal every imported runtime dependency. The clamp_min equality qualification is recorded separately.

Evidence: `train_selected_geometry_preservation.py:87-116`; `train_selected_geometry_preservation.py:181-214`; `train_selected_geometry_preservation.py:230-247`; `verify_m106_checkpoint_replay.py:54-87`; `accept_m106_selected_geometry.py:94-141`; `M106_REPLAY_RECEIPT.json:2-6`; `m106_completed/train_weight1/result.json:14-107`.
### E. Evaluation scope and scientific qualification: WARN

One seed (2027), one control and one weight1 final on fixed cached Train states: 2,544 fit states from 130 sequences and 495 development states from 22 sequences; training uses 2,033 eligible fitting states. Sampling/strata depend on historical native errors and GT, invalid current GT is excluded, tags overlap, and neither independent test generalization nor recursive/official/public/semantic performance is evaluated. Claims in RESULT.md correctly restrict this scope, but neither arm passes all four gates. Weight1 loses 11/4 correct fit/development states versus control, gains zero, and leaves both parent-correct healthy breaks unrepaired.

Evidence: `prepare_train_states.py:64-105`; `train_ab_visual_control.py:58-82`; `train_selected_geometry_preservation.py:138`; `train_selected_geometry_preservation.py:162-171`; `m106_completed/driver.json:5-146`; `m106_completed/train_weight0/result.json:281-285`; `m106_completed/train_weight1/result.json:281-285`; `m106_completed/RESULT.md:3-24`; `m106_completed/acceptance.json:2195-2330`.
### F. Evaluation-type classification: PASS

Fixed-cache localization evaluation against supplied dataset GT. Oracle maxima and the post-comparison fitting diagnostic remain GT-based counterfactual/localization analyses, not human_eval or physical/semantic identity truth. Model predictions are boxes being judged and historical event-selection inputs, not the truth target. No synthetic/self-supervised/simulation result is promoted to real tracking performance.

Evidence: `prepare_train_states.py:69-105`; `train_ab_visual_control.py:75`; `train_selected_geometry_preservation.py:64-66`; `train_selected_geometry_preservation.py:101-107`; `inspect_m106_fit_selection.py:19-40`; `M106_FIT_SELECTION_DIAGNOSTIC.json:2-19`; `m106_completed/RESULT.md:3-5`.
## Independent deterministic result
All 91,170 unique candidate IoUs (3,039 states × 10 candidates × parent/control/weight1) reproduce exactly with explicit float32 rounding. All 30,390 saved metric fields, 20 marginal summary rows, joint profiles, original gates and complete parent/control gain/harm lists agree. The executor acceptance file reports 182,340 repeated arm-specific checks; that is not 182,340 unique candidates. I did not execute its acceptance script.
| Split | Readout | Selected IoU≥.5 | Mean selected IoU | Parent rescues / breaks | Top10 oracle |
|---|---|---:|---:|---:|---:|
| fit | frozen parent | 1602 | 0.60831378877649755 | 0 / 0 | 1814 |
| fit | weight0 | 1652 | 0.61671053830527078 | 51 / 1 | 1866 |
| fit | weight1 | 1641 | 0.61658458192648635 | 40 / 1 | 1856 |
| development | frozen parent | 272 | 0.53070769798706730 | 0 / 0 | 305 |
| development | weight0 | 286 | 0.53529750435534307 | 15 / 1 | 323 |
| development | weight1 | 282 | 0.53500541429472803 | 11 / 1 | 317 |

Weight1 versus weight0: fit gained/lost = **0/11**, with IoU improved/decreased/equal = **1,046/1,042/456**; development gained/lost = **0/4**, with **170/185/140**. Every lost-correct row is intermediate-tagged. These are paired threshold outcomes, not a claim that all individual IoUs worsened.
Both arms retain `correct_exceeds_parent=true`, `mean_iou_exceeds_parent=true`, `healthy_new_breaks_zero=false`, `transition_at_least_parent=true`. They pass 3/4, not all four. Their transition correctness stays 22 fit / 4 development; the gate checks development correctness, not its mean IoU. Evidence: `train_selected_geometry_preservation.py:234-237`; both result files `:281-285`; `summary.csv:4`, `:11`, `:14`, `:21` under `m106_completed/`.
The same healthy parent-correct box still breaks in each arm:

| Event | Parent IoU | Weight0 | Weight1 | Exact row in each arm |
|---|---:|---:|---:|---|
| pine01_indoor@1276 | 0.5004430413246155 | 0.47923150658607483 | 0.49121901392936707 | `m106_completed/train_weight{0,1}/fit_events.jsonl:2436` |
| mobilephone02_indoor@65 | 0.5105010271072388 | 0.44826263189315796 | 0.4675859808921814 | `m106_completed/train_weight{0,1}/development_events.jsonl:462` |

Both are in the previously recorded eligible subset (`m103_completed/events.jsonl:2708`, `:3006`); the fitting pine example is not an unpenalized-ineligible exception. Its raw GT is `m106_gt_inputs/pine01_indoor.txt:1277`; the phone GT is `m106_gt_inputs/mobilephone02_indoor.txt:66`.
All lost-correct rows are preserved with exact locations in the JSON audit. Fit line numbers, in both arm files: 339, 709, 778, 886, 1123, 1141, 1332, 1369, 1522, 1825, 2300. Development: 167, 328, 394, 477. The ordered lists also match `m106_completed/acceptance.json:2195-2330`.
## Preservation-loss semantics
The implemented per-batch term is mean(1[parent IoU >= .5] * clamp_min(parent IoU - refined-selected IoU, 0)) on the existing eligible-fit batches. Before is a constant cached float32 IoU; after is the per-box IoU returned by giou_loss, not the GIoU loss. No GT is added to forward inputs or inference selection. The plan's mathematical relu notation is value-equivalent, but must not be interpreted as F.relu's zero boundary subgradient: standard clamp_min passes the gradient at equality. For a reliable row at before == after, scalar penalty is zero while dL/d(after) is -1/B; at before < after it is zero. This is corroborated by locally inspected PyTorch 2.13.0+cpu derivative source, not a test or source attestation of the stated remote Torch1.13.1 runtime.

Evidence: `M106_SELECTED_GEOMETRY_PLAN.md:15`; `train_selected_geometry_preservation.py:187-199`; `C:/Users/gb/.codex_track_publish_m29_20260902/lib/utils/box_ops.py:13-16`; `C:/Users/gb/.codex_track_publish_m29_20260902/lib/utils/box_ops.py:86-94`; `C:/Users/gb/AppData/Local/uv/cache/archive-v0/Fi_OUYJWFu9xFWZGDf8J2/torchgen/packaged/autograd/derivatives.yaml:432-434`; `C:/Users/gb/AppData/Local/uv/cache/archive-v0/Fi_OUYJWFu9xFWZGDf8J2/torchgen/packaged/autograd/derivatives.yaml:2119-2121`; `C:/Users/gb/AppData/Local/uv/cache/archive-v0/Fi_OUYJWFu9xFWZGDf8J2/torch/_decomp/decompositions.py:240-241`; `C:/Users/gb/AppData/Local/uv/cache/archive-v0/Fi_OUYJWFu9xFWZGDf8J2/torch/version.py:4`.
Exact zero-delta box equality does not guarantee equality of cached xywh IoU and the xyxy-area IoU inside the preservation term. A separate stdlib float32 arithmetic diagnostic using unchanged serialized parent boxes and conventional xyxy area finds 681 exact-equal, 439 before>after and 471 before<after among 1,591 eligible reliable fitting rows; largest absolute difference 1.0132789611816406e-6. This is a local source-path diagnostic, not observed initial-step remote gradients or proof of the negative result's cause.

Evidence: `analyze_train_states.py:20-25`; `train_selected_geometry_preservation.py:153-155`; `train_selected_geometry_preservation.py:189-197`; `C:/Users/gb/.codex_track_publish_m29_20260902/lib/utils/box_ops.py:13-16`; `C:/Users/gb/.codex_track_publish_m29_20260902/lib/utils/box_ops.py:43-55`.
The base GT-best smooth-L1 + 2×GIoU objective remains active (`train_selected_geometry_preservation.py:164`, `:174-184`). The extra term acts only in weight1, on the frozen actual-selected candidate of the same eligible fitting minibatch. Its mean divides by the full minibatch size, not the reliable count. Weight0 skips its forward entirely. Reliability is inclusive at parent IoU=.5; this is localization reliability, not identity confidence. No positive reward is paid for exceeding the parent IoU. It is a soft training term, not a hard guarantee that no healthy box will break.
The log's epoch `selected_preservation_mean` is an unweighted mean of 32 batch means, including a final batch of 49, not a uniform mean over 2,033 events. Every full weight1 epoch records 1,591 reliable calls; weight0's zero counter means the branch was skipped. Epoch1/12 weight1 values are 0.0018825707738869824 / 0.0035323620249982923. These training aggregates and nonzero sanity gradients are consistent with an active loss; they do not isolate the cause of harm. Evidence: `train_selected_geometry_preservation.py:170-214`; `m106_completed/train_weight1/result.json:14-107`; `m106_completed/sanity_weight{0,1}/result.json:29-32`.
## GT separation, scope and budget
The 152 raw files contain 219,993 rows. All 3,502 current and 3,502 previous cached entries match the supplied raw files exactly, including null validity outcomes (3,039 current and 2,852 previous valid). The receipt's 130/22 sequence order matches cached split assignments. This independently verifies local raw-file/cache alignment and manifest hashes. It does not independently authenticate the original official download or live server execution. The remote attester itself checks current entries only; this audit additionally checks previous entries.
GT legitimately affects event sampling/strata, current-valid filtering, the training eligibility subset and GT-best target choice. It is absent from the inspected refiner feature vector and frozen candidate ranking. Thus `no_GT_forward_gate` must not be expanded into 'GT-free experiment': this is supervised fitting with GT-conditioned state sampling. Sources: `prepare_train_states.py:79-105`; `train_ab_visual_control.py:58-94`; `collect_ab_interface_panel.py:52-62`; `instance_ab_prototype.py:108-125`; `train_selected_geometry_preservation.py:64-84`, `:162-199`. The original dataset localization boxes do not establish human-reviewed category, attribute or physical-identity truth.
The submitted run contains two two-update sanity jobs followed by two fixed 12-epoch, 384-update final arms: 772 total optimizer steps including sanity. Each arm has 59,540 optimized parameters, 2,033 eligible fit events per epoch, batch64 (last49), AdamW3e-4 and seed2027. Evaluation covers all 2,544/495 valid fit/development states, excluding 368/95 invalid-current events. Every sequence is represented (130/22), but this is not full-sequence training or a full-frame/recursive benchmark. Additional preservation forward computation means equal parameters/updates do not mean equal compute. Only a fixed final checkpoint is selected in the inspected runner; no sweep or retry is present. Evidence: `run_m106_selected_geometry.py:27-51`; trainer `:138`, `:162-171`, `:238-247`; `m106_completed/driver.json:5-151`.
## Source, control and runtime evidence limits
I did not connect to the server, use credentials, import Torch, train, or execute GPU replay. Locally checked bytes, tensor storages and arithmetic are independent deterministic evidence. Zero initialization, frozen parent throughout training, 495 parent-choice parity, final forward reload and replay-state immutability remain source assertions plus runtime receipt/log attestations. The bundle hashes trainer/runner/replayer, but not the complete imported runtime tree, feature/context/origin caches, M101 parent, image data or remote autograd kernels. Current local helper source and imported box_ops.py cannot independently prove their remote executed bytes.
The six M104 definitions (`LocalVisualRefiner`, `decode`, `normalized_target`, `metadata`, `encoded_inputs`, `evaluate`) are independently AST-identical. Control and M104 checkpoint bytes match SHA256 `546bd218a14906beb28033717bb53cc17721e9bb6b3fb53f36a4bac15dff9f6e`; the six stored float32 tensors total 59,540 finite values and match independently without importing Torch. Weight1 is a different finite checkpoint, SHA256 `57082505db00dd81467e5060a2145da6e365e3807f4580790d0a7c0cc9ccca25`. The final nonzero tensor state does not itself prove a zero initial state. A supplied identical control artifact supports exact artifact reproduction; it is not an independently observed training rerun or a universal determinism guarantee.
All 25 runtime download entries match bytes/hashes. All logs and exit files are consistent with four successful children; controller logs, launch snapshots and final driver agree. Hash consistency binds supplied artifacts; it cannot authenticate a remote process by itself. Driver elapsed is 38.52110741706565 seconds (`m106_completed/driver.json:149-151`). The controller uses the mplt Python while the actual training children use STTrack Python (`controller_launch.json:5`, `driver.json:10`); this is not an environment mismatch in the trainer command. The source plan records Torch1.13.1+cu116/Python3.8.20 (`M106_SELECTED_GEOMETRY_PLAN.md:38-41`), but the supplied deploy receipt independently reports only Python3.8.20 and trainer/runner hashes (`M106_DEPLOY_RECEIPT.json:45-48`).
## Later fitting-only diagnostic
All 2,033 fitting-diagnostic records and counters match independent recomputation and bound inputs. Correct/partial/low selected quality = 1,591/341/101; selected index equals the first GT-best argmax for 1,449; GT center is geometrically inside selected box/context-factor2 for 1,918/1,992. These were inspected after seeing the comparison, use no development rows, and are not a new accepted evaluation gate, causal explanation, usable-pixel evidence or semantic identity label. Probe global row order differs from primary geometry order; the code joins by unique key, and the exact key set/metadata match.
The 584 selected-versus-GT-best index differences, 115 centers outside the selected box and 41 outside its factor2 context are complements of these verified counts. A different GT-best index can also arise from tied IoUs (the diagnostic chooses the first maximum); it must not automatically be called a quality deficit. Geometry inclusion does not establish that unmasked, informative pixels/features were actually observed. Source/receipt chains and full 2,033-row checks are in the JSON report; `M106_FIT_SELECTION_DIAGNOSTIC.json:18-19` already states its proper limits.
## Claim impact and actions
- **C1:** M106 is a completed, locally arithmetically consistent fixed-cache negative comparison. — supported_with_runtime_provenance_qualifier.
- **C2:** Weight0 reproduces this archived M104 checkpoint, all 3,039 rows and summaries. — supported_by_independent_bytes_tensors_and_arithmetic.
- **C3:** Weight1 improves preservation capacity or clears the original promotion gate. — unsupported; 11/4 lost correct states, zero paired gains, both arms fail healthy_new_breaks_zero.
- **C4:** Preservation has zero gradient whenever its scalar value is zero. — unsupported_at_equality_for_clamp_min; remote boundary execution not independently tested.
- **C5:** Training/evaluation uses dataset localization GT, with no GT in refiner features or frozen selection. — supported_by_local_raw_GT_alignment_and_inspected_source; original runtime inputs not independently replayed.
- **C6:** The fitting-only GT-best mismatch/geometry-support counts diagnose the cause of harm. — descriptive_counts_supported; causal_or_semantic_interpretation_unsupported.
- **C7:** Recursive tracking, language contribution, robust multi-seed/public results, C-module readiness or nine-metric completion. — unsupported_and_not_claimed_in_RESULT.

- Retain the negative comparison and both failed healthy-break gates; do not promote C, recursive tracking, public metrics or semantic claims.
- Describe the executed loss as clamp_min with inclusive boundary subgradient semantics; a zero penalty is not proof of zero preservation gradient. Do not silently relabel this run as F.relu or rerun it under a changed loss.
- Keep deterministic arithmetic/byte checks distinct from source and runtime attestations. Before stronger runtime-causal claims, bind the relevant imported sources/runtime and preserve replay/input provenance; no new execution is needed to retain this negative result.
- Keep the additional fit-selection analysis explicitly post-comparison and fitting-only. The remaining 584 index disagreements and 115/41 geometric center exclusions are descriptive counts, not proof of usable visual evidence, cause, or physical identity.
- Preserve dataset-localization classification, GT-conditioned sampling, single-seed/cache scope and the absence of independent human semantic labels.

No local arithmetic, cache/raw-GT alignment or artifact-binding failure was found. The overall WARN preserves the semantic boundary/runtime limits and scientific gate failure; it does not turn this negative result into a positive claim. This same-family review remains provisional. The JSON contains independently run stdlib verification programs and their results.
## Audited input SHA256 manifest
Every listed file was read/hashed. All paths without a drive are relative to the evidence directory stated above. This manifest includes the 152 supplied raw GT text files, every runtime artifact, added diagnostics, source dependencies inspected, and review instructions. JSON also records absolute paths and byte lengths. Audit outputs are excluded from their own input manifest.

| Input | SHA256 |
|---|---|
| `C:/Users/gb/.codex/skills/experiment-audit/SKILL.md` | `0ab77f219f554cbe42f22b79d780528e61c5e58169af4c2f56acf59596ab8db7` |
| `C:/Users/gb/.codex/skills/shared-references/experiment-integrity.md` | `22150fbdc5867e7790c88f968bccb367079c2db4c190ddce980709aea7e76f60` |
| `C:/Users/gb/.codex/skills/shared-references/local-codex-policy.md` | `b0249bfccd48ff79a5976afa7a54c64a3cfb65a5496a2105c5bb21069d197783` |
| `C:/Users/gb/.codex/skills/shared-references/reviewer-independence.md` | `1aa1756a99dc47009072f0ee7bf570d797bf7e2cc84439401e2a9312d73659b0` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/traces/experiment-audit/2026-09-28_m106_completed/request.txt` | `401f7b63276a409197ab4a0328d8ac85021def5f88d7596a9bdef88b0ccf3077` |
| `C:/Users/gb/.codex_track_publish_m29_20260902/lib/utils/box_ops.py` | `b6dd8c9ca37ecc192c0ea2539aaff80385ff88a77ee56f3298ab5127f33785eb` |
| `accept_m106_selected_geometry.py` | `3ac16f9261d626e8b9dded42ae02bb3f1aaf3a6456c5cd12d70f58f098e69c68` |
| `analyze_train_states.py` | `2439a8d0cdb7a7d6c35bbf68aaacd28850bfb8db2211f07eb9e9e7dac66b6def` |
| `attest_m106_raw_gt.py` | `a920e86a79c7ef889353f6ab2b15129253ab798ec058ae53657de7c15cd81529` |
| `collect_ab_interface_panel.py` | `361559ef53623dc53665534243a0b4cfb078821dc852e20b34a75d14dca45086` |
| `diagnose_b_geometry.py` | `02e5481a4b65726675e16364881b89dc5fc0cb4b2a0f80be99c8afe7c96b2747` |
| `inspect_m106_fit_selection.py` | `fe097d123ca94d176c563ec4b10e9f88249cf187e8683412d83103c082b4f756` |
| `instance_ab_prototype.py` | `09db6833f161b6028e64905f2e73a69027e0caf28c611b6c9d81a9fa6862d9fb` |
| `m103_completed/events.jsonl` | `1a2d66f09bf1eef37b9ed59954725135298407fa4c59c04da1d232e89e689cd7` |
| `m104_completed/train/development_events.jsonl` | `c339e6ec301e16a630501ae5f7a07cc62ef6da604967bcb40e3b00b63a5bf136` |
| `m104_completed/train/final.pt` | `546bd218a14906beb28033717bb53cc17721e9bb6b3fb53f36a4bac15dff9f6e` |
| `m104_completed/train/fit_events.jsonl` | `0cf535db21f0baf6933686936e8acc129a2b7424c5ef08c230fe57b951a6676f` |
| `m104_completed/train/result.json` | `315ff9ae562e71878d36e8fe84b04343a285f5c295e0f62cce3b449e87f14a7f` |
| `m105_inputs/download_receipt.json` | `86f20ffbabf072c6655b1ce5e71d021c35b850d24b4eb6393b805e65e8cf1792` |
| `m105_inputs/preparation.json` | `59cedd45335583228f59451b818d9a4a5c06c3c67295bf2796c8bdefa18539aa` |
| `m105_inputs/training_labels.json` | `433226372eb045b0dede646ec1fa016506390728724ceaaf194701bb1b213a0f` |
| `m106_completed/acceptance.json` | `800c3cfffa1bcf4168f72de94ba0405f49ce378156138e2fdfd7027d713e6350` |
| `m106_completed/controller.log` | `5e150272cccd89192e01b0a3bcf9a04678d8ecaabdfd983a9778fbf9625e34c8` |
| `m106_completed/controller_launch.json` | `c6290cffcf2dad08b79848c9ff4d25576f11c2a8620b40f984e29b4f92d13650` |
| `m106_completed/download_receipt.json` | `8f5464358b6079fd7b25005c6f8967c0509643e9bb45a5e78e8a9954d24ad536` |
| `m106_completed/driver.exit` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `m106_completed/driver.json` | `06af89479eab0a0039e409016ca5b1283b9326ab5044704ad3ca910cdfacd5f2` |
| `m106_completed/launch.json` | `a48bdb183b07939094918603887d8073bca0c9dbb0d11c53a1bfd8a51e8402d4` |
| `m106_completed/RESULT.md` | `25917b4c3ad4dff2b46af65ba2df958392921bff022b4770235c5a4e839b07a6` |
| `m106_completed/sanity_weight0/result.json` | `556756de8c780c19526cc611afdba4cdf06c6edbfafb2913919138cc2a9489e4` |
| `m106_completed/sanity_weight0.exit` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `m106_completed/sanity_weight0.log` | `c8a85c2a7354f8a97ce0931cfa0d34da9285d870279650e8d96aee02c54decd7` |
| `m106_completed/sanity_weight1/result.json` | `6ece35e507d3b8dffddb35bd71fe6efdbbb82b5793009c2d34182a6a1a1782a7` |
| `m106_completed/sanity_weight1.exit` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `m106_completed/sanity_weight1.log` | `e8b02eb3dc2d4d5ff047cae67f79102e254a02ab472dd4acaa05b444c5a650d9` |
| `m106_completed/summary.csv` | `d3ac11a8605e210365c1dceb0592d922174de05df2b6c2a1082a71406eab2481` |
| `m106_completed/train_weight0/development_events.jsonl` | `c339e6ec301e16a630501ae5f7a07cc62ef6da604967bcb40e3b00b63a5bf136` |
| `m106_completed/train_weight0/final.pt` | `546bd218a14906beb28033717bb53cc17721e9bb6b3fb53f36a4bac15dff9f6e` |
| `m106_completed/train_weight0/fit_events.jsonl` | `0cf535db21f0baf6933686936e8acc129a2b7424c5ef08c230fe57b951a6676f` |
| `m106_completed/train_weight0/result.json` | `02604d2b3827569b2c77da4b252c15bfc6636eb6481869d7067a4be7c23cc96b` |
| `m106_completed/train_weight0.exit` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `m106_completed/train_weight0.log` | `57661e00746a0b2a8f38dc08f424bd7a0d4537d5566903c2b9797f5665dd7afd` |
| `m106_completed/train_weight1/development_events.jsonl` | `2087c9a905481b2922feaa95795b8216d5b253f66cdb6b518d2c5f2bb314994c` |
| `m106_completed/train_weight1/final.pt` | `57082505db00dd81467e5060a2145da6e365e3807f4580790d0a7c0cc9ccca25` |
| `m106_completed/train_weight1/fit_events.jsonl` | `329bcfc8b201e457e1ad5ca725698195326de900b0f8236062728542029f182d` |
| `m106_completed/train_weight1/result.json` | `bb76c51b6dd6aeea07ccb709338a4e513cb9fd254d9476694de9ad22ace469d4` |
| `m106_completed/train_weight1.exit` | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `m106_completed/train_weight1.log` | `3d9c9fb69dd44eb2068b556054ec60506365ba774a757a7746e6a34a1df3f1e5` |
| `m106_completed/verification/geometry_boxes.jsonl` | `725705548b9dca56cafe27af00cb851ec31afb11bc91a191644f3ad80fd2a164` |
| `m106_completed/verification/result.json` | `c06f83e7338724f424372749749d40029676eac136a56aaeb00ee68fe77bce8b` |
| `M106_DEPLOY_RECEIPT.json` | `2c25f2db60926d8f15e8249259f768bf33806c787261c376d9868c043abf0ebe` |
| `M106_FIT_SELECTION_DIAGNOSTIC.json` | `901afd7a7f1d5f5e53f32d7681b1d20e3a09d36cfc1b4b4c18c15c753eadb1ae` |
| `m106_gt_inputs/adapter02_indoor.txt` | `33a964dea164f87c4c4adaf720aeed9f6209f4f3f5025fc9d597c03f84d14e78` |
| `m106_gt_inputs/bag03_indoor.txt` | `fb4702c2f0e4c6a1e33251efb0f3c1c7f4eb5f26c1a3c79d9018913b2372d8ec` |
| `m106_gt_inputs/bag04_indoor.txt` | `0d06a12db2f75b4428d77d0983050e51ed2e56153c24ab74871ab4abca315267` |
| `m106_gt_inputs/bag05_indoor.txt` | `19fac45f89a26ebfc9f3609008139d917dd80c75c825b6fe9c9d085ec55ebf52` |
| `m106_gt_inputs/ball02_indoor.txt` | `2147f99e45f7af89f466d6bfbe718ca4a95a75827324f1c736eaef810d9f2083` |
| `m106_gt_inputs/ball03_indoor.txt` | `ce41f746332d2777ea7f5f97ae8f545bbfa9afbe93929eaca8f0f0ee70f057c7` |
| `m106_gt_inputs/ball04_indoor.txt` | `db05ab15d9143192adc6a248c4a8d1fa66c339d028d067d60cae72c01ff85661` |
| `m106_gt_inputs/ball05_indoor.txt` | `4a3bf9f327e4994423b75c3d1267b80ac71d05c5e9b93b08b509f6d83bff4a34` |
| `m106_gt_inputs/ball07_indoor.txt` | `c9cb5a6fcfd0bd3f5b3582292e5c30625a4c30fee1c08e5eb247f5ae6ffaaab2` |
| `m106_gt_inputs/ball08_wild.txt` | `a3570303bd54848f4b96234001d8a0f141460cf42effcb1ade9b2760e8e84bf9` |
| `m106_gt_inputs/ball09_wild.txt` | `a3647edc0b7e4ffcbedbc970b0e48e88aa11c5fea9cf9a2f22b490a43f0f1a01` |
| `m106_gt_inputs/ball12_wild.txt` | `feee35d47e25bcdfcaef049ebebd67f77892cb1b533c842ad5147c1cff6b8c9b` |
| `m106_gt_inputs/ball13_indoor.txt` | `99c5e76747408d0c120b92ce172d37cbd39abc2836207837633eb92698547446` |
| `m106_gt_inputs/ball14_wild.txt` | `f6d7da3590803baab88cadc089e76daa57b25fd1bcd2152abab118e41d085017` |
| `m106_gt_inputs/ball16_indoor.txt` | `e39af82dd9ae52bbf6734c63cc916dfa043129fbcc4e0d675de09535ce0180a4` |
| `m106_gt_inputs/ball17_wild.txt` | `a2e90cb9a2db72c3d4dd00bbeeb645b8f02b95094f714bb3f783ccb7e0dcd5e0` |
| `m106_gt_inputs/ball19_indoor.txt` | `21a1bddce56f75dd67ca5b9fbb14eccafc46bd36b24a468fb9483290d68e125d` |
| `m106_gt_inputs/ball21_indoor.txt` | `5230b453b3d069f8ade8e23aef0f3d18f9794d900a1a6264ebb830aff0ac0bb4` |
| `m106_gt_inputs/basket_indoor.txt` | `d681b5a242bed11da0f5fe714491b49651ef1d1c4b4f43b92cfc6263c9d1e9ca` |
| `m106_gt_inputs/beautifullight01_indoor.txt` | `0e89eed1894076dbfbbebed2f931163b27fa3572246d21b49662b5ccbd1a945a` |
| `m106_gt_inputs/bike01_wild.txt` | `cd70a8371938cdac020502cf2c4e0c77f77180eecd6baabee70b269b2ec47f24` |
| `m106_gt_inputs/bike02_wild.txt` | `4b90a9f1d4251f16d6dff2bf9864387d2954f0978145a524b0ed06ba3c03ea35` |
| `m106_gt_inputs/bike03_wild.txt` | `dad8d95321f350d2a347f4f71504d8d82b57fd7fdd08ef617b63e2929f57c3e3` |
| `m106_gt_inputs/book01_indoor.txt` | `9d0c8affacf73e01d0eed4d446aadaeb3e42f111d46b2e2d38ed86de40b9a19c` |
| `m106_gt_inputs/book02_indoor.txt` | `25fc50ddf6083b81f0e202fbfbc83561d4d2cb13d445d9adb398e798473a7fc1` |
| `m106_gt_inputs/book04_indoor.txt` | `d7009271319c37873f1ca7e4a615e7efed384818c4a5a4c36b02c3f4628cbfdc` |
| `m106_gt_inputs/book05_indoor.txt` | `8e2f3008f581de3517e2d19185080261a796ec36506a62f07824ecbdf34196f1` |
| `m106_gt_inputs/book06_indoor.txt` | `df6b93503aa8177ac8b0e32507124189f453bce6c5bcb5f993667850f77b58a1` |
| `m106_gt_inputs/bottle01_indoor.txt` | `e226e68e5c5ec7dbe8b380aaa0c37ad6444b683ca4f6fa84bfc59968ee818e83` |
| `m106_gt_inputs/bottle02_indoor.txt` | `48f14b53ca931b24f50e0e24cef612aaef8311a85736c0916f69b58d54c72522` |
| `m106_gt_inputs/bottle03_indoor.txt` | `8cb3e98365c17668ebbbb937431a5a38d22f05804b352f353d51d834fd3a117a` |
| `m106_gt_inputs/bottle05_indoor.txt` | `222999382c4ba7803d1ad6fd919d93fc290f1b0fd521f3c08dddd2c47e06b01f` |
| `m106_gt_inputs/bottle06_indoor.txt` | `7dce3716c2635f7801abaefa8cda804ab943aa16ac77024d1ce11fa871437812` |
| `m106_gt_inputs/box_indoor.txt` | `a34d48e631a3ec23016aa71f1951d350a72ad03beb878ed1ca18217875b98afd` |
| `m106_gt_inputs/candlecup_indoor.txt` | `2c2ba43df1406dd9fd0ac25d776dde07980efc6ae94900d3bbcdc453a6fbfc01` |
| `m106_gt_inputs/car01_indoor.txt` | `dca7e65de77cced65b4b06a589352fdde456152b9a75d9af8759082b8cf458d4` |
| `m106_gt_inputs/car02_indoor.txt` | `ed1171be55b7782858af063f174b2ece551f35b6ee2b5a4d0e81b5fc53f87ec7` |
| `m106_gt_inputs/cart_indoor.txt` | `e5b16fea25a7b2ddb32cf3132f97e385d6c27bd9af99c64cf12bd882b72c5226` |
| `m106_gt_inputs/cat02_indoor.txt` | `db4723e61d50464bc5124bc5d27ff1cdb9bbaefbacd76fd109c762490ebcfd1d` |
| `m106_gt_inputs/cat03_indoor.txt` | `d1290ff9f5837d882b274bdd1f19c9680bfab53b72c06aff547fc9e6953fe43f` |
| `m106_gt_inputs/cat04_indoor.txt` | `6421ade8ca96dbc2da52ca02cedd69c415f37498775732b0cf780d81f2d292af` |
| `m106_gt_inputs/cat05_indoor.txt` | `5d9144b7989d54a349cedafcf15f407db098d8af69019302bc05c2de0f09082a` |
| `m106_gt_inputs/chair01_indoor.txt` | `fc25354c8769860a8c4404019368e34c1415367828009e6503bb7695581bf62a` |
| `m106_gt_inputs/chair02_indoor.txt` | `4f05bc2ed056bdc6c9c5a2687ab6c074a2f32b95cae62bf6feebe89899f03ca2` |
| `m106_gt_inputs/clothes_indoor.txt` | `e0717f89c1ad115ce87bb252bc7915ee1d0b34ab73c72fc8ca5a47a07612094b` |
| `m106_gt_inputs/colacan01_indoor.txt` | `38443588eb1127d65c2ac800186759ad392581eef544588156c8c0f14b2ee866` |
| `m106_gt_inputs/colacan02_indoor.txt` | `ca6587fa0aa38475f51d7dc443398b8e716d7b0a175346f310c60bb8b8360f57` |
| `m106_gt_inputs/colacan04_indoor.txt` | `c556156a8252ea1e1b0c6a14cb4d63a374c8edf47b3dbdcdf129edfb562a0c2a` |
| `m106_gt_inputs/container01_indoor.txt` | `c08d2571102bb7106e6530a0dea8512f6206e15ee765a3a52b2734978fba7e1e` |
| `m106_gt_inputs/container02_indoor.txt` | `00cfbbbd0e55ccf24f191bda511d2a217c43a665f65ba0f64073d3b7b51fdccf` |
| `m106_gt_inputs/cube01_indoor.txt` | `c7e7f55e9d0ed4bac64c18d658796fa340fe6dddd2467a877015d42c71d4185e` |
| `m106_gt_inputs/cube04_indoor.txt` | `d32cecf11f9e9bfd15c7bed815ac0208c991df8a7b89c2388219907f6e6b6bb4` |
| `m106_gt_inputs/cube06_indoor.txt` | `19cf63610ae9b40e67f5e9b8f10ec62a42ad6662a693158eda64d18bb2b4c43a` |
| `m106_gt_inputs/cup03_indoor.txt` | `aef83d01e39f78bded1998acf5650504e4833292f9efaafc712094bca7eeda3d` |
| `m106_gt_inputs/cup05_indoor.txt` | `15d37ca8f694388415795e8df944d882347dbb5a2fd2e0a2e07e346164733149` |
| `m106_gt_inputs/cup06_indoor.txt` | `ebba4639a8d94969986dc9dd60960c43701d1a1fafefaec270cdac99cc1b9e5f` |
| `m106_gt_inputs/cup07_indoor.txt` | `a87da31a984f168cba69d158eb46e31373ef79d11b2d4fd50c4c1886737c8885` |
| `m106_gt_inputs/cup08_indoor.txt` | `b7c606e0373f5cdba15ecb21efa1ce293bc6620c5ffa60ddb567a1394c67f1c8` |
| `m106_gt_inputs/cup09_indoor.txt` | `6235e8bec9d88203c3bcc2adfa66db218730d79c619979fa575e77bd3cc2c5c5` |
| `m106_gt_inputs/cup10_indoor.txt` | `237081c1453dc80861c19444a63b2a4f63bd56c8703285c408c3e4d52a3d55ae` |
| `m106_gt_inputs/cup11_indoor.txt` | `14e516059c4ebb3500cb0626fa4ff5135cee275d9b2e3f8862faa83a4e4af761` |
| `m106_gt_inputs/cup13_indoor.txt` | `bd048b8a08b71b5616f185f8e8f1727972d7addf6251881c35f94d2911099864` |
| `m106_gt_inputs/cup14_indoor.txt` | `9ad3ecd17fdbf886c7edf1f56934e7d47ac192cf7b9b4c53d013451651656bda` |
| `m106_gt_inputs/duck01_wild.txt` | `83aca68e7101f1261e7bafcff766bad3a0a68301f10921ded08f11192497263b` |
| `m106_gt_inputs/duck02_wild.txt` | `ac48fd345b322c2bb305e68ec659c8872446f9023867e23851f3b6e12d11790e` |
| `m106_gt_inputs/duck04_wild.txt` | `29848ca243337725cfeae195563bd6035fae1f14f34c32a0441918112772c986` |
| `m106_gt_inputs/duck05_wild.txt` | `5ad1b542b94c9181d66bc1d84e033b747ecd9a8fa62fe4717d6a65f05d870004` |
| `m106_gt_inputs/duck06_wild.txt` | `dfd69275038758c44fd778ea88b899b28c17337be0ff03bdcd4189d008910e1e` |
| `m106_gt_inputs/dumbbells02_indoor.txt` | `18dac18d42fb4fa09483990feae0d084fff631c4629589fb4c7cfe880ff0052c` |
| `m106_gt_inputs/earphone02_indoor.txt` | `9178128a1ed5d23438515fdea6bac208c800122a1d38d0963d44f0eb700ba782` |
| `m106_gt_inputs/egg_indoor.txt` | `a4010da190dfc37a2c48efdc21c60993e398af6b9df87b91dc3aee813e5bc9bd` |
| `m106_gt_inputs/file02_indoor.txt` | `b8f7d0914b2162609f993e8445b92cef3e2e1bb6e388cd21d1e62d835a154dfb` |
| `m106_gt_inputs/flower01_indoor.txt` | `97ecc6e67a43c29a5ef5eceaff28a480f40487347788ff044bef19fcbaea13db` |
| `m106_gt_inputs/flower02_wild.txt` | `517aa37496ae936121cc6ddbfaee6e845616a64a137bdcb8a8cfee489d2cc644` |
| `m106_gt_inputs/flower03_indoor.txt` | `66480878783c4fcdf76d47234f1884cdffaddb775028bf2c7d8b12d2061da8ce` |
| `m106_gt_inputs/flowerbasket_indoor.txt` | `0cc6a4a0bec64e746491603b06b076fd41f7fbce4c86c0d1455b2440302f0d1b` |
| `m106_gt_inputs/ghostmask_indoor.txt` | `5aba29f532dd42f9df5bbfd1c6c5558c8632f9d62fbf5fc5d52470d799ef9406` |
| `m106_gt_inputs/glass02_indoor.txt` | `8e019d223669c6f31adf10d3ee4162fca5404fd40be892f1c44f411fcc03ae4f` |
| `m106_gt_inputs/glass03_indoor.txt` | `b4e6aa986c65391dfdb688c77403795f5626134f5088821ea3bd4b7284e184da` |
| `m106_gt_inputs/glass04_indoor.txt` | `d24775109819cb517cbe764c3ef7990d25fe925049538864f4770864b40afbde` |
| `m106_gt_inputs/glass05_indoor.txt` | `51cc8d500877bc4a321bc2bb6e810a30cb86711896764968d2700250f9352206` |
| `m106_gt_inputs/guitarbag_indoor.txt` | `0d80234a96568cf983950b1f8b5bda33a01cb8cffd5cf1705f5c93f5621f9618` |
| `m106_gt_inputs/gymring_wild.txt` | `b2df5de9aaaae941866b11586188ca7a5a94be1000a54e118df375cacf3d2346` |
| `m106_gt_inputs/hand02_indoor.txt` | `fdb582d198c0d0e843a21891c4e8966cced3c2330dc3ded5110029ac67ccfab5` |
| `m106_gt_inputs/hat01_indoor.txt` | `8898891868efd65a97b8138c99404a0ae24e760f808487870f1b7529a30270bf` |
| `m106_gt_inputs/hat02_indoor_320.txt` | `5851dd6051472626d3b0432e492fae38c4fa7a06d12caaab93f44a6c5c1dbbd5` |
| `m106_gt_inputs/hat03_indoor.txt` | `c7cd257073a4e20905e43f0bb3558cfb2d8bd4d391c6db18ae8bb8cf3cce9951` |
| `m106_gt_inputs/hat04_indoor.txt` | `2d71dd08fff19e4be6ba64b51dff107674ac7c18d50dfe5e924e23e58989eaa1` |
| `m106_gt_inputs/human01_indoor.txt` | `384febbcddeeafe3489749b1b143c5d357e0c2098b02f589f45f208a0f627271` |
| `m106_gt_inputs/human03_wild.txt` | `608ef24142202c0e8e9af1b3752ee0709e043a5f085e30bc66068bcaa4196543` |
| `m106_gt_inputs/human04_wild.txt` | `3a24191b7f4b4690c0c6c9c912754cdf658a702622686aa7891419f866a9a8b7` |
| `m106_gt_inputs/human05_wild.txt` | `65e455611bc7c342891c0bbe60ca0f876c181196251fa9d80f963ead74542f9e` |
| `m106_gt_inputs/human06_indoor.txt` | `0f723bba01cd765122da6a513f6e458623ba3345aafb5a94826d787c9b1ee134` |
| `m106_gt_inputs/leaves01_wild.txt` | `d34f88a2e29041dcc3f0921878f934bae4f654e12f9e43d14aaad17e00e83561` |
| `m106_gt_inputs/leaves02_indoor.txt` | `d6a2ad6c7f3a9720b75c5a2457201857edfba1beeeeea2c5e5e85a39807fd696` |
| `m106_gt_inputs/leaves03_wild.txt` | `a8169078aadfb3484b795038128cb1d314e04f2626a9f974253ac55c8dd69b5a` |
| `m106_gt_inputs/leaves04_indoor.txt` | `ede8be7594992f5bd8addd4a22c657ee5fa63857c5b54ac602f88381dd76dffb` |
| `m106_gt_inputs/leaves05_indoor.txt` | `aa0b613b169058dbf47eafd1c74e75e9e3d617e4f3ed21136bb5093e3ce8a3c1` |
| `m106_gt_inputs/leaves06_wild.txt` | `e5e14506d23aea54df25bd626b5d418240d15d1fa9b16739bac967fe464c7d74` |
| `m106_gt_inputs/lock01_wild.txt` | `8812b746751f5f59ea8a16764ec70b7d1ded6c3ffcf963ad9245ac7ce84ac577` |
| `m106_gt_inputs/mac_indoor.txt` | `7b7fbed57f079ebedae12870613d30719a69d98f458815d7a42c50be893adb1a` |
| `m106_gt_inputs/milkbottle_indoor.txt` | `1d677d5d444fd4e14aec00fcdd9c4fbf3af1efb32bb8bcc1b53490610ab54f55` |
| `m106_gt_inputs/mirror_indoor.txt` | `39399adcf38d1be5d83ed6bac42e777cb660e49f0c93b150a5e2f105f9674d41` |
| `m106_gt_inputs/mobilephone01_indoor.txt` | `572ff6fde310d831e0fd2dd5201805a4f9d2cf18ec1dd63a227266872bf218ce` |
| `m106_gt_inputs/mobilephone02_indoor.txt` | `d27bba52e1ebe7f816b9d2ce264d332404530765bcce830029070b146c52e45c` |
| `m106_gt_inputs/mobilephone04_indoor.txt` | `c18af5d30957a339497ba8235ecbf6986dbc7ce66773afcfabdbdb0a3b03d55c` |
| `m106_gt_inputs/mobilephone05_indoor.txt` | `70c425a3f893af73aca7bdac61cb111e47549d0c935b42af3a299a8f29919356` |
| `m106_gt_inputs/mobilephone06_indoor.txt` | `4d96750a2019b57ad1ab7cfe0076f3fea8a1b03a8d293aec7826dc25f188ceff` |
| `m106_gt_inputs/mushroom01_indoor.txt` | `fee7899ead7c9ba02f3afb6e6e0829656cff0ba0df64ca9fbef1304574661ced` |
| `m106_gt_inputs/mushroom02_wild.txt` | `b7e602085a0c3af4ddd417b9dea144eab7d9d8bfec6517db261227f46090716d` |
| `m106_gt_inputs/mushroom03_wild.txt` | `453dd8042c4d9f0decb712d3899e228197b62f432e3318bdd1da44f26bd172fd` |
| `m106_gt_inputs/mushroom04_indoor.txt` | `bc2f201ef68a86eca6b19abef4dd410ed50aaec2569d6163cc0b57913c32540f` |
| `m106_gt_inputs/mushroom05_indoor.txt` | `79887acbc43dc5468d4e9bdcf021df1347c869cc15ad336fc4e5fe7e42b5bb25` |
| `m106_gt_inputs/notebook02_indoor.txt` | `fa3202cc8968e24785620c772584bc28e2a353c06799a0859ee6a2b8d8a880a0` |
| `m106_gt_inputs/notebook03_indoor.txt` | `a7c2e9a80ae2a6303b3b06c21cf8c1ab2fd96d88ae77368eaa804f83644100a0` |
| `m106_gt_inputs/paintbottle_indoor.txt` | `138c6884d76f8385a71bfd04c0ccc83b835ce67b9f37a9e8880da2abb2caca05` |
| `m106_gt_inputs/painting_indoor_320.txt` | `452f8042e633005d48071913f4b63c71d24829f0a931891f142539d4664a2ec6` |
| `m106_gt_inputs/parkingsign_wild.txt` | `397c45dcc9273890a8a0364f59ccb58f4d485bf5454e04af27cf0aee7e3b8343` |
| `m106_gt_inputs/pigeon03_wild.txt` | `4281a4712dae248a85869058f68162c94984b6e287896287bf7e74e5440712fd` |
| `m106_gt_inputs/pigeon05_wild.txt` | `7ce05357a9a6e942c8dca07cc44507ec36f431b6920f6150c11629ab594e4f7d` |
| `m106_gt_inputs/pigeon06_wild.txt` | `d249393a979a8cfc4c541107b51329cc133e2e99c30f821c937359d5304df275` |
| `m106_gt_inputs/pigeon07_wild.txt` | `91f2a4d1cbdd2808af3c0fc42f7a2a26a728eee5246b32a8c3ef12a2c06883d3` |
| `m106_gt_inputs/pine01_indoor.txt` | `c06487c4c9e9de804050ebe5a7c25c90cc1dcb1d88c83279cd980cb0c5de5329` |
| `m106_gt_inputs/pine02_wild_320.txt` | `0555c5378049662102ef0f382e3a4ddc2e5abdcc4fc5deb6ce2a9966bf5be16f` |
| `m106_gt_inputs/shoes01_indoor.txt` | `cee9970156a03576505b05a59ecc55e8daf4ff5ef5f10ef23dbf5efad7123326` |
| `m106_gt_inputs/shoes03_indoor.txt` | `4f3bc65b100348817f2d1856613d54c1f783759ed1a282b8df28eded96467f7c` |
| `m106_gt_inputs/skateboard01_indoor.txt` | `afb682605337866d5ea72db4025dfbb70ee2acd2ea85719b5989a7626bb758d5` |
| `m106_gt_inputs/skateboard02_indoor.txt` | `304a07f4b196eb9a855d1a1857beec4757141965a09c5e635b80cede70938b68` |
| `m106_gt_inputs/speaker_indoor.txt` | `1a06abd953cb7833c232576238363cbe6c5248c486b3f221567d18dacf9a9a44` |
| `m106_gt_inputs/stand_indoor.txt` | `15219e425e311f79e435a1c7526b3d04de6dbe1a035a6a1b5a4614e0ed4cef21` |
| `m106_gt_inputs/suitcase_indoor.txt` | `946321102dd03c4d28f1f3ccb31434bb955982bcec7e88082ce86b1579442c6e` |
| `m106_gt_inputs/swing01_wild.txt` | `2518e4ffc0ba579cbcce533e6aa33792b9f2a9f676651c2b79347dd24629657b` |
| `m106_gt_inputs/swing02_wild.txt` | `9768e065c404b8acf47dc8dbe902f8eca8dd1736c501e97c5ad431cfac087741` |
| `m106_gt_inputs/teacup_indoor.txt` | `33ecb768c92c86e7106b05ce42527bf80db3c81c1f58b8332e22ccbb11625f97` |
| `m106_gt_inputs/thermos01_indoor.txt` | `4a9a8a13af7359b697edfffe887b26b8e316042171f2955b44ad874b8f25d99e` |
| `m106_gt_inputs/thermos02_indoor.txt` | `96b26f5f41d8ad70a8d1c1c34751aa224b2c8e447c007dc7dec6eb9ab90c5dbc` |
| `m106_gt_inputs/toiletpaper02_indoor.txt` | `23039f5b59a9259bbfc89dd6d38e3ed9dcce696e03b827fb8ea320eeaef5cd0c` |
| `m106_gt_inputs/toiletpaper03_indoor.txt` | `5b1f12533be6068b6db0ec4a06cf5702660e4a904c6385e78fce892a352a3cc4` |
| `m106_gt_inputs/toiletpaper04_indoor.txt` | `e01fe3e9d60dbfe0588a3a096ca3b136a5ed0e2017e8e401d2704e57caf1467d` |
| `m106_gt_inputs/toy01_indoor.txt` | `0f69714f77bc1cde2549ff14dc6474262aadaf78cae2ba74e7baf67bb56642c0` |
| `m106_gt_inputs/toy03_indoor.txt` | `aa4e58bcabae622280b2c73dcf1143118ff540a3cac0202f8eb5da8e779b6888` |
| `m106_gt_inputs/toy04_indoor.txt` | `bb2d9f68b10422815d088d3d6aa9ecac6eed2bd3eb64c2d5b32c3fd239fed6d7` |
| `m106_gt_inputs/toy05_indoor.txt` | `1bb2433f521ae44c146e6df9d7bbd825daa1d11e1b8df9f36d7d98e8ebd524ec` |
| `m106_gt_inputs/toy06_indoor.txt` | `21ec622063569d66a1d4135a7f86b2b983070fbd3f2331d2bd18ec7aacb4cfc6` |
| `m106_gt_inputs/toy07_indoor_320.txt` | `683e8ae7ae401b71b8d10e9bb489c3956a150163606f5bac925a911f395444e2` |
| `m106_gt_inputs/toy08_indoor.txt` | `6c471ef8b84b8cd168a4ea3b7e578131a659bc59ca4207afb6d54bc819feec4d` |
| `m106_gt_inputs/toy10_indoor.txt` | `018c4dd5c17df16c7e186a4ad2f48759ba4836cedfe8fbf43edc53ba40c61e14` |
| `m106_gt_inputs/toydog_indoor.txt` | `837f3df1f6bac05974c27339219bc9d61c8943a73ba03f8aa7f7cf4aaa6c4b01` |
| `m106_gt_inputs/trashbin_indoor.txt` | `a10a14dc93003e0b3f111f92ebe093ebff80d856f9d01637af43bc5dc057edb1` |
| `m106_gt_inputs/tree_wild.txt` | `3563aa656258cfab370ff9aa39a887500afa9d80a20aa39d273f02074cd8089e` |
| `m106_gt_inputs/trophy_indoor.txt` | `8cee409befcc8e7ddc51c9d971bcd323ae4b4d527607359c17421b222b1c9cf2` |
| `m106_gt_inputs/ukulele02_indoor.txt` | `fd075a6c2a64bd319316c63b6be190b8e16565e5e1af99147b08d65f703733d9` |
| `M106_RAW_GT_RECEIPT.json` | `dc456dfd37e626fcde809d31f7760eae301bf7da33238400439f28401f4f9bf2` |
| `M106_REPLAY_RECEIPT.json` | `bdf20d097364e885f6e9aa1a7eb380427fc980568e29d7e61898d4c2302f0652` |
| `M106_SELECTED_GEOMETRY_PLAN.md` | `9ced7ee2f76fe1ca9f5df3c038f6ec2a9e2f7b369a01d72fea02df932b7a7695` |
| `prepare_train_states.py` | `cceb53710cdbde8350000c25d6f0ed25b490e4084022b37c76062315ad641c79` |
| `run_m106_selected_geometry.py` | `9a3cd0fcc6d875256b7ae0cf7f8e1856f08ce5a8dcec7eb8218bae8c03bd9989` |
| `train_ab_visual_control.py` | `b1f5c1a458c380eae481b5f4ebccdc619524b8eb8a275ceb5d25fc7a24f49323` |
| `train_fixed_visual_selector.py` | `a8c644a8ca2201c6c36a67c7146725a08ee8a9624fd9ef111da9ca596a930784` |
| `train_local_visual_geometry.py` | `bfb63d22a4e3d2d5f86475854f31210000ebea3d14ce26b40b340b4695e562b8` |
| `train_selected_geometry_preservation.py` | `14bf528777efcf338642f49d8a4b9b00b6e7641912bb89fc3a421afeaff4e7b4` |
| `verify_m106_checkpoint_replay.py` | `7a7d4cfe069c80c329e8d4367f5e98649a493d8d6aec047dd4bbde9bc498a9d8` |
| `C:/Users/gb/AppData/Local/uv/cache/archive-v0/Fi_OUYJWFu9xFWZGDf8J2/torch/_decomp/decompositions.py` | `c07db919fdf2008e59d2254ab1e723c934ca1fc3cfc452fa4a6f20cdc87a9c56` |
| `C:/Users/gb/AppData/Local/uv/cache/archive-v0/Fi_OUYJWFu9xFWZGDf8J2/torch/version.py` | `2aba2b231a8cda9e560ecdf9eee9f34923f648a3bf7c9c156a690562a1cd6957` |
| `C:/Users/gb/AppData/Local/uv/cache/archive-v0/Fi_OUYJWFu9xFWZGDf8J2/torchgen/packaged/autograd/derivatives.yaml` | `04035b0c074f246e82c96ccf0a8ef7d43d7cf9d65d556c4242c854d28fdf81d5` |

