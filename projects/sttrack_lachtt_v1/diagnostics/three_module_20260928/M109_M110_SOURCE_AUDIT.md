# M109 / M110 completed experiment integrity audit

Date: 2026-10-02. Reviewer: **gpt-6-astra**, reasoning **max**, fresh read-only reviewer `/root/m109_integrity_m110_source`. Review independence: **same-family**. Acceptance status: **provisional**.

**Overall verdict: WARN.** The audited sources, recorded results, independently recounted numbers, hashes and checkpoint invariants are consistent. No implementation, data or artifact-integrity blocker was found. **Both M110 finals pass the original four fixed-state gates, but a useful reviewed-text increment over Empty remains unsupported.** M109 does not establish persistent historical gradient conflict.

The originally requested source audit was extended, on receipt of completed artifacts, to the completed M110 run. Relative paths below use the diagnostic directory D. The JSON companion records all 306 audited hashes, complete aggregate recounts and actual checks. No private captions, individual choices or event keys are reproduced.

## Actual checks and limitations

Independently recounted M109's **160 batch records / 176 aggregate values**, with maximum difference **4.45e-16**. Checked the norm/cosine identities and seed-2038 membership. Independently recounted all **8 M110 JSONLs / 8,058 rows**, plus the corresponding M108 rows for comparison; summary differences were at most **1.12e-16**. Logs, launch ordering, six exits, fixed budgets, gates and source hashes agree. All **28 M110 archive members** match local bytes; archive size is **2,027,862 bytes**.

Also checked 152 local GT arrays / 3,502 event targets, 14 source ASTs, the trainer diff, historical hashes and private supervision bindings. Existing CPU Torch strictly loaded all four M108/M110 finals: 35 frozen parameter tensors and 2 buffers equal M101; 19 tensors / 95,683 elements are in the permitted trainable set. The reused bank preserves all nine original M107 fields and masks.

No dependency installation, network/SSH, GPU launch, model forward/backward, optimizer update, feature extraction, recursive tracker or official evaluator was executed. Gradient vectors and candidate-box IoUs were not recomputed from the original feature cache. Full 3,039-state Empty score/quality equality remains **recorded GPU assertion evidence**, not reviewer forward replay. The JSON records executed checks, local inspection-command corrections, hashes and limitations in detail.

## A. Ground-truth provenance and fit/development separation — PASS

Dataset boxes are loaded and SHA-bound in `prepare_train_states.py:69-74`; current/previous targets are written at `:102-105`. Split construction is fixed at `:39-45,64`. My direct array/target checks confirm these bindings. Candidate IoU is calculated against the dataset target in `train_ab_visual_control.py:45-75`. Model input construction excludes the `iou` tensor and non-tensor keys/strata (`:86-94`). GT nevertheless determines offline sampling, strata and supervised targets; these are selected cached Train states, not an unselected tracking benchmark.

The text bank and A/B judgments are explicitly **model-weak, human-unconfirmed** (`m107_weak_inputs/weak_labels.json:2-4`; `train_m110_no_weak_rank.py:63-65,126`). I checked 152 initialization records, 24 candidate records and the 11 explicit fit pairs. Both private relation files agree and independently recount to ten preferred candidates with higher IoU and one with lower IoU. This is localization ordering, not physical-identity truth.

M109 takes only the fit panel into model/loss calls (`diagnose_m109_training_gradients.py:31,60-69`). The common loader constructs both splits; no development model call is made. M110 optimization uses fit tensors only (`train_m110_no_weak_rank.py:80-102`); development is used for Empty engineering checks and final readout (`:76-78,116-118,139-148`). No CDTB/VOT input occurs in the called training paths.

## B. Metric, norm, cosine and normalization calculations — PASS

IoU divides intersection by geometric union (`analyze_train_states.py:20-25`); reported means divide by state counts and threshold counts use IoU ≥ 0.5 (`train_ab_visual_control.py:114-128`). The candidate oracle is a separately reported GT upper bound, never a score denominator.

For M109, let l, p and w be flattened localization, preservation and weak-loss gradients on the same allowed coordinates. The code computes b = l + p, raw norms, w·b/(||w||||b||), ||w||/||b|| and the cosine of b+w with b (`diagnose_m109_training_gradients.py:64-83`). Cosines with zero vectors and ratios with zero denominator are null; a zero weak numerator with nonzero baseline legitimately gives ratio zero. The independently checked values obey these rules.

These are standard gradient diagnostics, not performance scores normalized to a model's maximum. Raw norms are available alongside ratios. Localization BCE, preservation and weak-pair softplus retain their original reductions and unit coefficients (`train_m108_frozen_semantics.py:95-102`; `train_ab_visual_control.py:148-153`). Weak loss averages over the pairs present in a batch; it is not a loss averaged over all 2,544 states.

The following statistics are **unweighted means of the nine paired batches**, not epoch-gradient or pair-weighted means (`summarize_m109_gradients.py:30-41`):

| Arm / snapshot | Mean ||w|| / ||l+p|| | Mean cosine w vs l+p | Opposing paired batches |
|---|---:|---:|---:|
| Generic / initial | 2.871629669 | -0.000995732 | 4/9 |
| Generic / final | 2.190598046 | -0.106145479 | 7/9 |
| Reviewed / initial | 3.801636755 | -0.007277166 | 4/9 |
| Reviewed / final | 0.915496069 | 0.187321780 | 1/9 |

Evidence: both `m109_completed/{generic,weak_text}/result.json:27-1710`; `M109_GRADIENT_SUMMARY.json:13-270`. All 40 batches per snapshot total 2,544 states and 11 weak pairs.

## C. Completion, files, numbers and provenance — PASS

M109's root result equals its two arm results; logs match the arm metadata; all three exits are zero. Launch arguments, source hashes, parent/bank/label hashes and both actual M108 final hashes agree (`run_m109_pair.py:11-34`; both arm `result.json:2-25`; `m109_completed/controller.log:1-2`). Recorded driver runtime is **16.775137901 seconds**. There is no M109 checkpoint or optimizer output.

M110 has two separate three-step sanity runs followed by two fresh fixed twelve-epoch runs. Both sanity processes finish before either training process starts. Each training history contains 12 × 40 = **480 updates** and 11 logged pair uses per epoch. Weak-pair counters describe logging, not optimization. Mean total loss equals mean localization plus preservation within 3.831e-9. Source, all logs, root/arm results, six zero exits and checkpoint hashes agree (`run_m110_no_rank_pair.py:35-54`; `m110_completed/controller.log:1-6`; both training `result.json:2-183,446-453`). Driver runtime is **48.982646227 seconds**.

`M110_ARTIFACT_MIRROR.json:3-32` agrees with all 28 local files. The archive SHA-256 is `094206e9a16acf6c533c8f7dc6c867bc4b07adb3bcf8d7641ff5397470e78c43`. Both completed finals have matching actual file hashes:

| Final | SHA-256 |
|---|---|
| M108 generic | `ec6af542b1cd4f7b10e5683ecacbe6b17c94c2799ed7c9b3bd372d5e3d054e48` |
| M108 reviewed | `f1a11bd9e3f98b5cbc4386bb497224d098c3beeec01103331001b53408cf5168` |
| M110 generic | `746ef5fc8d832627b07ffff17538b7d481379b9eac918bf62f9689a475df91e9` |
| M110 reviewed | `36f3cdf300522eeb68252f538d04c4b8e02eff6a69a45c09102c3bebf82da4cc` |

Of the prior M108 audit's 235 input hashes, **234 are unchanged**. The sole difference is the final sentence of `M108_RESULT_NOTE.md`, narrowed from global benchmark immutability to no new M108 official evaluation. I verified the original private trace copy, both hashes and the exact diff against `M108_AUDIT_CLOSURE.json:6-9`. Historical training sources, finals, raw results, bank and labels remain unchanged.

Two current wording findings were also corrected during this audit. `M109_RESULT_NOTE.md:30` now says the diagnostic performed no new official evaluation, without claiming a global artifact audit. `M109_M110_HANDOFF_APPEND.md:18` now reports terminal reading at about four minutes, matching the **239.749854565-second** receipt difference, after waiting at least three minutes. I directly verified their preserved originals, current files and `M109_M110_AUDIT_CLOSURE.json`. These were text-only corrections; no measured value, source or model changed. The complete polling history and narrated SSH timeout were not independently reconstructed.

## D. Actual-called math, weak-term exclusion, freeze and Empty checks — PASS

M109 directly evaluates each loss and calls `torch.autograd.grad` over the same 95,683 coordinates (`diagnose_m109_training_gradients.py:48-75`). It never creates an optimizer, asserts parameter gradients remain None and checks unchanged state after every snapshot (`:84-86`). The source records actual endpoint calculations, not unused helper formulas.

The trainer diff has one optimization change:

- M108 `train_m108_frozen_semantics.py:102`: loss = localization + preservation + weak.
- M110 `train_m110_no_weak_rank.py:102`: loss = localization + preservation.

Weak remains computed at lines 97-101 and is detached for logging at line 110; it is absent from the backward graph rooted at loss. All other trainer logic is AST-identical after accounting for the module description, experiment status and added provenance fields. There is no hidden nonzero weak coefficient.

The seed, order seed sequence, batch size, epochs, optimizer construction, initial M101 hash, zero-initialized readouts, allowed prefixes, mask handling and four gate formulas are preserved (`train_m110_no_weak_rank.py:12-24,60-90,145-148`). Fresh processes reload the parent for training, so sanity updates are not carried forward. The generic control retains each sequence's slot mask.

The centered full-minus-Empty branches use the same mask (`instance_ab_prototype.py:84-91`), while frozen selection layers remain differentiable with respect to semantic input (`:108-116`). Recorded sanity results show nonzero text/readout gradients after zero-readout startup. Direct CPU comparisons confirm all frozen tensors/buffers in the four finals equal M101. Stored Empty development choices and IoUs equal the parent for all 495 states.

Complete 3,039-state score/quality preservation is asserted in the called source (`train_m110_no_weak_rank.py:27-35,113-118`) and recorded in successful outputs (`m110_completed/train_generic/result.json:180-181` and the corresponding reviewed file). **I did not independently forward-replay that assertion.** The same distinction applies to original gradient and serialization-roundtrip assertions. Active evaluation calls and corresponding JSONLs exist at `train_m110_no_weak_rank.py:136-153`; no dead or phantom metric was found.

## E. Scope, causal claims and scientific interpretation — WARN

M109 reads initial and final weights on one recreated epoch-12 batch order. These are raw coordinate gradients, not historical intermediate gradients, AdamW moments, weight-decay updates or accumulated update directions. Endpoint conflict patterns cannot establish persistent training interference or uniquely explain generalization. The reviewed final has only one opposing paired batch; the generic final has seven despite its better M108 own-condition mean IoU. The qualified wording in `M109_RESULT_NOTE.md:21-25` and `M110_NO_WEAK_RANK_PLAN.md:3-7` respects this limit.

M110 is **one seed, two fixed-state finals, 130 fit / 22 development Train sequences, 2,544 / 495 valid states**, with ten fixed candidates per state. The development panel has been used across earlier experiments; it is not an untouched final test (`M108_COMPLETED_AUDIT.md:65-69`). Generic is a matched-mask control. No multiple-seed, human-identity, recursive or official tracking claim follows.

The independently recounted final results are:

| Fixed final / content | Correct / 495 | Mean IoU | Rescues / breaks | Healthy correct / 264 | Transition correct / 127 |
|---|---:|---:|---:|---:|---:|
| Native candidate | 268 | 0.524770723083 | — | 264 | 4 |
| Either M110 final / Empty | 272 | 0.530707697987 | 5 / 1 | 264 | 4 |
| Generic / generic | 272 | 0.530402967976 | 5 / 1 | 264 | 4 |
| Generic / reviewed | 272 | 0.530707697987 | 5 / 1 | 264 | 4 |
| Reviewed / generic | 272 | 0.530707697987 | 5 / 1 | 264 | 4 |
| Reviewed / reviewed | 272 | 0.530402967976 | 5 / 1 | 264 | 4 |

Evidence: six development JSONLs, lines 1-495 each; both M110 training `result.json:236-451`; `M110_CPU_RECOUNT.json:5-366`. Both own-condition finals pass the four existing gates. However, each own-condition mean is **0.0003047300107551344 below Empty**. For the reviewed final, reviewed input changes six selections versus Empty, with zero IoU improvements and one decrease. The generic final's reviewed input has equal mean IoU to Empty; it does not establish a positive increment over that parent. Fit correct counts are 1,608 and 1,605 versus M101's 1,602; these are in-sample results.

Relative to M108, generic own-condition mean IoU changes by **-0.002035586197**, correct count by -2, healthy breaks from 2 to 0, and transition-correct from 5 to 4. Reviewed own-condition mean changes by **+0.000144464064**, correct count is unchanged, and transition-correct rises from 3 to 4. Thus the observed gate deficits disappear, but the ablation does not deliver a net useful reviewed-text increment. This is a single-seed controlled objective comparison, not proof that the endpoint gradient statistic caused these differences.

The numerical tables in `M109_RESULT_NOTE.md:14-19`, `M110_RESULT_NOTE.md:16-31` and `M109_M110_HANDOFF_APPEND.md:5-31` match the raw outputs. The two scope/timing wording findings are closed as described above. No global official-result immutability claim is made by this audit.

## F. Diagnostic and evaluation classification — PASS

| Evidence | Classification | Allowed interpretation |
|---|---|---|
| Candidate localization targets and IoU comparisons | real_gt | Selected cached Train-state localization |
| Model-reviewed text and provisional A/B supervision | synthetic_proxy | Weak model supervision, not human identity GT |
| M109 gradient norms/cosines | Objective diagnostic with mixed real_gt/proxy supervision | Initial/final gradient geometry only |
| Frozen tensors, Empty and no-mutation checks | self_supervised_proxy / engineering invariant | Functional or state preservation, not tracking accuracy |

Evidence: `train_ab_visual_control.py:75`, `m107_weak_inputs/weak_labels.json:2-4`, `diagnose_m109_training_gradients.py:64-86`, `train_m110_no_weak_rank.py:38-50,113-118`. There is no human_eval, simulation-only claim, official DepthTrack Test/CDTB/VOT result or recursive tracking evaluation in this run.

## Actions and claim impact

- Preserve and report both M110 gate passage and the absence of a net semantic benefit. Gate passage alone does not justify promotion.
- Keep the single-seed, selected cached Train-state, matched-mask, model-provisional label and reused-development qualifications.
- Keep M109 claims at initial/final raw gradients. Do not present them as historical AdamW updates or a causal explanation of generalization.
- Distinguish my direct recount/hash/tensor checks from recorded GPU Empty/gradient assertions.
- The two wording corrections are verified and closed. No source fix, training replay or new GPU task is required to report these results.

This is finite same-family provisional assurance. The full original request, supplements and actual response are retained privately at `R/.aris/traces/experiment-audit/2026-10-02_m109_m110_source/`.

