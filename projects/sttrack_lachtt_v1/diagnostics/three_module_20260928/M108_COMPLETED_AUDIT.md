# M108 completed experiment integrity audit

Date: 2026-10-02. Auditor: **gpt-6-astra**, reasoning **max**, fresh reviewer `/root/m108_completed_integrity`. Review independence: **same-family**. Acceptance status: **provisional**.

**Overall verdict: WARN.** The stored results, numerical summaries and hashes are internally consistent. Independent checks verify dataset GT provenance, the frozen M101 tensors, and preservation of the original text bank. **Both scientific promotion gates fail.** This experiment supports a negative result on selected cached Train states; it does not establish a semantic benefit or official tracking improvement.

All relative paths below are under `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928` (D). “Private review root” means `C:/Users/gb/Desktop/document/RGBD_INITIALIZATION_REVIEW_20260928`. No raw private phrases, choices tied to identities, or review notes are reproduced. The JSON companion contains the complete audited-input hash inventory and independently recomputed per-stratum summaries. Trace path: `.aris/traces/experiment-audit/2026-10-02_m108_completed/`; the parent executor retains the actual request/response trace.

## What I executed

I independently recounted **all eight JSONL files**: 2 × 2,544 fit rows and 6 × 495 development rows, **8,058 stored rows**. I checked unique keys, candidate-index bounds, finite IoUs, oracle bounds, summaries, all four gates, same-weight comparisons, and parent references. Counts matched exactly; floating summaries matched within `1e-12` using an independent `math.fsum` aggregation.

I read all **152 saved GT arrays**, verified their SHA-256 values against the source spec, and checked all **3,502 current/previous event targets** against those arrays. The arrays contain 219,993 rows. All 3,039 valid states and their strata match the output sets; 463 events with invalid current GT are excluded.

An existing local environment successfully loaded **Torch 2.14.0+cpu** at `C:/Users/gb/AppData/Local/uv/cache/archive-v0/BhiD8AmYwGCba9pUBzFx6/Scripts/python.exe`. I strictly loaded both final state dictionaries, compared tensors to M101, and compared both private banks. No installation was performed. The initial managed-Python probe lacked Torch; the existing cached environment worked. Torch warned that NumPy was absent; these tensor comparisons did not require NumPy.

I did **not** execute SSH, a GPU job, model forward/backward, optimizer updates, CLIP encoding, feature extraction, an official evaluator, or a recursive tracker. No source/results were changed. The all-state Empty score/quality equality and original gradient checks therefore remain **recorded GPU assertion evidence**, distinct from my direct CPU checks.

## A. Ground-truth provenance — PASS

Localization targets come from dataset `groundtruth.txt`, not model predictions: `prepare_train_states.py:69-78` loads and hashes dataset boxes; `:102-105` writes current/previous targets. `../full152_paired_20260925/M82_training_spec.json:6-7` identifies the DepthTrack Train root and sequence inventory. The spec SHA matches `preparation.json:3`; `m105_inputs/training_labels.json:1` matches the label SHA in `preparation.json:10`. All 152 mirrored files under `m106_gt_inputs/` were opened, hashed and compared to the spec, then all 3,502 event targets were checked directly.

`train_ab_visual_control.py:45-75` computes candidate IoU against those dataset boxes. `:86-94` excludes `iou`, keys and strata from model input. `collect_train_states.py:99-153`, `collect_train_contexts.py:94-143`, and `collect_initial_feature_origins.py:50-90` collect visual inputs from images, tracker outputs and the initialization bbox. The initialization bbox is legitimately dataset provided. The narrower supported claim is that **current/future target coordinates are absent from the model feature dictionary**. GT still determines offline event sampling, strata and supervised losses. Candidate feature tensors were not inspected in this audit, so this part is source/receipt evidence, not a fresh feature-level replay.

Private identity/text supervision is explicitly model derived: `m107_weak_inputs/weak_labels.json:2-4` records model-weak status and `human_confirmed=false`; `:2398-2406` binds the inputs and phrase protocol. The private `初审/审核回执.json:3-8` describes model image-board review, no human confirmation and no complete original-video review. Its statement that current GT was not used for choices is receipt evidence, not something I personally witnessed.

I verified all four private input hashes; all 152 first category phrases and first four pipe-separated supported attributes exactly match the reviewed initialization CSV; all 24 candidate choices match the candidate CSV. The 11 explicit choices used for ranking lie in fit; 13 uncertain records are ignored (`train_m108_frozen_semantics.py:81-87`). The exact `initial_binding_sha256` target is **`multiframe_review/full152_multiframe_model_prefill.csv`**, not `audit_register.json`: its digest and all 152 ID/sequence/split mappings match. The candidate digest resolves to `sources/candidate_review_private_selection_v2.json:6`, whose 24 sequence/frame/A/B mappings match. A separate `audit_register.json:3-5` mapping and initial-bbox cross-check also passed. No semantic-correctness or human-identity certification follows from these structural checks.

## B. Score normalization — PASS

IoU divides intersection by geometric union (`analyze_train_states.py:20-25`). Summary means divide by state count, and threshold counts use IoU ≥ 0.5 (`train_ab_visual_control.py:114-128`). The oracle is a separately reported candidate upper bound, never a normalization denominator (`train_m108_frozen_semantics.py:45-49`).

The model's masked pooling and phrase-slot means use valid-support counts (`instance_ab_prototype.py:16-18,112`); native-preservation loss averages over eligible pairs (`train_ab_visual_control.py:148-153`). None rescales a reported performance metric by the model's own maximum, minimum or mean. Healthy 100% results are a GT-selected stratum property, not a claim of 100% overall tracking.

## C. Files, calls, metrics and hashes — PASS

All 32 completed artifacts exist. All seven exit files contain zero. Preparation, the four launches, launch ordering, two three-step sanity runs, twelve logged epochs per training arm, final summaries and driver receipt agree. Each training history records 40 steps and 11 weak-pair uses per epoch: 480 steps and 132 pair uses. The driver records 60.616015911 seconds. See `m108_completed/controller.log:1-6`, both `train_*.log:1-13`, both `sanity_*.log:1-2`, and `run_m108_frozen_pair.py:40-55`.

The compressed archive `m108_public_results_20261002.tar.gz` is **2,023,465 bytes**, SHA-256 `89cfbb29b813b4045dbd0bcbb8da4d779be241e43e38756cd659b05d1e37da17`, exactly matching `M108_ARTIFACT_MIRROR.json:2-5`. All 32 file members match local bytes. The internal manifest has 30 hashes (`m108_completed/archive_receipt.json:4-33`); its own file and `controller.log` are the other two archive members. This is accounted for, not a missing-member discrepancy.

Deployed source hashes match `M108_DEPLOY_RECEIPT.json:3-8`; recorded trainer/prototype/input-helper hashes match local source. Parent, label, bank and final hashes also match. The final checkpoint hashes are:

| Artifact | SHA-256 |
|---|---|
| M101 parent | `1d187d78c1dd76ebf59e41fd31af16fa69cecc37b4cb6c74fbdfe3ed5c237293` |
| Generic final | `ec6af542b1cd4f7b10e5683ecacbe6b17c94c2799ed7c9b3bd372d5e3d054e48` |
| Reviewed-text final | `f1a11bd9e3f98b5cbc4386bb497224d098c3beeec01103331001b53408cf5168` |

All fit/development summaries, strata, gates and same-weight content numbers independently match the JSONLs, both arm results, driver result, and `M108_CPU_RECOUNT.json`. The newly supplied `M108_RESULT_NOTE.md:32-50` table and numerical narrative match too. No separate paper or broader author claim set was supplied.

Logs are preserved evidence of execution, not a fresh hardware witness. Finals contain model state, not optimizer state, and fit JSONLs are post-training evaluation rows rather than step traces. I did not recreate 480 updates or independently recompute candidate-box IoUs without the feature caches.

## D. Active dataflow, frozen path and Empty — PASS

`train_m108_frozen_semantics.py:12-24` defines permitted semantic prefixes and the three content conditions. `:68-79` freezes the parent, enables only those prefixes and initializes both readouts to zero. Training is not wrapped in `no_grad` (`:93-110`). The frozen selection MLP remains differentiable with respect to its semantic input (`instance_ab_prototype.py:108-116`). Sanity logs show zero upstream text gradients at the first zero-head step, then nonzero text/attention/readout gradients; this is the expected zero-readout startup, not a dead semantic path.

My direct checkpoint comparison found **56 state entries**, including **35 frozen parameter tensors / 146,311 elements** and **2 buffers / 56 elements**, all bitwise equal to M101. There are **19 trainable tensors / 95,683 elements**; 18 differ from the parent and every changed tensor is inside an allowed prefix. Both final states load strictly and are finite. Tensor changes alone do not prove every changed value received a useful task gradient.

The M108 bank preserves **all nine original M107 fields**, including tokens, mask and Empty tensor, exactly. Only `generic`, `generic_phrase` and `m107_bank_sha256` are added; the finite 768-dimensional generic vector differs from Empty. Actual CLIP encoding was not replayed. See `prepare_m108_generic.py:13-30` and `m108_completed/text.json:2-8`.

The centered full-minus-Empty branches use the same mask (`instance_ab_prototype.py:84-91`). Both stored Empty development files are byte-identical and all 495 choices/IoUs equal the M101 rows (`m101_completed/train_weight1/result.json:222`). Full 3,039-state pre/post **scores and quality** are checked by code at `train_m108_frozen_semantics.py:27-35,113-118`, with successful receipts at both arm `result.json:174-176`. Those arrays are not serialized, so I do not claim to have personally replayed that equality. Historical save/load equality is likewise an assertion at `:147-150`; my independent check is strict loading and tensor comparison.

The active metric function is called for fit and all three development conditions (`:134-140`), with outputs present. No phantom/dead metric was found. Observation and the three phrase channels do not have independent identity/support/conflict/visibility evaluation; calibrated interpretations are unsupported.

## E. Scope and fixed gates — WARN

The scope is **one seed, two finals, 130 fit Train sequences / 2,544 valid cached states, and 22 disjoint Train development sequences / 495 states**. I checked sequence and key disjointness directly. Event/stratum selection uses GT and native behavior (`prepare_train_states.py:79-105`); strata overlap and cannot be summed as exclusive partitions. The fixed candidate count is 10.

The source uses seed 2027, fixed twelve epochs and final-only saving, without per-epoch development-based checkpoint selection (`train_m108_frozen_semantics.py:60-61,88-118,134-151`). Sanity states are not saved or carried into training. However, earlier M107 development results motivated M108 (`M108_FROZEN_SEMANTIC_PLAN.md:3-8`), so this development set is not an untouched final test set. Generic also retains the same per-sequence phrase masks; it is a matched-mask control.

| Fixed final / input | Correct / 495 | Mean IoU | Rescues / breaks | Healthy correct / 264 | Transition correct / 127 |
|---|---:|---:|---:|---:|---:|
| Native candidate | 268 | 0.524770723083 | — | 264 | 4 |
| Either final / Empty | 272 | 0.530707697987 | 5 / 1 | 264 | 4 |
| Generic / generic | 274 | 0.532438554174 | 9 / 3 | 262 | 5 |
| Generic / reviewed | 272 | 0.530010909379 | 5 / 1 | 264 | 4 |
| Reviewed / generic | 272 | 0.531629981246 | 5 / 1 | 264 | 4 |
| Reviewed / reviewed | 272 | 0.530258503913 | 5 / 1 | 264 | 3 |

Evidence: six `m108_completed/train_{generic,weak_text}/*_development.jsonl:1-495`; each arm `result.json:230-438`. Complete independently recounted all/healthy/intermediate/late-low/transition summaries, including fit, are in this audit's JSON.

| Own-condition fixed gate | Generic | Reviewed text |
|---|---|---|
| Correct count exceeds native | PASS: 274 > 268 | PASS: 272 > 268 |
| Mean IoU exceeds native | PASS | PASS |
| Zero healthy breaks | **FAIL: 2** | PASS: 0 |
| Transition correct ≥ native | PASS: 5 ≥ 4 | **FAIL: 3 < 4** |
| Joint decision | **FAIL** | **FAIL** |

Gate formulas match `train_m108_frozen_semantics.py:143-146`, the existing M101 definitions in `train_ab_visual_control.py:240-244`, and saved `result.json:440-445` in both arms.

| Same final, reviewed input versus | Changed selections | IoU improved / harmed | Mean-IoU change |
|---|---:|---:|---:|
| Generic final: generic | 21 | 3 / 9 | -0.002427644794 |
| Generic final: Empty | 10 | 1 / 2 | -0.000696788608 |
| Reviewed final: generic | 13 | 6 / 3 | -0.001371477334 |
| Reviewed final: Empty | 15 | 7 / 3 | -0.000449194074 |

All four changes are negative; more small improvements than harms does not imply positive mean IoU. The reviewed-text semantic-benefit claim is unsupported. The fit summaries are in-sample diagnostics (generic 1,617/2,544; reviewed 1,618/2,544), not development or official performance.

## F. Evaluation classification — PASS

| Evidence | Classification | Claim boundary |
|---|---|---|
| Fit localization | `real_gt` | In-sample cached Train states |
| Development content conditions and same-weight comparisons | `real_gt` | Selected, sequence-disjoint Train development states |
| Top-10 oracle | `real_gt` | GT upper bound over fixed model-generated candidates |
| Model-reviewed phrases and A/B supervision | `synthetic_proxy` | Weak model supervision; no standalone identity accuracy |
| Frozen/Empty/roundtrip checks | `self_supervised_proxy` | GT-free engineering invariants in this taxonomy, not tracking accuracy |

No `human_eval`, `simulation_only`, official DepthTrack Test/CDTB/VOT evaluation, or recursive M108 tracking performance was established. Source evidence: `train_m108_frozen_semantics.py:38-50,81-118,134-151`, `prepare_m107_weak_text.py:12-30`, and `run_m108_frozen_pair.py:49-54`.

## Claim impact and actions

- **Supported:** exact frozen parent tensors; preserved original bank; stored development metrics; both failed gate components; no positive reviewed-text mean-IoU increment.
- **Supported by recorded GPU assertions, not independently forward-replayed:** complete 3,039-state Empty score/quality equality, original finite gradients and save/load roundtrip.
- **Unsupported:** net semantic benefit, human-confirmed identity truth, calibrated phrase channels, multiple-seed robustness, recursive recovery or official benchmark gains.
- Preserve this negative result and its fixed gates. Reporting improvement over the native candidate alone would omit the stronger preserved Empty parent and generic controls.
- Keep the selected Train-state, single-seed, matched-mask and model-label qualifications. Separately version any later human-confirmed labels.
- Preserve exact binding paths: the initial digest belongs to the CSV; the candidate digest belongs to the private JSON. No unresolved digest discrepancy remains.
- `M108_RESULT_NOTE.md:70` says public benchmark results are unchanged. This audit establishes that the M108 runner performs no official evaluation; it did not compare all other public-result files before and after. Scope that sentence accordingly.
- No source change or rerun is needed merely to preserve and report the current negative result. The remaining execution limits should stay explicit.

No numerical discrepancy, failed hash check, split overlap, forbidden final tensor change or mislabeled human evaluation was found in the audited artifacts. This is finite, same-family provisional assurance, not certification of all historical execution or label semantics.

