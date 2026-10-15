# Experiment audit: M125 R0 fixed six-pass source

**Source verdict: PASS_SOURCE_ONLY. Unresolved blockers: 0. Required source fixes: none.**

**Overall assurance: WARN — native R0 training/evaluation is unexecuted in this review and requested model routing is UNATTESTED.** This is a fresh same-family review with provisional semantic acceptance. Requested routing was `gpt-6-astra` / `max`; the host supplied no attestation of actual model or effort, so this report does not attribute an actual Astra execution.

The nine candidate modules, 44 captured dependencies and original counterparts remained byte-identical. The executor's additional real64 admission script initially lacked a verdict/source-binding predicate; that concrete issue was corrected before this final review and the revised predicates pass negative fixtures. No unresolved source defect was found in the final reviewed bytes. No original experiment or input source was edited by the reviewer.

## Evidence and scope

- **73 primary local inputs hashed**. Candidate sources, dependencies, original source counterparts, payload/receipt and revised operators remained unchanged. One concurrent append to `EXPERIMENT_PLAN.md` was caught by the original guard and separately reviewed; both versions and the genuine failed guard are preserved. The smaller policy hash list is separate.
- **341 deterministic checks, 0 failures:** 207 inventory/hash/AST/payload checks; 36 training/control-flow/exit/arithmetic fixtures; 65 inherited metric/staging/operator checks; 33 preparation/wrapper/manifest fixtures.
- The supplied real staging receipt contains **53 copied source hashes and nine CPU `runpy` imports with exit 0**. Its candidate-gate digest is reproducible from the source payload: `18580735b4234ecbadbab8508f9344640a29695d10a52a995e9d686ac6424d16`.
- Reviewer actions: **0 NN calls, 0 SSH calls, 0 GPU queries, 0 installs, 0 source edits, 0 original-experiment mutations**. No public/external notification was sent.
- Synthetic fixture output is clearly marked and kept under `SYNTHETIC_CONTROL_FLOW_NO_NN`; it is not training, checkpoint, or performance evidence.

Candidate SHA-256: `a3d019bbf534d816c993c70987e4f3fefacfc10d636b5dbc06b6083149ac5ef2`.

The report covers R0 only. The 7,288-entry E6 artifact file was a lookup source for 12 direct inherited metric/metadata entries. No E6 numerical-result re-audit or R1/R2 experiment/design review was performed.

## Findings against the requested protocol

| Check | Source conclusion and evidence |
|---|---|
| Common warm initialization | `train_m125_R0_fixed_pass.py:24-94` binds the exact spec/bank/labels/native/CLIP/common-warm hashes. Pass one loads only common warm A_B/C, creates a fresh AdamW, asserts empty optimizer state and exact warm parameters, then reseeds Python, NumPy, CPU Torch and all CUDA RNGs after construction. Formal histories start at zero. |
| Complete budget | `train_m125_R0_fixed_pass.py:115-133,206` and `run_m125_R0_fixed_suite.py:50-67` visit the complete sealed sequence order in every pass; no retained 37-sequence prefix or extra sanity optimization enters formal history. The source spec has 152 unique sequences and 219,802 track calls per pass: **912 sequence runs and 1,318,812 track calls** over six passes. Real64 is separately discarded. |
| Resume state | Training `:54-59,85-93,194-205` rejects the old M124 format and partial boundaries, strictly restores A_B/C, all Adam state and four RNG classes, verifies model/frozen digests, and retains cumulative counters/records. Frozen backbones are reconstructed from immutable weight files and digest-checked; dynamic tracker state intentionally resets for the next sequence. Stub fixtures verify control flow, not numerical Torch equivalence. |
| Inherited update behavior | The complete per-frame prediction/loss/backward/normalization/clipping/optimizer-update AST is **identical** to the captured M124 implementation. `:133-180` retains accumulation 32, AdamW lr 3e-5 / weight decay .01, per-module gradient normalization and clipping, and current-IoU C loss. No extra objective, gradient path, or temporal backpropagation was introduced. |
| GT and own history | `m123_joint_full_tracker.py:34-84` commits model-selected bbox/query before the outer training loss. `train_m125_R0_fixed_pass.py:134-156` reads the current target only after `tracker.step`; `write_features` detaches current/past/reference inputs. The GT file is loaded once before the sequence (`load_truth:14-22`), so this is a causal-use statement, not a claim that the file is unopened before prediction. No GT reset occurs after legal sequence initialization. |
| Freeze and parameter counts | `full_dense_tracker.py:31-40,53-73` freezes/evaluates STTrack and CLIP and observes under `no_grad`. Training `:207-220` checks state digests, absence of frozen gradients and final serialization equality. Symbolic constructor arithmetic independently gives **A_B 380,167 + C 70,721 = 450,888** trainable parameters. |
| Real64 witness | `check_m125_R0_real64.py:31-108` starts from the common warm/fresh Adam state, uses the first sequence's own 64-frame history, checks 256 candidates, two AB/C updates with finite nonzero module gradient norms, frozen digests/no frozen gradients, changed A_B/C digests, serialized model/Adam equality, and optimizer reload. The first C batch has 31 targets and the second 32 because frame one has no previous feature. It does not assert per-parameter nonzero gradients or a resumed trajectory equivalence that it never measures. |
| Fixed final and no selection | Controller `:50-67` completes all six training processes before `prepare`; `:69-93` evaluates only pass-six final. Runtime `:23-37` rejects an earlier pass even after metadata hashes are recomputed in a fixture. There is no inter-pass OPE/VOT call, early stopping, external-score checkpoint selection, or sanity checkpoint resume in this path. |
| Full OPE50/80 | `run_m125_R0_fixed_official.py:12-61` requires 50/76,373 and 80/101,956 sequence/frame totals, seals prediction/confidence files, then validates real dataset GT and invokes the hash-bound long-term PR evaluator. All trajectories use the same final weight and bound legal initialization bank. |
| Full VOT127/1765 | Preparation `:55-94` preserves the inherited 127-sequence/1,765-anchor manifest and writes an R0-specific wrapper/INI. The inherited runner `run_m123_joint_vot_shards.py:23-61` waits for all four shard children, requires all 1,765 trajectories and 5,295 files, merges to a fresh master, and records exit codes. Controller official-analysis name and analyzer/collector lookup names agree. |
| Metric/manifest compatibility | The direct OPE metric and VOT failure collector sources match their bound hashes. The inherited manifest contains exactly 1,765 unique trajectories over 127 sequences. The actual R0 preparation/check functions passed synthetic filesystem fixtures preserving the same trajectory names and source/plan/wrapper bindings. The original M122 base-plan verifier itself was stubbed in that fixture; actual invocation still must pass its checks. |
| Source/import binding | The supplied stager `:25-32` copies all 44 dependencies beside the nine new modules and adds each copied path/hash to the candidate gate. Thus the sibling tracker hash and `code_root/run_m123_joint_vot_shards.py` resolve to actual copied sources. The staged 53-source map equals the payload and receipt exactly. |
| Status and failure handling | `run_m125_R0_fixed_suite.py:14-31` writes actual child return codes then fails without retry; the supervisor `:16-25` returns the controller's actual exit. Negative fixtures preserve exit 7 and prevent further training/evaluation or a success exit file. The revised real64/formal operators bind the review, source set, actual real64 result/exit and warm hash before a single formal launch. |

File/line references above are relative to `M125_R0_FIXED_SOURCE_20261015`; inherited files are under `frozen_dependency_snapshots`. Additional operator paths are in its parent directory. Exact absolute paths and digests are in `EXPERIMENT_AUDIT.json`.

## Experiment-audit A–F

| Item | Status | Evidence boundary |
|---|---|---|
| A. Ground-truth provenance | PASS | Dataset GT supplies training IoU/loss and formal evaluation. No model-generated target is treated as ground truth. OPE annotations are opened only after sealed predictions; legal VOT anchor initialization is part of the protocol. |
| B. Score normalization | PASS | Bound OPE `depthtrack_pr.py:117-153` uses per-sequence overlaps/visible counts, macro-averaged PR and the standard max-F operating point. VOT `multistart.py:115-188,318-442` uses dataset overlap and run-length aggregation. Reporting only converts fractions to percentages. No self-max metric rescaling is present. |
| C. Result existence | PASS for claim boundary | Candidate status and actual staging/import receipt are real inputs. No new trained R0 weights or nine metrics are claimed to exist. Future result-writing branches are not evidence they have executed. |
| D. Reachability | PASS | All intended OPE/VOT metric, sealing and collection entry points are called by the formal controller; the VOT generated wrapper imports the R0 runtime. The separately invoked witness is bound by the formal operator. |
| E. Scope | WARN | This is one-seed prospective source coverage. Complete training, real64 and full metric artifacts are not certified. Historical Test/CDTB engineering exposure and multiframe human-review aids remain disclosed; this is not a clean-unseen-test claim. |
| F. Evaluation type | `real_gt`, prospective | Local reviewer fixtures are explicitly synthetic control-flow/metadata tests and supply no performance measurements. |

The OPE max-F confidence threshold is part of reporting the locked long-term metric. It does not select among training checkpoints. The cached-teacher history of the common warm C also remains historical provenance; `no_cached_feature_fit` refers to the new six-pass stage.

## Resolved admission issue and limits

The first supplied `launch_M125_R0_real64_20261015.py` read and copied audit bytes without evaluating them. The executor added strict assertions for `PASS_SOURCE_ONLY`, an empty blocker list, the unchanged candidate digest and the exact staged source map. Current local lines 18–20 and remote lines 33–37 implement those checks; the formal operator repeats the relevant binding at lines 21–25. Four negative predicate fixtures reject invalid verdict/blocker/candidate/source-map inputs. The first operator version was read before revision and was not hash-captured; this limitation is retained rather than inventing a prior digest.

No further source fix is required by this review. A source PASS permits the already-authorized native witness/admission workflow; it is **not a certificate that real64, six-pass training, an optimizer resume, complete evaluation, target attainment, or an effective scientific contribution has occurred**. Real64's saved-model/Adam equality describes serialization and load checks, not a separately measured identical resumed rollout. Requested reviewer model/effort routing also remains unverified.

Finalization initially stopped because the executor published an execution-status/diagnostic append to `EXPERIMENT_PLAN.md` during this review. The original **11,318 bytes** are an exact unchanged prefix: old SHA `5f5870228f4a5568275b3230b743cd8e5309462cf98e07f90352b2bc2e672e80`; current **12,221 bytes** SHA `83aac5c20281311c209eaf87260e29fce0b1d5624919bea4e4538bbc9c9c8dea`. The 903-byte suffix exactly matches the supplied publication packet, whose `before_sha256` matches the original audit hash; the supplied publication receipt binds the new digest and commit `1c002d00579ce323a83d65bb6ee68d15f21bb80e`. The appended lines 91–93 explicitly retain seed2027/common warm/fresh Adam and RNG/six-pass final-six protocol. Its historical diagnostic numbers were not re-audited here. `input_change_incident.json` and `write_audit_report.first_failure.txt` preserve this real guard failure. The report does **not** claim every input byte was unchanged.

The original M122 base-plan bodies were not present in the supplied E6 closure. The captured E6 plans reference the same base root used by R0, and inherited `checked_base_plan` verifies their hashes at runtime. This report inspected that direct contract and did not manufacture a full original-metadata/runtime execution claim.

Required next evidence is the original supervised real64 exit and bound witness result, followed by original formal pass/checkpoint/child receipts and the same final's complete OPE/VOT artifacts. A fresh result audit is required before any performance claim. Existing source flags correctly leave `fresh_result_audit_complete` / `independent_completed_audit` false.

## Artifacts

- `EXPERIMENT_AUDIT.json`: machine verdict, exact candidate/staged-source binding, all input digests and claim limits.
- `input_hashes.before.json`, `additional_input_hashes.json`, `input_hashes.after.json`: immutable input guard.
- `inventory_checks.json`, `deterministic_checks.json`, `metric_operator_checks.json`, `planner_checks.json`, `verification_summary.json`: 341 deterministic checks and explicit test scope.
- `inherited_metric_sources/`: exact 12 retrieved source/metadata bodies and their index in `metric_operator_checks.json`.
- `.aris/traces/experiment-audit/2026-10-15_run01/`: request, supplemental path messages, response and honest routing metadata.

Reviewer tool limitations retained: the first bare `python` metadata command failed with `No pyvenv.cfg file` before any check executed. The installed `uv python find` located an existing CPython 3.13 executable; subsequent stdlib-only fixture runs used that exact interpreter without installation. The first publication-packet text read also failed under Windows' GBK default; an explicit UTF-8 read then verified the exact append. All four evidence scripts completed with exit 0. The original report guard failure is retained separately as described above. Intentional negative fixtures are recorded as expected rejections, not runtime training failures.
