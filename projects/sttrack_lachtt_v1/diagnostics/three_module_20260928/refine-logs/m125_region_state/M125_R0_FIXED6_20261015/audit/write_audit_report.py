"""Finalize this review and its forensic trace, writing only inside the audit directory."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

OUT=Path(__file__).resolve().parent
BASE=OUT.parent
SRC=BASE/'M125_R0_FIXED_SOURCE_20261015'
POLICY=Path('C:/Users/gb/.codex/skills')
TRACE=OUT/'.aris/traces/experiment-audit/2026-10-15_run01'
TRACE.mkdir(parents=True,exist_ok=True)
NOW=datetime.now(timezone.utc).isoformat()
AGENT='/root/audit_m125_r0_fixed_source'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value): Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


before=read(OUT/'input_hashes.before.json')
additional=read(OUT/'additional_input_hashes.json')
inputs=dict(before,**additional)
after={p:dict(sha256=sha(p),bytes=Path(p).stat().st_size) for p in inputs}
changed=[p for p in inputs if inputs[p]!=after[p]]
incident=read(OUT/'input_change_incident.json')
assert changed==[incident['path']],changed
assert incident['original_prefix_byte_exact'] and incident['packet_append_exact'] and incident['packet_before_sha_matches_initial']
assert after[incident['path']]==incident['after']
assert sha(OUT/'EXPERIMENT_PLAN.final_read.md')==incident['after']['sha256']
publication=read(BASE/'M125_R2_result_publication_packet_20261015_publication.json')
relative_plan='projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/refine-logs/m125_region_state/EXPERIMENT_PLAN.md'
assert publication['server']['files_sha256'][relative_plan]==incident['after']['sha256']
incident['publication_commit']=publication['commit']
incident['receipt_final_plan_sha_verified']=True
write(OUT/'input_hashes.after.json',after)
inventory=read(OUT/'inventory_checks.json')
deterministic=read(OUT/'deterministic_checks.json')
metrics=read(OUT/'metric_operator_checks.json')
planner=read(OUT/'planner_checks.json')
counts=[inventory['counts']['checks'],deterministic['count'],metrics['count'],planner['count']]
assert counts==[207,36,65,33]
assert all(r['status']=='PASS' for group in [inventory,deterministic,metrics,planner] for r in group['checks'])
candidate=read(SRC/'CANDIDATE.json')
candidate_sha=sha(SRC/'CANDIDATE.json')
policies=[POLICY/'experiment-audit/SKILL.md']+[POLICY/'shared-references'/n for n in
    ['local-codex-policy.md','reviewer-independence.md','experiment-integrity.md','review-tracing.md','integration-contract.md']]
write(OUT/'policy_input_hashes.json',{str(p):sha(p) for p in policies})
write(OUT/'verification_summary.json',dict(
    completed_at=NOW,primary_input_files=len(inputs),primary_inputs_unchanged=False,
    source_inputs_unchanged=True,reviewed_document_append=incident,
    checks=dict(inventory_hash_AST_payload=207,training_controller_exit_fixtures=36,
                inherited_metrics_staging_operator_checks=65,planner_wrapper_manifest_fixtures=33,total=341,failed=0),
    actual_supplied_CPU_imports=9,reviewer_NN_calls=0,reviewer_SSH_calls=0,reviewer_installs=0,
    reviewer_source_edits=0,reviewer_experiment_mutations=0,
    fixture_warning='All synthetic metadata are under SYNTHETIC_CONTROL_FLOW_NO_NN; they are not experiment results.'))

checks={
 'gt_provenance':dict(status='PASS',evaluation_type='real_gt',execution='PROSPECTIVE_SOURCE_ONLY',
    evidence=['train_m125_R0_fixed_pass.py:120-156','frozen_dependency_snapshots/train_m122_full_causal.py:14-31',
              'frozen_dependency_snapshots/m123_joint_full_tracker.py:34-84','run_m125_R0_fixed_official.py:23-61',
              'inherited_metric_sources/depthtrack_pr.py:95-129','inherited_metric_sources/finalize_vot_transaction_low22.py:193-258'],
    details='Dataset annotations provide targets. Training loads a GT file before the sequence but consumes frame targets for loss only after the actual step; no current GT is passed into step or online state. OPE opens subsequent GT after predictions are sealed; VOT uses legal multi-start initialization and official dataset GT.'),
 'score_normalization':dict(status='PASS',evidence=['inherited_metric_sources/depthtrack_pr.py:117-153',
    'inherited_metric_sources/multistart.py:115-188','inherited_metric_sources/multistart.py:318-442','analyze_m125_R0_fixed_vot.py:22-35'],
    details='No reported metric divides by a model-output maximum. OPE uses confidence-threshold PR curves, dataset-visible frame denominators, macro averaging, and max-F operating point; this is not checkpoint selection. VOT uses official overlap/length aggregation; only final percentage conversion multiplies by 100.'),
 'result_existence':dict(status='PASS',evidence=['CANDIDATE.json:status','M125_R0_candidate_staging_20261015_stdout.json',
    'run_m125_R0_fixed_suite.py:95-101','collect_m125_R0_fixed_results.py:15-49'],
    details='Only staging and nine CPU imports are supplied actual R0 execution evidence. Candidate/report do not claim completed real64, six-pass training, or nine metrics. Result writers and completeness checks are reachable future code, not already-existing NN results.'),
 'dead_code':dict(status='PASS',evidence=['launch_M125_R0_real64_20261015.py:50-64','launch_M125_R0_fixed_full_suite_20261015.py:26-43',
    'run_m125_R0_fixed_suite.py:50-93','run_m125_R0_fixed_official.py:81-84','prepare_m125_R0_fixed_evaluation.py:59-88'],
    details='Real64 has its separate operator; formal controller calls six train processes, prepare, both OPE track/analyze pairs, inherited VOT shards, official analysis, seal and collector. The generated VOT wrapper imports the R0 entry. Copied legacy selection/cache helpers are not executed by this path.'),
 'scope':dict(status='WARN',evidence=['EXPERIMENT_PLAN.md:27-35','EXPERIMENT_PLAN.md:85-89','run_m125_R0_fixed_suite.py:50-100'],
    details='Source supports one seed, six complete Train152 passes, one final pass-six checkpoint, OPE50/80 and VOT127/1765. No new NN/performance evidence is supplied; historical Test/CDTB engineering exposure, warm-start history and multiframe human-review aids remain disclosed. Source audit cannot establish full training/evaluation success or scientific effectiveness.'),
 'eval_type':dict(status='PASS',classification='real_gt',execution='PLANNED_NOT_RUN_IN_THIS_REVIEW',
    details='Prospective formal evaluation uses dataset-provided GT. Reviewer fixtures are explicitly synthetic control-flow/metadata tests and have no performance claim.')}

limits=[
 'Requested reviewer routing gpt-6-astra / max is not host-attested. Actual model and actual reasoning routing remain UNATTESTED; semantic review is same-family/provisional.',
 'Nine supplied CPU runpy imports prove import compatibility only. No torch/NN, SSH, GPU query, deployment, package installation, or experiment execution was performed by this reviewer.',
 'Real64, native gradients, frozen-state equality, checkpoint/Adam tensor round-trip, actual pass-boundary resume, all six complete passes, full OPE/VOT metrics and performance are not established by these deterministic fixtures.',
 'Real64 checks finite module-level A_B/C gradients and nonzero norms, changed module digests, frozen model digests/no gradients, saved tensor/optimizer equality, and optimizer load. It does not claim per-parameter nonzero gradients, bit-identical resumed trajectories, or full-boundary RNG numerical equivalence.',
 'The 7,288-entry E6 closure was used only to retrieve 12 relevant inherited source/metadata entries and inspect direct metric/manifest interfaces. This is not a re-audit of E6 numerical results or the entire official toolkit.',
 'The original M122 base-plan bodies were not in the supplied E6 closure. Direct E6 base-plan references match the candidate base root; the inherited checked_base_plan performs their SHA checks on actual execution. The local planner fixture deliberately substitutes that upstream verifier.',
 'Frozen parameters are reconstructed from bound native/CLIP weights on each pass and compared by full state digests; they are not duplicated inside the learning checkpoint. Online sequence state is intentionally reinitialized at each sequence boundary.',
 'no_cached_feature_fit and no_development_split describe the new six-pass stage, not the entire historical provenance of the common warm model. The warm C was previously fit on cached Train152 teacher events.',
]

resolved=[dict(id='OPERATOR-01',status='RESOLVED_BY_EXECUTOR_BEFORE_FINAL_REVIEW',severity='blocking_before_fix',
  issue='The first provided real64 operator copied audit bytes and marked the gate reviewed without parsing the verdict or binding reviewed sources.',
  initial_evidence='First read of launch_M125_R0_real64_20261015.py lines 13-17 and 34-40; that version was read before the executor revision and was not hash-captured.',
  final_evidence='Current launch_M125_R0_real64_20261015.py:18-20 and :33-37; formal launch :21-25.',
  fix='Require PASS_SOURCE_ONLY, no blocking findings, exact candidate digest and exact 53-source staging map locally/remotely before launch.',
  validation='Actual predicate AST accepted the matching fixture and rejected FAIL, a blocking finding, wrong candidate SHA and a mismatched source map; current bytes included in immutable final input inventory.')]

report=dict(audit_skill='experiment-audit',date='2026-10-15',generated_at=NOW,
    verdict='WARN',overall_verdict='WARN',integrity_status='warn',source_verdict='PASS_SOURCE_ONLY',
    reason_code='source_pass_runtime_unexecuted_and_review_route_unattested',
    summary='No unresolved source blocker in the fixed-six-pass R0 pipeline and supplied revised operators. 341 bounded deterministic checks pass. Only actual staging/CPU imports are evidenced; no new training or performance result is certified.',
    blocking_findings=[],required_source_fixes=[],resolved_findings=resolved,
    candidate_sha256=candidate_sha,audited_staged_source_sha256=metrics['audited_staged_source_sha256'],
    audited_input_hashes={p:row['sha256'] for p,row in after.items()},
    input_hashes_before_path=str(OUT/'input_hashes.before.json'),input_hashes_after_path=str(OUT/'input_hashes.after.json'),
    audited_input_count=len(inputs),audited_inputs_unchanged=False,audited_source_inputs_unchanged=True,
    reviewed_input_changes=[incident],
    trace_path=str(TRACE),agent_id=AGENT,verdict_id=AGENT+'/2026-10-15_source',agent_id_kind='canonical task path; no separate host opaque id supplied',
    executor_model='UNATTESTED',executor_family='openai',reviewer_model='UNATTESTED',reviewer_reasoning='UNATTESTED',
    reviewer_model_requested='gpt-6-astra',reviewer_reasoning_requested='max',model_routing_attestation='UNATTESTED',
    reviewer_family='openai',review_independence='same-family',acceptance_status='provisional',
    deterministic_evidence_acceptance=dict(review_independence='deterministic',acceptance_status='accepted',checks=341,failed=0,
        limitations='Acceptance covers exact static/fixture assertions only, not NN/runtime/performance.'),
    scope='Nine candidate Python files, CANDIDATE/FROZEN_DEPENDENCIES, 44 immutable dependencies, eight original counterparts, supplied common inputs/plan, staging payload/receipt, revised real64/formal operators, direct inherited metric/manifest interfaces.',
    runtime_evidence=dict(staging_receipt=str(BASE/'M125_R0_candidate_staging_20261015_stdout.json'),CPU_imports=9,CPU_import_exit_codes=[0]*9,
        actual_R0_NN_result_evidence='NONE_SUPPLIED',real64='NOT_CERTIFIED',formal_training='NOT_CERTIFIED',official_metrics='NOT_CERTIFIED'),
    reviewer_actions=dict(NN_calls=0,SSH_calls=0,GPU_queries=0,installs=0,source_edits=0,original_experiment_mutations=0,external_notifications=0),
    checks=checks,limitations=limits,
    claims=[dict(id='source_protocol',impact='supported_source_only',text='Fixed six complete passes and fixed final-six evaluation are implemented in the audited source.'),
            dict(id='actual_real64_or_complete_training',impact='unsupported_by_this_audit',text='Requires actual original child exit, witness/result/checkpoint evidence and subsequent result review.'),
            dict(id='nine_metrics_or_effective_modules',impact='unsupported_by_this_audit',text='No R0 performance numbers or effectiveness claims are established.'),
            dict(id='clean_heldout_protocol',impact='not_supported',text='Historical Test/CDTB engineering exposure and multiframe human-review aids remain disclosed.')],
    next_required_runtime_evidence=['Successful original real64 supervised exit and bound witness tensors/gradient/freeze/Adam checks before formal launch.',
        'Original formal supervisor and all child terminal receipts, six complete pass records/checkpoints, final-pass-six source/input/weight bindings.',
        'Complete OPE50/76373, CDTB80/101956 and VOT127/1765 trajectory/metric artifacts for the same final, followed by a fresh result integrity audit.'])

md=f'''# Experiment audit: M125 R0 fixed six-pass source

**Source verdict: PASS_SOURCE_ONLY. Unresolved blockers: 0. Required source fixes: none.**

**Overall assurance: WARN — native R0 training/evaluation is unexecuted in this review and requested model routing is UNATTESTED.** This is a fresh same-family review with provisional semantic acceptance. Requested routing was `gpt-6-astra` / `max`; the host supplied no attestation of actual model or effort, so this report does not attribute an actual Astra execution.

The nine candidate modules, 44 captured dependencies and original counterparts remained byte-identical. The executor's additional real64 admission script initially lacked a verdict/source-binding predicate; that concrete issue was corrected before this final review and the revised predicates pass negative fixtures. No unresolved source defect was found in the final reviewed bytes. No original experiment or input source was edited by the reviewer.

## Evidence and scope

- **{len(inputs)} primary local inputs hashed**. Candidate sources, dependencies, original source counterparts, payload/receipt and revised operators remained unchanged. One concurrent append to `EXPERIMENT_PLAN.md` was caught by the original guard and separately reviewed; both versions and the genuine failed guard are preserved. The smaller policy hash list is separate.
- **341 deterministic checks, 0 failures:** 207 inventory/hash/AST/payload checks; 36 training/control-flow/exit/arithmetic fixtures; 65 inherited metric/staging/operator checks; 33 preparation/wrapper/manifest fixtures.
- The supplied real staging receipt contains **53 copied source hashes and nine CPU `runpy` imports with exit 0**. Its candidate-gate digest is reproducible from the source payload: `{metrics['candidate_gate_sha256']}`.
- Reviewer actions: **0 NN calls, 0 SSH calls, 0 GPU queries, 0 installs, 0 source edits, 0 original-experiment mutations**. No public/external notification was sent.
- Synthetic fixture output is clearly marked and kept under `SYNTHETIC_CONTROL_FLOW_NO_NN`; it is not training, checkpoint, or performance evidence.

Candidate SHA-256: `{candidate_sha}`.

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

Finalization initially stopped because the executor published an execution-status/diagnostic append to `EXPERIMENT_PLAN.md` during this review. The original **11,318 bytes** are an exact unchanged prefix: old SHA `{incident['before']['sha256']}`; current **12,221 bytes** SHA `{incident['after']['sha256']}`. The 903-byte suffix exactly matches the supplied publication packet, whose `before_sha256` matches the original audit hash; the supplied publication receipt binds the new digest and commit `{publication['commit']}`. The appended lines 91–93 explicitly retain seed2027/common warm/fresh Adam and RNG/six-pass final-six protocol. Its historical diagnostic numbers were not re-audited here. `input_change_incident.json` and `write_audit_report.first_failure.txt` preserve this real guard failure. The report does **not** claim every input byte was unchanged.

The original M122 base-plan bodies were not present in the supplied E6 closure. The captured E6 plans reference the same base root used by R0, and inherited `checked_base_plan` verifies their hashes at runtime. This report inspected that direct contract and did not manufacture a full original-metadata/runtime execution claim.

Required next evidence is the original supervised real64 exit and bound witness result, followed by original formal pass/checkpoint/child receipts and the same final's complete OPE/VOT artifacts. A fresh result audit is required before any performance claim. Existing source flags correctly leave `fresh_result_audit_complete` / `independent_completed_audit` false.

## Artifacts

- `EXPERIMENT_AUDIT.json`: machine verdict, exact candidate/staged-source binding, all input digests and claim limits.
- `input_hashes.before.json`, `additional_input_hashes.json`, `input_hashes.after.json`: immutable input guard.
- `inventory_checks.json`, `deterministic_checks.json`, `metric_operator_checks.json`, `planner_checks.json`, `verification_summary.json`: 341 deterministic checks and explicit test scope.
- `inherited_metric_sources/`: exact 12 retrieved source/metadata bodies and their index in `metric_operator_checks.json`.
- `.aris/traces/experiment-audit/2026-10-15_run01/`: request, supplemental path messages, response and honest routing metadata.

Reviewer tool limitations retained: the first bare `python` metadata command failed with `No pyvenv.cfg file` before any check executed. The installed `uv python find` located an existing CPython 3.13 executable; subsequent stdlib-only fixture runs used that exact interpreter without installation. The first publication-packet text read also failed under Windows' GBK default; an explicit UTF-8 read then verified the exact append. All four evidence scripts completed with exit 0. The original report guard failure is retained separately as described above. Intentional negative fixtures are recorded as expected rejections, not runtime training failures.
'''
(OUT/'EXPERIMENT_AUDIT.md').write_text(md,encoding='utf-8')
write(OUT/'EXPERIMENT_AUDIT.json',report)

prompt='''Perform a fresh source integrity/correctness review required by installed experiment-audit skill. Read C:/Users/gb/.codex/skills/experiment-audit/SKILL.md and applicable native Codex reviewer/tracing policy. Requested gpt-6-astra/max, actual model routing UNATTESTED absent host evidence, same-family/provisional. Do not SSH, run NN, install packages, edit input sources, notify anyone, or touch original experiments. Read artifact paths only, verify them directly. Scope: proposed fixed-six-pass full Train152 R0 baseline + own-history real64 witness + official complete three-dataset evaluation pipeline. All candidate new source files (nine .py) and CANDIDATE.json are in C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_R0_FIXED_SOURCE_20261015. FROZEN_DEPENDENCIES.json in that folder lists 44 exact captured immutable dependency sources under frozen_dependency_snapshots. Read them as needed for actual call graph. Original source counterparts are in C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928 (train_m124_joint_epoch.py, m124_epochwise_official_runtime.py, run_m124_epochwise_official.py, prepare_m123_joint_evaluation.py, run_m123_joint_full_suite.py, analyze_m124_epochwise_vot.py, collect_m124_epochwise_results.py, m124_epochwise_process_supervisor.py). Common original spec/gate/warm-result raw texts are in C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_fixed_R0_common_inputs_20261015.json. Protocol proposal: C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/refine-logs/m125_region_state/EXPERIMENT_PLAN.md. Checklist: fresh common A_B/C warm weights and fresh Adam/RNG on first pass; complete sequence/order/frame budget 152*6 and 219802*6 with no retained37-sequence legacy or extra sanity training; later pass complete-boundary checkpoint restore includes all Adam/RNG/model/frozen state and accurate cumulative histories; actual training update/loss/gradient behavior matches inherited M124 except declared initialization/protocol changes; GT only after prediction for loss and no GT state reset; freeze backbone/CLIP; parameter counts; snapshots/reload and truthful flags/status; no Test/CDTB/VOT checkpoint selection or inter-pass evaluations; exactly finalpass6 is evaluated on complete OPE50/80 and VOT127/1765; call paths/imports/metadata/manifest/wrapper source binding; all subprocess status/no autoretry and preservation of originals. Audit real64 checks for meaningful gradient/freeze/save/Adam behavior, no phantom checks, and any real likely runtime defects. Candidate is not reviewed/deployed/launched; source-only PASS must not imply actual runtime or performance. Use deterministic AST/text fixtures or local stubs where meaningful without torch/NN; preserve failures honestly. Write reports (MD+JSON), input hashes, trace, deterministic checks to C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_R0_FIXED_SOURCE_AUDIT_20261015. Report concrete blockers first and required minimal fixes. Do not expand to R1 design or new experiments.'''
supplemental=[
 '''Additional concrete deployment source path for your scope: C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/stage_M125_R0_candidate_remote_20261015.py and M125_R0_staging_payload_20261015.json in same folder. Candidate CANDIDATE.json remains unchanged. Stager copies the 44 exact FROZEN_DEPENDENCIES into new code_root alongside nine new modules and includes all actual copied paths/SHA in candidate_gate before CPU runpy-importing all nine; no NN/launch/GPU queries. Please include this actual path contract in review and test fixtures. Not executed yet. If this does not satisfy a concrete dependency/source-binding requirement, identify minimal fix; do not presume omitted staging as successful runtime.''',
 '''Actual static staging and all nine CPU runpy imports completed exit0, no NN/GPU queries. Additional primary receipt: C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_R0_candidate_staging_20261015_stdout.json (53 actual isolated source hashes, nine complete command/exit/stdout/stderr rows). Original candidate and all audited sources unchanged. Please scope this as CPU import evidence only; no training/metric dataset execution.''',
 '''Exact additional primary closure, already local/no SSH: C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_E6_terminal_audit_artifacts_20261015.json. Its files map contains 7,288 exact text/SHA entries, including bound depthtrack_pr.py, failure audit/common sources and VOT official toolkit Python source, evaluation bundle/plan/shard_manifest/execution/seals. Core original gate/warm/spec in M125_fixed_R0_common_inputs_20261015.json remains supplied. Smaller extracted exact metric also at C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_R2_RESULT_INPUTS_20261015/locked_depthtrack_pr.py. Read only relevant inherited sources/metadata for R0 path integrity; no need rerun previous E6 full-result audit. All inputs unchanged.''',
 '''One additional operator source for final path/exit/admission review, not executed: C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/launch_M125_R0_real64_20261015.py. It expects final EXPERIMENT_AUDIT.md/json filenames, pins actual staging/candidate and all source/input SHA, requires original suites terminal and GPU memory admission, emits an actual-waiting child supervisor preserving real64 exit, no formal training launch. Source review JSON schema/verdict predicate will be pinned to your actual final fields before invocation; not assumed PASS from existence. Remote and generated supervisor AST parsing checked locally only. Please write report filenames EXPERIMENT_AUDIT.md/.json or tell final names.''',
 '''Patched real64 launcher to parse your promised exact fields: source_verdict PASS_SOURCE_ONLY + blocking_findings[] + candidate SHA equals unchanged local CANDIDATE bytes + audited_staged_source_sha256 equals actual staging53. Duplicated predicate and hashes remotely before gate/NN launch, added source_review_candidate_sha256 to gate. Actual helper still NOT invoked. Additional planned formal launch operator for your bounded path/exit check: C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/launch_M125_R0_fixed_full_suite_20261015.py. It requires same audit predicate/53 hashes, gate matched to actual real64 result+exit/receipt,64calls/two AB+C steps/freshinit/frozen+Adam roundtrip, warmSHA unchanged, no reuse sanitycheckpoint, no existing launch/passes, memory+disk admission, then one original supervisor; no retry logic. Please include latest operator bytes in final inventory/review.''',
 '''Keep the remaining review bounded to the nine new sources and their direct inherited metric/manifest interfaces. The 7,288-file closure is a lookup source, not a request to audit every VOT toolkit file or re-audit E6 numerical results. All earlier completed reviews remain separate evidence. Once direct path/schema/metric call compatibility is checked, finalize with honest inherited/runtime limitations; no additional broad test expansion is needed.''',
 '''Please finalize the source report now from completed direct checks, preserving any remaining inherited/runtime boundary as WARN/UNVERIFIED. The actual real64 witness will test execution before full launch. Do not add another fixture family or delay on numerical re-evaluation of inherited official metrics; that is outside this source gate. Report any actual unresolved blocker explicitly instead of extending scope.''',
 '''That plan change is the executor's completed diagnostic publication, committed/pushed while this review ran. Exact primary packet C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_R2_result_publication_packet_20261015.json contains append[EXPERIMENT_PLAN.md] and before_sha256; publication receipt same stem _publication.json binds final SHA, commit1c002d00579ce323a83d65bb6ee68d15f21bb80e. No earlier plan bytes were edited, only the diagnostic execution-status/coverage/scale paragraph appended; fixed6/commonwarm/freshAdam/seed/budget unchanged. Please verify exact append/prefix SHA, retain the genuine initial guard failure, and explicitly bind both plan versions/change in final report. All nine sources, candidate, staging and operators remain unchanged. Do not claim every input byte was unchanged.''']
meta=dict(skill='experiment-audit',run_id='2026-10-15_run01',executor='codex',
    executor_model='UNATTESTED',executor_family='openai',reviewer_model_requested='gpt-6-astra',
    reviewer_reasoning_requested='max',reviewer_model_actual='UNATTESTED',reviewer_reasoning_actual='UNATTESTED',
    review_independence='same-family',acceptance_status='provisional',project_dir=str(SRC),
    agent_id=AGENT,completed_at=NOW,status='completed_source_review')
write(TRACE/'run.meta.json',meta)
write(TRACE/'001-fixed-r0-source.request.json',dict(call_number=1,purpose='fixed-six-pass-R0-source-integrity',
    timestamp='NOT_SUPPLIED_BY_HOST',tool='spawn_agent',model_requested='gpt-6-astra',reasoning_effort_requested='max',
    model_actual='UNATTESTED',files_referenced=list(inputs),prompt=prompt,supplemental_messages=supplemental,
    tracing_note='Request/parent supplements transcribed from received messages; no host opaque agent id or route attestation supplied.'))
(TRACE/'001-fixed-r0-source.response.md').write_text(md,encoding='utf-8')
write(TRACE/'001-fixed-r0-source.meta.json',dict(meta,call_number=1,purpose='fixed-six-pass-R0-source-integrity',
    source_verdict=report['source_verdict'],overall_verdict=report['overall_verdict'],
    report_sha256=sha(OUT/'EXPERIMENT_AUDIT.json'),report_markdown_sha256=sha(OUT/'EXPERIMENT_AUDIT.md')))
events=OUT/'.aris/meta';events.mkdir(parents=True,exist_ok=True)
with (events/'events.jsonl').open('a',encoding='utf-8') as stream:
    stream.write(json.dumps(dict(event='review_trace',skill='experiment-audit',purpose='fixed-six-pass-R0-source-integrity',
        agent_id=AGENT,trace_path=str(TRACE),status='ok',source_verdict=report['source_verdict']))+'\n')
print(json.dumps(dict(source_verdict=report['source_verdict'],overall_verdict=report['overall_verdict'],
    blocking_findings=0,checks=341,input_files=len(inputs),all_inputs_unchanged=False,sources_unchanged=True,
    report_sha256=sha(OUT/'EXPERIMENT_AUDIT.json'),report_markdown_sha256=sha(OUT/'EXPERIMENT_AUDIT.md'))))
