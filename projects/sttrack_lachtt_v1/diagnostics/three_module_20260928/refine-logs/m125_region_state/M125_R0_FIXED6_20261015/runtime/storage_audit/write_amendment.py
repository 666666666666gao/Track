from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

OUT = Path(__file__).resolve().parent
A = OUT.parent
OLD = A / 'M125_R0_FIXED_SOURCE_AUDIT_20261015'
SRC = A / 'M125_R0_FIXED_SOURCE_20261015'
TRACE = OUT / '.aris/traces/experiment-audit/2026-10-15_storage_r2'
TRACE.mkdir(parents=True, exist_ok=True)
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
load = lambda path: json.loads(Path(path).read_bytes())
now = datetime.now(timezone.utc).isoformat()
checks = load(OUT / 'CHECKS.json')
before = load(OUT / 'input_hashes.before.json')
after = load(OUT / 'input_hashes.after.json')
staging = load(A / 'M125_R0_storage_R2_staging_20261015_stdout.json')
prior = load(OLD / 'EXPERIMENT_AUDIT.json')
assert checks['failed'] == 0 and checks['total'] == 56
assert before == after
assert all(sha(p) == row['sha256'] for p, row in after.items())
prior_sha = sha(OLD / 'EXPERIMENT_AUDIT.json')
candidate_sha = sha(SRC / 'CANDIDATE_STORAGE_R2.json')
request = '''Bounded storage-only admission revision review. Preserve the previous source audit unchanged. Verify the sole candidate policy delta required_free_disk_bytes 20 GiB to 2 GiB, measured output/free-space evidence, exact enriched-gate serialization, unchanged 53 sources/scientific protocol, first failed admission before GPU/NN/writes, exact operator-reference changes, inherited CPU imports, and unchanged formal admission interface. No rerun of prior 341 fixtures, no SSH/NN/install/input edits/notifications/additional agents. Write compatible EXPERIMENT_AUDIT.md/json. Requested reviewer model/effort gpt-6-astra/max; actual route UNATTESTED, same-family/provisional. This amendment permits source admission for the real64 witness, not a runtime/performance result.'''
md = f'''# M125 R0 storage admission amendment R2

**PASS_SOURCE_ONLY; zero blocking findings.** This bounded amendment supports proceeding to the actual real64 witness. It certifies no NN, training, metric or performance result. Overall integrity status remains **WARN** because runtime evidence and reviewer-route attestation are absent.

The prior [source audit]({(OLD / 'EXPERIMENT_AUDIT.json').as_posix()}) is reused by SHA-256 `{prior_sha}`. Its JSON and Markdown remain byte-exact; its 341 fixtures were not rerun. The amendment performed 56 local static/hash/arithmetic checks, all passing, over 71 input files. See [CHECKS.json](CHECKS.json), [input hashes](input_hashes.after.json), [53 source bindings](unchanged_source_bindings.json) and [operator diff](real64_operator.diff).

- The only candidate JSON change is `required_free_disk_bytes`: 21,474,836,480 to 2,147,483,648 bytes. New candidate SHA-256: `{candidate_sha}`. All other values, including raw input/model bindings, seed 2027, six passes, final-pass-six selection, and source maps, are identical. All nine scientific modules and 44 frozen dependencies match their previous hashes. Optimizer/training behavior inherits the unchanged source audit; raw datasets were not independently reread by this reviewer.
- Reconstructing the original enriched gate gives `18580735b4234ecbadbab8508f9344640a29695d10a52a995e9d686ac6424d16`. Changing only that threshold reproduces the actual R2 receipt's gate SHA `{staging['candidate_gate_sha256']}`. Original metadata is retained.
- Actual `du` evidence measures one previous fit directory at 8,409,076 bytes and its complete evaluation at 170,452,733 bytes. The budget extrapolates **6 × one measured fit + one measured evaluation = 220,907,189 bytes**; it does not measure six new fits. The 2 GiB threshold is 9.72 times this estimate. Output-mount free space was 11,800,924,160 bytes in diagnosis and {staging['actual_free_bytes']:,} bytes at static R2 staging. This is storage planning evidence, not an exact future peak bound or performance result. Live disk checks remain before real64 and formal launch.
- The preserved first-attempt stderr reports `AssertionError` at remote line 17, exactly the disk assertion. It precedes GPU queries, gate/review writes and NN launch; original stdout is empty. The subsequent read-only diagnosis records all seven launch/gate/pass artifacts absent and NN/GPU/mutations zero. The failed admission is retained as a real failure, resolved here by the explicit reviewed threshold revision.
- The new real64 operator is byte-equivalent to the previously reviewed operator after six explicit filename/reference substitutions (seven occurrences). Verdict/no-blocker, candidate SHA, all-source/input SHA, disk/GPU admission, original-suite terminal checks, waiting child supervisor and actual exit preservation remain intact. The unchanged formal operator still requires the bound gate/review, successful real64 exit/results, 64 calls, two A_B/C steps, fresh initialization, frozen/Adam checks and live disk/GPU admission. There is no automatic retry, fallback, deletion, source edit or data move. The observer enforces a 180-second minimum and reads once; its existence or observation does not establish completion.
- R2 staging changes only new metadata files. Its 53-source map equals the original actual staging map. The nine original CPU-import rows are retained exactly with `CPU_imports_reexecuted=false`; the original remote receipt serialization SHA is reproduced. **No CPU imports were rerun in R2, and no successful real64 or formal result is established by this amendment.**

Requested review routing remains `gpt-6-astra` / `max`; actual model and effort are **UNATTESTED**, semantic review **same-family / provisional**. Deterministic assertions are accepted only for their exact bounded scope. The reviewer used no SSH, NN, GPU query, package installation, source/input edit or external notification. One preliminary read-only AST extraction failed because it treated a `.replace(...)` assignment as a literal; this was corrected before the checks and is retained in the trace.

Required source fixes: **none**. Next evidence is the original supervised real64 exit and bound witness result. Full training/official evaluation retain the prior audit's separate runtime and result-review requirements.
'''
report = dict(
    audit_skill='experiment-audit', date='2026-10-15', generated_at=now,
    verdict='WARN', overall_verdict='WARN', integrity_status='warn',
    source_verdict='PASS_SOURCE_ONLY', blocking_findings=[], required_source_fixes=[],
    reason_code='storage_only_source_amendment_pass_runtime_unverified_route_unattested',
    summary='Only the measured disk admission threshold changed; all 53 source hashes and scientific protocol are unchanged. No unresolved source blocker. Runtime and performance remain uncertified.',
    revision_scope='storage admission only: required_free_disk_bytes 20 GiB to 2 GiB; explicit R2 metadata/operator input-output references',
    candidate_path=str(SRC / 'CANDIDATE_STORAGE_R2.json'), candidate_sha256=candidate_sha,
    prior_source_audit_path=str(OLD / 'EXPERIMENT_AUDIT.json'), prior_source_audit_sha256=prior_sha,
    prior_source_audit_md_sha256=sha(OLD / 'EXPERIMENT_AUDIT.md'), prior_source_audit_unchanged=True,
    prior_checks_rerun=False, prior_checks_count=341,
    audited_staged_source_sha256=staging['source_sha256'],
    audited_input_hashes={p:row['sha256'] for p,row in after.items()},
    input_hashes_before_path=str(OUT / 'input_hashes.before.json'), input_hashes_after_path=str(OUT / 'input_hashes.after.json'),
    audited_input_count=len(after), audited_inputs_unchanged=True,
    candidate_policy_delta={'required_free_disk_bytes': {'old': 21474836480, 'new': 2147483648}},
    original_enriched_gate_sha256='18580735b4234ecbadbab8508f9344640a29695d10a52a995e9d686ac6424d16',
    revised_enriched_gate_sha256=staging['candidate_gate_sha256'],
    storage_evidence=dict(measured_one_fit_bytes=8409076, measured_one_evaluation_bytes=170452733,
        extrapolated_six_fit_plus_one_evaluation_bytes=220907189, required_free_disk_bytes=2147483648,
        diagnosis_output_mount_free_bytes=11800924160, static_R2_stage_output_mount_free_bytes=staging['actual_free_bytes'],
        interpretation='Six-fold extrapolation of one existing fit plus one existing full evaluation; not measured R0 output, exact future peak guarantee or a performance result.'),
    original_admission_failure=dict(status='PRESERVED_ACTUAL_FAILURE_RESOLVED_BY_EXPLICIT_STORAGE_POLICY_REVISION',
        stderr_path=str(A / 'M125_R0_real64_actual_launch_20261015_stderr.log'), remote_line=17,
        assertion="assert shutil.disk_usage(root).free>=gate['required_free_disk_bytes']",
        before_GPU_NN_and_launch_writes=True, subsequent_all_seven_unstarted_artifacts_absent=True),
    runtime_evidence=dict(static_R2_staging_receipt=str(A / 'M125_R0_storage_R2_staging_20261015_stdout.json'),
        CPU_imports_reexecuted=False, inherited_CPU_imports=9, inherited_CPU_import_exit_codes=[0]*9,
        inherited_CPU_import_receipt_sha256=staging['inherited_CPU_import_receipt_sha256'],
        R2_staging_NN_calls=0, R2_staging_GPU_queries=0, real64='NOT_CERTIFIED', formal_training='NOT_CERTIFIED', official_metrics='NOT_CERTIFIED'),
    next_action_supported='actual original supervised real64 witness, subject to unchanged live admission predicates',
    formal_training_authorized_by_this_report=False,
    checks=dict(path=str(OUT / 'CHECKS.json'), total=checks['total'], passed=checks['passed'], failed=checks['failed']),
    agent_id='/root/audit_m125_r0_fixed_source', agent_id_kind='canonical task path; no separate host opaque id supplied',
    verdict_id='/root/audit_m125_r0_fixed_source/2026-10-15_storage_r2',
    executor_model='UNATTESTED', executor_family='openai', reviewer_model='UNATTESTED', reviewer_reasoning='UNATTESTED',
    reviewer_model_requested='gpt-6-astra', reviewer_reasoning_requested='max', model_routing_attestation='UNATTESTED',
    reviewer_family='openai', review_independence='same-family', acceptance_status='provisional',
    deterministic_evidence_acceptance=dict(review_independence='deterministic', acceptance_status='accepted', checks=56, failed=0, limitations='Only bounded static/hash/arithmetic assertions; no new NN/runtime/performance attestation.'),
    reviewer_actions=dict(NN_calls=0, SSH_calls=0, GPU_queries=0, installs=0, source_edits=0, input_edits=0, original_experiment_mutations=0, external_notifications=0, additional_agents=0),
    limitations=[
        'Prior scientific source audit and its limitations are inherited by SHA; not rerun or enlarged.',
        'The actual requested reviewer model/effort is unverified; semantic assessment remains same-family/provisional.',
        'Raw input/model paths and SHA bindings are unchanged; reviewer did not reread remote raw dataset/model bytes.',
        'The storage budget extrapolates an existing fit directory six times; future exact peak usage is not measured.',
        'R2 static receipt inherits prior CPU imports. No new successful real64, formal training or official metric result is certified.',
        'Observer performs a single scheduled read after at least 180 seconds; terminal success still requires actual exit/result evidence.'
    ], trace_path=str(TRACE))
(OUT / 'EXPERIMENT_AUDIT.md').write_text(md, encoding='utf-8')
(OUT / 'EXPERIMENT_AUDIT.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
(TRACE / 'request.md').write_text(request + '\n', encoding='utf-8')
(TRACE / 'response.md').write_text(md, encoding='utf-8')
meta = {k:report[k] for k in ['generated_at','agent_id','verdict_id','reviewer_model_requested','reviewer_reasoning_requested','reviewer_model','reviewer_reasoning','model_routing_attestation','review_independence','acceptance_status','prior_source_audit_sha256','revision_scope']}
meta.update(request_sha256=sha(TRACE / 'request.md'), response_sha256=sha(TRACE / 'response.md'), report_sha256=sha(OUT / 'EXPERIMENT_AUDIT.json'), checks_sha256=sha(OUT / 'CHECKS.json'))
(TRACE / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
(TRACE / 'execution_notes.json').write_text(json.dumps(dict(
    preliminary_readonly_extraction_failure=dict(exit_code=1, tool_chunk='f1d128', raw_error="AttributeError: 'Call' object has no attribute 'value'", reason='remote is a literal followed by .replace(...), not a bare Constant', resolution='Read literal from the known AST Call.func.value; no operator executed.'),
    bounded_checks_execution=dict(exit_code=0, tool_chunk='f45ac5', total=56, failed=[], input_paths=71),
    prohibited_actions_performed=False), indent=2) + '\n', encoding='utf-8')
assert all(sha(p) == row['sha256'] for p, row in after.items())
print(json.dumps(dict(source_verdict=report['source_verdict'], blocking_findings=[], report_json=str(OUT / 'EXPERIMENT_AUDIT.json'), report_json_sha256=sha(OUT / 'EXPERIMENT_AUDIT.json'), report_md_sha256=sha(OUT / 'EXPERIMENT_AUDIT.md'), prior_source_audit_sha256=prior_sha)))
