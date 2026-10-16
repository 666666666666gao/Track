"""Write only the independent R3 report and trace; verify its real bytes in isolation."""
import ast
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import zlib

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
CAD = BASE / 'M125_CAD_UNIFIED_SOURCE_R2_20261016'
NEW = BASE / 'M125_CAD_DEPLOYMENT_SOURCE_R2_20261016/invoke_M125_CAD_reviewed_suite.py'
OLD = BASE / 'M125_CAD_DEPLOYMENT_SOURCE_20261016/invoke_M125_CAD_reviewed_suite.py'
OP = BASE / 'M125_REGION_DEPLOYMENT_SOURCE_R2_20261015/deploy_m125_region_fixed.py'
R1 = BASE / 'M125_CAD_UNIFORM_SOURCE_AUDIT_20261016'
R2 = BASE / 'M125_CAD_UNIFORM_SOURCE_AUDIT_R2_20261016'
LAYOUT = BASE / 'M125_CAD_R2_final66_actual_CPU_layout_20261016.json'
TRACE = OUT / '.aris/traces/experiment-audit/2026-10-16_run03'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def refs(path, *lines):
    return [str(path) + ':' + str(line) for line in lines]


now = datetime.now(timezone.utc).isoformat()
before = read(OUT / 'input_hashes_before.json')
inputs = before['files']
after = [dict(path=r['path'], bytes=Path(r['path']).stat().st_size, sha256=sha(r['path'])) for r in inputs]
assert after == inputs
det = read(OUT / 'deterministic_checks.json')
byte = read(OUT / 'actual_CRLF_transport_verification.json')
layout = read(LAYOUT)
prior = read(R2 / 'EXPERIMENT_AUDIT.json')
unchanged = read(OUT / 'unchanged_input_receipt.json')
assert unchanged['files'] == inputs and det['total'] == det['passed'] == 533
assert det['failed'] == []

checks = [
    dict(id='A', status='PASS_SOURCE_ONLY', subject='Ground-truth provenance and training scope',
         evidence_basis='Hash-bound reuse of the direct R1/R2 source review; scientific 16 files unchanged.',
         details='One fixed six-pass DepthTrack Train152 R0 final is reused. Tracking receives legal initialization annotations and subsequent RGB/depth observations; OPE GT is opened for evaluation after prediction receipts. VOT initialization comes from the official bank. No later GT is introduced by this operator revision. Existing test-aware engineering and annotation-assistance disclosures remain applicable.',
         evidence=refs(CAD / 'run_m125_CAD_official.py', 23, 27, 55, 56, 72, 79) + refs(CAD / 'PROTOCOL.md', 3, 7, 23)),
    dict(id='B', status='PASS_SOURCE_ONLY', subject='Official metrics and normalization',
         evidence_basis='Unchanged scientific source and frozen dependencies, independently rehashed.',
         details='The official OPE and VOT analysis paths and nine-metric thresholds are unchanged. No metric is divided by the model output maximum/mean. Full U1 and U2 collection still requires DT50/76373 frames, CDTB80/101956 frames and VOT127/1765 anchors/5295 trajectories. This does not establish any new metric value.',
         evidence=refs(CAD / 'run_m125_CAD_official.py', 15, 16, 57, 59) + refs(CAD / 'analyze_m125_CAD_vot.py', 19, 24)
                  + refs(CAD / 'collect_m125_CAD_results.py', 52, 53, 54)),
    dict(id='C', status='PASS_RECEIPT_SCOPE', subject='Version and artifact existence',
         evidence_basis='Fresh raw-byte hashes of all 16 scientific files, 66 local sources and the actual R2 CPU receipt.',
         details='All 16 scientific inputs and the R2 CPU receipt exactly match their prior R2 audit seals. The primary CPU receipt retains its own R2 staging root, 66 source hashes, 16 locked-input hashes and 12 successful import rows, with zero main executions and zero tracking. Its exact absolute-key bare-digest map is copied into this report only after independent verification of all 66 frozen local sources. The launch operator is outside this 66-file scientific map and is separately sealed by its actual absolute path in audited_input_hashes.',
         evidence=[str(LAYOUT) + ':1'] + refs(NEW, 24, 27, 28, 29, 30, 35, 39, 40)
                  + [str(OUT / 'local_66_source_verification.json')]),
    dict(id='D', status='PASS_SOURCE_ONLY', subject='Complete scientific path and actual supervisor interface',
         evidence_basis='Previously exercised unchanged scientific functions; direct reread of current coordinator/supervisor and operator.',
         details='The R2 prepare fix uses bundle["training_result_path"], and collector uses event_name without overwriting the model name. Their prior complete stdlib fixtures (3 prepare profiles, 2 full-shaped collectors) remain valid for the identical bytes. The new operator launches m125_CAD_process_supervisor.py with the candidate root; that supervisor reads gate.json and invokes run_m125_CAD_uniform_suite.py with the same root. Its child return code is written to terminal receipts and returned via sys.exit. Both prior stubbed exit-0 and exit-7 cases are reused, not rerun. The coordinator calls qualification, tracking, analysis, VOT sealing and collection stages.',
         evidence=refs(CAD / 'prepare_m125_CAD_evaluation.py', 80) + refs(CAD / 'collect_m125_CAD_results.py', 43, 46)
                  + refs(NEW, 88, 92, 94, 96) + refs(CAD / 'm125_CAD_process_supervisor.py', 11, 12, 15, 16, 17, 18, 20, 23)
                  + refs(CAD / 'run_m125_CAD_uniform_suite.py', 58, 61, 84, 90, 92, 93)),
    dict(id='E', status='WARN_SCOPE', subject='U0 reference and actual execution scope',
         evidence_basis='Source inspection and supplied historical CPU receipts only.',
         details='U0 remains the original full R0 three-dataset reference, coupled prospectively to a new 64-step/65-row saved-prefix qualification; it is not a new full U0 evaluation. U1/U2 each require a full three-dataset uniform-profile row using the same final. No actual new real64 witness, full benchmark result, current R0 completion, real resource admission or deployment was read or produced in R3. The time boundary alone is not admission.',
         evidence=refs(CAD / 'run_m125_CAD_uniform_suite.py', 61, 64, 69, 70, 71, 75, 95, 102) + refs(NEW, 79, 80, 102)),
    dict(id='F', status='PASS_CLASSIFICATION', subject='Evaluation classification',
         evidence_basis='Explicit fixture/source/receipt scopes.',
         details='R3 evidence is local raw-byte hashing, AST inspection and isolated stdlib transport/date fixtures. Prior scientific interface fixtures remain simulation_only. The supplied actual CPU receipt is import/layout evidence. The intended official benchmark is real_gt; no new GPU/NN or formal metric is claimed.',
         evidence=[str(OUT / 'deterministic_checks.json'), str(OUT / 'actual_CRLF_transport_verification.json'),
                   str(R2 / 'fixture_results.json'), str(LAYOUT)]),
    dict(id='G', status='PASS_SOURCE_ONLY', subject='One final, uniform profiles and CAD adaptation semantics',
         evidence_basis='Hash-bound reuse of direct scientific review; no scientific changes in R3.',
         details='U0/U1/U2 retain one profile per complete row, one R0 final and no per-dataset seed/layer/weight/profile routing. Strict response/C thresholds, native response at the ABC-selected cell, unchanged final ABC reported confidence and prediction/state/query/features, retained initial template and dynamic-slot replacement are unchanged. C participates in U0/U1/U2; optional U4 bypass is disclosed and unqueued. The implementation remains a CAD-inspired adaptation, not a claim of official CAD network reproduction.',
         evidence=refs(CAD / 'cad_template_profiles.py', 14, 16, 22, 25, 32)
                  + refs(CAD / 'profiled_template_tracker.py', 41, 44, 45, 60, 61, 65, 70, 71)
                  + refs(CAD / 'PROTOCOL.md', 19, 21, 27) + refs(NEW, 70, 71)),
    dict(id='H', status='PASS_SOURCE_AND_BYTE_FIXTURE', subject='Actual CRLF bytes and R3 self-binding',
         evidence_basis='Fresh exact-expression transport fixtures against the actual R1/R2 report bytes, followed by actual R3 artifact binding.',
         details='Only lines 21, 43 and 44 change: the selected audit directory becomes R3 and both report reads become read_bytes().decode("utf-8"). The remote program and encoding expression remain exactly unchanged. All four historical CRLF reports preserve every raw byte after the exact JSON/zlib/base64/inverse pipeline. Both exact remote JSON hash assertions pass with the revised reader; the old reader fails on both JSON artifacts. Old FAIL reports are used only as byte fixtures, never admitted. The final R3 JSON/MD are checked by the exact safe local prefix (lines 19-46) and isolated remote report-binding statements. That includes the absolute-path SHA key of this operator, candidate, manifest, locked inputs and exact 66-map.',
         evidence=refs(NEW, 21, 25, 26, 27, 30, 32, 34, 35, 40, 43, 44, 45, 62, 65, 66, 67, 85, 89, 105)
                  + [str(OUT / 'actual_CRLF_transport_verification.json'), str(OUT / 'R3_report_self_binding_receipt.json')]),
    dict(id='I', status='PASS_SOURCE_AND_BOUNDARY_FIXTURE', subject='Time, source/input checks and full R0 prerequisite ordering',
         evidence_basis='Direct source inspection; date assertions only are executed with controlled timestamps.',
         details='The first local statement and first remote non-import statement require 2026-10-17 12:45:00 +08:00 (04:45 UTC). Both reject one second before and accept the exact boundary in isolated fixtures. Remote staging source/locked-input hashes and report/candidate bindings precede all outside-code source and asset hashes; then verify_predecessor, resource_admission, exclusive root creation and Popen occur in that order. The frozen predecessor requires both controller exits zero, supervisor terminal zero, 15 successful stages, all six passes with cumulative 152 sequences/219802 track calls per pass, fixed final, and three same-final dataset receipts. None of those live conditions was evaluated by this review.',
         evidence=refs(NEW, 18, 56, 60, 61, 62, 66, 67, 73, 74, 76, 79, 80, 81, 96)
                  + refs(OP, 59, 62, 68, 69, 71, 72, 76, 77, 78, 80, 82, 84, 86, 87, 89)),
    dict(id='J', status='PASS_SOURCE_ONLY', subject='Resources, exclusive attempt/root and no automatic retry',
         evidence_basis='Unchanged remote launch logic and frozen prerequisite source; no live resource/process reads.',
         details='Resource admission checks candidate free-disk requirement and exactly GPU indices 0/1 with memory.used <500 MB using only index,memory.used,memory.total. The local attempt uses open("x") before one transport call; remote root.mkdir/code.mkdir and xb/x artifact writes are exclusive. The inherited static SSH args specify ConnectionAttempts=1. No automatic retry exists in the invoker, remote program, supervisor or coordinator. The launch receipt records a Popen start and explicitly withholds real64/formal-metric completion; a successful SSH exit is not suite success. The supervisor preserves the actual coordinator terminal code.',
         evidence=refs(OP, 112, 113, 114, 117, 118) + refs(NEW, 48, 51, 79, 80, 81, 83, 87, 91, 96, 102, 103, 107, 111, 116, 120)
                  + refs(CAD / 'run_m125_CAD_uniform_suite.py', 31, 32, 33, 34, 35)
                  + refs(CAD / 'm125_CAD_process_supervisor.py', 15, 16, 17, 18, 20)),
    dict(id='K', status='WARN_RESULT_SCOPE', subject='Inherited limits and truthful metadata',
         evidence_basis='Unchanged R2 scientific source and prior direct review.',
         details='R2 correctly discloses inherited per-invocation rehashing of baseline final/native/CLIP assets; no one-time-only or measured runtime benefit is claimed. VOT event sealing binds initialization keys/profile/final/bundle and 1765 unique events, but independent trajectory-by-trajectory track-call reconciliation remains a completed-result review limit. These scope qualifiers do not create a speculative source blocker or require unrelated refactoring.',
         evidence=refs(CAD / 'SOURCE_SPEC.json', 56, 57) + refs(CAD / 'PROTOCOL.md', 35)
                  + refs(CAD / 'run_m125_CAD_official.py', 81, 84, 85)
                  + refs(CAD / 'analyze_m125_CAD_vot.py', 35, 41, 43, 44, 50)
                  + refs(CAD / 'collect_m125_CAD_results.py', 42, 43)),
    dict(id='L', status='PASS_LOCAL_PRESERVATION', subject='Original inputs and reports preserved',
         evidence_basis='250 task-artifact before/after raw-byte seals.',
         details='All 250 sealed local task artifacts are unchanged, including the full original R1/R2 report/evidence trees, R2 scientific 16 files, primary CPU receipts, old and new operators, revision record and necessary frozen dependencies. All R3 writes are confined to the new R3 audit directory. The old R1/R2 verdicts remain unchanged; this report accepts only the separate revised operator.',
         evidence=[str(OUT / 'input_hashes_before.json'), str(OUT / 'unchanged_input_receipt.json')]),
]

report = dict(
    audit_skill='experiment-audit', date='2026-10-16', generated_at=now,
    verdict='WARN', overall_verdict='WARN', integrity_status='warn',
    source_verdict='PASS_SOURCE_ONLY', scientific_source_verdict='PASS_SOURCE_ONLY',
    admission_operator_source_verdict='PASS_SOURCE_ONLY',
    execution_verdict='NOT_EXECUTED', admission_verdict='NOT_PERFORMED_SOURCE_REVIEW_ONLY',
    overall_scope='The reviewed source path passes. Overall WARN reflects unperformed real admission/execution and result-evidence limits, not a remaining source blocker.',
    execution_scope='No SSH, actual transport, live R0 inspection, GPU query, NN, installation or real launch; current checks are local stdlib/AST/byte fixtures only.',
    admission_scope='Source ordering and local boundary/binding fixtures verified. Real date eligibility, R0 terminal state, asset hashes on host, resources and actual launch have not been established.',
    summary='The independent launch revision preserves raw UTF-8/CRLF report bytes and selects the R3 review. The prior report-byte blocker is resolved. The unchanged R2 scientific path and revised admission operator pass this bounded source review; actual admission and benchmark execution remain unperformed.',
    blocking_findings=[], checks=checks,
    resolved_findings=[
        dict(id='R2-B1', status='RESOLVED_SOURCE_AND_ACTUAL_BYTE_FIXTURE',
             details='Both report read expressions preserve original raw UTF-8 bytes; four existing CRLF artifacts and the final R3 artifacts are byte-exact after encoding/decoding.',
             evidence=refs(NEW, 43, 44, 45, 62, 85, 105)),
        dict(id='original-B1', status='REMAINS_RESOLVED_IN_UNCHANGED_R2',
             details='prepare uses bundle training_result_path; prior three-profile complete fixture reused.',
             evidence=refs(CAD / 'prepare_m125_CAD_evaluation.py', 80)),
        dict(id='original-B2', status='REMAINS_RESOLVED_IN_UNCHANGED_R2',
             details='collector event_name preserves model name; prior two complete collector fixtures reused.',
             evidence=refs(CAD / 'collect_m125_CAD_results.py', 43, 46)),
    ],
    nonblocking_findings=[
        dict(id='R3-W1', finding='Source approval is not actual execution or resource/predecessor admission; no new performance metrics exist in this review.'),
        dict(id='R3-W2', finding='Review is same-family/provisional, requested Astra/max but exact actual route is UNATTESTED.'),
        dict(id='R3-W3', finding='U0 full result is the original R0 reference; new U0 work is a planned 64-step prefix qualification only.'),
        dict(id='R3-W4', finding='Inherited repeated asset hashing and VOT trajectory-counter reconciliation limit remain accurately qualified.'),
    ],
    candidate_sha256=sha(CAD / 'CANDIDATE.json'),
    admission_operator_sha256=sha(NEW),
    audited_staged_source_sha256=layout['source_sha256'],
    audited_staged_source_sha256_origin=str(LAYOUT) + ':source_sha256',
    audited_staged_source_map_preserved_exactly=True,
    audited_input_hashes={r['path']: 'sha256:' + r['sha256'] for r in inputs},
    input_hashes_after={r['path']: 'sha256:' + r['sha256'] for r in after},
    inputs_unchanged=True, scientific_16_and_CPU_receipt_unchanged=True,
    input_hashes_before_path=str(OUT / 'input_hashes_before.json'),
    input_hashes_before_sha256=sha(OUT / 'input_hashes_before.json'),
    unchanged_input_receipt_path=str(OUT / 'unchanged_input_receipt.json'),
    unchanged_input_receipt_sha256=sha(OUT / 'unchanged_input_receipt.json'),
    scientific_evidence_reuse=dict(
        source_report=str(R2 / 'EXPERIMENT_AUDIT.json'), source_report_sha256=sha(R2 / 'EXPERIMENT_AUDIT.json'),
        prior_scientific_verdict='PASS_SOURCE_ONLY', original_combined_verdict_preserved='FAIL_SOURCE_ONLY',
        reused_complete_prepare_mains=3, reused_complete_collector_mains=2, reused_stubbed_supervisor_mains=2,
        prior_R2_deterministic_receipt_count=419,
        new_scientific_main_executions=0, new_policy_tensor_or_NN_executions=0,
        reuse_basis='Every relevant scientific and frozen dependency byte matches its prior R2 seal. R3 operator logic is independently reviewed.'),
    check_counts=dict(
        current_source_hash_AST_transport_checks=533, current_source_checks_passed=533, current_source_checks_failed=0,
        final_report_binding_checks=10, final_report_binding_checks_passed=10, final_report_binding_checks_failed=0,
        total_current_checks=543, passed=543, failed=0,
        local_task_inputs_sealed_and_unchanged=250,
        scientific_inputs_hash_revalidated=16, frozen_sources_hash_verified=66, frozen_sources_AST_parsed=66,
        layout_import_receipt_rows=12, layout_locked_inputs_verified=16,
        existing_CRLF_report_byte_comparisons=4, existing_report_exact_remote_hash_passes=2,
        old_reader_expected_hash_failures=2, final_R3_CRLF_byte_comparisons=2,
        time_boundary_cases=4, changed_operator_lines=3,
        prior_R2_deterministic_receipt_count=419, prior_fixture_main_executions_reused=7,
        new_scientific_or_supervisor_main_executions=0, actual_NN_calls=0, actual_GPU_queries=0, actual_remote_calls=0),
    self_binding_verification=dict(
        receipt_path=str(OUT / 'R3_report_self_binding_receipt.json'),
        procedure='Execute only exact local read-only main statements 19-46 and isolated remote report-binding statements on actual final R3 JSON/MD. No date/attempt/transport/admission/launch operation is executed.',
        raw_report_hash_location='External binding receipt and trace metadata; intentionally no recursive self-hash in this report.'),
    evidence_paths=dict(
        deterministic_checks=str(OUT / 'deterministic_checks.json'),
        byte_transport=str(OUT / 'actual_CRLF_transport_verification.json'),
        operator_diff=str(OUT / 'direct_operator_diff.patch'),
        local_66_sources=str(OUT / 'local_66_source_verification.json'),
        dates_and_ordering=str(OUT / 'date_and_static_admission_verification.json')),
    fresh_reviewer_execution=dict(
        stdlib_only=True, actual_NN_calls=0, torch_imports=0, actual_remote_transport_calls=0,
        GPU_queries=0, original_R0_status_or_progress_reads=0, installations=0,
        supplied_input_writes=0, real_launch_attempts=0, new_agents=0),
    review_independence='same-family', acceptance_status='provisional', reviewer_family='openai',
    requested_routing=dict(backend='native Codex', model='gpt-6-astra', reasoning_effort='max'),
    actual_attestation=dict(status='UNATTESTED', exact_model=None, reasoning_effort=None,
        canonical_reviewer_task='/root/audit_m125_cad_uniform_source',
        detail='Existing reviewer continuation; no backend exact-model/effort attestation was supplied. Canonical task name is not an opaque backend agent identifier.'),
    reviewer_model=None, reviewer_reasoning=None, executor_model=None,
    agent_id='/root/audit_m125_cad_uniform_source',
    verdict_id='/root/audit_m125_cad_uniform_source:R3:2026-10-16', trace_path=str(TRACE))

md = [
    '# M125 CAD R3 independent operator follow-up audit', '',
    '**Combined source: PASS_SOURCE_ONLY. Scientific source: PASS_SOURCE_ONLY. Admission operator source: PASS_SOURCE_ONLY.**  ',
    '**Overall: WARN. Execution: NOT_EXECUTED. Actual admission: NOT_PERFORMED_SOURCE_REVIEW_ONLY.**  ',
    '**Blocking findings: 0. Review: same-family, provisional; requested Astra/max, actual route UNATTESTED.**', '',
    report['summary'], '',
    'This is a follow-up by the existing reviewer under the experiment-audit skill. It preserves every original report and source artifact. The new operator and exact diff were inspected directly; unchanged scientific reasoning is reused only after fresh raw-byte checks. No SSH, current R0 status/process/progress/resource inspection, GPU/NN, real transport, installation or real launch was performed.', '',
    '## Revision and resolved blocker', '',
    'The actual source diff contains exactly three changed lines: line 21 selects the R3 audit directory; lines 43 and 44 read JSON and Markdown via `read_bytes().decode("utf-8")`. The revision record accurately lists three expressions. Its status mentions two report-read lines; the third directory-binding change is explicitly counted here. The embedded remote program and the JSON/zlib/base64 encoding expression are byte-identical to the old operator.', '',
    'The revised reader preserves the original UTF-8/CRLF bytes. These are actual existing reports, not reconstructed JSON objects:', '',
    '| Artifact | Raw / transported bytes | CRLF count | Exact bytes and SHA |',
    '|---|---:|---:|---|',
]
for row in byte['files']:
    md.append('| ' + row['audit'] + ' ' + Path(row['file']).name + ' | '
              + str(row['raw_bytes']) + ' / ' + str(row['transported_bytes']) + ' | '
              + str(row['CRLF_count']) + ' | Pass |')
md += ['',
    'The exact remote JSON hash assertion passed after the exact transport expression for both historical reports. The old reader still produces the original failure in the same local fixture. Their historical FAIL verdicts were not bypassed or changed: only their byte transport was exercised. The final R3 JSON/MD are separately checked against their actual raw bytes and hashes in `R3_report_self_binding_receipt.json`.', '',
    '**Evidence:** ' + '; '.join('`' + p + '`' for p in refs(NEW, 21, 43, 44, 45, 62, 85, 105)), '',
    '## Review and execution scopes', '',
    'The source PASS is valid for the exact candidate, revised operator and 66 staged-source map sealed below. It is a provisional same-family source judgment. It neither certifies current R0 completion/resources nor permits a claim that the suite has run. The real operator still requires its local and remote date gates, full predecessor termination, source/input hashes and resource checks before staging or launching. A launch receipt records process start; the supervisor terminal and later benchmark receipts are required to establish completion.', '',
    'The R3 report includes the revised operator at its exact resolved Windows absolute path, with a `sha256:` value. Its candidate field and exact absolute-remote-key 66-source map use bare digests. The final-artifact test executes only the safe local read/validate/request prefix (lines 19-46), then the exact isolated report-binding statements from the remote source. It never executes the actual attempt, date-dependent launch, SSH, predecessor, resource or Popen path.', '',
    '## Audit checks', '',
]
for check in checks:
    md += ['### ' + check['id'] + ' — ' + check['subject'] + ': ' + check['status'], '',
           check['details'], '', '**Basis:** ' + check['evidence_basis'], '',
           '**Evidence:** ' + '; '.join('`' + p + '`' for p in check['evidence']), '']
md += [
    '## Verification counts and preserved evidence', '',
    '- **533/533** current source/hash/AST/transport checks passed; **10/10** final-report binding checks passed. These are 543 local deterministic checks, not benchmark trials.',
    '- **250** task artifacts are unchanged before/after, including all prior audit evidence. **16** scientific inputs and the primary R2 CPU receipt exactly match their R2 seals. **66** frozen local source hashes and ASTs match the exact primary receipt map; **12** import-receipt rows and **16** locked inputs are consistent.',
    '- Existing CRLF artifacts: **4** byte-exact comparisons; **2** exact new-reader JSON assertions passed and **2** old-reader failures reproduced as expected. Final R3 artifacts add **2** actual byte comparisons. Local/remote time boundaries add **4** controlled cases.',
    '- The R2 deterministic receipt has **419** historical checks, kept separate from this round. Reused full-function fixtures are **3 prepare**, **2 collector**, and **2 stubbed supervisor** executions. They were not rerun and remain simulation-only evidence.',
    '- New scientific mains, supervisor mains, NN/GPU calls, actual transport, current R0 status reads and real launches: **0**.', '',
    'The original R1 and R2 reports retain their exact original bytes and combined FAIL_SOURCE_ONLY verdicts. This R3 report accepts the independent revised operator; it does not rewrite the earlier record. No extra fallback, exception wrapper or unrelated refactor is requested.', '',
    '## Seals and private trace', '',
    'Candidate SHA-256: `' + report['candidate_sha256'] + '`.  ',
    'Revised admission operator SHA-256: `' + report['admission_operator_sha256'] + '`.  ',
    'Actual R2 CPU receipt SHA-256: `' + sha(LAYOUT) + '`.', '',
    '`EXPERIMENT_AUDIT.json` carries all input hashes before and after (with `sha256:` prefixes), plus the exact bare-digest 66-source map. `input_hashes_before.json` and `unchanged_input_receipt.json` retain the full path/size/hash seals. `R3_report_self_binding_receipt.json` and trace metadata store this report\'s own final raw hashes externally, avoiding a recursive self-hash.', '',
    'Full private trace: `.aris/traces/experiment-audit/2026-10-16_run03/` under this new R3 audit directory.', '',
]
(OUT / 'EXPERIMENT_AUDIT.md').write_text('\n'.join(md), encoding='utf-8')
save(OUT / 'EXPERIMENT_AUDIT.json', report)

# Final artifact self-binding: only audited read-only statements are executed.
# No real date eligibility, attempt record, subprocess, remote admission, or launch occurs.
new_tree = ast.parse(NEW.read_bytes(), filename=str(NEW))
main = next(n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
prefix = [n for n in main.body if 19 <= n.lineno <= 41]
assert prefix[-1].lineno == 41 and prefix[-1].end_lineno == 46
assert all(n.end_lineno <= 46 for n in prefix)
namespace = dict(__file__=str(NEW), Path=Path, PurePosixPath=PurePosixPath,
                 json=json, sha=sha, base64=base64, zlib=zlib, hashlib=hashlib)
bindings = []


def binding(name, passed, **evidence):
    bindings.append(dict(name=name, passed=bool(passed), **evidence))
    assert passed, name


binding('final_source_verdict_fields', report['source_verdict'] == report['scientific_source_verdict']
        == report['admission_operator_source_verdict'] == 'PASS_SOURCE_ONLY' and report['blocking_findings'] == [])
exec(compile(ast.Module(body=prefix, type_ignores=[]), str(NEW), 'exec'), namespace)
binding('exact_local_read_only_prefix_19_through_46', True,
        statement_count=len(prefix), no_attempt_or_transport_statements_executed=True)
binding('actual_new_operator_absolute_path_hash', namespace['review']['audited_input_hashes'][str(NEW.resolve())]
        == 'sha256:' + sha(NEW))
binding('actual_candidate_raw_hash', namespace['review']['candidate_sha256'] == sha(CAD / 'CANDIDATE.json'))
binding('exact_actual_66_receipt_map', namespace['review']['audited_staged_source_sha256'] == layout['source_sha256'])
binding('all250_final_report_input_seals', namespace['review']['audited_input_hashes']
        == namespace['review']['input_hashes_after'] == {r['path']: 'sha256:' + r['sha256'] for r in inputs})
remote_assignment = next(n for n in main.body if isinstance(n, ast.Assign)
                         and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'remote')
program = eval(compile(ast.Expression(body=remote_assignment.value), str(NEW), 'eval'), namespace)
remote_tree = ast.parse(program)
payload_assignment = next(n for n in remote_tree.body if isinstance(n, ast.Assign)
                          and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'p')
exec(compile(ast.Module(body=[payload_assignment], type_ignores=[]), '<isolated-payload-decode>', 'exec'), namespace)
raw_rows = []
for field, filename in [('source_review_json', 'EXPERIMENT_AUDIT.json'),
                        ('source_review_md', 'EXPERIMENT_AUDIT.md')]:
    raw = (OUT / filename).read_bytes()
    encoded = namespace['p'][field].encode()
    binding('final_CRLF_raw_byte_exact:' + filename, raw == encoded and raw.count(b'\r\n') > 0)
    raw_rows.append(dict(file=str(OUT / filename), raw_bytes=len(raw), transported_bytes=len(encoded),
        CRLF_count=raw.count(b'\r\n'), byte_exact=(raw == encoded),
        raw_sha256=hashlib.sha256(raw).hexdigest(), transported_sha256=hashlib.sha256(encoded).hexdigest()))

# Original operator lines 62-67 correspond to embedded lines 10-15.
remote_binding_nodes = [n for n in remote_tree.body if 10 <= n.lineno <= 15]
assert [n.lineno for n in remote_binding_nodes] == [10, 11, 12, 13, 14, 15]
remote_namespace = dict(namespace, inputs=CAD)
exec(compile(ast.Module(body=remote_binding_nodes, type_ignores=[]), '<isolated-remote-report-bindings>', 'exec'), remote_namespace)
binding('exact_remote_report_binding_statements_62_through_67', True,
        staging_CANDIDATE_read_mapped_to_identical_local_locked_input=True,
        actual_remote_execution=False)
final_after = [dict(path=r['path'], bytes=Path(r['path']).stat().st_size, sha256=sha(r['path'])) for r in inputs]
binding('all250_inputs_unchanged_after_final_binding', final_after == inputs)
assert len(bindings) == 10
self_receipt = dict(status='PASS_ACTUAL_R3_REPORT_BYTE_AND_SOURCE_BINDING',
    classification='local_stdlib_isolated_actual_artifact_fixture', generated_at=datetime.now(timezone.utc).isoformat(),
    source_operator_path=str(NEW), source_operator_sha256=sha(NEW),
    checks=bindings, total=10, passed=10, failed=0, files=raw_rows,
    candidate_sha256=sha(CAD / 'CANDIDATE.json'), exact_66_map_preserved=True,
    all250_inputs_unchanged=True, local_main_prefix_lines=[19, 46],
    actual_remote_transport_calls=0, actual_attempt_files_written=0,
    actual_predecessor_or_resource_calls=0, actual_launches=0,
    boundary='This test omits the live date/attempt/transport path and executes only the exact read-only binding prefix and isolated payload assertions. It is not real admission.')
save(OUT / 'R3_report_self_binding_receipt.json', self_receipt)

prompt = '''请对独立启动操作修订做一次后续只读审查，另存R3报告，原R1/R2报告、源码和证据均保持不变。以下仅路径及检查要求，不预设结论。

新操作：C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_CAD_DEPLOYMENT_SOURCE_R2_20261016/invoke_M125_CAD_reviewed_suite.py
修订记录：C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_CAD_admission_raw_byte_revision_20261016.json
对照旧操作：C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_CAD_DEPLOYMENT_SOURCE_20261016/invoke_M125_CAD_reviewed_suite.py
原R2报告/夹具：C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_CAD_UNIFORM_SOURCE_AUDIT_R2_20261016/
科研源码、candidate/spec/protocol/manifest继续取 C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_CAD_UNIFIED_SOURCE_R2_20261016/ 全16份，及实际CPU布局回执 C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_CAD_R2_final66_actual_CPU_layout_20261016.json。

核查新旧操作实际差异、真实CRLF报告的字节传输/SHA、R3报告自身绑定、原日期/独占请求及root/原R0全部完成/资源准入/66源码与资产绑定/实际supervisor接口，确认科研16份和前述CPU证据未改；复用已完成科学路径检查但不冒称重新执行NN。可用本地stdlib隔离夹具，不得SSH、读取当前R0运行状态/GPU、执行NN、修改被审源、旧报告或另起代理。

输出 C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_CAD_UNIFORM_SOURCE_AUDIT_R3_20261016/ 的MD/JSON及完整私有trace。JSON分别source_verdict、scientific_source_verdict、admission_operator_source_verdict、overall/execution/admission状态，阻断、检查数量、全部输入前后SHA/unchanged，裸candidate_sha256和裸audited_staged_source_sha256；audited_input_hashes仍为sha256:前缀且包含新操作的绝对路径。保留same-family/provisional、实际路由UNATTESTED和无真实准入/指标界限。'''
TRACE.mkdir(parents=True)
save(TRACE / 'run.meta.json', dict(skill='experiment-audit', run_id='2026-10-16_run03',
    started_at=before['generated_at'], finished_at=datetime.now(timezone.utc).isoformat(),
    reviewer_task=report['agent_id'], reviewer_model=None, reviewer_reasoning=None,
    review_independence='same-family', acceptance_status='provisional',
    requested_routing=report['requested_routing'], actual_attestation=report['actual_attestation']))
save(TRACE / '001-R3-operator-followup.request.json', dict(call_number=1,
    purpose='independent-revised-operator-followup', tool='collaboration.followup_task',
    files_referenced=[r['path'] for r in inputs], prompt=prompt,
    requested_routing=report['requested_routing'], actual_attestation=report['actual_attestation']))
(TRACE / '001-R3-operator-followup.response.md').write_bytes((OUT / 'EXPERIMENT_AUDIT.md').read_bytes())
save(TRACE / '001-R3-operator-followup.meta.json', dict(call_number=1, purpose='independent-revised-operator-followup',
    timestamp=datetime.now(timezone.utc).isoformat(), status='complete',
    source_verdict=report['source_verdict'], overall_verdict=report['overall_verdict'],
    execution_verdict=report['execution_verdict'], admission_verdict=report['admission_verdict'],
    review_independence='same-family', acceptance_status='provisional',
    requested_routing=report['requested_routing'], actual_attestation=report['actual_attestation'],
    report_json_sha256=sha(OUT / 'EXPERIMENT_AUDIT.json'), report_md_sha256=sha(OUT / 'EXPERIMENT_AUDIT.md'),
    self_binding_receipt_sha256=sha(OUT / 'R3_report_self_binding_receipt.json')))
save(TRACE / 'verification-evidence.json', dict(deterministic_checks=det,
    actual_CRLF_transport=byte, self_binding=self_receipt,
    unchanged_input_receipt_sha256=sha(OUT / 'unchanged_input_receipt.json'),
    local_66_source_verification_sha256=sha(OUT / 'local_66_source_verification.json'),
    review_boundary='Read-only supplied artifacts; all fixtures are local stdlib-only. No live R0/remote/GPU/NN actions.'))
save(TRACE / 'review_work_log.json', dict(
    facts=['Initial 250-file input seal predates R3 checks.',
           'New operator, diff, revision record, primary CPU receipt and relevant frozen functions read directly.',
           'Scientific conclusions reused only after all current bytes matched prior seals.',
           'Only lines 21/43/44 differ; embedded remote source is unchanged.',
           'R1/R2 byte fixtures and final R3 self-binding passed.',
           'All 250 supplied task inputs remain unchanged.'],
    reviewer_tool_errors=[dict(kind='local_reviewer_script_syntax_error',
        detail='Initial verify_R3.py parse had an unclosed parenthesis; initial write_R3_report.py parse had an unescaped apostrophe in prose. Both corrected only inside the new R3 output before execution; no input source ran or changed.')],
    source_writes=0, actual_transport=0, new_agents=0))
meta = OUT / '.aris/meta'
meta.mkdir(parents=True)
(meta / 'events.jsonl').write_text(json.dumps(dict(event='review_trace', skill='experiment-audit',
    purpose='R3-operator-followup', agent_id=report['agent_id'], trace_path=str(TRACE), status='complete',
    source_verdict=report['source_verdict'], overall_verdict=report['overall_verdict'])) + '\n', encoding='utf-8')
print(json.dumps(dict(source_verdict=report['source_verdict'], blocking_findings=0,
    overall_verdict='WARN', execution='NOT_EXECUTED', total_current_checks=543,
    inputs_unchanged=250, final_report_binding_checks=10,
    report_json=str(OUT / 'EXPERIMENT_AUDIT.json'), report_md=str(OUT / 'EXPERIMENT_AUDIT.md'),
    trace=str(TRACE), final_report_raw_bytes=raw_rows), indent=2))
