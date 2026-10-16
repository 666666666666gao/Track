"""Write this review's original report and private trace, without changing inputs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
CAD = BASE / 'M125_CAD_UNIFIED_SOURCE_20261016'
R0 = BASE / 'M125_R0_FIXED_SOURCE_20261015'
SNAP = R0 / 'frozen_dependency_snapshots'
OP = BASE / 'M125_REGION_DEPLOYMENT_SOURCE_R2_20261015/deploy_m125_region_fixed.py'
TRACE = OUT / '.aris/traces/experiment-audit/2026-10-16_run01'
TRACE.mkdir(parents=True)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def evidence(file, *lines):
    return [str(file) + ':' + str(line) for line in lines]


before = read(OUT / 'input_hashes_before.json')
checks = read(OUT / 'deterministic_checks.json')
versions = read(OUT / 'receipt_version_relationships.json')
reproductions = read(OUT / 'interface_reproductions.json')
final66 = read(BASE / 'M125_CAD_final66_actual_CPU_layout_20261016.json')
now = datetime.now(timezone.utc).isoformat()
after = [dict(path=row['path'], bytes=Path(row['path']).stat().st_size, sha256=sha(row['path'])) for row in before['files']]
changed = [new['path'] for old, new in zip(before['files'], after) if old != new]
assert not changed
unchanged = dict(status='UNCHANGED', generated_at=now, files_checked=len(after), changed_files=changed,
    before_receipt_sha256=sha(OUT / 'input_hashes_before.json'), files=after,
    scope='All 79 sealed local scientific inputs; no remote source or running R0 inspection.')
save(OUT / 'unchanged_input_receipt.json', unchanged)

blocking = [
    dict(id='B1', severity='blocking', title='Preparation uses an undefined training CLI attribute',
         finding='The submitted CLI creates output/profile/reference_evaluation/source_gate only. The selection writer reads args.training, so the coordinator prepare_U0 stage raises AttributeError before creating selection.json or starting qualification.',
         evidence=evidence(CAD / 'prepare_m125_CAD_evaluation.py', 78, 80, 93, 95, 96) + evidence(CAD / 'run_m125_CAD_uniform_suite.py', 58, 61),
         reproduction=reproductions['prepare_missing_training'],
         minimal_fix='Use the training_result_sha256 already bound into the immutable reference bundle; do not add a second independently selectable training input.'),
    dict(id='B2', severity='blocking', title='VOT event loop overwrites the model name before collection',
         finding='The model name is assigned at collector line 12, then overwritten by each event filename at line 43. Line 46 seeks <last-event-filename>.json_full127.json instead of the official CAD_U1_same_R0_fixed6_full127.json (or U2 equivalent). After B1 is fixed, the first complete profile cannot be collected and the suite cannot advance to U2. The same overwritten name would also contaminate VOT rows and gates if an incorrectly named analysis file existed.',
         evidence=evidence(CAD / 'collect_m125_CAD_results.py', 12, 43, 46, 48, 55) + evidence(CAD / 'run_m125_CAD_uniform_suite.py', 89, 90, 93),
         reproduction=reproductions['collector_model_name_shadowing'],
         minimal_fix='Rename the event-loop variable (for example event_name) and preserve the model name for analysis lookup, result rows, and gates.')
]

warnings = [
    dict(id='W1', severity='warning', title='The one-time asset-hashing claim is contradicted by the inherited runtime',
         finding='SOURCE_SPEC declares one-time asset hashing and no unused training-warm rehash per VOT anchor. Every vot_track calls CAD checked_plan; it invokes checked_R0_plan and thereby m122 checked_plan, which hashes the baseline final, training result, native checkpoint and CLIP weights. CAD make_bundle also calls verify_predecessor, which rehashes R0 final and result files. There is no memoization or immutable cached verification in these functions. This is an unsupported performance/source-description claim, not a measured runtime regression or a reason to modify the current R0.',
         evidence=evidence(CAD / 'SOURCE_SPEC.json', 42, 43) + evidence(CAD / 'run_m125_CAD_official.py', 68) +
             evidence(CAD / 'm125_CAD_official_runtime.py', 26, 44, 47) + evidence(R0 / 'm125_R0_fixed_official_runtime.py', 11, 20) +
             evidence(SNAP / 'm122_official_runtime.py', 12, 13) + evidence(OP, 82, 89),
         action='Correct the claim, or make a separately reviewed CAD-side change if avoiding repeated hashing is a requirement. Preserve frozen R0 sources.'),
    dict(id='W2', severity='scope', title='CPU/import evidence does not establish executable end-to-end admission',
         finding='All three receipts explicitly record zero real tracking. Both pipeline import receipts record zero entrypoint main executions; the early import uses an older runtime hash. The final66 receipt fixes the layout/version scope but still does not exercise preparation, qualification, collection, TraX, resources, or launch admission.',
         evidence=[str(BASE / name) + ':1' for name in ['M125_CAD_uniform_actual_CPU_checks_20261016.json', 'M125_CAD_uniform_actual_pipeline_CPU_imports_20261016.json', 'M125_CAD_final66_actual_CPU_layout_20261016.json']],
         action='Keep these as CPU/source checks. Do not promote them to actual deployment, real64 qualification, full U0 parity, or new formal metrics.'),
    dict(id='W3', severity='scope', title='Scientific coordinator exists; a CAD-specific local admission/deployment operator is absent from the package',
         finding='CANDIDATE is still candidate_M125_CAD_uniform_inference_source_gate. The supervisor/coordinator consume a pre-existing reviewed gate but do not verify this report or create/admit that gate. The reused operator main is explicitly regional: it requires an arm, different candidate files/status, 11 new/64 staged files, and launches regional training. Only verify_predecessor/resource_admission are reused by the CAD coordinator. No CAD-specific staging/review-bound gate producer or actual deployment/admission receipt is supplied.',
         evidence=evidence(CAD / 'CANDIDATE.json', 2) + evidence(CAD / 'm125_CAD_process_supervisor.py', 11, 12) +
             evidence(CAD / 'run_m125_CAD_uniform_suite.py', 43, 47, 49) + evidence(OP, 32, 41, 44, 99, 125, 168, 218),
         action='Treat the source package as implemented scientific orchestration requiring correction and later explicit admission, not as an admitted launch. The absence is an execution boundary, separate from B1/B2.'),
    dict(id='W4', severity='scope', title='VOT event hashes are designed, but actual event-to-trajectory evidence is unavailable',
         finding='The source emits one exclusive event file keyed by the legal initialization key, validates profile/final/bundle and unique cardinality 1765, sums counters, and seals/rechecks event hashes. The analyzer does not independently match its key set or tracking-call totals to each official trajectory. No actual events or results are in the allowed evidence. Therefore the source supports implemented counter instrumentation, not an independently recounted completed VOT counter result.',
         evidence=evidence(CAD / 'm125_CAD_official_runtime.py', 67) + evidence(R0 / 'm125_R0_fixed_official_runtime.py', 76, 77) +
             evidence(CAD / 'run_m125_CAD_official.py', 81, 84, 85) + evidence(CAD / 'analyze_m125_CAD_vot.py', 35, 41, 43, 44, 50) +
             evidence(CAD / 'collect_m125_CAD_results.py', 42, 43),
         action='At completed-result review, distinguish sealed emitted counts from independently reconciled trajectory counts. No speculative fallback or unrelated refactor is requested.')
]

af = {
    'A_ground_truth_provenance': dict(status='PASS_SOURCE_ONLY',
        details='OPE tracking uses the sealed initialization box plus RGB/depth frames; subsequent groundtruth.txt is opened only in post-prediction analysis. VOT consumes the official initial rectangle and subsequent images. The bank maps legal RGB-byte/box keys, with historical multi-frame human review disclosed. Training provenance checks require the original Train152 six-pass final. No actual dataset or running R0 artifacts were opened in this audit.',
        evidence=evidence(CAD / 'run_m125_CAD_official.py', 23, 27, 29, 55, 56, 72, 74, 79) +
            evidence(SNAP / 'bind_m122_official_initializations.py', 13, 28, 39, 46, 49, 54) + evidence(R0 / 'm125_R0_fixed_official_runtime.py', 23, 24, 25, 26, 34)),
    'B_score_normalization': dict(status='PASS_SOURCE_ONLY',
        details='The OPE metric invocation, resolution and hash remain unchanged from R0. VOT reads the same official baseline result indices and converts fractions to percent with multiplication by 100. No own-prediction max/min/mean denominator is introduced. The selected native response controls writes while best_score remains the final selected ABC score. Official external metric implementations are hash-pinned and were not fetched/rerun.',
        evidence=evidence(CAD / 'run_m125_CAD_official.py', 57, 59) + evidence(CAD / 'analyze_m125_CAD_vot.py', 23, 24) +
            evidence(CAD / 'profiled_template_tracker.py', 44, 45, 51, 78)),
    'C_result_existence_and_version': dict(status='WARN',
        details='Primary CPU receipts exist and are hash-verified. Four policy/tensor sources match current source; the early import records 12 import successes but its runtime digest is old (010dfd61...), with the other 11 CAD modules and operator matching. Final66 records all 12 imports, exact 66-source layout and 16 locked inputs. Its full raw source map is copied unchanged into audited_staged_source_sha256. No new CAD GPU/formal metric result is claimed or established; source completeness is blocked by B1/B2.',
        evidence=[str(BASE / name) + ':1' for name in ['M125_CAD_uniform_actual_CPU_checks_20261016.json', 'M125_CAD_uniform_actual_pipeline_CPU_imports_20261016.json', 'M125_CAD_final66_actual_CPU_layout_20261016.json']] + evidence(CAD / 'SOURCE_SPEC.json', 37, 38, 45)),
    'D_reachable_execution_and_dead_code': dict(status='FAIL',
        details='The scientific coordinator explicitly reaches qualification, OPE analysis, official VOT analysis, sealing and collection; these are not unused metric definitions. However preparation always fails on the unsupported training attribute and collection later fails because event filenames shadow the model name. Import-only checks cannot detect these failures.',
        evidence=evidence(CAD / 'prepare_m125_CAD_evaluation.py', 80, 93) + evidence(CAD / 'collect_m125_CAD_results.py', 43, 46) +
            evidence(CAD / 'run_m125_CAD_uniform_suite.py', 58, 61, 84, 90, 92, 93)),
    'E_scope_assessment': dict(status='WARN',
        details='Observed evidence is scalar policy tests, synthetic tensor interface tests and CPU imports. Planned U1/U2 rows require DT Test50/76373 frames, CDTB80/101956 frames and VOT127/1765 anchors/5295 files. U0 is expressly the original complete R0 result; the new U0 path adds only 64 tracking calls and a 65-row six-decimal box/score prefix comparison. None proves a new full U0 rerun, hidden-state bit parity, tracking gains, C usefulness, or full nine-metric success.',
        evidence=evidence(CAD / 'PROTOCOL.md', 11, 23, 29) + evidence(CAD / 'run_m125_CAD_uniform_suite.py', 61, 69, 70, 71, 75) +
            evidence(CAD / 'check_m125_CAD_real64.py', 27, 57, 62) + evidence(CAD / 'run_m125_CAD_official.py', 15, 16) + evidence(CAD / 'prepare_m125_CAD_evaluation.py', 39, 73, 74)),
    'F_evaluation_type': dict(status='PASS', actual='simulation_only', planned='real_gt',
        details='Policy/tensor/interface fixtures are synthetic logic simulations, with the old tracker as a consistency reference, not performance ground truth. Import checks are static interface evidence. The intended full benchmark pipeline is real_gt with dataset/official protocol annotations used after predictions. No simulation score is represented as dataset performance.',
        evidence=evidence(CAD / 'check_cad_template_profiles.py', 18, 37, 39, 40) + evidence(CAD / 'check_m125_CAD_CPU_tracker.py', 125, 126, 128, 130) +
            evidence(CAD / 'run_m125_CAD_official.py', 55, 59))
}

supplemental = [
    dict(id='S1', status='PASS_SOURCE_ONLY', subject='One final and one uniform profile per complete row',
        details='Runtime reconstructs the exact bundle/plan from the original completed R0 final and one named profile. No dataset/sequence-specific weight, seed, layer, or profile branch is present. Optional U3/U4 are excluded from the primary gate and CLI.',
        evidence=evidence(CAD / 'cad_template_profiles.py', 25, 32) + evidence(CAD / 'm125_CAD_official_runtime.py', 21, 29, 30, 47, 49, 54) + evidence(CAD / 'run_m125_CAD_uniform_suite.py', 44, 75, 102)),
    dict(id='S2', status='PASS_SOURCE_ONLY', subject='Official CAD versus this adaptation',
        details='Official CADTrack uses a Hann response maximum, interval 5, strict threshold 0.65 and template slot preservation. The local adaptation uses the native response at the ABC-selected cell and retains current-quality C for U0/U1/U2. Official LasHeR/RGBT training/normalization and CAD network are not imported. This is a strategy comparison, not a CAD network reproduction or a new original module.',
        evidence=evidence(BASE / 'CADTrack_PRIMARY_20261016/lib/test/tracker/cadtrack.py', 103, 113, 114, 120, 122) +
            evidence(BASE / 'CADTrack_PRIMARY_20261016/experiments/cadtrack/cadtrack.yaml', 28, 31, 86, 87) + evidence(CAD / 'PROTOCOL.md', 21, 27)),
    dict(id='S3', status='PASS_SOURCE_ONLY', subject='Strict thresholds and selected-cell qualification',
        details='Response qualification and C acceptance both use strict >. Profiled tracker samples native_response at selected[0], never an unrelated native maximum. Pure policy 5416 checks were independently rerun with stdlib, and the supplied mock has an unrelated larger native peak at cell 3 to exercise this distinction.',
        evidence=evidence(CAD / 'cad_template_profiles.py', 14, 16, 18, 22) + evidence(CAD / 'profiled_template_tracker.py', 44, 46) + evidence(CAD / 'check_m125_CAD_CPU_tracker.py', 63, 66, 67, 117)),
    dict(id='S4', status='PASS_SOURCE_ONLY', subject='Prediction, state, query, features and template semantics',
        details='The first 20 statements of old/new step have identical normalized AST, covering the observation/decoder and committed selected box, native query, selected feature and selected native response. Reporting keeps final ABC confidence; only the write gate changes. A qualified accepted write appends the six-channel crop and pops slot 1, retaining the initial slot. Synthetic U0 receipt covers 100 steps and matching C inputs; no real long-run parity is inferred.',
        evidence=evidence(CAD / 'profiled_template_tracker.py', 25, 41, 42, 43, 44, 61, 67, 70, 71) +
            evidence(SNAP / 'trusted_template_tracker.py', 28, 44, 45, 46, 47, 58, 66, 67) + evidence(SNAP / 'full_dense_tracker.py', 36, 69)),
    dict(id='S5', status='PASS_SOURCE_ONLY', subject='C participation and counters',
        details='U0/U1/U2 call C on every qualified write event using the unchanged 519-value current/previous/initial/quality/motion feature function and threshold 0.5. U4 bypasses C and is explicitly optional/unqueued. Counters reset per initialization and increment actual code paths; OPE collector checks tracking_calls=frames-1. Supplied synthetic receipt records U4 C_evaluations=0.',
        evidence=evidence(CAD / 'profiled_template_tracker.py', 21, 48, 49, 60, 61, 64, 65, 72) +
            evidence(SNAP / 'template_write_features.py', 29, 35, 36, 38) + evidence(CAD / 'collect_m125_CAD_results.py', 30) + evidence(CAD / 'PROTOCOL.md', 19, 21)),
    dict(id='S6', status='PASS_SOURCE_ONLY', subject='Final66 dependency/sibling resolution',
        details='All 66 prospective sources hash-match local files: 12 CAD modules, 53 original R0/frozen dependency sources and one frozen R2 operator. All parse with Python AST. Final66 recorded package origins are internally consistent with that separate layout. All 12 modules and 9 guarded entrypoints were inspected; the library-only modules have no CLI. Import success does not establish main behavior.',
        evidence=evidence(CAD / 'CANDIDATE.json', 180, 233, 245) + [str(BASE / 'M125_CAD_final66_actual_CPU_layout_20261016.json') + ':1']),
    dict(id='S7', status='FAIL', subject='End-to-end final/profile/bundle/plan/receipt/collection binding',
        details='Bindings reconstruct the exact reference plan, final, source gate, profile and output paths and check receipt/result hashes. The intended chain is meaningful, but B1 prevents selection creation and B2 breaks official analysis lookup/model identity at collection. Source cannot be accepted end-to-end until both are corrected.',
        evidence=evidence(CAD / 'm125_CAD_official_runtime.py', 42, 45, 47, 48, 50, 54) + evidence(CAD / 'run_m125_CAD_official.py', 40, 43, 61, 63) + evidence(CAD / 'collect_m125_CAD_results.py', 21, 25, 38, 41, 43, 46)),
    dict(id='S8', status='PASS_SOURCE_ONLY', subject='Official metric and scope locks',
        details='Full OPE cardinalities and frame totals are unchanged. VOT uses frozen 127-sequence/1765-anchor manifest, official analysis command and 5295 result-file merge. OPE call, VOT metric extraction and all nine pass thresholds are AST-identical to original R0. Counter files are outside official prediction result directories.',
        evidence=evidence(CAD / 'run_m125_CAD_official.py', 15, 16, 57, 59) + evidence(CAD / 'run_m125_CAD_uniform_suite.py', 90) +
            evidence(SNAP / 'run_m123_joint_vot_shards.py', 45, 59, 60) + evidence(CAD / 'analyze_m125_CAD_vot.py', 19, 24) + evidence(CAD / 'collect_m125_CAD_results.py', 52, 53, 54)),
    dict(id='S9', status='PASS_SOURCE_ONLY', subject='Time, source, predecessor and resource ordering; no automatic retry',
        details='Suite checks optimization flag and not-before time before reading its gate, then source/input hashes, completed predecessor and GPU-memory/disk admission before launching stages. Runtime bundle creation also checks time first. Frozen predecessor verifier requires both zero exits, terminal receipt, all six complete passes and three result receipts. Only GPU index/memory fields are queried. Subprocess nonzero exits assert and VOT restart=false; no autoretry loop is introduced. None of these admissions was executed by this audit.',
        evidence=evidence(CAD / 'run_m125_CAD_uniform_suite.py', 35, 40, 41, 45, 46, 47, 49) + evidence(CAD / 'm125_CAD_official_runtime.py', 16) +
            evidence(OP, 59, 61, 68, 71, 72, 83, 112, 114, 118) + evidence(CAD / 'prepare_m125_CAD_evaluation.py', 55) + evidence(SNAP / 'run_m123_joint_vot_shards.py', 45)),
    dict(id='S10', status='PASS_LOCAL_HASH_SCOPE', subject='Preservation of fixed R0 and audit inputs',
        details='All 79 local inputs are byte-identical before/after, including all 53 original R0 Python/frozen sources. All reviewer writes remain under the new audit output directory. No current R0 status/log/process/weights/resources were read; remote immutability or progress is not newly attested.',
        evidence=[str(OUT / 'input_hashes_before.json'), str(OUT / 'unchanged_input_receipt.json')])
]

semantic_files = list(sorted(CAD.glob('*.py'))) + [R0 / name for name in [
    'm125_R0_fixed_official_runtime.py', 'prepare_m125_R0_fixed_evaluation.py', 'run_m125_R0_fixed_official.py',
    'analyze_m125_R0_fixed_vot.py', 'collect_m125_R0_fixed_results.py', 'run_m125_R0_fixed_suite.py']] + [SNAP / name for name in [
    'trusted_template_tracker.py', 'full_dense_tracker.py', 'template_write_features.py', 'template_write_C.py',
    'run_m123_joint_vot_shards.py', 'bind_m122_official_initializations.py', 'm122_official_runtime.py', 'prepare_m122_evaluation_suite.py']] + [
    OP, BASE / 'CADTrack_PRIMARY_20261016/lib/test/tracker/cadtrack.py']

report = dict(
    audit_skill='experiment-audit', date='2026-10-16', generated_at=now,
    verdict='FAIL', overall_verdict='FAIL', source_verdict='FAIL_SOURCE_ONLY', integrity_status='fail',
    execution_verdict='NOT_EXECUTED_NOT_ADMITTED', deployment_admission='NOT_ESTABLISHED',
    reason_code='undefined_preparation_argument_and_collector_name_shadowing',
    summary='Two reproducible source blockers prevent preparation and complete-profile collection. CPU/import evidence and the 66-source map are valid within their stated narrow scopes; no new real tracking or formal metrics are established.',
    blocking_findings=blocking, nonblocking_findings=warnings, checks=af, supplemental_checks=supplemental,
    candidate_sha256=sha(CAD / 'CANDIDATE.json'),
    audited_staged_source_sha256=final66['source_sha256'],
    audited_staged_source_sha256_origin=str(BASE / 'M125_CAD_final66_actual_CPU_layout_20261016.json') + ':source_sha256',
    audited_staged_source_map_preserved_exactly=True,
    audited_input_hashes={row['path']: 'sha256:' + row['sha256'] for row in before['files']},
    input_hashes_before_path=str(OUT / 'input_hashes_before.json'),
    unchanged_input_receipt_path=str(OUT / 'unchanged_input_receipt.json'),
    unchanged_input_receipt_sha256=sha(OUT / 'unchanged_input_receipt.json'),
    inputs_unchanged=True, local_input_files=79,
    check_counts=dict(primary_python_files_read=12, all_python_files_semantically_read=len(semantic_files),
        prospective_python_files_AST_parsed=66, new_deterministic_checks=checks['total'],
        new_deterministic_checks_passed=checks['passed'], new_deterministic_checks_failed=0,
        new_pure_policy_assertions=5416, reproduced_blockers=2,
        supplied_policy_check_count=5416, supplied_mock_tensor_check_count=2700,
        supplied_synthetic_tracker_steps=600, supplied_early_import_rows=12, supplied_final66_import_rows=12,
        input_manifest_entries_verified=15, final66_locked_inputs_verified=16,
        all_local_inputs_unchanged=79, source_A_to_F_categories=6, supplemental_categories=10),
    fresh_reviewer_execution=dict(stdlib_only=True, actual_neural_calls=0, torch_imports=0,
        real_tracking_calls=0, actual_candidate_main_executions=0, synthetic_collector_main_executions=1,
        original_R0_progress_queries=0, remote_queries=0, child_transport_calls=0, GPU_queries=0,
        installations=0, supplied_input_writes=0, new_agents=0),
    receipt_version_relationships_path=str(OUT / 'receipt_version_relationships.json'),
    interface_reproductions_path=str(OUT / 'interface_reproductions.json'),
    deterministic_checks_path=str(OUT / 'deterministic_checks.json'),
    direct_diffs_path=str(OUT / 'direct_source_diffs.patch'),
    semantically_read_python_files=[str(p) for p in semantic_files],
    review_independence='same-family', acceptance_status='provisional', reviewer_family='openai',
    requested_routing=dict(backend='native Codex agent', model='gpt-6-astra', reasoning_effort='max',
        source='experiment-audit SKILL.md and local-codex-policy.md'),
    actual_attestation=dict(agent_id='/root/audit_m125_cad_uniform_source', agent_id_kind='canonical_task_path',
        backend='native Codex collaboration task', exact_model=None, reasoning_effort=None,
        status='UNATTESTED', detail='The task context has no backend model/effort attestation or parent spawn arguments. Requested routing is not evidence of actual routing.'),
    agent_id='/root/audit_m125_cad_uniform_source', verdict_id='/root/audit_m125_cad_uniform_source:original:2026-10-16',
    reviewer_model=None, reviewer_reasoning=None, trace_path=str(TRACE),
    claims=[dict(id='policy_and_synthetic_U0_parity', impact='supported_with_synthetic_scope'),
        dict(id='final66_CPU_import_layout', impact='supported_receipt_and_local_hash_scope'),
        dict(id='complete_executable_scientific_pipeline', impact='unsupported_until_B1_B2_fixed'),
        dict(id='one_time_asset_hashing', impact='contradicted_by_source_call_chain'),
        dict(id='real64_or_full_U1_U2_or_new_metrics', impact='not_executed_not_claimed'),
        dict(id='full_new_U0_parity', impact='unsupported; planned prefix only, full original reference separate'),
        dict(id='CAD_network_reproduction_or_new_original_module', impact='not_supported_and_disclaimed'),
        dict(id='actual_CAD_deployment_admission', impact='not_established')])

md = [
    '# Experiment audit — original M125 CAD uniform source', '',
    '**Date:** 2026-10-16  ',
    '**Overall verdict: FAIL. Source verdict: FAIL_SOURCE_ONLY.**  ',
    '**Execution/admission: NOT_EXECUTED_NOT_ADMITTED.**  ',
    '**Independence:** fresh native Codex task; same-family; acceptance provisional. Requested `gpt-6-astra` / `max` by the installed skill; actual model/effort are **UNATTESTED** in this reviewer context.', '',
    report['summary'], '',
    'The review is bounded to the supplied local static artifacts. All 12 CAD Python files, four metadata files, three primary CPU receipts, the two official CAD source files and receipt, the used frozen operator, the six corresponding R0 pipeline files, and eight relevant frozen runtime/binding/feature/shard sources were read directly. The original R0 running task was not inspected. There were no SSH/remote calls, neural executions, GPU/resource queries, installations, actual child transport, or supplied-source writes.', '',
    '## Blocking findings', ''
]
for finding in blocking:
    md.extend(['### ' + finding['id'] + ' — ' + finding['title'], '', finding['finding'], '',
        '**Evidence:** ' + '; '.join('`' + ref + '`' for ref in finding['evidence']), '',
        '**Reproduction:** `' + finding['reproduction']['error_type'] + ': ' + finding['reproduction']['error'] + '`. ' + finding['reproduction']['scope'], '',
        '**Minimal correction:** ' + finding['minimal_fix'], ''])
md.extend(['The reproductions are in `interface_reproductions.json`. The first uses the exact submitted CLI/attribute AST; the second executes the complete unchanged collector `main` AST against in-memory synthetic receipts shaped as 50 OPE sequences, 80 OPE sequences and 1765 VOT events. These are diagnostic simulations, not real evaluation results.', '',
    '## A–F integrity checks', ''])
for name, result in af.items():
    md.extend(['### ' + name.replace('_', ' ') + ': ' + result['status'], '', result['details'], '',
        '**Evidence:** ' + '; '.join('`' + ref + '`' for ref in result['evidence']), ''])
md.extend(['## Source and interface findings', ''])
for result in supplemental:
    md.extend(['### ' + result['id'] + ' — ' + result['subject'] + ': ' + result['status'], '', result['details'], '',
        '**Evidence:** ' + '; '.join('`' + ref + '`' for ref in result['evidence']), ''])
md.extend(['## Nonblocking findings and limits', ''])
for finding in warnings:
    md.extend(['### ' + finding['id'] + ' — ' + finding['title'], '', finding['finding'], '',
        '**Evidence:** ' + '; '.join('`' + ref + '`' for ref in finding['evidence']), '', finding['action'], ''])
md.extend(['## Primary receipt versions and verification counts', '',
    '| Evidence | Verified scope | Relationship to audited source |',
    '|---|---|---|',
    '| Actual policy/tensor CPU receipt, 03:37:37 UTC | 5416 scalar checks; 600 synthetic steps / 2700 counted interface checks; two recorded exit-0 commands | All four recorded source hashes match |',
    '| Early pipeline CPU imports, 03:46:08 UTC | 12 import rows; zero main executions | Runtime hash `010dfd61bf6261b9895916c898cac4550c753d74bc78de5e054aaf2bbadf5b25` differs; other 11 CAD modules and frozen operator match |',
    '| Final66 CPU layout, 03:49:28 UTC | 12 import rows, 66 staged source hashes, 16 locked input hashes; zero main executions | All 66 sources and all 16 locked inputs match current sealed local files |', '',
    'The final runtime digest is `6c330dda1c38f0a021a0e9e7ad926f9b01f0c7f01ccd6c097b189cb714503e7e`. The older import receipt must not be presented as testing that version. Final66 supplies the current import evidence, with its explicit zero-main/zero-real-tracking limits.', '',
    '- Fresh deterministic checks: **339 passed, 0 failed**, including 66 AST parses, source/manifest/receipt comparisons, package-origin consistency, unchanged metric/threshold AST checks, and successful reproduction of both defects.',
    '- Fresh stdlib policy run: **5416 assertions passed**. This repeats the same scalar policy scope; it is not independent learned-model evidence.',
    '- Full semantic Python reads: **28**, including all **12** CAD files. All **66** prospective staged Python sources were parsed and hash-verified.',
    '- All **79** local inputs are unchanged before/after. Existing tensor/import receipts were read and verified locally; no remote test was rerun.', '',
    'A default local Python shim failed with `No pyvenv.cfg file`; the audit used the already installed Python 3.12.12 executable. The first audit-helper pass incorrectly assumed final66 map keys were basenames, then stopped on KeyError before a fixture executed. The helper was corrected to compare normalized names while preserving the exact original absolute-key map in the report; the recorded 339-check run is the successful final pass. No supplied input was changed.', '',
    '## Admission and claim boundary', '',
    'The candidate describes an implemented scientific coordinator, not an admitted deployment. The frozen R2 regional `main` is not a CAD launch operator. Its two reusable checks are source-visible but have not been executed here. The current fixed R0 remains untouched; reaching the not-before timestamp alone is not admission. A later corrected CAD package still needs review-bound source staging/gate creation, original R0 full-suite completion and resource admission, then actual 3×64 prefix qualification and full U1/U2 three-dataset inference.', '',
    'U0 must remain labelled as the original complete R0 result. A new U0 65-row saved-prefix match would not establish hidden-state bit parity or a new full U0 run. U4 must remain labelled as bypassing C. CAD-inspired interval/threshold engineering is not evidence of a new original module, CAD network reproduction, improved tracking, or effective learned memory. Full-result claims remain unavailable.', '',
    '## Seals and artifacts', '',
    '- Candidate SHA-256 (bare digest): `' + report['candidate_sha256'] + '`.',
    '- `EXPERIMENT_AUDIT.json` contains all 79 actual input hashes and the **exact**, unmodified, bare-digest `source_sha256` map from the final66 receipt in `audited_staged_source_sha256`.',
    '- `input_hashes_before.json` and `unchanged_input_receipt.json` provide the before/after seal.',
    '- `deterministic_checks.json`, `reviewer_policy_fixture_result.json`, `receipt_version_relationships.json`, `interface_reproductions.json`, and `direct_source_diffs.patch` preserve verification evidence.',
    '- Full private request/response and metadata trace: `.aris/traces/experiment-audit/2026-10-16_run01/` beneath this audit output directory.', '',
    '**Required source corrections are B1 and B2.** W1 requires correcting its unsupported claim or a separately reviewed CAD-side implementation change. This original audit does not approve launch and does not change any current source or running experiment.', ''
])
md_text = '\n'.join(md)
(OUT / 'EXPERIMENT_AUDIT.md').write_text(md_text, encoding='utf-8')
save(OUT / 'EXPERIMENT_AUDIT.json', report)

prompt = r'''Apply C:/Users/gb/.codex/skills/experiment-audit/SKILL.md and local-codex-policy/reviewer-independence/review-tracing references. Fresh bounded read-only source/interface audit; read primary artifacts directly. Do not trust the executor narrative. No SSH, remote queries, current R0 logs/process/weights/progress/resource reads, actual NN/child transport/installation, source writes, or new agents. Local stdlib/AST/synthetic fixtures are allowed only inside your new output directory. Hash all inputs before/after; preserve originals.

Read ALL 12 Python files, CANDIDATE.json, SOURCE_SPEC.json, PROTOCOL.md and INPUT_MANIFEST.json in C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_CAD_UNIFIED_SOURCE_20261016/.
Actual CPU primary receipts (same parent directory): M125_CAD_uniform_actual_CPU_checks_20261016.json, M125_CAD_uniform_actual_pipeline_CPU_imports_20261016.json, M125_CAD_final66_actual_CPU_layout_20261016.json. Read directly, classify scope and version relationships.
Official primary reference directory: C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/CADTrack_PRIMARY_20261016/ (source_receipt.json, lib/test/tracker/cadtrack.py, experiments/cadtrack/cadtrack.yaml).
Frozen reusable operator: C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_REGION_DEPLOYMENT_SOURCE_R2_20261015/deploy_m125_region_fixed.py. Its prior report is optional lookup at M125_REGION_DEPLOYMENT_AUDIT_R2_20261015/EXPERIMENT_AUDIT.json, not a replacement for reading used functions.
Original R0 source directory: C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_R0_FIXED_SOURCE_20261015/. Read corresponding original runtime/prepare/run/analyze/collect/suite for direct diff; in frozen_dependency_snapshots/ at minimum trusted_template_tracker.py, full_dense_tracker.py, template_write_features.py, template_write_C.py, run_m123_joint_vot_shards.py, bind_m122_official_initializations.py and relevant base evaluation/runtime source. Other immutable dependencies only when needed for concrete unresolved behavior. Do not audit unrelated files or inspect any running task.

Checklist A-F from experiment-audit plus: one DepthTrack-trained final and one uniform profile per full three-dataset row; no per-dataset weight/seed/layer/profile routing; fixed current R0 not hot-edited; official CAD vs adaptation distinctions; strict threshold boundaries and selected-native-cell versus unrelated native max, reported confidence versus write qualification; unchanged prediction/state/query/features and first-template/dynamic-slot semantics; C actual participation and U4 bypass qualifier; counters and U0 synthetic parity scope; original full U0 reference versus new 64-prefix qualification scope; valid initial annotations only, no subsequent GT in tracking; all12 CLI/sibling dependency resolution and prospective66 source maps; end-to-end reference final/profile/bundle/plan/receipt/collection binding; full U1/U2 DT50/CDTB80/VOT127/1765 anchors, official metrics unchanged; VOT event attribution/count sealing; source/admission/time/resource ordering and no autoretry; no source/CPU test claimed as real GPU or new metrics. Identify concrete blocking bugs/unsupported claims; do not demand speculative fallback/extra try/except or unrelated refactor. Distinguish implemented scientific coordinator from any absent local launch operator and actual deployment admission.

Write original MD/JSON to C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_CAD_UNIFORM_SOURCE_AUDIT_20261016/EXPERIMENT_AUDIT.md and .json, with full private trace beneath .aris/traces/experiment-audit/2026-10-16_run01/. Include exact file:line evidence, source_verdict, separate overall/execution verdict, blocking_findings, check counts, actual input hashes and unchanged-input receipt, candidate_sha256 (bare digest of the supplied CANDIDATE.json), audited_staged_source_sha256 (the exact bare-digest map from final66 primary receipt after verification), requested routing versus actual attestation (do not fabricate), review_independence same-family/acceptance provisional. Do not modify any supplied input or old report. Return concise verdict, blockers and paths.'''
save(TRACE / 'run.meta.json', dict(skill='experiment-audit', run_id='2026-10-16_run01',
    started_at=before['generated_at'], finished_at=now, project_dir=str(BASE), output_dir=str(OUT),
    reviewer_task=report['agent_id'], requested_routing=report['requested_routing'],
    actual_attestation=report['actual_attestation'], review_independence='same-family', acceptance_status='provisional'))
save(TRACE / '001-source-review.request.json', dict(call_number=1, purpose='fresh-original-source-interface-audit',
    timestamp=before['generated_at'], tool='native collaboration NEW_TASK',
    requested_routing=report['requested_routing'], actual_attestation=report['actual_attestation'],
    files_referenced=[row['path'] for row in before['files']], prompt=prompt))
(TRACE / '001-source-review.response.md').write_text(md_text, encoding='utf-8')
save(TRACE / '001-source-review.meta.json', dict(call_number=1, purpose='fresh-original-source-interface-audit',
    timestamp=now, agent_id=report['agent_id'], requested_routing=report['requested_routing'],
    actual_attestation=report['actual_attestation'], review_independence='same-family', acceptance_status='provisional',
    status='complete', source_verdict=report['source_verdict'], overall_verdict=report['overall_verdict'],
    execution_verdict=report['execution_verdict'], report_json_sha256=sha(OUT / 'EXPERIMENT_AUDIT.json'),
    report_md_sha256=sha(OUT / 'EXPERIMENT_AUDIT.md')))
save(TRACE / '002-parent-followup.request.json', dict(call_number=2, purpose='preserve-original-review-boundary',
    tool='collaboration.send_message', prompt='已收到。原12源码、冻结metadata及全部原CPU回执保持不变；请将实际发现和复现分别写入原稿最终报告，包含你指出的重复hash事实与尚未执行的部署边界。你完成原稿FINAL后，我再提供独立R2路径进行后续审查。'))
save(TRACE / 'verification-evidence.json', dict(deterministic_checks=checks, interface_reproductions=reproductions,
    version_relationships=versions, unchanged_input_receipt_sha256=sha(OUT / 'unchanged_input_receipt.json'),
    audit_tool_diagnostics=[
        dict(command='python --version', result='No pyvenv.cfg file; no audit source executed'),
        dict(command='existing Python 3.12.12 audit helper initial pass', result='KeyError on final66 absolute-path map keys; only audit helper corrected'),
        dict(command='existing Python 3.12.12 audit helper final pass', result='339 checks passed; 5416 pure-policy assertions; two blockers reproduced')]))
event_dir = OUT / '.aris/meta'
event_dir.mkdir(parents=True)
save(event_dir / 'event.json', dict(event='review_trace', skill='experiment-audit', purpose='fresh-original-source-interface-audit',
    agent_id=report['agent_id'], trace_path=str(TRACE), status='complete', source_verdict=report['source_verdict']))
(event_dir / 'events.jsonl').write_text(json.dumps(read(event_dir / 'event.json'), ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps(dict(verdict=report['verdict'], source_verdict=report['source_verdict'], execution_verdict=report['execution_verdict'],
    blockers=len(blocking), local_inputs_unchanged=len(after), report=str(OUT / 'EXPERIMENT_AUDIT.md'),
    machine_report=str(OUT / 'EXPERIMENT_AUDIT.json'), trace=str(TRACE)), indent=2))
