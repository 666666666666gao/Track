"""Read-only R3 review: actual local bytes, AST, and isolated stdlib transport fixtures.

Never imports/executes the reviewed launch main, remote main, scientific runtime,
predecessor/resource checks, subprocess, SSH, GPU, or neural-network code.
All writes are confined to this new audit output directory.
"""
import ast
import base64
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path, PurePosixPath
import zlib

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
CAD = BASE / 'M125_CAD_UNIFIED_SOURCE_R2_20261016'
R0 = BASE / 'M125_R0_FIXED_SOURCE_20261015'
SNAP = R0 / 'frozen_dependency_snapshots'
OP = BASE / 'M125_REGION_DEPLOYMENT_SOURCE_R2_20261015/deploy_m125_region_fixed.py'
NEW = BASE / 'M125_CAD_DEPLOYMENT_SOURCE_R2_20261016/invoke_M125_CAD_reviewed_suite.py'
OLD = BASE / 'M125_CAD_DEPLOYMENT_SOURCE_20261016/invoke_M125_CAD_reviewed_suite.py'
R1_AUDIT = BASE / 'M125_CAD_UNIFORM_SOURCE_AUDIT_20261016'
R2_AUDIT = BASE / 'M125_CAD_UNIFORM_SOURCE_AUDIT_R2_20261016'
LAYOUT = BASE / 'M125_CAD_R2_final66_actual_CPU_layout_20261016.json'
REVISION = BASE / 'M125_CAD_admission_raw_byte_revision_20261016.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def save(name, obj):
    (OUT / name).write_bytes((json.dumps(obj, indent=2, ensure_ascii=False,
                                        allow_nan=False) + '\n').encode('utf-8'))


checks = []


def check(name, passed, **evidence):
    checks.append(dict(name=name, passed=bool(passed), **evidence))
    assert passed, name


def tree(path):
    return ast.parse(Path(path).read_bytes(), filename=str(path))


def main_node(path):
    return next(n for n in tree(path).body if isinstance(n, ast.FunctionDef) and n.name == 'main')


def assignment(nodes, name):
    return next(n for n in nodes if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name)


def expression(node, namespace):
    return eval(compile(ast.Expression(body=node), '<reviewed-pure-expression>', 'eval'), namespace)


def single_statement(node, namespace):
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<reviewed-isolated-statement>', 'exec'), namespace)


before = read(OUT / 'input_hashes_before.json')
inputs = before['files']
check('250_unique_presealed_inputs', len(inputs) == len({r['path'] for r in inputs}) == 250)
for row in inputs:
    check('input_seal:' + row['path'], sha(row['path']) == row['sha256']
          and Path(row['path']).stat().st_size == row['bytes'])
sealed = {r['path']: r['sha256'] for r in inputs}
r2_report = read(R2_AUDIT / 'EXPERIMENT_AUDIT.json')
for path, digest in r2_report['audited_input_hashes'].items():
    check('prior_R2_input_unchanged:' + path, 'sha256:' + sealed[path] == digest)
check('R2_report_still_original_failure', r2_report['source_verdict'] == 'FAIL_SOURCE_ONLY'
      and r2_report['scientific_source_verdict'] == 'PASS_SOURCE_ONLY'
      and [b['id'] for b in r2_report['blocking_findings']] == ['R2-B1'])
prior_checks = read(R2_AUDIT / 'deterministic_checks.json')
check('prior_419_check_receipt_intact', prior_checks['total'] == prior_checks['passed'] == 419
      and prior_checks['failed'] == [])

layout = read(LAYOUT)
candidate = read(CAD / 'CANDIDATE.json')
spec = read(CAD / 'SOURCE_SPEC.json')
manifest = read(CAD / 'INPUT_MANIFEST.json')
metadata_names = ['CANDIDATE.json', 'SOURCE_SPEC.json', 'PROTOCOL.md', 'INPUT_MANIFEST.json']
scientific_paths = sorted(CAD.glob('*.py')) + [CAD / n for n in metadata_names]
check('scientific_inputs_exact16', len(scientific_paths) == 16
      and {p.name for p in scientific_paths} == set(layout['locked_inputs_sha256']))
for p in scientific_paths:
    raw = p.read_bytes()
    check('science16_exact:' + p.name, sha(p) == layout['locked_inputs_sha256'][p.name]
          and 'sha256:' + sha(p) == r2_report['audited_input_hashes'][str(p)])
    if p.suffix == '.py':
        ast.parse(raw, filename=str(p))
check('CPU_receipt_unchanged_since_R2', 'sha256:' + sha(LAYOUT)
      == r2_report['audited_input_hashes'][str(LAYOUT)])
for path, digest in manifest.items():
    check('manifest_exact:' + path, sha(path) == digest and sealed[path] == digest)
for name, digest in spec['source_sha256'].items():
    check('SOURCE_SPEC_exact:' + name, sha(CAD / name) == digest)

sources = list(CAD.glob('*.py')) + list(R0.glob('*.py')) + list(SNAP.glob('*.py')) + [OP]
local_sources = {p.name: p for p in sources}
receipt_by_name = {PurePosixPath(p).name: d for p, d in layout['source_sha256'].items()}
future = {PurePosixPath(p).name: d for p, d in candidate['source_sha256'].items()
          if PurePosixPath(p).parent == PurePosixPath(candidate['code_root'])}
check('66_unique_local_sources', len(sources) == len(local_sources) == 66)
check('66_exact_receipt_names', set(local_sources) == set(receipt_by_name)
      and len(layout['source_sha256']) == 66)
check('candidate_future_exact66', future == receipt_by_name)
check('R2_review_exact_original_remote_key_map', r2_report['audited_staged_source_sha256']
      == layout['source_sha256'])
source_rows = []
for name, path in sorted(local_sources.items()):
    check('staged_source_hash:' + name, sha(path) == receipt_by_name[name])
    ast.parse(path.read_bytes(), filename=str(path))
    source_rows.append(dict(name=name, local_path=str(path), sha256=sha(path)))
check('12_imports_zero_main_zero_tracking', layout['code_files'] == 66
      and layout['CAD_entrypoints'] == len(layout['rows']) == 12
      and all(r['exit_code'] == 0 for r in layout['rows'])
      and layout['entrypoint_main_executions'] == layout['real_tracking_calls'] == 0)
for row in layout['rows']:
    check('receipt_origin_row:' + row['source'], json.loads(row['stdout'])
          == row['verified_packaged_module_origins']
          and all(p == layout['root'] + '/code/' + name + '.py'
                  for name, p in row['verified_packaged_module_origins'].items()))
save('verified_staged_source_sha256.json', layout['source_sha256'])
save('local_66_source_verification.json', dict(exact_remote_key_map_preserved=True, files=source_rows))

old_raw = OLD.read_bytes()
new_raw = NEW.read_bytes()
old_lines = old_raw.decode('utf-8').splitlines(keepends=True)
new_lines = new_raw.decode('utf-8').splitlines(keepends=True)
changed_lines = [i + 1 for i, (old, new) in enumerate(zip(old_lines, new_lines)) if old != new]
check('only_three_changed_source_lines', len(old_lines) == len(new_lines) == 125
      and changed_lines == [21, 43, 44])
check('R3_audit_path_change', "audit = A / 'M125_CAD_UNIFORM_SOURCE_AUDIT_R3_20261016'" in new_lines[20])
check('exact_two_raw_byte_read_changes', all(
    new_lines[i] == old_lines[i].replace("read_text(encoding='utf-8')", "read_bytes().decode('utf-8')")
    for i in [42, 43]))
(OUT / 'direct_operator_diff.patch').write_bytes(''.join(difflib.unified_diff(
    old_lines, new_lines, fromfile=str(OLD), tofile=str(NEW))).encode('utf-8'))
revision = read(REVISION)
check('revision_hashes_match_actual', revision['original_sha256'] == sha(OLD)
      and revision['revised_sha256'] == sha(NEW))
check('revision_three_expressions_disclosed', revision['changed_expressions'] == len(changed_lines) == 3)
check('revision_actual_original_CRLF_count', revision['actual_prior_audit_JSON_CRLF_lines']
      == (R1_AUDIT / 'EXPERIMENT_AUDIT.json').read_bytes().count(b'\r\n') == 747)
new_main = main_node(NEW)
old_main = main_node(OLD)
new_remote_node = assignment(new_main.body, 'remote')
old_remote_node = assignment(old_main.body, 'remote')
new_template = new_remote_node.value.func.value.value
old_template = old_remote_node.value.func.value.value
check('embedded_remote_exactly_unchanged', new_template == old_template)
check('transport_encoding_expression_unchanged', ast.dump(new_remote_node.value)
      == ast.dump(old_remote_node.value))
remote_tree = ast.parse(new_template)
(OUT / 'embedded_remote_source.txt').write_bytes(new_template.encode('utf-8'))
request_new = assignment(new_main.body, 'request')
request_old = assignment(old_main.body, 'request')
remote_hash_assert = next(n for n in remote_tree.body if isinstance(n, ast.Assert)
                          and "source_review_sha256" in ast.unparse(n))
transport_rows = []
for label, directory in [('R1', R1_AUDIT), ('R2', R2_AUDIT)]:
    env = dict(audit=directory, layout=layout, source=CAD, __file__=str(NEW), sha=sha,
               base64=base64, zlib=zlib, json=json, hashlib=hashlib)
    env['request'] = expression(request_new.value, env)
    program = expression(new_remote_node.value, env)
    program_ast = ast.parse(program)
    p_assign = assignment(program_ast.body, 'p')
    single_statement(p_assign, env)
    single_statement(remote_hash_assert, env)
    check('exact_remote_review_byte_assertion:' + label, True)
    new_payload = env['p']
    old_env = dict(env, __file__=str(OLD))
    old_env['request'] = expression(request_old.value, old_env)
    old_program = expression(old_remote_node.value, old_env)
    single_statement(assignment(ast.parse(old_program).body, 'p'), old_env)
    old_failed = False
    try:
        single_statement(remote_hash_assert, old_env)
    except AssertionError:
        old_failed = True
    check('old_assertion_reproduces_CRLF_failure:' + label, old_failed)
    for field, filename in [('source_review_json', 'EXPERIMENT_AUDIT.json'),
                            ('source_review_md', 'EXPERIMENT_AUDIT.md')]:
        raw = (directory / filename).read_bytes()
        transported = new_payload[field].encode()
        old_transport = old_env['p'][field].encode()
        check('exact_original_report_bytes:' + label + ':' + filename,
              transported == raw and len(raw) > 0 and raw.count(b'\r\n') > 0)
        transport_rows.append(dict(audit=label, file=str(directory / filename),
            raw_bytes=len(raw), transported_bytes=len(transported), CRLF_count=raw.count(b'\r\n'),
            byte_exact=(transported == raw), raw_sha256=hashlib.sha256(raw).hexdigest(),
            transported_sha256=hashlib.sha256(transported).hexdigest(),
            old_transported_bytes=len(old_transport), old_byte_exact=(old_transport == raw),
            old_transported_sha256=hashlib.sha256(old_transport).hexdigest()))
save('actual_CRLF_transport_verification.json', dict(
    classification='local_stdlib_transport_fixture', actual_remote_calls=0,
    exact_read_expression_source=[str(NEW) + ':43', str(NEW) + ':44'],
    exact_encode_expression_source=str(NEW) + ':105',
    exact_remote_hash_assertion_source=str(NEW) + ':62',
    new_assertions_passed=2, old_assertions_failed_as_expected=2,
    files=transport_rows, historical_FAIL_reports_not_admitted=True))

# Recheck the exact date assertions without executing either main or clock-dependent side effects.
class Clock:
    value = None
    @classmethod
    def now(cls, tz):
        return cls.value
    fromisoformat = staticmethod(datetime.fromisoformat)

local_time = new_main.body[0]
remote_effects = [n for n in remote_tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
remote_time = remote_effects[0]
check('first_local_statement_is_date_gate', isinstance(local_time, ast.Assert))
check('first_remote_nonimport_statement_is_date_gate', isinstance(remote_time, ast.Assert))
time_results = []
for side, node in [('local', local_time), ('remote', remote_time)]:
    for expected, timestamp in [(False, '2026-10-17T12:44:59+08:00'),
                                (True, '2026-10-17T12:45:00+08:00')]:
        Clock.value = datetime.fromisoformat(timestamp)
        accepted = True
        try:
            single_statement(node, dict(datetime=Clock, timezone=timezone))
        except AssertionError:
            accepted = False
        check('date_boundary:' + side + ':' + timestamp, accepted == expected)
        time_results.append(dict(side=side, time=timestamp, accepted=accepted, simulation_only=True))

new_text = new_raw.decode('utf-8')
check('exclusive_attempt_before_only_transport_call',
      "open('x', encoding='utf-8', newline='\\n')" in new_text
      and new_text.index("with (A / (stem + '_attempt.json'))") < new_text.index('completed = subprocess.run(')
      and len([n for n in ast.walk(new_main) if isinstance(n, ast.Call)
               and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
               and n.func.value.id == 'subprocess' and n.func.attr == 'run']) == 1)
check('date_source_inputs_predecessor_resource_root_launch_order',
      new_template.index('assert datetime.now') < new_template.index("for path,digest in p['code_sha256']")
      < new_template.index("for path,digest in candidate['input_sha256']")
      < new_template.index('predecessor=verify_predecessor(candidate)')
      < new_template.index('gpu_rows=resource_admission(candidate)')
      < new_template.index('root.mkdir();code.mkdir()') < new_template.index('child=subprocess.Popen('))
check('fixed_root_and_exclusive_create', "root==Path('/root/autodl-tmp/m125_CAD_same_R0_uniform_profiles_20261016')" in new_template
      and 'root.mkdir();code.mkdir()' in new_template and "(code/name).open('xb')" in new_template)
check('no_invoker_or_remote_retry_loop', not any(isinstance(n, ast.While) for n in ast.walk(new_main))
      and not any(isinstance(n, ast.While) for n in ast.walk(remote_tree)))
check('supervisor_command_and_assertions_enabled', "str(code/'m125_CAD_process_supervisor.py')" in new_template
      and "PYTHONOPTIMIZE='0'" in new_template and "gate['python'],'-B','-u'" in new_template)
check('launch_receipt_disclaims_complete', 'real64_complete_at_launch=False,formal_metrics_available_at_launch=False' in new_template
      and 'automatic_retries=0' in new_template)
op_text = OP.read_text(encoding='utf-8')
check('frozen_R0_full6_full3_source_intact', 'sha256:' + sha(OP) == r2_report['audited_input_hashes'][str(OP)]
      and all(s in op_text for s in ["range(1, 7)", "len(result['stages']) == 15",
                  "len(combined['results']) == 3", "{'depthtrack_test', 'cdtb', 'vot'}"]))
check('frozen_resource_memory_fields_intact', '--query-gpu=index,memory.used,memory.total' in op_text
      and '[row[0] for row in rows] == [0, 1]' in op_text)
observer = BASE / 'M124_fiftyfirst_hourly_compressed_native_observer_20261014.py'
args = ast.literal_eval(assignment(tree(observer).body, 'args').value)
check('static_transport_single_attempt_unchanged', args[0].endswith('/ssh.exe')
      and 'ConnectionAttempts=1' in args and 'sha256:' + sha(observer)
      == r2_report['audited_input_hashes'][str(observer)])
save('date_and_static_admission_verification.json', dict(
    boundary_cases=time_results, embedded_remote_unchanged=True,
    actual_predecessor_calls=0, actual_resource_calls=0, actual_transport_calls=0,
    supervisor_runtime_cases_reused_from_R2=2, new_supervisor_runtime_calls=0,
    prior_R2_deterministic_check_receipt_count=419,
    prior_complete_prepare_mains_reused=3, prior_complete_collector_mains_reused=2))

after = [dict(path=r['path'], bytes=Path(r['path']).stat().st_size, sha256=sha(r['path'])) for r in inputs]
check('all_250_inputs_unchanged_after_checks', after == inputs)
save('unchanged_input_receipt.json', dict(status='UNCHANGED', generated_at=datetime.now(timezone.utc).isoformat(),
    files_checked=len(inputs), changed_files=[], files=after, original_R1_R2_reports_and_evidence_preserved=True,
    scientific_R2_16_inputs_and_CPU_receipt_unchanged=True,
    scope='250 sealed local files; no remote or current running-task inspection'))
save('deterministic_checks.json', dict(total=len(checks), passed=sum(r['passed'] for r in checks), failed=[], checks=checks,
    check_type='read_only_hash_AST_and_stdlib_transport_fixture', inputs_unchanged=len(inputs),
    revised_operator_sha256=sha(NEW), old_operator_sha256=sha(OLD), candidate_sha256=sha(CAD / 'CANDIDATE.json'),
    changed_lines=changed_lines, prior_R2_deterministic_check_receipt_count=419,
    current_real_NN_calls=0, current_GPU_queries=0, current_SSH_calls=0, current_actual_launches=0))
print(json.dumps(dict(passed=len(checks), inputs_unchanged=len(inputs), changed_lines=changed_lines,
    existing_CRLF_reports_preserved=len(transport_rows), candidate_sha256=sha(CAD / 'CANDIDATE.json'),
    revised_operator_sha256=sha(NEW), actual_remote_or_neural_calls=0), indent=2))
