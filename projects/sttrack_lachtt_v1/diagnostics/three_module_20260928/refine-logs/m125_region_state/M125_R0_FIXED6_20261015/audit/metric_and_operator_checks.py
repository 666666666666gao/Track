"""Hash-bound inherited metric snapshots and local gate fixtures; no source imports or children."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
SRC = BASE / 'M125_R0_FIXED_SOURCE_20261015'
SNAP = OUT / 'inherited_metric_sources'
SNAP.mkdir(exist_ok=False)
closure_path = BASE / 'M124_E6_terminal_audit_artifacts_20261015.json'
closure = json.loads(closure_path.read_text())
candidate = json.loads((SRC / 'CANDIDATE.json').read_text())
payload = json.loads((BASE / 'M125_R0_staging_payload_20261015.json').read_text())
receipt_path = BASE / 'M125_R0_candidate_staging_20261015_stdout.json'
receipt = json.loads(receipt_path.read_text())
checks = []


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check(name, actual, expected=True, details=None):
    checks.append(dict(name=name, status='PASS' if actual == expected else 'FAIL', actual=actual,
                       expected=expected, details=details))


selected = {}
for remote, row in closure['files'].items():
    if (remote.endswith(('/depthtrack_pr.py', '/finalize_vot_full127.py', '/finalize_vot_transaction_low22.py',
                         '/vot/analysis/multistart.py', '/vot/analysis/longterm.py', '/vot/analysis/supervised.py'))
        or ('/epochs/epoch-06/evaluation/' in remote and remote.endswith(('plan.json', 'bundle.json', 'shard_manifest.json', 'execution.json')))):
        check('inherited_text_hash:' + remote, digest(row['text'].encode()), row['sha256'])
        check('inherited_text_bytes:' + remote, len(row['text'].encode()), row['bytes'])
        if remote in candidate['source_sha256']:
            check('inherited_candidate_binding:' + remote, row['sha256'], candidate['source_sha256'][remote])
        filename = PurePosixPath(remote).name if remote.endswith('.py') else str(len(selected)) + '_' + PurePosixPath(remote).name
        path = SNAP / filename
        path.write_bytes(row['text'].encode())
        selected[remote] = dict(local_snapshot=str(path), sha256=row['sha256'], bytes=row['bytes'])
        if remote.endswith('.py'):
            ast.parse(row['text'])
            check('inherited_AST:' + remote, True)

manifest_key = next(k for k in selected if k.endswith('shard_manifest.json'))
manifest = json.loads(closure['files'][manifest_key]['text'])
check('inherited_vot_unique_127_sequences', len(set(manifest['sequences'])), 127)
check('inherited_vot_full_1765_anchors', manifest['total_anchor_count'], 1765)
check('inherited_vot_full_four_shards', [s['index'] for s in manifest['shards']], [0,1,2,3])
all_trajectories = [t for s in manifest['shards'] for t in s['expected_trajectories']]
check('inherited_vot_exact_unique_1765_trajectories', len(set(all_trajectories)), 1765)
check('inherited_vot_no_duplicate_trajectories', len(all_trajectories), len(set(all_trajectories)))
check('inherited_vot_shard_anchor_sum', sum(s['anchor_count'] for s in manifest['shards']), 1765)
check('inherited_vot_trajectory_sequences', sorted(set(t.rsplit('_',1)[0] for t in all_trajectories)), sorted(manifest['sequences']))
# The inherited E6 descriptor identifies the same M122 base used by new R0 preparation.
for dataset in ['depthtrack_test','cdtb','vot']:
    key = next(k for k in selected if k.endswith('/' + dataset + '/plan.json'))
    plan = json.loads(closure['files'][key]['text'])
    check('inherited_plan_base_path:' + dataset, plan['base_plan_path'], candidate['base_evaluation'] + '/' + dataset + '/plan.json')

staged = {candidate['code_root'] + '/' + PurePosixPath(r['remote']).name: r['sha256'] for r in payload['frozen_dependencies']}
staged.update({candidate['code_root'] + '/' + name: r['sha256'] for name,r in payload['sources'].items()})
check('actual_staging_receipt_53_sources', len(receipt['source_sha256']), 53)
check('actual_staging_source_map_equals_payload', receipt['source_sha256'], staged)
check('actual_staging_nine_CPU_imports', len(receipt['CPU_imports']), 9)
check('actual_staging_CPU_names_match_nine_sources', sorted(r['source'] for r in receipt['CPU_imports']), sorted(payload['sources']))
check('actual_staging_all_CPU_import_exits_zero', [r['exit_code'] for r in receipt['CPU_imports']], [0] * 9)
check('actual_staging_not_training', receipt['training_launched'], False)
check('actual_staging_no_NN_calls', receipt['NN_calls'], 0)
stage_gate = copy.deepcopy(candidate)
stage_gate.pop('new_source_local_paths')
stage_gate['source_sha256'].update(staged)
check('reconstructed_candidate_gate_exact_sha', digest((json.dumps(stage_gate,indent=2)+'\n').encode()), receipt['candidate_gate_sha256'])

# Exercise the actual local launcher admission assertions with both accepted and rejected fixtures.
launcher_path = BASE / 'launch_M125_R0_real64_20261015.py'
launcher_tree = ast.parse(launcher_path.read_text())
gate_asserts = [n for n in launcher_tree.body if isinstance(n, ast.Assert) and 'audit' in ast.unparse(n.test)]
check('real64_launcher_has_three_local_review_binding_asserts', len(gate_asserts), 3)
audit = dict(source_verdict='PASS_SOURCE_ONLY', blocking_findings=[], candidate_sha256=digest((SRC/'CANDIDATE.json').read_bytes()),
             audited_staged_source_sha256=staged)
env = dict(audit=audit, candidate_path=SRC/'CANDIDATE.json', hashlib=hashlib, staging=receipt)
gate_code = compile(ast.Module(body=gate_asserts, type_ignores=[]), str(launcher_path), 'exec')
exec(gate_code, env)
check('real64_launcher_accepts_matching_source_only_review', True)
for field,value in [('source_verdict','FAIL'), ('blocking_findings',[{'id':'block'}]),
                    ('candidate_sha256','wrong'), ('audited_staged_source_sha256',{})]:
    invalid = dict(audit, **{field:value})
    rejected = False
    try:
        exec(gate_code, dict(env,audit=invalid))
    except AssertionError:
        rejected = True
    check('real64_launcher_rejects:' + field, rejected)

operator_paths = [BASE / 'stage_M125_R0_candidate_remote_20261015.py', launcher_path,
                  BASE / 'launch_M125_R0_fixed_full_suite_20261015.py']
operator_records = {}
for path in operator_paths:
    text = path.read_text()
    tree = ast.parse(text)
    remote_assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='remote' for t in n.targets))
    value = remote_assignment.value
    remote = value.value if isinstance(value, ast.Constant) else value.func.value.value
    remote_tree = ast.parse(remote)
    check('embedded_operator_AST:' + path.name, True)
    outpath = OUT / (path.stem + '.remote.txt')
    outpath.write_text(remote, encoding='utf-8')
    operator_records[str(path)] = dict(sha256=digest(path.read_bytes()), remote_snapshot=str(outpath))
    if path == launcher_path:
        assignment = next(n for n in remote_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='supervisor' for t in n.targets))
        generated = eval(compile(ast.Expression(assignment.value), str(path), 'eval'),
                         dict(root=Path('/SYNTHETIC_NO_NN'), command=['NEVER_RUN'], gate=dict(code_root='/SYNTHETIC_CODE')))
        ast.parse(generated)
        check('generated_real64_supervisor_AST', True)
        check('generated_real64_supervisor_returns_child_exit', 'sys.exit(completed.returncode)' in generated)
        (OUT/'generated_real64_supervisor.fixture.txt').write_text(generated)

# Confirm failure counting uses official GT/grace semantics with small pure-Python fixtures.
failure_path = SNAP/'finalize_vot_transaction_low22.py'
tree = ast.parse(failure_path.read_text())
node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='failure_progress')
scope = {}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(failure_path),'exec'),scope)

class Proxy:
    def __init__(self, values): self.values=values
    def __len__(self): return len(self.values)
    def groundtruth(self,index):
        value=self.values[index]
        return type('GT',(),{'is_empty':lambda self: value})()

fn=scope['failure_progress']
check('failure_grace_confirmed_after_ten_low_overlaps', fn([1.]+[0.]*10, Proxy([False]*11),10,.1),1)
check('failure_grace_not_confirmed_after_nine', fn([1.]+[0.]*9, Proxy([False]*10),10,.1),10)
check('failure_absent_GT_does_not_count_as_failure', fn([0.]*12,Proxy([True]*12),10,.1),12)

result=dict(scope='Source/hash/AST and explicit synthetic predicate/failure fixtures only; no original experiment/result audit rerun, no NN/SSH/children',
            inherited_sources=selected, operator_records=operator_records,
            audited_staged_source_sha256=staged, candidate_gate_sha256=receipt['candidate_gate_sha256'],
            checks=checks, count=len(checks), failed=sum(r['status']!='PASS' for r in checks))
(OUT/'metric_operator_checks.json').write_text(json.dumps(result,indent=2)+'\n')
(OUT/'additional_input_hashes.json').write_text(json.dumps({str(p):dict(sha256=digest(p.read_bytes()),bytes=p.stat().st_size)
    for p in [closure_path,receipt_path]+operator_paths},indent=2)+'\n')
print(json.dumps(dict(checks=len(checks),failures=[r['name'] for r in checks if r['status']!='PASS'],snapshots=len(selected))))
