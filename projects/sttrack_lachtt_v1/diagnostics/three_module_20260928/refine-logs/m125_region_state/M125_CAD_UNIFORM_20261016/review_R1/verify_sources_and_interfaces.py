"""Read-only audit of supplied sources; stdlib fixtures remain in this folder."""
import argparse
import ast
import contextlib
from datetime import datetime, timezone
import difflib
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
CAD = BASE / 'M125_CAD_UNIFIED_SOURCE_20261016'
R0 = BASE / 'M125_R0_FIXED_SOURCE_20261015'
SNAP = R0 / 'frozen_dependency_snapshots'
OP = BASE / 'M125_REGION_DEPLOYMENT_SOURCE_R2_20261015/deploy_m125_region_fixed.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


checks = []
def check(name, value, **evidence):
    checks.append(dict(name=name, passed=bool(value), **evidence))


baseline = read(OUT / 'input_hashes_before.json')
check('baseline_present', len(baseline['files']) > 0)
for row in baseline['files']:
    check('baseline_' + row['path'], sha(row['path']) == row['sha256'])

candidate = read(CAD / 'CANDIDATE.json')
manifest = read(CAD / 'INPUT_MANIFEST.json')
spec = read(CAD / 'SOURCE_SPEC.json')
early_policy = read(BASE / 'M125_CAD_uniform_actual_CPU_checks_20261016.json')
early_import = read(BASE / 'M125_CAD_uniform_actual_pipeline_CPU_imports_20261016.json')
final_import = read(BASE / 'M125_CAD_final66_actual_CPU_layout_20261016.json')
final_by_name = {Path(name).name: digest for name, digest in final_import['source_sha256'].items()}
for path, digest in manifest.items():
    check('input_manifest_' + Path(path).name, sha(path) == digest)
for name, digest in spec['source_sha256'].items():
    check('source_spec_' + name, sha(CAD / name) == digest)
for name, digest in final_import['locked_inputs_sha256'].items():
    check('final66_locked_input_' + name, sha(CAD / name) == digest)

paths = list(SNAP.glob('*.py')) + list(R0.glob('*.py')) + list(CAD.glob('*.py')) + [OP]
stage = {p.name: p for p in paths}
check('prospective_stage_no_collisions', len(stage) == len(paths) == 66)
check('final66_name_set', set(stage) == set(final_by_name))
candidate_stage = {Path(p).name: d for p, d in candidate['source_sha256'].items()
                   if str(Path(p).parent).replace('\\', '/') == candidate['code_root']}
check('candidate_stage_map_exact', candidate_stage == final_by_name)
for name, path in stage.items():
    check('stage_hash_' + name, sha(path) == final_by_name[name], local=str(path))
    ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    check('AST_' + name, True)

receipts = {}
for label, receipt in [('policy_tensor', early_policy), ('early_import', early_import), ('final66_import', final_import)]:
    matches, different = [], []
    for raw_name, digest in receipt['source_sha256'].items():
        name = Path(raw_name).name
        target = stage[name]
        (matches if sha(target) == digest else different).append(name)
    rows = receipt['rows']
    check(label + '_all_recorded_exit0', all(row['exit_code'] == 0 for row in rows))
    check(label + '_zero_real_tracking', receipt['real_tracking_calls'] == 0)
    if 'entrypoint_main_executions' in receipt:
        check(label + '_zero_mains', receipt['entrypoint_main_executions'] == 0)
    receipts[label] = dict(status=receipt['status'], at=receipt['at'], matches=matches, different=different,
        rows=[{k:r[k] for k in ['source', 'command', 'exit_code', 'stderr']} for r in rows],
        real_tracking_calls=receipt['real_tracking_calls'],
        entrypoint_main_executions=receipt.get('entrypoint_main_executions'),
        source_sha256=receipt['source_sha256'])
    if label == 'final66_import':
        for row in rows:
            origins = json.loads(row['stdout'])
            check('recorded_origins_' + row['source'], origins == row['verified_packaged_module_origins'])
            for module, origin in origins.items():
                check('packaged_origin_' + row['source'] + '_' + module,
                    origin == receipt['root'] + '/code/' + module + '.py' and module + '.py' in stage)

official_root = BASE / 'CADTrack_PRIMARY_20261016'
for row in read(official_root / 'source_receipt.json')['files']:
    path = official_root / row['path']
    check('official_source_' + row['path'], sha(path) == row['sha256'] and path.stat().st_size == row['bytes'])

diffs = []
for old_name, new_name in [
    ('m125_R0_fixed_official_runtime.py', 'm125_CAD_official_runtime.py'),
    ('prepare_m125_R0_fixed_evaluation.py', 'prepare_m125_CAD_evaluation.py'),
    ('run_m125_R0_fixed_official.py', 'run_m125_CAD_official.py'),
    ('analyze_m125_R0_fixed_vot.py', 'analyze_m125_CAD_vot.py'),
    ('collect_m125_R0_fixed_results.py', 'collect_m125_CAD_results.py'),
    ('run_m125_R0_fixed_suite.py', 'run_m125_CAD_uniform_suite.py')]:
    diffs.extend(difflib.unified_diff((R0 / old_name).read_text(encoding='utf-8').splitlines(True),
        (CAD / new_name).read_text(encoding='utf-8').splitlines(True), fromfile=old_name, tofile=new_name))
diffs.extend(difflib.unified_diff((SNAP / 'trusted_template_tracker.py').read_text(encoding='utf-8').splitlines(True),
    (CAD / 'profiled_template_tracker.py').read_text(encoding='utf-8').splitlines(True),
    fromfile='trusted_template_tracker.py', tofile='profiled_template_tracker.py'))
(OUT / 'direct_source_diffs.patch').write_text(''.join(diffs), encoding='utf-8')

# Execute only the supplied pure stdlib policy module and checker; no torch import.
import types
policy_module = types.ModuleType('cad_template_profiles')
policy_module.__file__ = str(CAD / 'cad_template_profiles.py')
sys.modules['cad_template_profiles'] = policy_module
exec(compile((CAD / 'cad_template_profiles.py').read_text(), policy_module.__file__, 'exec'), policy_module.__dict__)
policy_scope = {'__name__': 'audit_policy_checker', '__file__': str(CAD / 'check_cad_template_profiles.py')}
exec(compile((CAD / 'check_cad_template_profiles.py').read_text(), policy_scope['__file__'], 'exec'), policy_scope)
policy_stdout = io.StringIO()
with contextlib.redirect_stdout(policy_stdout):
    policy_scope['main']()
policy_result = json.loads(policy_stdout.getvalue())
check('reviewer_pure_policy_5416', policy_result['checks'] == 5416 and policy_result['NN_calls'] == 0)
save('reviewer_policy_fixture_result.json', policy_result)

# Use the exact original CLI AST to obtain the supported Namespace.
prep_path = CAD / 'prepare_m125_CAD_evaluation.py'
prep_tree = ast.parse(prep_path.read_text(), filename=str(prep_path))
cli_node = next(node for node in prep_tree.body if isinstance(node, ast.If))
capture = {}
cli_scope = {'argparse': argparse, 'Path': Path, 'PRIMARY_PROFILES': ('U0', 'U1', 'U2'),
             'main': lambda args: capture.update(args=vars(args)), '__name__': '__main__'}
argv_before = sys.argv
sys.argv = [str(prep_path), '--reference-evaluation', str(OUT / 'fixtures/reference'),
    '--source-gate', str(OUT / 'fixtures/gate.json'), '--profile', 'U1', '--output', str(OUT / 'fixtures/prepared')]
exec(compile(ast.Module(body=[cli_node], type_ignores=[]), str(prep_path), 'exec'), cli_scope)
sys.argv = argv_before
args = argparse.Namespace(**capture['args'])
training_attr = next(n for n in ast.walk(prep_tree) if isinstance(n, ast.Attribute) and
                     isinstance(n.value, ast.Name) and n.value.id == 'args' and n.attr == 'training')
try:
    eval(compile(ast.Expression(body=training_attr), str(prep_path), 'eval'), {'args': args})
    missing_training = {'reproduced': False}
except AttributeError as error:
    missing_training = dict(reproduced=True, error_type=type(error).__name__, error=str(error),
        source=str(prep_path), line=training_attr.lineno, actual_cli_attributes=sorted(vars(args)),
        scope='Exact supplied CLI AST and exact unresolved attribute AST; no preparation main, transport, or neural call.')
check('prepare_missing_training_reproduced', missing_training['reproduced'])

# Run the complete unchanged collector main with in-memory synthetic receipts.
# sha/read are intentionally synthetic and all represented paths stay in OUT.
collector_path = CAD / 'collect_m125_CAD_results.py'
collector_tree = ast.parse(collector_path.read_text(), filename=str(collector_path))
collector_main = next(n for n in collector_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
fixture_root = OUT / 'fixtures/collector'
digest = 'a' * 64
profile = policy_module.profile_dict('U1')
name = 'CAD_U1_same_R0_fixed6'
model = dict(name=name, final_sha256=digest, bundle_sha256=digest, plans={})
selection = dict(status='M125_CAD_uniform_profile_before_metrics', models=[model],
    inference_profile_id='U1', inference_profile=profile, one_profile_for_all_datasets=True,
    per_dataset_profile_selection=False, bank_sha256=digest, binding_sha256=digest)
objects = {
    str(fixture_root / 'selection.json'): selection,
    str(fixture_root / 'bundle.json'): dict(inference_profile_id='U1', inference_profile=profile,
        final_path=str(fixture_root / 'final.pt'), source_sha256={}),
}
for dataset, count, frames in [('depthtrack_test', 50, 76373), ('cdtb', 80, 101956)]:
    plan_path = fixture_root / dataset / 'plan.json'
    model['plans'][dataset] = dict(path=str(plan_path), sha256=digest)
    folder = fixture_root / dataset / 'predictions'
    rows = [dict(sequence='synthetic_%03d' % i, frames=2, inference_profile_id='U1',
        template_counts=dict(tracking_calls=1), bbox_sha256=digest, confidence_sha256=digest) for i in range(count)]
    objects[str(folder / 'receipt.json')] = dict(inference_profile_id='U1', inference_profile=profile,
        one_profile_for_all_datasets=True, bank_sha256=digest, sequences=rows, frames=frames,
        frozen_before_after_exact=True, optimizer_steps=0, subsequent_GT_opened=False)
    objects[str(folder / 'metrics.json')] = dict(status='complete', final_sha256=digest,
        bundle_sha256=digest, inference_profile_id='U1', inference_profile=profile,
        plan_sha256=digest, receipt_sha256=digest, metrics=dict(sequences=count, frames=frames,
        precision_percent=50., recall_percent=50., f_score_percent=50.))
event_map = {('%064x.json' % i): digest for i in range(1765)}
objects[str(fixture_root / 'vot/result.json')] = dict(status='complete_full127', final_sha256=digest,
    bundle_sha256=digest, bank_sha256=digest, binding_sha256=digest, anchors=1765, sequences=127,
    external_optimizer_steps=0, inference_profile_id='U1', inference_profile=profile,
    profile_event_files_sha256=event_map, merge_sha256=digest, analysis_sha256=digest,
    metrics_percent=dict(EAO=50., ACC=50., ROB=50.), confirmed_failures=0)
objects[str(fixture_root / 'vot/run/merge_result.json')] = dict(result_sha256={})
requested_analysis = []
expected_analysis = fixture_root / 'vot/run/master/analysis' / (name + '_full127.json')
def fixture_sha(path):
    path = Path(path)
    if path.parent.name == 'analysis':
        requested_analysis.append(str(path))
        if path != expected_analysis:
            raise FileNotFoundError(str(path))
    return digest
collector_scope = dict(Path=Path, read=lambda path: objects[str(path)], sha=fixture_sha,
    write=lambda *args: None, math=math)
exec(compile(ast.Module(body=[collector_main], type_ignores=[]), str(collector_path), 'exec'), collector_scope)
try:
    collector_scope['main'](fixture_root)
    shadowing = dict(reproduced=False)
except FileNotFoundError as error:
    shadowing = dict(reproduced=True, error_type=type(error).__name__, error=str(error),
        traceback=traceback.format_exc(), expected_analysis_path=str(expected_analysis),
        requested_analysis_paths=requested_analysis, event_file_count=len(event_map),
        scope='Complete unchanged collector main AST over in-memory synthetic receipts; no real results or metrics.')
check('collector_event_name_shadowing_reproduced', shadowing['reproduced'])

# Check exact normalized AST identity for metric calls/thresholds and prediction commits.
def function_tree(path, function):
    tree = ast.parse(path.read_text(), filename=str(path))
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function)
def dump(node):
    return ast.dump(node, include_attributes=False)
old_run = function_tree(R0 / 'run_m125_R0_fixed_official.py', 'ope_analyze')
new_run = function_tree(CAD / 'run_m125_CAD_official.py', 'ope_analyze')
old_metric = next(n for n in ast.walk(old_run) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'evaluate_depthtrack_results')
new_metric = next(n for n in ast.walk(new_run) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'evaluate_depthtrack_results')
check('OPE_metric_call_AST_unchanged', dump(old_metric) == dump(new_metric))
old_collect = function_tree(R0 / 'collect_m125_R0_fixed_results.py', 'main')
old_passed = next(n for n in ast.walk(old_collect) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'passed' for t in n.targets))
new_passed = next(n for n in ast.walk(collector_main) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'passed' for t in n.targets))
check('nine_metric_gate_AST_unchanged', dump(old_passed) == dump(new_passed))
old_analyze = function_tree(R0 / 'analyze_m125_R0_fixed_vot.py', 'main')
new_analyze = function_tree(CAD / 'analyze_m125_CAD_vot.py', 'main')
for variable in ['metrics', 'analysis_path']:
    old_node = next(n for n in ast.walk(old_analyze) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == variable for t in n.targets))
    new_node = next(n for n in ast.walk(new_analyze) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == variable for t in n.targets))
    check('VOT_' + variable + '_AST_unchanged', dump(old_node) == dump(new_node))
old_step = function_tree(SNAP / 'trusted_template_tracker.py', 'step')
new_step = function_tree(CAD / 'profiled_template_tracker.py', 'step')
old_initial_statements = old_step.body[:20]
new_initial_statements = new_step.body[:20]
check('prediction_state_query_feature_first20_statements_AST_unchanged',
      [dump(n) for n in old_initial_statements] == [dump(n) for n in new_initial_statements])

save('interface_reproductions.json', dict(prepare_missing_training=missing_training, collector_model_name_shadowing=shadowing))
save('receipt_version_relationships.json', receipts)
save('verified_staged_source_sha256.json', final_import['source_sha256'])
save('deterministic_checks.json', dict(generated_at=datetime.now(timezone.utc).isoformat(),
    checks=checks, total=len(checks), passed=sum(c['passed'] for c in checks),
    failed=[c for c in checks if not c['passed']], candidate_sha256=sha(CAD / 'CANDIDATE.json'),
    stage_files=len(stage), primary_CAD_files=len(list(CAD.glob('*.py'))),
    entrypoint_main_executions=0, torch_imports=0, actual_neural_calls=0, remote_queries=0,
    local_fixture_collector_main_executions=1, reviewer_policy_assertions=policy_result['checks']))
print(json.dumps(dict(checks=len(checks), failed=[c['name'] for c in checks if not c['passed']],
    candidate_sha256=sha(CAD / 'CANDIDATE.json'), version_differences={k:v['different'] for k,v in receipts.items()},
    prepare_blocker=missing_training, collector_blocker=shadowing, policy_checks=policy_result['checks']), indent=2))
