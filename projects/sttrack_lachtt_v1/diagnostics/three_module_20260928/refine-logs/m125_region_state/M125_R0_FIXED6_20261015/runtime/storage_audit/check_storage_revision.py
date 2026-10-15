"""Read-only, standard-library checks of the explicit storage-only amendment.

Does not import experiment modules, invoke operators, use SSH, or execute NN.
Writes only inside this audit directory.
"""
from pathlib import Path
from datetime import datetime, timezone
import ast
import copy
import difflib
import hashlib
import json

OUT = Path(__file__).resolve().parent
A = OUT.parent
SRC = A / 'M125_R0_FIXED_SOURCE_20261015'
OLD = A / 'M125_R0_FIXED_SOURCE_AUDIT_20261015'
checks = []

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
    return json.loads(Path(path).read_bytes())

def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

def check(name, passed, evidence):
    checks.append(dict(name=name, passed=bool(passed), evidence=evidence))

def key(path):
    return str(Path(path).resolve()).casefold()

def remote_assignment(source):
    return next(n.value for n in ast.parse(source).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'remote' for t in n.targets))

old = load(SRC / 'CANDIDATE.json')
new = load(SRC / 'CANDIDATE_STORAGE_R2.json')
prior = load(OLD / 'EXPERIMENT_AUDIT.json')
frozen = load(SRC / 'FROZEN_DEPENDENCIES.json')
oldstage = load(A / 'M125_R0_candidate_staging_20261015_stdout.json')
newstage = load(A / 'M125_R0_storage_R2_staging_20261015_stdout.json')
diagnosis = load(A / 'M125_R0_storage_admission_diagnosis_retry1_20261015.json')
budget = load(A / 'M125_R0_storage_R2_budget_20261015.json')
original_launch_path = A / 'launch_M125_R0_real64_20261015.py'
launch_path = A / 'launch_M125_R0_real64_storage_R2_20261015.py'
formal_path = A / 'launch_M125_R0_fixed_full_suite_20261015.py'
stage_path = A / 'stage_M125_R0_storage_R2_20261015.py'
observer_path = A / 'observe_M125_R0_real64_storage_R2_once_20261015.py'
primary = [OLD / 'EXPERIMENT_AUDIT.json', OLD / 'EXPERIMENT_AUDIT.md',
           SRC / 'CANDIDATE.json', SRC / 'CANDIDATE_STORAGE_R2.json', SRC / 'FROZEN_DEPENDENCIES.json',
           A / 'M125_R0_candidate_staging_20261015_stdout.json',
           A / 'M125_R0_storage_R2_staging_20261015_stdout.json',
           A / 'M125_R0_storage_R2_staging_20261015_stderr.log',
           A / 'M125_R0_storage_admission_diagnosis_retry1_20261015.json',
           A / 'M125_R0_storage_R2_budget_20261015.json',
           original_launch_path, launch_path, formal_path, stage_path, observer_path,
           A / 'stage_M125_R0_candidate_remote_20261015.py',
           A / 'M125_R0_real64_actual_launch_20261015_stderr.log',
           A / 'M125_R0_real64_actual_launch_20261015_stdout.json']
sources = [Path(p) for p in old['new_source_local_paths'].values()] + [Path(r['local']) for r in frozen]
paths = sorted(set(p.resolve() for p in primary + sources), key=str)
before = {str(p): dict(sha256=sha(p), bytes=p.stat().st_size) for p in paths}
write('input_hashes.before.json', before)
prior_inventory = {key(p): h for p, h in prior['audited_input_hashes'].items()}

check('prior_json_byte_exact', sha(OLD / 'EXPERIMENT_AUDIT.json') == 'e592627e72d6b4f7fff240899868c57bd6b3068721be140d398ce186fe57775b', str(OLD / 'EXPERIMENT_AUDIT.json'))
check('prior_md_byte_exact', sha(OLD / 'EXPERIMENT_AUDIT.md') == 'd6551958d16a9396407041c837baf9a067cb597e98b09f4bb163bbf90173dd5e', str(OLD / 'EXPERIMENT_AUDIT.md'))
check('prior_source_pass_zero_blockers', prior['source_verdict'] == 'PASS_SOURCE_ONLY' and prior['blocking_findings'] == [], prior['reason_code'])
check('old_candidate_bound_to_prior', sha(SRC / 'CANDIDATE.json') == prior['candidate_sha256'] == budget['original_candidate_sha256'], prior['candidate_sha256'])
check('new_candidate_bound_to_budget', sha(SRC / 'CANDIDATE_STORAGE_R2.json') == budget['corrected_candidate_sha256'], sha(SRC / 'CANDIDATE_STORAGE_R2.json'))
delta = {k: dict(old=old.get(k), new=new.get(k)) for k in set(old) | set(new) if old.get(k) != new.get(k)}
check('candidate_exact_single_field_delta', delta == {'required_free_disk_bytes': {'old': 20 * 1024**3, 'new': 2 * 1024**3}}, delta)
check('fixed6_final6_seed_unchanged', new['fixed_budget_passes'] == new['fixed_final_pass'] == 6 and new['seed'] == 2027, {k: new[k] for k in ['fixed_budget_passes', 'fixed_final_pass', 'seed']})
check('raw_input_and_all_source_bindings_identical', old['input_sha256'] == new['input_sha256'] and old['source_sha256'] == new['source_sha256'], {'input_bindings': len(new['input_sha256']), 'source_bindings': len(new['source_sha256'])})
check('new_nine_and_frozen44_counts', len(old['new_source_local_paths']) == 9 and len(frozen) == 44 and len(sources) == 53, {'new': 9, 'frozen': 44})
source_rows = []
for remote, local in old['new_source_local_paths'].items():
    actual = sha(local)
    source_rows.append(dict(local=str(Path(local).resolve()), staged_path=remote, actual_sha256=actual,
        passes=actual == old['source_sha256'][remote] == newstage['source_sha256'][remote] == prior_inventory[key(local)]))
for row in frozen:
    local = row['local']
    staged = new['code_root'] + '/' + Path(local).name
    actual = sha(local)
    source_rows.append(dict(local=str(Path(local).resolve()), staged_path=staged, actual_sha256=actual,
        passes=actual == row['sha256'] == newstage['source_sha256'][staged] == prior_inventory[key(local)]))
write('unchanged_source_bindings.json', source_rows)
check('all53_local_sources_byte_exact', all(r['passes'] for r in source_rows), 'unchanged_source_bindings.json: all 53 actual local file hashes compared to prior audit and actual R2 receipt')
check('actual53_staged_maps_identical', len(newstage['source_sha256']) == 53 and oldstage['source_sha256'] == newstage['source_sha256'] == prior['audited_staged_source_sha256'], 'original staging, R2 staging, prior audit maps')
check('frozen_manifest_unchanged', sha(SRC / 'FROZEN_DEPENDENCIES.json') == prior_inventory[key(SRC / 'FROZEN_DEPENDENCIES.json')], sha(SRC / 'FROZEN_DEPENDENCIES.json'))

enriched = copy.deepcopy(old)
enriched.pop('new_source_local_paths')
enriched['source_sha256'].update(oldstage['source_sha256'])
serialized = lambda value: (json.dumps(value, indent=2) + '\n').encode()
digest = lambda raw: hashlib.sha256(raw).hexdigest()
old_gate_sha = digest(serialized(enriched))
check('original_enriched_gate_reproduced', old_gate_sha == oldstage['candidate_gate_sha256'] == newstage['original_candidate_gate_sha256'], old_gate_sha)
revised = copy.deepcopy(enriched)
revised['required_free_disk_bytes'] = 2 * 1024**3
new_gate_sha = digest(serialized(revised))
check('new_enriched_gate_reproduced_from_single_delta', new_gate_sha == newstage['candidate_gate_sha256'], new_gate_sha)
check('new_gate_metadata_retained', {k:v for k,v in enriched.items() if k != 'required_free_disk_bytes'} == {k:v for k,v in revised.items() if k != 'required_free_disk_bytes'}, 'Every original enriched key/value retained except disk threshold')
check('actual_gate_reference_and_threshold', newstage['candidate_gate_path'] == new['root'] + '/candidate_gate_storage_R2.json' and newstage['required_free_disk_bytes'] == new['required_free_disk_bytes'] and newstage['old_required_free_disk_bytes'] == old['required_free_disk_bytes'], newstage['candidate_gate_path'])
check('inherited_import_receipt_serialization_reproduced', digest(serialized(oldstage)) == newstage['inherited_CPU_import_receipt_sha256'], newstage['inherited_CPU_import_receipt_sha256'])
check('CPU_imports_explicitly_inherited', newstage['CPU_imports_reexecuted'] is False and newstage['CPU_imports'] == oldstage['CPU_imports'], {'CPU_imports_reexecuted': newstage['CPU_imports_reexecuted'], 'rows': len(newstage['CPU_imports'])})
check('original_nine_imports_exit0', len(oldstage['CPU_imports']) == 9 and all(r['exit_code'] == 0 for r in oldstage['CPU_imports']), [r['exit_code'] for r in oldstage['CPU_imports']])
check('R2_receipt_no_NN_GPU_training', newstage['NN_calls'] == newstage['GPU_queries'] == 0 and newstage['training_launched'] is False, {k:newstage[k] for k in ['status', 'NN_calls', 'GPU_queries', 'training_launched']})
check('static_R2_stderr_empty', (A / 'M125_R0_storage_R2_staging_20261015_stderr.log').read_bytes() == b'', 'local static-staging stderr')

du = {line.split('\t')[1]: int(line.split('\t')[0]) for line in diagnosis['actual_existing_output_sizes'].splitlines()}
fit_path = '/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/epochs/epoch-06/fit'
eval_path = '/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/epochs/epoch-06/evaluation'
estimate = 6 * du[fit_path] + du[eval_path]
free = diagnosis['mounts'][new['root']]['free']
check('budget_embeds_actual_diagnosis_exactly', budget['actual_measurement'] == diagnosis, diagnosis['at'])
check('storage_estimate_arithmetic', estimate == budget['measured_six_fit_plus_one_evaluation_bytes'] == 220907189, {'measured_one_fit': du[fit_path], 'measured_one_evaluation': du[eval_path], 'six_fit_extrapolation_plus_evaluation': estimate})
check('2GiB_exceeds_estimated_output', new['required_free_disk_bytes'] > estimate, {'threshold_bytes': new['required_free_disk_bytes'], 'estimate_bytes': estimate, 'ratio': new['required_free_disk_bytes'] / estimate})
check('actual_free_failed20GiB_but_exceeds2GiB', old['required_free_disk_bytes'] > free >= new['required_free_disk_bytes'], {'free': free, 'old_required': old['required_free_disk_bytes'], 'new_required': new['required_free_disk_bytes']})
check('static_stage_actual_free_exceeds2GiB', newstage['actual_free_bytes'] >= new['required_free_disk_bytes'], newstage['actual_free_bytes'])
check('prelaunch_diagnosis_all_unstarted', set(diagnosis['unstarted_artifacts']) == {'real64.actual_launch.json','real64.exit','real64','gate.json','source_review.json','actual_launch.json','passes'} and all(v is False for v in diagnosis['unstarted_artifacts'].values()), diagnosis['unstarted_artifacts'])
check('readonly_diagnosis_no_NN_GPU_mutation', diagnosis['NN_calls'] == diagnosis['GPU_queries'] == diagnosis['filesystem_mutations'] == 0, {k:diagnosis[k] for k in ['NN_calls','GPU_queries','filesystem_mutations']})

original = original_launch_path.read_text(encoding='utf-8')
current = launch_path.read_text(encoding='utf-8')
replacements = [
    ('M125_R0_FIXED_SOURCE_AUDIT_20261015', 'M125_R0_STORAGE_ADMISSION_AUDIT_R2_20261015', 1),
    ('M125_R0_FIXED_SOURCE_20261015/CANDIDATE.json', 'M125_R0_FIXED_SOURCE_20261015/CANDIDATE_STORAGE_R2.json', 1),
    ('M125_R0_candidate_staging_20261015_stdout.json', 'M125_R0_storage_R2_staging_20261015_stdout.json', 1),
    ('candidate_gate.json', 'candidate_gate_storage_R2.json', 2),
    ('M125_R0_real64_actual_launch_20261015_stdout.json', 'M125_R0_real64_actual_launch_R2_20261015_stdout.json', 1),
    ('M125_R0_real64_actual_launch_20261015_stderr.log', 'M125_R0_real64_actual_launch_R2_20261015_stderr.log', 1),
]
expected = original
for old_text, new_text, count in replacements:
    check('launcher_replacement_count_' + old_text, expected.count(old_text) == count, {'expected_occurrences': count, 'actual_occurrences': expected.count(old_text)})
    expected = expected.replace(old_text, new_text)
check('launcher_only_explicit_path_substitutions', expected == current, replacements)
(OUT / 'real64_operator.diff').write_text(''.join(difflib.unified_diff(original.splitlines(keepends=True), current.splitlines(keepends=True), fromfile=original_launch_path.name, tofile=launch_path.name)), encoding='utf-8')
check('original_real64_and_formal_operators_unchanged', all(sha(p) == prior_inventory[key(p)] for p in [original_launch_path, formal_path]), {p.name:sha(p) for p in [original_launch_path, formal_path]})
check('original_stager_unchanged', sha(A / 'stage_M125_R0_candidate_remote_20261015.py') == prior_inventory[key(A / 'stage_M125_R0_candidate_remote_20261015.py')], sha(A / 'stage_M125_R0_candidate_remote_20261015.py'))
oldremote = remote_assignment(original).func.value.value
newremote = remote_assignment(current).func.value.value
stageremote = remote_assignment(stage_path.read_text(encoding='utf-8')).value
observerremote = remote_assignment(observer_path.read_text(encoding='utf-8')).func.value.value
formalremote = remote_assignment(formal_path.read_text(encoding='utf-8')).value
for label, body in [('new_launcher', current), ('new_launcher_remote', newremote), ('new_stager', stage_path.read_text(encoding='utf-8')), ('new_stager_remote', stageremote), ('new_observer', observer_path.read_text(encoding='utf-8')), ('new_observer_remote', observerremote), ('unchanged_formal_remote', formalremote)]:
    ast.parse(body)
    check('AST_parse_' + label, True, 'Parsed only; never executed/imported')
failure = (A / 'M125_R0_real64_actual_launch_20261015_stderr.log').read_text(encoding='utf-8')
disk_line = "assert shutil.disk_usage(root).free>=gate['required_free_disk_bytes']"
check('original_failure_exact_disk_assertion', failure == 'Traceback (most recent call last):\n  File "<stdin>", line 17, in <module>\nAssertionError\n' and oldremote.splitlines()[16] == disk_line, {'stderr': failure, 'remote_line17': oldremote.splitlines()[16]})
check('failed_admission_produced_no_stdout', (A / 'M125_R0_real64_actual_launch_20261015_stdout.json').read_bytes() == b'', 'Original failed stdout remains empty and preserved')
check('disk_admission_precedes_GPU_writes_NN', oldremote.index(disk_line) < oldremote.index("query=subprocess.run(['nvidia-smi'") < oldremote.index("target=root/name") < oldremote.index("command=[gate['python']"), 'Original exact line 17 precedes GPU query and all launch/gate writes')
check('R2_verdict_candidate_sources_admission_preserved', all(text in newremote for text in ["assert review['source_verdict']=='PASS_SOURCE_ONLY' and review['blocking_findings']==[]", "assert review['candidate_sha256']==p['candidate_sha256']", "assert review['audited_staged_source_sha256']==p['staging']['source_sha256']", disk_line, "all(r[1]<500 for r in gpu_rows)", "sys.exit(completed.returncode)", "automatic_retries=0"]), 'Exact operator delta also establishes inherited predicates/waiting supervisor unchanged')
check('formal_still_uses_bound_gate_disk_and_real64', all(text in formalremote for text in [disk_line, "assert review['candidate_sha256']==gate['source_review_candidate_sha256']", "assert sanity['gate_sha256']==sha(root/'gate.json')", "assert (root/'real64.exit').read_text().strip()=='0'", "assert sanity['track_calls']==64 and sanity['AB_optimizer_steps']==sanity['C_optimizer_steps']==2", "assert not sanity['formal_training_resume_uses_this_sanity']"]), 'Unchanged original formal source hash plus direct admission interface')
check('stager_exact_one_policy_mutation', "candidate=dict(original,required_free_disk_bytes=2*1024**3)" in stageremote and "assert original['required_free_disk_bytes']==20*1024**3" in stageremote and "for path,digest in original['input_sha256'].items():assert sha(path)==digest,path" in stageremote and "for path,digest in original['source_sha256'].items():assert sha(path)==digest,path" in stageremote, 'Original enriched metadata retained through shallow dict copy; all values otherwise unchanged')
check('stager_only_new_metadata_writes', "with target.open('x')" in stageremote and "with (root/'storage_R2_staging.json').open('x')" in stageremote and 'subprocess' not in stageremote and not any(s in stageremote for s in ['unlink(', 'remove(', 'rmtree(', 'move(', 'copy(', 'copy2(']), 'No experiment imports, NN invocation, deletion, source copying or data movement in remote stager')
check('stager_truthfully_inherits_CPU_imports', 'CPU_imports_reexecuted=False' in stageremote and "inherited_CPU_import_receipt_sha256=sha(root/'staging.json')" in stageremote, 'Original CPU rows retained; no reexecution')
observer = observer_path.read_text(encoding='utf-8')
check('observer_180_second_local_and_remote_gate', 'timedelta(seconds=180)' in observer and 'assert datetime.now(timezone.utc) >= not_before' in observer and "assert datetime.now(timezone.utc)>=datetime.fromisoformat('NOT_BEFORE')" in observerremote, 'At least 180 seconds from actual R2 launch; one observation, not a completion guarantee')
check('observer_remote_readonly_no_NN_GPU', 'subprocess' not in observerremote and all(s not in observerremote for s in ['write_text(', 'write_bytes(', ".open('w", ".open('x", 'unlink(', 'remove(', 'rmtree(', 'move(']) and 'NN_calls=0,GPU_queries=0' in observerremote, 'Reads existing witness/gate/receipts and bounded log tail')
check('new_operators_no_retry_or_fallback', all(not any(isinstance(n, (ast.Try, ast.While)) for n in ast.walk(ast.parse(body))) for body in [current, newremote, stage_path.read_text(encoding='utf-8'), stageremote, observer, observerremote]), 'No try/fallback or retry loop; explicit R2 action is a separately named operator')
after = {str(p): dict(sha256=sha(p), bytes=p.stat().st_size) for p in paths}
write('input_hashes.after.json', after)
check('all_amendment_inputs_unchanged_during_check', before == after, {'paths': len(paths), 'prior_MD_JSON_included': True})
write('CHECKS.json', dict(generated_at=datetime.now(timezone.utc).isoformat(), kind='bounded_storage_revision_static_and_hash_checks', prior_341_fixtures_rerun=False, total=len(checks), passed=sum(c['passed'] for c in checks), failed=sum(not c['passed'] for c in checks), checks=checks))
write('REPRODUCED_GATE_HASHES.json', dict(kind='local_deterministic_serialization_check_not_remote_execution', original_gate_sha256=old_gate_sha, revised_gate_sha256=new_gate_sha, original_staging_serialized_sha256=digest(serialized(oldstage)), candidate_delta=delta))
print(json.dumps(dict(total=len(checks), failed=[c for c in checks if not c['passed']], input_paths=len(paths), candidate_sha256=sha(SRC / 'CANDIDATE_STORAGE_R2.json'), gate_sha256=new_gate_sha)))
raise SystemExit(0 if all(c['passed'] for c in checks) else 1)
