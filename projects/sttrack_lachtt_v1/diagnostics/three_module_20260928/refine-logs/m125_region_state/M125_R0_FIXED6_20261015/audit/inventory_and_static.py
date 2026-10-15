"""Source-only reviewer evidence. Standard library only; never imports project code."""
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
SRC = BASE / 'M125_R0_FIXED_SOURCE_20261015'
ORIGINAL = BASE.parent.parent / 'projects/sttrack_lachtt_v1/diagnostics/three_module_20260928'
COMMON = BASE / 'M125_fixed_R0_common_inputs_20261015.json'
PLAN = ORIGINAL / 'refine-logs/m125_region_state/EXPERIMENT_PLAN.md'
STAGER = BASE / 'stage_M125_R0_candidate_remote_20261015.py'
PAYLOAD = BASE / 'M125_R0_staging_payload_20261015.json'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


candidate = read(SRC / 'CANDIDATE.json')
deps = read(SRC / 'FROZEN_DEPENDENCIES.json')
common = read(COMMON)
payload = read(PAYLOAD)
original_names = [
    'train_m124_joint_epoch.py', 'm124_epochwise_official_runtime.py',
    'run_m124_epochwise_official.py', 'prepare_m123_joint_evaluation.py',
    'run_m123_joint_full_suite.py', 'analyze_m124_epochwise_vot.py',
    'collect_m124_epochwise_results.py', 'm124_epochwise_process_supervisor.py',
]
paths = sorted(SRC.glob('*.py')) + [SRC / 'CANDIDATE.json', SRC / 'FROZEN_DEPENDENCIES.json']
paths += [Path(row['local']) for row in deps]
paths += [ORIGINAL / name for name in original_names]
paths += [COMMON, PLAN, STAGER, PAYLOAD]
hashes = {str(path): {'sha256': sha(path.read_bytes()), 'bytes': path.stat().st_size} for path in paths}
write('input_hashes.before.json', hashes)

checks = []


def check(name, actual, expected=True, details=None):
    checks.append(dict(name=name, status='PASS' if actual == expected else 'FAIL',
                       actual=actual, expected=expected, details=details))


check('nine_new_modules', len(list(SRC.glob('*.py'))), 9)
check('44_frozen_dependencies', len(deps), 44)
for row in deps:
    check('frozen_hash:' + PurePosixPath(row['remote']).name,
          sha(Path(row['local']).read_bytes()), row['sha256'])
    check('frozen_gate_binding:' + row['remote'], candidate['source_sha256'][row['remote']], row['sha256'])
for remote, local in candidate['new_source_local_paths'].items():
    check('new_hash:' + PurePosixPath(remote).name, sha(Path(local).read_bytes()), candidate['source_sha256'][remote])
for name in original_names:
    check('original_vs_snapshot:' + name, sha((ORIGINAL / name).read_bytes()),
          sha((SRC / 'frozen_dependency_snapshots' / name).read_bytes()))
parsed = {}
for path in paths:
    if path.suffix == '.py':
        parsed[str(path)] = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        check('AST_parse:' + str(path), True)

import_graph = {}
for name, tree in parsed.items():
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    import_graph[name] = sorted(set(imports), key=str)
write('import_graph.json', import_graph)

common_decoded = {}
for name, row in common['files'].items():
    data = row['text'].encode()
    check('common_raw_sha:' + name, sha(data), row['sha256'])
    check('common_raw_length:' + name, len(data), row['bytes'])
    common_decoded[name] = json.loads(row['text'])
spec = common_decoded[candidate['spec']]
warm = common_decoded[candidate['warm_result']]
check('spec_gate_sha', common['files'][candidate['spec']]['sha256'], candidate['input_sha256'][candidate['spec']])
check('spec_sequences', len(spec['sequence_order']), 152)
check('spec_sequence_names_unique', len({row['sequence'] for row in spec['sequence_order']}), 152)
check('spec_track_calls_one_pass', sum(row['rgb_frames'] - 1 for row in spec['sequence_order']), 219802)
check('spec_seed', spec['seed'], 2027)
check('warm_final_binding', warm['final_sha256'], candidate['input_sha256'][candidate['warm_final']])
check('warm_steps', warm['optimizer_steps'], 400)
check('payload_candidate_exact', payload['candidate'], candidate)
check('payload_frozen_dependencies_exact', payload['frozen_dependencies'], deps)
check('payload_new_sources_names', sorted(payload['sources']), sorted(path.name for path in SRC.glob('*.py')))
for name, row in payload['sources'].items():
    check('payload_text:' + name, sha(row['text'].encode()), sha((SRC / name).read_bytes()))
    check('payload_digest:' + name, row['sha256'], sha((SRC / name).read_bytes()))

remote_node = next(n for n in parsed[str(STAGER)].body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'remote' for t in n.targets))
remote_script = remote_node.value.func.value.value
ast.parse(remote_script)
check('embedded_remote_stager_AST_parse', True)
(OUT / 'stager_remote_source.txt').write_text(remote_script, encoding='utf-8')
staged_names = {PurePosixPath(row['remote']).name for row in deps} | set(payload['sources'])
check('staged_basename_collision_free', len(staged_names), len(deps) + len(payload['sources']))
check('staged_tracker_sibling_present', 'm123_joint_full_tracker.py' in staged_names)
check('staged_vot_runner_present', 'run_m123_joint_vot_shards.py' in staged_names)

write('inventory_checks.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
      scope='static source/hash/payload checks; no imports, torch, NN, network, SSH, or deployment',
      counts=dict(inputs=len(hashes), parsed_python=len(parsed), checks=len(checks)), checks=checks))
write('common_summary.json', dict(
      common_status=common['status'], input_names=list(common_decoded),
      spec_keys=list(spec), first_sequence=spec['sequence_order'][0],
      complete_passes=6, sequence_runs=152 * 6, base_track_calls=219802 * 6,
      warm_result=warm, candidate_status=candidate['status'],
      gate_source_count=len(candidate['source_sha256']), stage_copied_count=len(staged_names)))
print(json.dumps(dict(inputs=len(hashes), checks=len(checks),
      failures=[row['name'] for row in checks if row['status'] == 'FAIL'],
      first_sequence=spec['sequence_order'][0]['sequence'], warm_status=warm['status'])))
