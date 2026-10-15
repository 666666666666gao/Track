"""Reviewer-owned stdlib-only source/hash/receipt checks. No project imports."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

BASE = Path(r'C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007')
NEW = BASE / 'M125_REGION_SOURCE_20261015'
OLD = BASE / 'M125_R0_FIXED_SOURCE_20261015'
OUT = BASE / 'M125_REGION_SOURCE_AUDIT_20261015'
TRACE = OUT / '.aris/traces/experiment-audit/2026-10-15_run01'

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

before = read(TRACE / 'input_hashes_before.json')['sha256']
checks = []
def check(name, passed, details=None):
    checks.append(dict(name=name, passed=bool(passed), details=details))

python_files = [Path(p) for p in before if p.endswith('.py')]
trees = {}
for p in sorted(python_files):
    trees[p] = ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))
    check('AST:' + str(p.relative_to(BASE)), True)
manifest = read(NEW / 'INPUT_MANIFEST.json')
for p, digest in manifest.items():
    check('manifest:' + Path(p).name, sha(Path(p)) == digest)
spec = read(NEW / 'REGION_SPEC.json')
for name, digest in spec['source_sha256'].items():
    check('spec_source:' + name, sha(NEW / name) == digest)
for key, name in [('prior_CPU_receipt_sha256', 'M125_region_CPU_interface_actual_20261015.json'),
                  ('CPU_import_receipt_sha256', 'M125_region_full_protocol_CPU_imports_20261015.json')]:
    check('spec_receipt:' + name, sha(BASE / name) == spec[key])

local_by_name = {p.name: p for p in python_files if p.parent == OLD / 'frozen_dependency_snapshots'}
local_by_name.update({p.name: p for p in python_files if p.parent in [NEW, OLD]})
candidates = {}
for name in ['CANDIDATE_ROI.json', 'CANDIDATE_R1.json', 'CANDIDATE_V.json']:
    candidate = read(NEW / name)
    candidates[name] = candidate
    mapping = {p: h for p, h in candidate['source_sha256'].items() if p.startswith(candidate['code_root'] + '/')}
    check('staged_source_count:' + name, len(mapping) == len(python_files), {'mapped': len(mapping), 'local_python': len(python_files)})
    for p, digest in mapping.items():
        key = p.rsplit('/', 1)[-1]
        check('staged_source:' + name + ':' + key, key in local_by_name and sha(local_by_name[key]) == digest)

normalized = {}
for name, candidate in candidates.items():
    text = json.dumps(candidate, sort_keys=True).replace(candidate['root'], '<ARM_ROOT>')
    normalized[name] = json.loads(text)
roi = normalized['CANDIDATE_ROI.json']
differences = {name: {key: value for key, value in candidate.items() if roi.get(key) != value}
               for name, candidate in normalized.items()}
old_candidate = read(OLD / 'CANDIDATE_STORAGE_R2.json')

unit = read(BASE / 'M125_region_CPU_interface_actual_20261015.json')
for name, row in unit['files'].items():
    check('embedded_receipt:' + name, hashlib.sha256(row['text'].encode()).hexdigest() == row['sha256'])
for name, digest in unit['source_sha256'].items():
    check('unit_source:' + name, sha(NEW / name) == digest)
unit_result = json.loads(unit['files']['synthetic_unit/result.json']['text'])
check('unit_exit_zero', unit['receipt']['exit_code'] == 0 and unit['files']['unit.exit']['text'].strip() == '0')
check('unit_check_count', len(unit_result['checks']) == unit_result['check_count'] == 10)
check('unit_forward_count', len(unit_result['forward_log']) == unit_result['decoder_forward_calls'] == 15)
check('unit_no_real_scope', unit_result['real_tracking_calls'] == unit_result['dataset_images_opened'] == unit_result['STTrack_CLIP_models_instantiated'] == 0)
imports = read(BASE / 'M125_region_full_protocol_CPU_imports_20261015.json')
check('import_count', len(imports['rows']) == len(imports['source_sha256']) == 11)
for row in imports['rows']:
    check('import_receipt:' + row['source'], row['exit_code'] == 0 and not row['stdout'] and not row['stderr'] and '-B' in row['command'])
for name, digest in imports['source_sha256'].items():
    check('import_source:' + name, sha(NEW / name) == digest)

new_params = 512 * 64 + 64 + 2 * 64 + (64 + 1) + (64 * 4 + 4) + (64 + 1)
c_params = 519 * 128 + 128 + 128 * 32 + 32 + 32 + 1
check('new_parameter_arithmetic', new_params == 33350)
check('C_parameter_arithmetic', c_params == 70721)
check('total_parameter_arithmetic', 380167 + new_params + c_params == 484238)
check('six_pass_arithmetic', 219802 * 6 == 1318812 and 152 * 6 == 912)

diffs = []
for old in sorted(OLD.glob('*.py')):
    new = NEW / old.name.replace('M125_R0', 'M125_region').replace('m125_R0', 'm125_region')
    if new.is_file():
        left = old.read_text().replace('M125_R0', 'M125_region').replace('m125_R0', 'm125_region')
        right = new.read_text()
        diffs.extend(difflib.unified_diff(left.splitlines(keepends=True), right.splitlines(keepends=True), fromfile=old.name, tofile=new.name, n=3))
(TRACE / 'R0_to_region.diff').write_text(''.join(diffs), encoding='utf-8')

result = dict(mode='stdlib_AST_JSON_hash_only_no_project_imports_no_NN',
    input_file_count=len(before), python_AST_count=len(python_files),
    checks=checks, check_count=len(checks), pass_count=sum(r['passed'] for r in checks),
    fail_count=sum(not r['passed'] for r in checks),
    candidate_non_hash_fields={name: {k: v for k, v in candidate.items() if k not in ['source_sha256', 'input_sha256']} for name, candidate in candidates.items()},
    normalized_candidate_differences=differences,
    inherited_R0_candidate_non_hash_fields={k: v for k, v in old_candidate.items() if k not in ['source_sha256', 'input_sha256']},
    cpu_unit_result=unit_result, import_receipt=imports)
(TRACE / 'deterministic_checks.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps({k: result[k] for k in ['mode', 'input_file_count', 'python_AST_count', 'check_count', 'pass_count', 'fail_count']}, indent=2))
print(json.dumps({'failed': [r for r in checks if not r['passed']], 'candidate_differences': differences,
                  'R0_non_hash_fields': result['inherited_R0_candidate_non_hash_fields']}, indent=2))
