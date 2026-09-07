from pathlib import Path
import hashlib
import json
import shutil
import tarfile

BASE = Path('/root/autodl-tmp')
PARENT = BASE / 'sttrack_m67_supervised_semantic_support_20260907'
ROOT = PARENT / 'training_completion_audit'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
read = lambda path: json.loads(Path(path).read_text())
result = read(ROOT / 'result.json')
assert result['status'] == 'completed_paired_training_artifact_audit_recursive_live'
assert result['source_sha256'] == sha(BASE / 'audit_m67_training_finish_20260907.py')
assert result['frozen_auditor_sha256'] == sha(BASE / 'audit_m67_completed_20260907.py')
assert result['new_tracking_calls'] == result['new_optimizer_steps'] == 0 and not result['new_formal_metrics']
out = ROOT / 'publication_evidence'
out.mkdir()
shutil.copyfile(ROOT / 'result.json', out / 'result.json')
shutil.copyfile(BASE / 'audit_m67_training_finish_20260907.py', out / 'audit_m67_training_finish_20260907.py')
shutil.copyfile(__file__, out / Path(__file__).name)
for arm in ['control', 'support']:
    assert sha(PARENT / 'training' / arm / 'final.pth') == result['training'][arm]['final_checkpoint_sha256']
    assert sha(PARENT / 'training' / arm / 'result.json') == result['training'][arm]['result_sha256']
    shutil.copyfile(PARENT / 'training' / arm / 'result.json', out / (arm + '_training_result.json'))
(out / 'evidence_manifest.json').write_text(json.dumps([dict(path=p.name, bytes=p.stat().st_size, sha256=sha(p))
    for p in sorted(out.iterdir())], indent=2) + '\n')
archive = ROOT / 'evidence.tar.gz'
with tarfile.open(str(archive), 'w:gz') as tar:
    for path in sorted(out.iterdir()): tar.add(str(path), arcname=path.name)
print(json.dumps(dict(archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
    training_audit_sha256=sha(ROOT / 'result.json'), training=result['training'],
    recursive_processes=result['live_recursive_processes']), indent=2))
