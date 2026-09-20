"""Export sealed M84 evidence after the original queue has completed."""
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path('/root/autodl-tmp/sttrack_m84_centered_20260920')
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
exits = ['training_category', 'category_recursive', 'category_empty_recursive',
         'category_swapped_recursive', 'recursive_analysis', 'controller']
for stage in exits:
    assert (ROOT / (stage + '.exit')).read_text().strip() == '0', stage
spec = read(ROOT / 'recursive_spec.json')
training = read(ROOT / 'training_spec.json')
result = read(ROOT / 'recursive_result.json')
frozen = read(ROOT / 'frozen.json')
assert sha(ROOT / 'recursive_spec.json') == frozen['recursive_spec_sha256']
assert sha(ROOT / 'training_spec.json') == frozen['training_spec_sha256']
assert result['status'] == 'complete_recursive_development'
assert result['recursive_spec_sha256'] == sha(ROOT / 'recursive_spec.json')
assert result['training_spec_sha256'] == sha(ROOT / 'training_spec.json')

files = {}
def add(name, path):
    assert name not in files and path.is_file(), (name, str(path))
    files[name] = path

for name in ['recursive_spec.json', 'training_spec.json', 'frozen.json',
             'integration.json', 'run_recursive.py', 'recursive_metric.py',
             'run_m84.sh', 'recursive_result.json', 'EXPERIMENT_PLAN.md',
             'training_saved_verification.json', 'verify_training_complete.py']:
    add(name, ROOT / name)
for stage in exits:
    add(stage + '.exit', ROOT / (stage + '.exit'))
for path in ROOT.glob('*.log'):
    add(path.name, path)
for name in ['result.json', 'final.pth', 'sequence_log.jsonl']:
    add('training/category/' + name, ROOT / 'training/category' / name)
assert sha(ROOT / 'training/category/final.pth') == result['head_sha256']

for arm in ['category', 'category_empty', 'category_swapped']:
    receipt_path = ROOT / (arm + '_recursive_receipt.json')
    receipt = read(receipt_path)
    assert sha(receipt_path) == result['receipts'][arm]
    assert receipt['status'] == 'complete' and receipt['total_frames'] == 33130
    assert len(receipt['sequences']) == len(spec['cases']) == 22
    assert receipt['head_sha256'] == result['head_sha256']
    add(receipt_path.name, receipt_path)
    for case, item in zip(spec['cases'], receipt['sequences']):
        assert case['sequence'] == item['sequence']
        assert case['frames'] == item['frames']
        name = 'recursive/' + arm + '/' + case['sequence'] + '.json'
        path = ROOT / name
        assert sha(path) == item['sha256']
        add(name, path)

# Dataset labels are exported only after all three prediction receipts seal.
for case in spec['cases']:
    path = Path(training['dataset_root']) / case['sequence'] / 'groundtruth.txt'
    assert sha(path) == case['gt_sha256']
    add('dataset_gt/' + case['sequence'] + '.txt', path)
for name, path, digest in [
    ('native_result.json', Path(spec['native_result_path']), training['native_result_sha256']),
    ('M82_recursive_result.json', Path(training['parent_result_path']), training['parent_result_sha256'])
]:
    assert sha(path) == digest
    add('controls/' + name, path)
for name, digest in read(ROOT / 'integration.json')['source_sha256'].items():
    path = ROOT / 'code' / name
    assert sha(path) == digest
    add('code/' + name, path)
add('export_completed.py', Path(__file__))
manifest = {name: {'sha256': sha(path), 'bytes': path.stat().st_size}
            for name, path in sorted(files.items())}
archive = ROOT / 'completed_evidence.tar.gz'
assert not archive.exists()
with tarfile.open(str(archive), 'w:gz') as tar:
    for name, path in sorted(files.items()):
        tar.add(str(path), arcname=name, recursive=False)
    raw = (json.dumps(manifest, indent=2) + '\n').encode()
    info = tarfile.TarInfo('evidence_manifest.json')
    info.size = len(raw)
    tar.addfile(info, io.BytesIO(raw))
print(json.dumps({'status': 'sealed_evidence_exported', 'archive': str(archive),
                  'sha256': sha(archive), 'bytes': archive.stat().st_size,
                  'files': len(files)}, sort_keys=True))
