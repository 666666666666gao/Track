"""Seal completed M56 records without re-running tracking or choosing a model."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile


root = Path('/root/autodl-tmp/sttrack_m56_attribute_alignment_v1_20260906')
destination = Path('/root/autodl-tmp/sttrack_m56_recursive_completed_20260906')
sys.path.insert(0, str(root))
from run_recursive import bound


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


spec, frozen, training, cases = bound(root)
exits = ['recursive_analysis.exit']
exits += ['queue_gpu%d%s.exit' % (gpu, suffix) for gpu in [0, 1] for suffix in ['', '_controller']]
exits += [arm + '_recursive.exit' for arm in spec['variants']]
for name in exits:
    assert (root / name).read_text().strip() == '0', name
result = json.loads((root / 'recursive_result.json').read_text())
assert result['status'] == 'complete_recursive_development'
assert result['recursive_spec_sha256'] == digest(root / 'recursive_spec.json')
assert result['training_result_sha256'] == digest(root / 'training_result.json')
assert result['primary_pass'] == all(result['gates'].values())
assert not result['public_evaluation'] and not result['text_strings_updated_online']
names = {case['sequence'] for case in cases}
predictions = []
for arm in spec['variants']:
    path = root / (arm + '_recursive_receipt.json')
    receipt = json.loads(path.read_text())
    assert digest(path) == result['receipts'][arm]
    assert receipt['status'] == 'complete' and receipt['variant'] == arm
    assert receipt['frames'] == 33130 and len(receipt['sequences']) == 22
    assert {row['sequence'] for row in receipt['sequences']} == names
    assert receipt['head_sha256'] == training['variants'][arm]['checkpoint_sha256']
    assert receipt['base_sha256'] == spec['base_checkpoint_sha256']
    assert receipt['native_default_template_updates'] and not receipt['subsequent_gt_opened']
    for row in receipt['sequences']:
        path = root / 'recursive' / arm / (row['sequence'] + '.json')
        assert digest(path) == row['sha256']
        predictions.append(path)

files = exits + ['spec.json', 'recursive_spec.json', 'recursive_inputs.json', 'training_result.json',
    'recursive_result.json', 'recursive_analysis.log', 'recursive_queue_launch.json',
    'queue_gpu0.log', 'queue_gpu1.log', 'queue_gpu0_plan.json', 'queue_gpu1_plan.json']
files += [arm + suffix for arm in spec['variants'] for suffix in ['_recursive.log', '_recursive_receipt.json']]
destination.mkdir()
for name in files:
    shutil.copyfile(root / name, destination / name)
shutil.copyfile(__file__, destination / 'export_completion.py')
record = dict(status='complete_recursive_artifacts_verified', observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    root=str(root), recursive_result_sha256=digest(root / 'recursive_result.json'),
    recursive_spec_sha256=digest(root / 'recursive_spec.json'), primary_pass=result['primary_pass'],
    prediction_files_verified=len(predictions), prediction_frames=3 * 33130,
    exporter_reads_subsequent_gt=False, new_independent_result_review=False,
    public_evaluation=False, adopted_weight=False)
(destination / 'completion_binding.json').write_text(json.dumps(record, indent=2) + '\n')
manifest = [dict(path=path.relative_to(destination).as_posix(), bytes=path.stat().st_size, sha256=digest(path))
    for path in sorted(destination.iterdir())]
(destination / 'bundle_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
archive = destination.with_suffix('.tar.gz')
with tarfile.open(archive, 'w:gz') as stream:
    for path in sorted(destination.iterdir()):
        stream.add(path, arcname=path.name)
    for path in predictions:
        stream.add(path, arcname=path.relative_to(root).as_posix())
print(json.dumps(dict(**record, archive=str(archive), archive_bytes=archive.stat().st_size,
    archive_sha256=digest(archive), record_files=len(manifest)), indent=2))
