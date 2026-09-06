"""Archive both completed M55 recursive arms using their frozen binding."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile


root = Path('/root/autodl-tmp/sttrack_m55_tsg_direction_v2_20260906')
destination = Path('/root/autodl-tmp/sttrack_m55_recursive_completed_20260906')
sys.path.insert(0, str(root))
from run_recursive import binding


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


frozen, train, cases, training = binding(root)
exits = ['recursive_analysis.exit']
exits += ['recursive_queue_gpu%d%s.exit' % (gpu, suffix) for gpu in [0, 1] for suffix in ['', '_controller']]
exits += [arm + '_recursive.exit' for arm in ['control', 'clone']]
for name in exits:
    assert (root / name).read_text().strip() == '0', name
result = json.loads((root / 'recursive_result.json').read_text())
assert result['status'] == 'complete'
assert result['recursive_spec_sha256'] == sha(root / 'recursive_spec.json')
assert result['training_spec_sha256'] == sha(root / 'training_spec.json')
assert result['primary_pass'] == all(result['gates'].values())
assert not result['language_enabled'] and not result['public_evaluation']
names = {case['sequence'] for case in cases}
predictions = []
for arm in ['control', 'clone']:
    receipt_path = root / (arm + '_recursive_receipt.json')
    receipt = json.loads(receipt_path.read_text())
    assert sha(receipt_path) == result['recursive_receipt_sha256'][arm]
    assert receipt['status'] == 'complete' and receipt['variant'] == arm
    assert receipt['frames'] == 33130 and len(receipt['sequences']) == 22
    assert {row['sequence'] for row in receipt['sequences']} == names
    assert receipt['weight_sha256'] == training[arm]['weight_sha256'] == result['weights'][arm]
    assert receipt['training_result_sha256'] == result['training_result_sha256'][arm]
    assert not receipt['subsequent_gt_opened']
    for row in receipt['sequences']:
        path = root / 'recursive' / arm / (row['sequence'] + '.json')
        assert sha(path) == row['sha256']
        predictions.append(path)
files = exits + ['training_spec.json', 'recursive_spec.json', 'recursive_inputs.json',
    'recursive_result.json', 'recursive_analysis.log', 'recursive_queue_launch.json',
    'recursive_queue_launch_correction.json', 'recursive_queue_gpu0.log', 'recursive_queue_gpu1.log',
    'recursive_queue_gpu0_plan.json', 'recursive_queue_gpu1_plan.json']
files += [arm + suffix for arm in ['control', 'clone'] for suffix in ['_recursive.log', '_recursive_receipt.json']]
destination.mkdir()
for name in files:
    shutil.copyfile(root / name, destination / name)
for arm in ['control', 'clone']:
    shutil.copyfile(root / 'training' / arm / 'result.json', destination / (arm + '_training_result.json'))
shutil.copyfile(__file__, destination / 'export_completion.py')
record = dict(status='complete_recursive_artifacts_verified', observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    root=str(root), recursive_result_sha256=sha(root / 'recursive_result.json'),
    primary_pass=result['primary_pass'], prediction_files_verified=len(predictions), prediction_frames=2 * 33130,
    both_internal_final_checkpoints_and_training_sources_verified=True, exporter_reads_subsequent_gt=False,
    language_enabled=False, new_independent_result_review=False, public_evaluation=False, adopted_weight=False)
(destination / 'completion_binding.json').write_text(json.dumps(record, indent=2) + '\n')
manifest = [dict(path=path.name, bytes=path.stat().st_size, sha256=sha(path)) for path in sorted(destination.iterdir())]
(destination / 'bundle_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
archive = destination.with_suffix('.tar.gz')
with tarfile.open(archive, 'w:gz') as stream:
    for path in sorted(destination.iterdir()):
        stream.add(path, arcname=path.name)
    for path in predictions:
        stream.add(path, arcname=path.relative_to(root).as_posix())
print(json.dumps(dict(**record, archive=str(archive), archive_bytes=archive.stat().st_size,
    archive_sha256=sha(archive), record_files=len(manifest)), indent=2))
