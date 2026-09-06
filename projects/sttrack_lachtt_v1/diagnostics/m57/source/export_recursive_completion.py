"""Seal every completed M57 recursive arm and its posthoc divergence evidence."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tarfile

import numpy as np
import torch

root = Path('/root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906')
destination = Path('/root/autodl-tmp/sttrack_m57_recursive_completed_20260906')
sys.path.insert(0, str(root))
from run_recursive import bound
from train import sha, write

spec, frozen, training, cases = bound(root)
post = json.loads((root / 'completion_tools_spec.json').read_text())
assert sha(__file__) == post['exporter_sha256']
assert sha(root / 'diagnose_divergence.py') == post['diagnostic_sha256']
assert sha(root / 'recursive_spec.json') == post['recursive_spec_sha256']
exits = ['analysis_scheduler.exit', 'recursive_queue0.exit', 'recursive_queue1.exit']
exits += [arm + '_recursive.exit' for arm in spec['variants']]
for name in exits:
    assert (root / name).read_text().strip() == '0', name
result = json.loads((root / 'recursive_result.json').read_text())
assert result['status'] == 'complete_recursive_development'
assert result['recursive_spec_sha256'] == sha(root / 'recursive_spec.json')
assert result['training_result_sha256'] == sha(root / 'training_result.json')
assert result['primary_variant'] == 'initial_text'
assert result['primary_pass'] == all(result['gates'].values())
assert not result['public_evaluation'] and not result['text_strings_updated_online']
diagnostic = json.loads((root / 'divergence_diagnostic.json').read_text())
assert diagnostic['status'] == 'posthoc_diagnostic_complete'
assert diagnostic['source_result_sha256'] == sha(root / 'recursive_result.json')
assert not diagnostic['inference_or_training_changed']
names = {case['sequence'] for case in cases}
assert set(diagnostic['sequences']) == names
predictions = []
head_metadata = {}
for arm in spec['variants']:
    path = root / 'training' / (arm + '_final.pth')
    saved = torch.load(path, map_location='cpu')
    assert saved['m57_spec_sha256'] == sha(root / 'spec.json') and saved['variant'] == arm
    assert saved['epochs'] == 20 and saved['optimizer_steps'] == 960
    assert saved['reference_mode'] == arm.split('_')[0] and saved['use_text'] == arm.endswith('_text')
    assert saved['base_checkpoint_sha256'] == spec['base_checkpoint_sha256']
    assert saved['text_bank_sha256'] == sha(root / 'text_fit.pt')
    assert saved['initial_fit_sha256'] == sha(root / 'initial_references/initial_fit.pt')
    assert sha(path) == training['variants'][arm]['checkpoint_sha256']
    head_metadata[arm] = dict(sha256=sha(path), epochs=20, optimizer_steps=960,
        reference_mode=saved['reference_mode'], use_text=saved['use_text'])
    receipt_path = root / (arm + '_recursive_receipt.json')
    assert sha(receipt_path) == result['receipts'][arm]
    receipt = json.loads(receipt_path.read_text())
    assert receipt['status'] == 'complete' and receipt['variant'] == arm
    assert receipt['head_sha256'] == head_metadata[arm]['sha256']
    assert receipt['base_sha256'] == spec['base_checkpoint_sha256']
    assert receipt['recursive_spec_sha256'] == sha(root / 'recursive_spec.json')
    assert receipt['training_result_sha256'] == sha(root / 'training_result.json')
    assert receipt['t0_reference_exactly_verified_for_all_22_sequences'] and receipt['same_slot_mask']
    assert receipt['frames'] == 33130 and len(receipt['sequences']) == 22
    assert {row['sequence'] for row in receipt['sequences']} == names
    assert not receipt['subsequent_gt_opened'] and receipt['native_default_template_updates']
    for item in receipt['sequences']:
        path = root / 'recursive' / arm / (item['sequence'] + '.json')
        assert sha(path) == item['sha256']
        data = json.loads(path.read_text())
        case = next(case for case in cases if case['sequence'] == item['sequence'])
        rows = data['rows']
        assert data['sequence'] == case['sequence']
        assert len(rows) == case['frames'] == item['frames']
        assert [row['frame'] for row in rows] == list(range(case['frames']))
        assert rows[0]['bbox'] == case['init_bbox']
        boxes = np.asarray([row['bbox'] for row in rows])
        assert np.isfinite(boxes).all() and (boxes[:, 2:] > 0).all()
        assert sum(row['chosen'] != 0 for row in rows) == item['nondefault']
        assert sum(row['none'] for row in rows) == item['none']
        assert diagnostic['sequences'][item['sequence']][arm]['nondefault_frames'] == item['nondefault']
        predictions.append(path)
for arm, values in result['per_sequence'].items():
    aggregate = result['aggregates'][arm]
    assert set(values) == names
    for field in ['valid_frames', 'iou_sum', 'low_iou_frames', 'failure_episodes']:
        assert sum(row[field] for row in values.values()) == aggregate[field], (arm, field)
    assert aggregate['valid_frames'] == 28897
    assert abs(aggregate['mean_iou'] - aggregate['iou_sum'] / aggregate['valid_frames']) < 1e-12
    assert abs(aggregate['macro_sequence_mean_iou'] - sum(row['mean_iou'] for row in values.values()) / 22) < 1e-12
    for name, row in values.items():
        assert len(diagnostic['sequences'][name][arm]['H10_spans']) == row['failure_episodes']
assert len(predictions) == 88
files = exits + ['spec.json', 'recursive_spec.json', 'recursive_inputs.json', 'training_result.json',
    'training_seal.json', 'recursive_result.json', 'divergence_diagnostic.json',
    'analysis_started.json', 'analysis_wait_record.json', 'analysis_scheduler_launch.json',
    'analysis_scheduler.log', 'recursive_queue0.log', 'recursive_queue1.log',
    'recursive_queue0_launch.json', 'recursive_queue1_launch.json', 'completion_tools_spec.json', 'publication_git.json']
files += [arm + suffix for arm in spec['variants'] for suffix in ['_recursive.log', '_recursive_receipt.json', '_recursive_launch.json']]
destination.mkdir()
for name in files:
    shutil.copyfile(root / name, destination / name)
(destination / 'source').mkdir()
for name in ['run_recursive.py', 'queue_recursive.py', 'finish_recursive.py', 'diagnose_divergence.py']:
    shutil.copyfile(root / name, destination / 'source' / name)
shutil.copyfile(__file__, destination / 'source/export_recursive_completion.py')
record = dict(status='all_four_complete_recursive_artifacts_verified',
    observed_utc=datetime.now(timezone.utc).isoformat(),
    recursive_result_sha256=sha(root / 'recursive_result.json'), recursive_spec_sha256=sha(root / 'recursive_spec.json'),
    divergence_diagnostic_sha256=sha(root / 'divergence_diagnostic.json'),
    source_files_verified=len(spec['source_sha256']), head_metadata_verified=head_metadata,
    prediction_files_verified=88, prediction_frames=132520, primary_pass=result['primary_pass'],
    exporter_opens_full_frame_gt=False, public_evaluation=False, adopted_weight=False,
    new_independent_result_review=False)
write(destination / 'completion_binding.json', record)
manifest = [dict(path=path.relative_to(destination).as_posix(), bytes=path.stat().st_size, sha256=sha(path))
            for path in sorted(destination.rglob('*')) if path.is_file()]
write(destination / 'bundle_manifest.json', manifest)
archive = destination.with_suffix('.tar.gz')
with tarfile.open(archive, 'w:gz') as stream:
    for path in sorted(destination.rglob('*')):
        if path.is_file():
            stream.add(path, arcname=path.relative_to(destination).as_posix())
    for path in predictions:
        stream.add(path, arcname=path.relative_to(root).as_posix())
print(json.dumps(dict(**record, archive=str(archive), archive_bytes=archive.stat().st_size,
    archive_sha256=sha(archive), record_files=len(manifest)), indent=2))
