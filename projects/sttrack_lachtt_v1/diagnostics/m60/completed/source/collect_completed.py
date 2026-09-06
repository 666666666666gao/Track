"""Verify the sealed M60 control and export complete diagnostic evidence."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tarfile
import numpy as np

ROOT = Path('/root/autodl-tmp/sttrack_m60_category_isolation_20260906')
M59 = Path('/root/autodl-tmp/sttrack_m59_fixed_head_content_diagnostic_20260906')
PARENT = Path('/root/autodl-tmp/sttrack_m58_semantic_spatial_v2_20260906')
OUT = Path('/root/autodl-tmp/sttrack_m60_completed_export_20260907')
SOURCE = Path('/root/autodl-tmp/m60_category_isolation_20260906.py')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    for name in ['parity.exit', 'tracking.exit', 'analysis.exit', 'job.exit']:
        assert (ROOT / name).read_text().strip() == '0', name
    assert sha(SOURCE) == 'f8c1f789a7f927121e9f50cefc1d2995774d6bd9bf73d047ccb11e7ea920303b'
    runtime = load_module('category_isolation', SOURCE)
    spec, training, m59 = runtime.plans()
    result = json.loads((ROOT / 'result.json').read_text())
    receipt = json.loads((ROOT / 'receipt.json').read_text())
    assert result['status'] == 'completed_fixed_weight_category_content_isolation'
    assert result['spec_sha256'] == receipt['spec_sha256'] == sha(ROOT / 'spec.json')
    assert result['receipt_sha256'] == sha(ROOT / 'receipt.json')
    assert result['head_sha256'] == receipt['head_sha256'] == spec['head_sha256']
    assert receipt['bank_sha256'] == spec['bank_sha256']
    assert receipt['new_optimizer_steps'] == 0 and receipt['subsequent_gt_opened'] is False
    assert result['public_evaluation_allowed'] is False
    assert result['independent_review_pass'] is False
    assert receipt['status'] == 'complete' and receipt['frames'] == 33130
    reference_receipt = M59 / 'category_receipt.json'
    assert sha(reference_receipt) == m59['receipt_sha256']['category']
    reference_files = {r['sequence']: r for r in json.loads(reference_receipt.read_text())['sequences']}
    independent_source = Path('/root/autodl-tmp/audit_m58_completion_v3_20260906.py')
    assert sha(independent_source) == '5181b20f456e3150d714528d229d0fecc53ce1b9310d61b55f7e4ecda14fdae6'
    metric = load_module('completion_metrics', independent_source)
    cases = spec['cases']
    assert [r['sequence'] for r in receipt['sequences']] == [c['sequence'] for c in cases]
    recomputed = {}
    changes = {}
    templates = {}
    for case, item in zip(cases, receipt['sequences']):
        name = case['sequence']
        path = ROOT / 'predictions' / (name + '.json')
        assert sha(path) == item['sha256']
        data = json.loads(path.read_text())
        assert data['sequence'] == name and data['control'] == 'swapped_category'
        rows = data['rows']
        assert len(rows) == item['frames'] == case['frames']
        assert rows[0]['bbox'] == case['init_bbox'] and rows[0]['score'] is None
        assert [r['frame'] for r in rows] == list(range(case['frames']))
        boxes = np.asarray([r['bbox'] for r in rows])
        scores = np.asarray([r['score'] for r in rows[1:]])
        assert np.isfinite(boxes).all() and (boxes[:, 2:] > 0).all() and np.isfinite(scores).all()
        gt_path = Path(training['dataset_root']) / name / 'groundtruth.txt'
        assert sha(gt_path) == case['gt_sha256']
        gt = np.loadtxt(gt_path, delimiter=',').reshape(-1, 4)
        assert len(gt) == len(rows)
        valid, ious = metric.values(boxes, gt)
        low = valid & (ious <= .1)
        h10 = [(a, b) for a, b in metric.spans(low) if b - a >= 10]
        current = dict(valid_frames=int(valid.sum()), iou_sum=float(ious[valid].sum()),
                       mean_iou=float(ious[valid].mean()), low_iou_frames=int(low.sum()),
                       failure_episodes=len(h10))
        expected = result['per_sequence']['swapped_category'][name]
        for key, value in current.items():
            assert abs(value - expected[key]) < 1e-8, (name, key, value, expected[key])
        recomputed[name] = dict(**current, h10_low_frame_count=sum(b - a for a, b in h10))
        ids = np.arange(1, len(rows))
        writes = ids[(ids % 50 == 0) & (scores > .75)]
        templates[name] = dict(checks=int((ids % 50 == 0).sum()), writes=len(writes))
        assert len(writes) == result['per_sequence_swapped_writes'][name]
        path = M59 / 'predictions/category' / (name + '.json')
        assert sha(path) == reference_files[name]['sha256']
        old = json.loads(path.read_text())['rows']
        old_boxes = np.asarray([r['bbox'] for r in old])
        diff = np.flatnonzero(np.max(np.abs(boxes - old_boxes), axis=1) > 1e-4)
        original = result['per_sequence']['original_category'][name]
        changes[name] = dict(original_minus_swapped_mean_iou=original['mean_iou'] - current['mean_iou'],
                             original_minus_swapped_H10=original['failure_episodes'] - current['failure_episodes'],
                             changed_bbox_frames=len(diff),
                             first_bbox_difference_frame=int(diff[0]) if len(diff) else None)
    aggregate = result['aggregates']['swapped_category']
    for key in ['valid_frames', 'iou_sum', 'low_iou_frames', 'failure_episodes']:
        assert abs(aggregate[key] - sum(row[key] for row in recomputed.values())) < 1e-8
    assert abs(aggregate['mean_iou'] - aggregate['iou_sum'] / aggregate['valid_frames']) < 1e-12
    assert abs(aggregate['macro_sequence_mean_iou'] - np.mean([r['mean_iou'] for r in recomputed.values()])) < 1e-12
    for name, old in [('original_category', 'category'), ('empty', 'empty')]:
        assert result['aggregates'][name] == m59['aggregates'][old]
        assert result['per_sequence'][name] == m59['per_sequence'][old]
    original = result['aggregates']['original_category']
    assert abs(result['original_minus_swapped_mean_iou'] - (original['mean_iou'] - aggregate['mean_iou'])) < 1e-12
    assert result['original_minus_swapped_H10'] == original['failure_episodes'] - aggregate['failure_episodes']
    assert result['original_minus_swapped_low_frames'] == original['low_iou_frames'] - aggregate['low_iou_frames']
    assert result['descriptive_category_margin_pass'] == (result['original_minus_swapped_mean_iou'] >= spec['primary_descriptive_margin'])
    assert sum(r['writes'] for r in templates.values()) == result['swapped_category_reconstructed_template_writes']
    native_path = PARENT / 'recursive_result.json'
    assert sha(native_path) == '54fd8445297e1eef761fa36e8a17def01afa027f71371984998f08229f6f5ee9'
    native = json.loads(native_path.read_text())['per_sequence']['native']
    protected = [name for name, row in native.items() if row['failure_episodes'] == 0]
    assert len(protected) == 3
    harms = {arm: [name for name in protected if rows[name]['failure_episodes'] > 0]
             for arm, rows in result['per_sequence'].items()}
    audit = dict(status='completed_category_isolation_verified', observed_utc=datetime.now(timezone.utc).isoformat(),
                 auditor_sha256=sha(Path(__file__)), result_sha256=sha(ROOT / 'result.json'),
                 spec_sha256=sha(ROOT / 'spec.json'), receipt_sha256=sha(ROOT / 'receipt.json'),
                 head_sha256=spec['head_sha256'], new_control_trajectories=22, new_control_image_frames=33130,
                 new_control_track_calls=33108, new_prediction_hashes_and_structure_verified=True,
                 reference_category_prediction_hashes_verified=True, independent_continuous_IoU_H10_recomputed=True,
                 GT_opened_only_after_complete=True, recomputed_swapped_per_sequence=recomputed,
                 original_category_better_sequences=sum(r['original_minus_swapped_mean_iou'] > 1e-9 for r in changes.values()),
                 original_category_worse_sequences=sum(r['original_minus_swapped_mean_iou'] < -1e-9 for r in changes.values()),
                 per_sequence_content_differences=changes, native_zero_H10_sequences=protected,
                 new_failure_sequences_within_native_zero_H10=harms, per_sequence_reconstructed_templates=templates,
                 source_and_bank_hashes_verified=True, public_evaluation_allowed=False, independent_review_pass=False,
                 scope='One frozen category-slot intervention on reused development22. Automatic categories are not semantic ground truth. No public metric or promotion implied.')
    OUT.mkdir()
    write(OUT / 'audit_result.json', audit)
    evidence = OUT / 'evidence'; evidence.mkdir()
    for name in ['result.json', 'receipt.json', 'tracking.log', 'tracking.exit', 'analysis.log', 'analysis.exit', 'job.exit']:
        shutil.copyfile(ROOT / name, evidence / name)
    shutil.copyfile(OUT / 'audit_result.json', evidence / 'audit_result.json')
    shutil.copyfile(Path(__file__), evidence / 'collect_completed.py')
    manifest = [dict(path=p.name, bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(evidence.iterdir())]
    write(evidence / 'evidence_manifest.json', manifest)
    archive = OUT / 'evidence.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for path in sorted(evidence.iterdir()):
            tar.add(path, arcname=path.name)
    print(json.dumps(dict(status=audit['status'], result_sha256=audit['result_sha256'],
                          audit_sha256=sha(OUT / 'audit_result.json'), archive=str(archive),
                          archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
                          original_better_sequences=audit['original_category_better_sequences'],
                          original_worse_sequences=audit['original_category_worse_sequences'],
                          native_zero_H10_harms=harms), indent=2))


if __name__ == '__main__':
    main()
