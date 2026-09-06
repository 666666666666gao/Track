"""Posthoc timing of M57 trajectory changes; never used for inference or training."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


root = Path('/root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906')
result_path = root / 'recursive_result.json'
assert (root / 'analysis_scheduler.exit').read_text().strip() == '0'
result = json.loads(result_path.read_text())
assert result['status'] == 'complete_recursive_development'
assert result['primary_variant'] == 'initial_text'
assert result['recursive_spec_sha256'] == hashlib.sha256((root / 'recursive_spec.json').read_bytes()).hexdigest()
assert result['training_result_sha256'] == hashlib.sha256((root / 'training_result.json').read_bytes()).hexdigest()
spec = json.loads((root / 'spec.json').read_text())
frozen = json.loads((root / 'recursive_spec.json').read_text())
cases = json.loads((root / 'recursive_inputs.json').read_text())
native = defaultdict(list)
for filename, digest in frozen['baseline_trace_sha256'].items():
    path = Path(filename)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    for row in json.loads(path.read_text())['rows']:
        native[row['sequence']].append(row)


def ious(boxes, gt):
    valid = np.isfinite(gt).all(1) & (gt[:, 2:] > 0).all(1)
    valid[0] = False
    a, b = boxes[valid], gt[valid]
    intersection = np.maximum(0, np.minimum(a[:, :2] + a[:, 2:], b[:, :2] + b[:, 2:])
                              - np.maximum(a[:, :2], b[:, :2])).prod(1)
    values = np.full(len(boxes), np.nan)
    values[valid] = intersection / (a[:, 2:].prod(1) + b[:, 2:].prod(1) - intersection)
    return values


def spans(values):
    mask = np.isfinite(values) & (values <= .1)
    changes = np.diff(np.r_[False, mask, False].astype(int))
    return [dict(start=int(a), end=int(b - 1), frames=int(b - a))
            for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)) if b - a >= 10]


records = {}
for case in cases:
    name = case['sequence']
    baseline = sorted(native[name], key=lambda row: row['frame_index'])
    assert [row['frame_index'] for row in baseline] == list(range(case['frames']))
    base_boxes = np.asarray([row['public_bbox'] for row in baseline], dtype=np.float64)
    gt_path = Path(spec['dataset_root']) / name / 'groundtruth.txt'
    assert hashlib.sha256(gt_path.read_bytes()).hexdigest() == frozen['development_gt_sha256'][name]
    gt = np.loadtxt(gt_path, delimiter=',')
    values = {'native': ious(base_boxes, gt)}
    record = {}
    for arm in spec['variants']:
        receipt_path = root / (arm + '_recursive_receipt.json')
        assert hashlib.sha256(receipt_path.read_bytes()).hexdigest() == result['receipts'][arm]
        receipt = json.loads(receipt_path.read_text())
        path = root / 'recursive' / arm / (name + '.json')
        expected = next(row['sha256'] for row in receipt['sequences'] if row['sequence'] == name)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        rows = json.loads(path.read_text())['rows']
        boxes = np.asarray([row['bbox'] for row in rows], dtype=np.float64)
        values[arm] = ious(boxes, gt)
        differences = np.flatnonzero(np.max(np.abs(boxes - base_boxes), axis=1) > 1e-6)
        changed_choices = [row['frame'] for row in rows if row['chosen'] != 0]
        first = int(differences[0]) if len(differences) else None
        entry = dict(first_nondefault_frame=changed_choices[0] if changed_choices else None,
                     first_bbox_difference_frame=first, nondefault_frames=len(changed_choices),
                     none_frames=sum(row['none'] for row in rows), H10_spans=spans(values[arm]))
        if first is not None:
            following = [span['start'] for span in entry['H10_spans'] if span['start'] >= first]
            entry['first_H10_start_at_or_after_divergence'] = min(following) if following else None
            entry['delay_to_first_H10_start'] = min(following) - first if following else None
            entry['first_difference'] = dict(chosen=rows[first]['chosen'], none=rows[first]['none'],
                score=rows[first]['score'], gt_valid=bool(np.isfinite(values[arm][first])),
                native_iou=float(values['native'][first]) if np.isfinite(values['native'][first]) else None,
                arm_iou=float(values[arm][first]) if np.isfinite(values[arm][first]) else None)
        entry['framewise_disagreement'] = dict(
            native_correct_arm_severe=int(((values['native'] >= .5) & (values[arm] <= .1)).sum()),
            native_severe_arm_correct=int(((values['native'] <= .1) & (values[arm] >= .5)).sum()))
        record[arm] = entry
    record['native'] = dict(H10_spans=spans(values['native']))
    for arm, data in values.items():
        metrics = result['per_sequence'][arm][name]
        assert abs(float(np.nansum(data)) - metrics['iou_sum']) < 1e-8
        assert len(record[arm]['H10_spans']) == metrics['failure_episodes']
    records[name] = record
output = dict(status='posthoc_diagnostic_complete', source_result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
    frame_index='zero based; initialization is frame 0', bbox_difference_tolerance=1e-6,
    subsequent_gt_used_posthoc=True, inference_or_training_changed=False,
    scope='All 22 sequences, all four completed arms; divergence timing and later H10 do not establish causation',
    sequences=records)
target = root / 'divergence_diagnostic.json'
assert not target.exists()
target.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
print(json.dumps(dict(path=str(target), sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    sequence_arm_pairs=len(cases)*len(spec['variants']),
    first_difference_matches_first_nondefault=sum(data['first_bbox_difference_frame'] == data['first_nondefault_frame']
        for row in records.values() for arm, data in row.items() if arm != 'native')), indent=2))
