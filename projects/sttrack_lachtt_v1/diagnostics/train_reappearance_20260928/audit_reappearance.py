"""Inventory GT-invalid-to-valid events in the sealed DepthTrack Train 152 list."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np


SPEC = Path('/root/autodl-tmp/sttrack_full152_paired_20260925/M82/training_spec.json')
OUT = Path('/root/autodl-tmp/sttrack_train_reappearance_20260928')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def search_crop(previous):
    x, y, width, height = previous
    side = math.ceil(math.sqrt(width * height) * 4)
    return (round(x + width / 2 - side / 2),
            round(y + height / 2 - side / 2), side)


def main():
    spec = json.loads(SPEC.read_text())
    assert len(spec['sequence_order']) == 152
    # prepare_full152.py appends the former development 22 after the first 130.
    prior_development = {row['sequence'] for row in spec['sequence_order'][130:]}
    assert len(prior_development) == 22
    rows = []
    sequence_counts = []
    total_invalid = 0
    for row in spec['sequence_order']:
        name = row['sequence']
        gt_path = Path(spec['dataset_root']) / name / 'groundtruth.txt'
        assert sha(gt_path) == row['groundtruth_sha256']
        gt = np.loadtxt(gt_path, delimiter=',').reshape(-1, 4)
        if name == 'toy07_indoor_320':
            assert len(gt) == 1406 and row['rgb_frames'] == 1367
            gt = gt[:row['rgb_frames']]
        assert len(gt) == row['rgb_frames']
        valid = np.isfinite(gt).all(axis=1) & (gt[:, 2:] > 0).all(axis=1)
        assert valid[0]
        total_invalid += int((~valid).sum())
        sequence_events = 0
        i = 1
        while i < len(gt):
            if valid[i]:
                i += 1
                continue
            start = i
            while i < len(gt) and not valid[i]:
                i += 1
            if i == len(gt):
                break
            previous = gt[start - 1]
            current = gt[i]
            x1, y1, side = search_crop(previous)
            gx, gy, width, height = current
            cx, cy = gx + width / 2, gy + height / 2
            center_in = x1 <= cx < x1 + side and y1 <= cy < y1 + side
            full_in = x1 <= gx and y1 <= gy and gx + width <= x1 + side and gy + height <= y1 + side
            rows.append(dict(sequence=name, prior_development_22=name in prior_development,
                             invalid_start=start,
                             invalid_end=i, invalid_frames=i - start,
                             previous_valid_frame=start - 1, return_frame=i,
                             center_in_last_gt_crop=bool(center_in),
                             full_box_in_last_gt_crop=bool(full_in),
                             last_gt_box=previous.tolist(), return_gt_box=current.tolist()))
            sequence_events += 1
            i += 1
        sequence_counts.append(dict(sequence=name, prior_development_22=name in prior_development, frames=len(gt),
                                    invalid_frames=int((~valid).sum()),
                                    return_events=sequence_events))
    assert sum(row['return_events'] for row in sequence_counts) == len(rows)
    buckets = {}
    for label, selected in (
        ('1_to_9', [e for e in rows if e['invalid_frames'] < 10]),
        ('10_to_49', [e for e in rows if 10 <= e['invalid_frames'] < 50]),
        ('50_plus', [e for e in rows if e['invalid_frames'] >= 50]),
    ):
        buckets[label] = dict(events=len(selected),
                              center_outside=sum(not e['center_in_last_gt_crop'] for e in selected),
                              full_box_outside=sum(not e['full_box_in_last_gt_crop'] for e in selected))
    report = dict(status='train_only_gt_event_inventory_complete',
                  training_spec_sha256=sha(SPEC), sequences=len(sequence_counts),
                  frames=sum(row['frames'] for row in sequence_counts),
                  gt_invalid_frames=total_invalid, return_events=len(rows),
                  sequences_with_return_events=sum(row['return_events'] > 0 for row in sequence_counts),
                  center_outside_last_gt_crop=sum(not e['center_in_last_gt_crop'] for e in rows),
                  full_box_outside_last_gt_crop=sum(not e['full_box_in_last_gt_crop'] for e in rows),
                  invalid_gap_buckets=buckets, sequence_counts=sequence_counts, events=rows,
                  limitation='GT invalidity does not prove physical absence. The crop is centered '
                             'on the last valid GT box held fixed through the interval; it is '
                             'a train-only geometric reference, not a model trajectory or a '
                             'deployable recovery result.')
    OUT.mkdir(exist_ok=True)
    destination = OUT / 'train_reappearance_inventory.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('sequences', 'frames', 'gt_invalid_frames',
        'return_events', 'sequences_with_return_events', 'center_outside_last_gt_crop',
        'full_box_outside_last_gt_crop', 'invalid_gap_buckets')}))


if __name__ == '__main__':
    main()
