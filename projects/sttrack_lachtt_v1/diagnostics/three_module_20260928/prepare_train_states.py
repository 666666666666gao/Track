"""Prepare Train-only native candidate-state collection without leaking event GT to inference."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def overlap(a, b):
    width = max(0., min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    height = max(0., min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    area = width * height
    return area / (a[2] * a[3] + b[2] * b[3] - area)


def spaced(values, count):
    if not values:
        return []
    indices = np.linspace(0, len(values) - 1, min(count, len(values))).round().astype(int)
    return [values[index] for index in sorted(set(indices.tolist()))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full152-spec', type=Path, required=True)
    parser.add_argument('--former-m82-spec', type=Path, required=True)
    parser.add_argument('--trace-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    spec = json.loads(args.full152_spec.read_text())
    former = json.loads(args.former_m82_spec.read_text())
    sequence_order = spec['sequence_order']
    assert len(sequence_order) == 152
    assert len({row['sequence'] for row in sequence_order}) == 152
    assert [row['sequence'] for row in sequence_order[:130]] == [row['sequence'] for row in former['sequence_order']]
    assert {row['sequence'] for row in sequence_order[130:]} == set(former['development_sequences'])
    by_name = {row['sequence']: row for row in sequence_order}

    trace_paths = [args.trace_root / f'shard{index}.json' for index in (0, 1)]
    traces = defaultdict(list)
    for path in trace_paths:
        data = json.loads(path.read_text())
        assert data['complete']
        for row in data['rows']:
            if row['sequence'] in by_name:
                traces[row['sequence']].append(dict(frame_index=row['frame_index'],
                    bbox=row['public_bbox'], score=row['public_score']))
    assert set(traces) == set(by_name)

    plans, labels = [], {}
    load = [0, 0]
    strata = Counter()
    for index, entry in enumerate(sequence_order):
        sequence = entry['sequence']
        split = 'fit' if index < 130 else 'development'
        rows = sorted(traces[sequence], key=lambda row: row['frame_index'])
        frames = entry['rgb_frames']
        assert [row['frame_index'] for row in rows] == list(range(frames))
        assert rows[0]['bbox'] == entry['first_box']
        gt_path = Path(spec['dataset_root']) / sequence / 'groundtruth.txt'
        assert sha(gt_path) == entry['groundtruth_sha256']
        gt = np.loadtxt(gt_path, delimiter=',').reshape(-1, 4)
        assert len(gt) >= frames
        gt = gt[:frames]
        valid = np.isfinite(gt).all(axis=1) & (gt[:, 2] > 0) & (gt[:, 3] > 0)
        iou = np.full(frames, np.nan)
        for frame in range(1, frames):
            if valid[frame]:
                iou[frame] = overlap(rows[frame]['bbox'], gt[frame])
        low = valid & (iou <= .1)
        edges = np.diff(np.r_[False, low, False].astype(int))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        onsets = [int(start) for start, end in zip(starts, ends)
                  if end - start >= 10 and 10 <= start < frames - 4]
        groups = dict(
            healthy=spaced([frame for frame in range(10, frames - 4) if iou[frame] >= .5], 12),
            intermediate=spaced([frame for frame in range(10, frames - 4) if .1 < iou[frame] < .5], 4),
            late_low=spaced([frame for frame in range(10, frames - 4) if low[frame]], 2),
            unavailable=spaced([frame for frame in range(10, frames - 4) if not valid[frame]], 2),
            transition=sorted({onset + delta for onset in spaced(onsets, 6)
                               for delta in (-2, 0, 2) if 2 <= onset + delta < frames}),
        )
        selected = sorted({frame for group in groups.values() for frame in group})
        assert selected
        shard = load.index(min(load))
        load[shard] += selected[-1] + 1
        plans.append(dict(sequence=sequence, split=split, shard=shard,
                          frames=frames, event_frames=selected, init_bbox=entry['first_box'],
                          expected_rows=rows[:selected[-1] + 1]))
        for frame in selected:
            tags = [name for name, values in groups.items() if frame in values]
            strata.update(tags)
            labels[f'{sequence}@{frame}'] = dict(sequence=sequence, split=split,
                frame=frame, strata=tags,
                current=gt[frame].tolist() if valid[frame] else None,
                previous=gt[frame - 1].tolist() if valid[frame - 1] else None)

    inputs = args.output / 'inference_inputs.json'
    targets = args.output / 'training_labels.json'
    inputs.write_text(json.dumps(plans) + '\n')
    targets.write_text(json.dumps(labels) + '\n')
    summary = dict(status='prepared_cpu_only', source_spec_sha256=sha(args.full152_spec),
                   former_m82_spec_sha256=sha(args.former_m82_spec),
                   source_trace_sha256={path.name: sha(path) for path in trace_paths},
                   inference_inputs_sha256=sha(inputs), training_labels_sha256=sha(targets),
                   sequence_count=len(plans), split_sequences=dict(Counter(row['split'] for row in plans)),
                   event_count=len(labels), split_events=dict(Counter(row['split'] for row in labels.values())),
                   event_tags=dict(strata), planned_shard_frame_prefixes=load,
                   labels_separate_from_inference=True,
                   no_candidate_features_collected=True, no_model_training=True)
    (args.output / 'preparation.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
