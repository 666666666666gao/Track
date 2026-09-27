"""Read-only Top-10 NMS capacity on the 57 sealed M89 VOT failure onsets."""

import hashlib
import json
from pathlib import Path

import numpy as np
from vot.region import RegionType, calculate_overlaps
from vot.region.shapes import Rectangle
from vot.workspace import Workspace

from analyze import ARMS, MASTER, ROOT, dense_boxes


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def top10(response):
    suppressed = response.reshape(16, 16).copy()
    indexes = []
    for _ in range(10):
        index = int(suppressed.argmax())
        row, column = divmod(index, 16)
        indexes.append(index)
        suppressed[max(0, row - 1):min(16, row + 2),
                   max(0, column - 1):min(16, column + 2)] = -np.inf
    return indexes


def main():
    spec = read(ROOT / 'spec.json')
    analysis = read(ROOT / 'analysis.json')
    assert analysis['status'] == 'complete'
    assert analysis['spec_sha256'] == sha(ROOT / 'spec.json')
    onsets = {row['anchor_key']: row for row in analysis['rows']
              if row['relative_to_failure'] == 0}
    assert len(onsets) == len(spec['cases']) == 57
    receipts = {shard: read(ROOT / 'predictions' / str(shard) / 'receipt.json')
                for shard in (0, 1)}
    items = {item['anchor_key']: item for receipt in receipts.values()
             for item in receipt['cases']}
    assert set(items) == set(onsets)
    workspace = Workspace.load(str(MASTER))
    sequences = {sequence.name: sequence for sequence in
                 workspace.stack.experiments['baseline'].transform(workspace.dataset)}
    rows = []
    for case in spec['cases']:
        key = case['anchor_key']
        folder = ROOT / 'predictions' / str(case['shard'])
        assert sha(folder / (key + '.npz')) == items[key]['dense_sha256']
        assert sha(folder / (key + '.json')) == items[key]['json_sha256']
        observed = read(folder / (key + '.json'))['rows']
        index = next(i for i, row in enumerate(observed)
                     if row['run_index'] == case['failure_start'])
        row = observed[index]
        sequence = sequences[case['sequence']]
        gt = sequence.groundtruth(row['source_frame'])
        ignore = sequence.object('_ignore', row['source_frame'])
        assert not gt.is_empty()
        result = dict(anchor_key=key, center_inside=onsets[key]['center_inside'],
                      arms={})
        with np.load(folder / (key + '.npz'), allow_pickle=False) as archive:
            window = archive['window'].reshape(256)
            for arm in ARMS:
                values = archive[arm][index]
                response = values[0].reshape(256) * window
                indexes = top10(response)
                assert indexes[0] == row['variants'][arm]['hann_peak']
                boxes = dense_boxes(values, row)[indexes]
                overlaps = np.asarray(calculate_overlaps(
                    [Rectangle(*box) for box in boxes], [gt] * 10,
                    sequence.size, ignore=[ignore] * 10))
                first_correct = np.flatnonzero(overlaps >= .5)
                full = onsets[key]['capacity'][arm]
                assert (len(first_correct) > 0) <= (full['correct_count'] > 0)
                result['arms'][arm] = dict(
                    top10_best_iou=float(overlaps.max()),
                    top10_correct=bool(len(first_correct)),
                    first_correct_rank=None if not len(first_correct)
                    else int(first_correct[0] + 1),
                    full256_correct=bool(full['correct_count']))
        rows.append(result)
    summaries = {}
    for arm in ARMS:
        summaries[arm] = dict(
            full256_correct=sum(row['arms'][arm]['full256_correct'] for row in rows),
            top10_correct=sum(row['arms'][arm]['top10_correct'] for row in rows),
            top10_correct_center_inside=sum(
                row['center_inside'] and row['arms'][arm]['top10_correct']
                for row in rows),
            first_correct_ranks={
                str(rank): sum(row['arms'][arm]['first_correct_rank'] == rank
                               for row in rows)
                for rank in range(1, 11)})
    assert summaries['category']['full256_correct'] == 36
    report = dict(status='read_only_complete',
                  scope='57 sealed M89 Candidate VOT failure onsets, fixed Category histories; Top-10 NMS capacity is GT posthoc and not an executed recovery or a Train selector result.',
                  spec_sha256=sha(ROOT / 'spec.json'),
                  analysis_sha256=sha(ROOT / 'analysis.json'),
                  summaries=summaries, rows=rows)
    path = ROOT / 'top10_onset_capacity.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
