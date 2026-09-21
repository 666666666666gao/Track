"""Descriptive CPU analysis of sealed three-content recursions and their M84 control.

This does not run a tracker, change acceptance gates, or establish causal effects.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def intervals(mask):
    start = None
    found = []
    for i, value in enumerate(mask + [False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i - start >= 10:
                found.append((start, i))
            start = None
    return found


def overlap(box, gt):
    assert len(box) == len(gt) == 4
    assert all(math.isfinite(x) for x in box) and box[2] > 0 and box[3] > 0
    if not all(math.isfinite(x) for x in gt) or gt[2] <= 0 or gt[3] <= 0:
        return None
    w = max(0, min(box[0]+box[2], gt[0]+gt[2])-max(box[0], gt[0]))
    h = max(0, min(box[1]+box[3], gt[1]+gt[3])-max(box[1], gt[1]))
    area = w*h
    return area/(box[2]*box[3]+gt[2]*gt[3]-area)


def statistics(values):
    valid = [v for v in values if v is not None]
    return dict(valid_frames=len(valid), iou_sum=sum(valid), mean_iou=sum(valid)/len(valid),
                low_iou_frames=sum(v <= .1 for v in valid),
                failure_episodes=len(intervals([v is not None and v <= .1 for v in values])))


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--control-rows', type=Path, required=True)
    parser.add_argument('--control-receipt', type=Path, required=True)
    parser.add_argument('--control-result', type=Path, required=True)
    args = parser.parse_args()
    e, out = args.evidence, args.output
    assert not out.exists(), 'Use a new output directory; preserve previous analysis.'
    result = read(e/'recursive_result.json')
    spec = read(e/'recursive_spec.json')
    control = read(args.control_result)
    assert result['status'] == control['status'] == 'complete_recursive_development'
    arms = ['category', 'category_empty', 'category_swapped']
    streams = {a: {} for a in arms + ['M84_control']}
    inputs = {'recursive_result': sha(e/'recursive_result.json'),
              'recursive_spec': sha(e/'recursive_spec.json'),
              'control_result': sha(args.control_result)}
    # Verify every sealed prediction set before opening any subsequent GT.
    for arm in streams:
        if arm == 'M84_control':
            receipt_path, folder = args.control_receipt, args.control_rows
            expected = control['receipts']['category']
        else:
            receipt_path, folder = e/(arm+'_recursive_receipt.json'), e/'recursive'/arm
            expected = result['receipts'][arm]
        assert sha(receipt_path) == expected
        receipt = read(receipt_path)
        assert receipt['status'] == 'complete' and receipt['total_frames'] == 33130
        assert len(receipt['sequences']) == len(spec['cases']) == 22
        for case, item in zip(spec['cases'], receipt['sequences']):
            name = case['sequence']
            assert item['sequence'] == name and item['frames'] == case['frames']
            path = folder/(name+'.json')
            assert sha(path) == item['sha256']
            rows = read(path)['rows']
            assert [r['frame'] for r in rows] == list(range(case['frames']))
            streams[arm][name] = rows
        inputs[arm+'_receipt'] = sha(receipt_path)
    per, events, writes = [], [], []
    metrics = {a: {} for a in streams}
    for case in spec['cases']:
        name = case['sequence']
        gt_path = e/'development_gt'/(name+'.txt')
        assert sha(gt_path) == case['gt_sha256']
        gt = [[float(v) for v in line.split(',')] for line in gt_path.read_text().splitlines()]
        assert len(gt) == case['frames']
        values = {a: [None] + [overlap(r['bbox'], gt[i]) for i, r in enumerate(data[name]) if i > 0]
                  for a, data in streams.items()}
        row = {'sequence': name}
        for arm in streams:
            measured = statistics(values[arm])
            expected = control['per_sequence']['category'][name] if arm == 'M84_control' else result['per_sequence'][arm][name]
            for key, val in measured.items():
                assert abs(val-expected[key]) < 1e-8, (arm, name, key, val, expected[key])
                row[arm+'_'+key] = val
            metrics[arm][name] = measured
            for i, pred in enumerate(streams[arm][name]):
                if i and i % 50 == 0 and pred['score'] > .75:
                    val = values[arm][i]
                    quality = 'invalid_gt' if val is None else 'correct' if val >= .5 else 'severe' if val <= .1 else 'intermediate'
                    writes.append(dict(sequence=name, arm=arm, frame=i, score=pred['score'], iou=val, quality=quality))
        for other in ['M84_control', 'category_empty', 'category_swapped']:
            row['delta_mean_vs_'+other] = row['category_mean_iou']-row[other+'_mean_iou']
            row['first_bbox_difference_vs_'+other] = next((i for i in range(1, len(gt))
                if streams['category'][name][i]['bbox'] != streams[other][name][i]['bbox']), None)
            for direction in ['damage', 'improvement']:
                bad, good = ('category', other) if direction == 'damage' else (other, 'category')
                mask = [b is not None and g is not None and b <= .1 and g >= .5
                        for b, g in zip(values[bad], values[good])]
                for start, end in intervals(mask):
                    events.append(dict(sequence=name, reference=other, direction=direction,
                                       start=start, end_exclusive=end, frames=end-start))
        per.append(row)
    aggregates = {}
    for arm, seqs in metrics.items():
        total = {k: sum(v[k] for v in seqs.values()) for k in ['valid_frames', 'iou_sum', 'low_iou_frames', 'failure_episodes']}
        assert total['valid_frames'] == 28897
        total['mean_iou'] = total['iou_sum']/total['valid_frames']
        total['macro_sequence_mean_iou'] = sum(v['mean_iou'] for v in seqs.values())/22
        expected = control['aggregates']['category'] if arm == 'M84_control' else result['aggregates'][arm]
        for key, val in total.items():
            assert abs(val-expected[key]) < 1e-8, (arm, key)
        aggregates[arm] = total
    out.mkdir(parents=True)
    write_csv(out/'per_sequence.csv', per, list(per[0]))
    write_csv(out/'strict_intervals.csv', events, ['sequence', 'reference', 'direction', 'start', 'end_exclusive', 'frames'])
    write_csv(out/'template_write_events.csv', writes, ['sequence', 'arm', 'frame', 'score', 'iou', 'quality'])
    report = dict(scope='Posthoc descriptive analysis of all22 repeated Train development sequences; no inference or gate changes.',
        limitations='First bbox difference is not first identity error. Strict intervals and write counts compare different recursive histories, not causal interventions. Automatic original captions are not semantic ground truth.',
        inputs_sha256=inputs, source_sha256=sha(Path(__file__)), aggregates=aggregates,
        strict_intervals={other: {direction: dict(segments=sum(x['reference']==other and x['direction']==direction for x in events),
            frames=sum(x['frames'] for x in events if x['reference']==other and x['direction']==direction))
            for direction in ['damage', 'improvement']} for other in ['M84_control', 'category_empty', 'category_swapped']},
        template_writes={arm: {quality: sum(x['arm']==arm and x['quality']==quality for x in writes)
            for quality in ['correct', 'severe', 'intermediate', 'invalid_gt']} for arm in streams},
        frozen_gates_as_recorded=result['gates'], gates_independently_recomputed=False,
        artifacts_sha256={n: sha(out/n) for n in ['per_sequence.csv', 'strict_intervals.csv', 'template_write_events.csv']})
    (out/'analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(out), 'aggregates': aggregates, 'strict_intervals': report['strict_intervals']}))


if __name__ == '__main__':
    main()
