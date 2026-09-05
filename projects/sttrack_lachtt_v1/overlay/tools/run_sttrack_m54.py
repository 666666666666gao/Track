"""Run the one frozen M54 reader over all development trajectories, then analyze."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from tools.sttrack_m54_common import check_sources, parameters, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--analyze', action='store_true')
    args = parser.parse_args()
    root = args.root
    plan, parent, spec = check_sources(root)
    training = json.loads((root / 'training_result.json').read_text())
    assert (root / 'training.exit').read_text().strip() == '0'
    assert training['status'] == 'complete' and training['spec_sha256'] == sha(root / 'spec.json')
    checkpoint = root / 'reader_final.pth'
    assert sha(checkpoint) == training['checkpoint_sha256']
    cases = [c for c in json.loads((parent / 'inference_inputs.json').read_text()) if c['split'] == 'development']
    names = {c['sequence'] for c in cases}
    assert len(cases) == len(names) == 22
    if args.analyze:
        from tools.analyze_sttrack_m42_recursive import statistics
        assert (root / 'recursive.exit').read_text().strip() == '0'
        receipt = json.loads((root / 'recursive_receipt.json').read_text())
        assert receipt['status'] == 'complete' and receipt['spec_sha256'] == sha(root / 'spec.json')
        assert receipt['checkpoint_sha256'] == training['checkpoint_sha256']
        assert {r['sequence'] for r in receipt['sequences']} == names
        predicted = {}
        for item in receipt['sequences']:
            path = root / 'recursive' / (item['sequence'] + '.json')
            assert sha(path) == item['sha256']
            data = json.loads(path.read_text())
            rows = data['rows']
            case = next(c for c in cases if c['sequence'] == item['sequence'])
            assert len(rows) == case['frames'] and [r['frame'] for r in rows] == list(range(case['frames']))
            assert rows[0]['bbox'] == case['init_bbox']
            boxes = np.asarray([r['bbox'] for r in rows])
            assert np.isfinite(boxes).all() and (boxes[:, 2:] > 0).all()
            assert all(r['choice'] in (0, 1) and np.isfinite(r['score']) for r in rows)
            predicted[case['sequence']] = rows
        baseline = defaultdict(list)
        for path, digest in spec['baseline_trace_sha256'].items():
            assert sha(path) == digest
            for row in json.loads(Path(path).read_text())['rows']:
                if row['sequence'] in names:
                    baseline[row['sequence']].append(row)
        # Both result families are verified before loading development GT.
        per = {'default': {}, 'reader': {}}
        for case in cases:
            name = case['sequence']
            original = sorted(baseline[name], key=lambda x: x['frame_index'])
            assert len(original) == case['frames']
            gt = np.loadtxt(Path(spec['dataset_root']) / name / 'groundtruth.txt', delimiter=',')
            per['default'][name] = statistics([r['public_bbox'] for r in original], gt)
            per['reader'][name] = statistics([r['bbox'] for r in predicted[name]], gt)
        aggregates = {}
        for arm, values in per.items():
            totals = {k: sum(v[k] for v in values.values()) for k in ['valid_frames', 'iou_sum', 'low_iou_frames', 'failure_episodes']}
            totals['mean_iou'] = totals['iou_sum'] / totals['valid_frames']
            totals['macro_sequence_mean_iou'] = float(np.mean([v['mean_iou'] for v in values.values()]))
            aggregates[arm] = totals
        base, method = aggregates['default'], aggregates['reader']
        positive = sum(per['reader'][n]['mean_iou'] > per['default'][n]['mean_iou'] for n in names)
        broken = sorted(n for n in names if per['default'][n]['failure_episodes'] == 0 and per['reader'][n]['failure_episodes'] > 0)
        rule = plan['recursive_gate']
        gates = dict(mean_iou=method['mean_iou'] >= base['mean_iou'] + rule['mean_iou_gain_at_least'],
            fewer_low_frames=method['low_iou_frames'] < base['low_iou_frames'],
            no_episode_increase=method['failure_episodes'] <= base['failure_episodes'],
            sequence_coverage=positive >= rule['positive_sequences_at_least'], successful_sequence_protection=not broken)
        result = dict(status='complete', spec_sha256=sha(root / 'spec.json'), checkpoint_sha256=sha(checkpoint),
            training_result_sha256=sha(root / 'training_result.json'), recursive_receipt_sha256=sha(root / 'recursive_receipt.json'),
            aggregates=aggregates, per_sequence=per, gates=gates, primary_pass=all(gates.values()),
            positive_sequences=positive, new_failure_sequences=broken, public_evaluation=False,
            scope='Full 22-sequence DepthTrack Train development recursion; repeatedly used development split',
            next='Freeze low22 comparison' if all(gates.values()) else 'Stop this frozen reader; no public evaluation')
        check_sources(root)
        (root / 'recursive_result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: v for k, v in result.items() if k != 'per_sequence'}, indent=2), flush=True)
        return
    from lib.models.sttrack.lachtt_template_reader import TemplateReader
    from lib.test.tracker.sttrack_template_reader import STTrackTemplateReader
    from lib.train.dataset.depth_utils import get_rgbd_frame
    torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location='cpu')
    assert saved['spec_sha256'] == sha(root / 'spec.json') and saved['base_checkpoint_sha256'] == spec['checkpoint_sha256']
    reader = TemplateReader()
    reader.load_state_dict(saved['model'], strict=True)
    tracker = STTrackTemplateReader(parameters(spec), reader)
    outdir = root / 'recursive'
    outdir.mkdir()
    receipts = []
    started = time.time()
    for case in cases:
        folder = Path(spec['dataset_root']) / case['sequence']

        def image_at(frame):
            return get_rgbd_frame(str(folder / 'color' / f'{frame+1:08d}.jpg'),
                str(folder / 'depth' / f'{frame+1:08d}.png'), dtype='rgbcolormap', depth_clip=True)

        tracker.initialize(image_at(0), dict(init_bbox=list(case['init_bbox'])))
        rows = [dict(frame=0, bbox=list(case['init_bbox']), score=1., choice=0)]
        for frame in range(1, case['frames']):
            out = tracker.track(image_at(frame))
            rows.append(dict(frame=frame, bbox=out['target_bbox'], score=out['best_score'], choice=out['template_read']))
        path = outdir / (case['sequence'] + '.json')
        path.write_text(json.dumps(dict(sequence=case['sequence'], rows=rows), allow_nan=False) + '\n')
        item = dict(sequence=case['sequence'], frames=len(rows), initial_reads=sum(r['choice'] == 1 for r in rows),
            sha256=sha(path), elapsed_seconds=time.time() - started)
        receipts.append(item)
        print(json.dumps(item), flush=True)
    check_sources(root)
    assert sha(checkpoint) == training['checkpoint_sha256']
    (root / 'recursive_receipt.json').write_text(json.dumps(dict(status='complete', sequences=receipts,
        spec_sha256=sha(root / 'spec.json'), checkpoint_sha256=sha(checkpoint), source_unchanged=True,
        subsequent_gt_opened=False, elapsed_seconds=time.time() - started), indent=2) + '\n')


if __name__ == '__main__':
    main()
