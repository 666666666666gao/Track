"""Frozen paired full-trajectory Train-development evaluation for M55."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def binding(root):
    frozen = json.loads((root / 'recursive_spec.json').read_text())
    assert sha(__file__) == frozen['runner_sha256']
    assert sha(root / 'training_spec.json') == frozen['training_spec_sha256']
    train = json.loads((root / 'training_spec.json').read_text())
    assert sha(root / 'preparation.json') == train['preparation_sha256']
    preparation = json.loads((root / 'preparation.json').read_text())
    assert sha(train['fitting_manifest']) == train['fitting_manifest_sha256']
    cases = json.loads((root / 'recursive_inputs.json').read_text())
    assert sha(root / 'recursive_inputs.json') == frozen['inputs_sha256']
    assert [x['sequence'] for x in cases] == sorted(train['development_sequences'])
    assert len(cases) == 22 and not set(train['fit_sequences']) & set(train['development_sequences'])
    results = {}
    for arm in ['control', 'clone']:
        assert (root / ('train_' + arm + '.exit')).read_text().strip() == '0'
        assert (root / ('train_' + arm + '_controller.exit')).read_text().strip() == '0'
        result = json.loads((root / 'training' / arm / 'result.json').read_text())
        assert result['status'] == 'complete_train' and result['variant'] == arm
        assert result['spec_sha256'] == frozen['training_spec_sha256']
        assert result['optimizer_steps'] == 3840 and result['clips'] == 30720
        assert result['microbatches'] == 15360 and result['search_frames'] == 122880
        assert len(result['epochs']) == 15 and result['epochs'][-1]['epoch'] == 15
        assert sha(result['weight_path']) == result['weight_sha256']
        for name, digest in preparation['code_sha256'][arm].items():
            assert sha(root / 'code' / arm / name) == digest, name
        for filename, field in [('batches.jsonl', 'batches_sha256'), ('steps.jsonl', 'steps_sha256')]:
            assert sha(root / 'training' / arm / filename) == result[field]
        results[arm] = result
    assert results['control']['batches_sha256'] == results['clone']['batches_sha256']
    assert results['control']['data_stream_sha256'] == results['clone']['data_stream_sha256']
    initial = [sha(root / 'training' / arm / 'initial_state_sha256.json') for arm in ['control', 'clone']]
    assert initial[0] == initial[1]
    return frozen, train, cases, results


def run(root, arm):
    frozen, train, cases, results = binding(root)
    code = root / 'code' / arm
    sys.path.insert(0, str(code))
    import numpy as np
    import torch
    from lib.config.sttrack.config import cfg, update_config_from_file
    from lib.test.tracker.sttrack import STTrack
    from lib.train.dataset.depth_utils import get_rgbd_frame
    torch.set_num_threads(1)
    update_config_from_file(str(code / 'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    assert cfg.DATA.TEMPLATE.NUMBER == 2
    assert cfg.TEST.UPDATE_INTERVALS == 50 and cfg.TEST.UPDATE_THRESHOLD == .75
    params = SimpleNamespace(cfg=cfg, checkpoint=results[arm]['weight_path'],
        template_factor=2., template_size=128, search_factor=4., search_size=256,
        save_all_boxes=False, debug=0)
    tracker = STTrack(params)
    directory = root / 'recursive' / arm
    directory.mkdir(parents=True)
    started = time.time()
    receipt = []
    for case in cases:
        folder = Path(train['dataset_root']) / case['sequence']

        def image_at(frame):
            return get_rgbd_frame(str(folder / 'color' / ('%08d.jpg' % (frame + 1))),
                str(folder / 'depth' / ('%08d.png' % (frame + 1))), dtype='rgbcolormap', depth_clip=True)

        tracker.initialize(image_at(0), dict(init_bbox=list(case['init_bbox'])))
        rows = [dict(frame=0, bbox=list(case['init_bbox']), score=1.)]
        for frame in range(1, case['frames']):
            out = tracker.track(image_at(frame))
            box = [float(x) for x in out['target_bbox']]
            score = float(out['best_score'])
            assert np.isfinite(box).all() and min(box[2:]) > 0 and np.isfinite(score)
            rows.append(dict(frame=frame, bbox=box, score=score))
        path = directory / (case['sequence'] + '.json')
        write(path, dict(sequence=case['sequence'], rows=rows))
        item = dict(sequence=case['sequence'], frames=len(rows), sha256=sha(path),
                    elapsed_seconds=time.time() - started)
        receipt.append(item)
        print(json.dumps(item), flush=True)
    binding(root)
    write(root / (arm + '_recursive_receipt.json'), dict(status='complete', variant=arm,
        sequences=receipt, recursive_spec_sha256=sha(root / 'recursive_spec.json'),
        training_result_sha256=sha(root / 'training' / arm / 'result.json'),
        weight_sha256=results[arm]['weight_sha256'], subsequent_gt_opened=False,
        frames=sum(x['frames'] for x in receipt), elapsed_seconds=time.time() - started))


def analyze(root):
    frozen, train, cases, training = binding(root)
    repository = Path(frozen['metric_repository'])
    assert sha(repository / 'tools/analyze_sttrack_m42_recursive.py') == frozen['metric_sha256']
    sys.path.insert(0, str(repository))
    import numpy as np
    from tools.analyze_sttrack_m42_recursive import statistics
    names = {case['sequence'] for case in cases}
    predicted = {'default': {}, 'control': {}, 'clone': {}}
    baseline = defaultdict(list)
    for path, digest in frozen['baseline_trace_sha256'].items():
        assert sha(path) == digest
        for row in json.loads(Path(path).read_text())['rows']:
            if row['sequence'] in names:
                baseline[row['sequence']].append(row)
    for case in cases:
        rows = sorted(baseline[case['sequence']], key=lambda x: x['frame_index'])
        assert [x['frame_index'] for x in rows] == list(range(case['frames']))
        predicted['default'][case['sequence']] = [x['public_bbox'] for x in rows]
    for arm in ['control', 'clone']:
        assert (root / (arm + '_recursive.exit')).read_text().strip() == '0'
        receipt = json.loads((root / (arm + '_recursive_receipt.json')).read_text())
        assert receipt['status'] == 'complete' and receipt['variant'] == arm
        assert receipt['recursive_spec_sha256'] == sha(root / 'recursive_spec.json')
        assert receipt['weight_sha256'] == training[arm]['weight_sha256']
        assert receipt['training_result_sha256'] == sha(root / 'training' / arm / 'result.json')
        assert len(receipt['sequences']) == len(names) and {x['sequence'] for x in receipt['sequences']} == names
        for item in receipt['sequences']:
            path = root / 'recursive' / arm / (item['sequence'] + '.json')
            assert sha(path) == item['sha256']
            data = json.loads(path.read_text())
            assert data['sequence'] == item['sequence']
            case = next(c for c in cases if c['sequence'] == item['sequence'])
            rows = data['rows']
            assert len(rows) == item['frames'] == case['frames']
            assert [x['frame'] for x in rows] == list(range(case['frames']))
            assert rows[0]['bbox'] == case['init_bbox']
            boxes = np.asarray([x['bbox'] for x in rows])
            assert np.isfinite(boxes).all() and (boxes[:, 2:] > 0).all()
            predicted[arm][case['sequence']] = boxes
    # Only after all three trajectory families are sealed and verified, open subsequent GT.
    per = {arm: {} for arm in predicted}
    for case in cases:
        sequence = case['sequence']
        path = Path(train['dataset_root']) / sequence / 'groundtruth.txt'
        assert sha(path) == frozen['development_gt_sha256'][sequence]
        gt = np.loadtxt(path, delimiter=',')
        assert len(gt) == case['frames']
        for arm, boxes in predicted.items():
            per[arm][sequence] = statistics(boxes[sequence], gt)
    aggregates = {}
    for arm, values in per.items():
        totals = {k: sum(v[k] for v in values.values())
                  for k in ['valid_frames', 'iou_sum', 'low_iou_frames', 'failure_episodes']}
        totals['mean_iou'] = totals['iou_sum'] / totals['valid_frames']
        totals['macro_sequence_mean_iou'] = float(np.mean([v['mean_iou'] for v in values.values()]))
        aggregates[arm] = totals
    baseline, control, clone = (aggregates[x] for x in ['default', 'control', 'clone'])
    rule = train['recursive_validation']
    broken = [n for n in sorted(names) if per['default'][n]['failure_episodes'] == 0
              and per['clone'][n]['failure_episodes'] > 0]
    gates = dict(mean_vs_native=clone['mean_iou'] >= baseline['mean_iou'] + rule['clone_required_gain_vs_native'],
        mean_vs_control=clone['mean_iou'] >= control['mean_iou'] + rule['clone_required_gain_vs_control'],
        low_frames=clone['low_iou_frames'] <= min(baseline['low_iou_frames'], control['low_iou_frames']),
        H10=clone['failure_episodes'] <= min(baseline['failure_episodes'], control['failure_episodes']),
        successful_sequence_protection=not broken)
    result = dict(status='complete', scope='Repeatedly used 22-sequence DepthTrack Train development only',
        recursive_spec_sha256=sha(root / 'recursive_spec.json'),
        training_spec_sha256=sha(root / 'training_spec.json'),
        training_result_sha256={arm: sha(root / 'training' / arm / 'result.json') for arm in training},
        recursive_receipt_sha256={arm: sha(root / (arm + '_recursive_receipt.json')) for arm in training},
        weights={arm: training[arm]['weight_sha256'] for arm in training},
        aggregates=aggregates, per_sequence=per, gates=gates, primary_pass=all(gates.values()),
        new_failure_sequences=broken, language_enabled=False, public_evaluation=False,
        metric='Continuous xywh IoU; init and invalid GT excluded; invalid GT breaks H10 low-overlap runs',
        next='Eligible for a frozen low22 comparison' if all(gates.values()) else 'Stop this frozen model revision')
    write(root / 'recursive_result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_sequence'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arm', choices=['control', 'clone'])
    parser.add_argument('--analyze', action='store_true')
    arguments = parser.parse_args()
    assert arguments.analyze != (arguments.arm is not None)
    if arguments.analyze:
        analyze(arguments.root)
    else:
        run(arguments.root, arguments.arm)
