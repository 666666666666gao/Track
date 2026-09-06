"""M56 full causal development trajectories, with independent native comparison."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from train import binding, sha, write


def bound(root):
    spec = binding(root)
    frozen = json.loads((root / 'recursive_spec.json').read_text())
    assert sha(__file__) == frozen['runner_sha256']
    assert sha(root / 'spec.json') == frozen['training_spec_sha256']
    assert sha(root / 'recursive_inputs.json') == frozen['inputs_sha256']
    training = json.loads((root / 'training_result.json').read_text())
    assert training['status'] == 'complete_train' and training['equal_initialization_order_budget_parameters']
    assert training['spec_sha256'] == sha(root / 'spec.json')
    for arm in spec['variants']:
        result_path = root / 'training' / (arm + '_result.json')
        assert sha(result_path) == training['variants'][arm]['result_sha256']
        result = json.loads(result_path.read_text())
        assert result['status'] == 'complete_train' and result['optimizer_steps'] == 960 and result['epochs'] == 20
        assert result['spec_sha256'] == sha(root / 'spec.json')
        assert sha(root / 'training' / (arm + '_final.pth')) == result['checkpoint_sha256'] == training['variants'][arm]['checkpoint_sha256']
    return spec, frozen, training, json.loads((root / 'recursive_inputs.json').read_text())


def run(root, arm):
    spec, frozen, training, cases = bound(root)
    assert arm in spec['variants']
    sys.path.insert(0, spec['repository'])
    import numpy as np
    import torch
    from lib.config.sttrack.config import cfg, update_config_from_file
    from lib.test.tracker.sttrack_attribute_candidate_set import STTrackAttributeCandidateSet
    from lib.train.dataset.depth_utils import get_rgbd_frame
    torch.set_num_threads(1)
    update_config_from_file(str(Path(spec['repository']) / 'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    assert cfg.DATA.TEMPLATE.NUMBER == 2 and cfg.TEST.UPDATE_INTERVALS == 50 and cfg.TEST.UPDATE_THRESHOLD == .75
    params = SimpleNamespace(cfg=cfg, checkpoint=spec['base_checkpoint'], template_factor=2., template_size=128,
        search_factor=4., search_size=256, save_all_boxes=False, debug=0)
    checkpoint = root / 'training' / (arm + '_final.pth')
    saved = torch.load(checkpoint, map_location='cpu')
    assert saved['m56_spec_sha256'] == sha(root / 'spec.json') and saved['variant'] == arm
    assert saved['base_checkpoint_sha256'] == spec['base_checkpoint_sha256']
    assert saved['text_bank_sha256'] == sha(root / 'text_bank.pt')
    del saved
    tracker = STTrackAttributeCandidateSet(params, str(checkpoint))
    assert tracker.variant == arm
    bank = torch.load(root / 'text_bank.pt', map_location='cpu')
    directory = root / 'recursive' / arm
    directory.mkdir(parents=True)
    started = time.time()
    receipts = []
    for case in cases:
        folder = Path(spec['dataset_root']) / case['sequence']

        def image_at(frame):
            return get_rgbd_frame(str(folder / 'color' / ('%08d.jpg' % (frame + 1))),
                str(folder / 'depth' / ('%08d.png' % (frame + 1))), dtype='rgbcolormap', depth_clip=True)

        index = bank['sequences'].index(case['sequence'])
        tracker.initialize(image_at(0), dict(init_bbox=list(case['init_bbox']), text_tokens=bank['tokens'][index],
            text_mask=bank['mask'][index], empty_text=bank['empty']))
        rows = [dict(frame=0, bbox=list(case['init_bbox']), score=1., chosen=0, none=False)]
        sequence_started = time.time()
        for frame in range(1, case['frames']):
            output = tracker.track(image_at(frame))
            box = [float(x) for x in output['target_bbox']]
            score = float(output['best_score'])
            assert np.isfinite(box).all() and min(box[2:]) > 0 and np.isfinite(score)
            rows.append(dict(frame=frame, bbox=box, score=score, chosen=int(output['association_candidate']),
                             none=bool(output['association_none'])))
        path = directory / (case['sequence'] + '.json')
        write(path, dict(sequence=case['sequence'], rows=rows))
        receipt = dict(sequence=case['sequence'], frames=len(rows), sha256=sha(path),
            nondefault=sum(row['chosen'] != 0 for row in rows), none=sum(row['none'] for row in rows),
            elapsed_seconds=time.time() - sequence_started)
        receipts.append(receipt)
        print(json.dumps(dict(variant=arm, **receipt)), flush=True)
    bound(root)
    write(root / (arm + '_recursive_receipt.json'), dict(status='complete', variant=arm, sequences=receipts,
        frames=sum(x['frames'] for x in receipts), elapsed_seconds=time.time() - started,
        recursive_spec_sha256=sha(root / 'recursive_spec.json'), training_result_sha256=sha(root / 'training_result.json'),
        head_sha256=sha(checkpoint), base_sha256=spec['base_checkpoint_sha256'],
        subsequent_gt_opened=False, text_strings_updated_online=False, native_default_template_updates=True))


def analyze(root):
    spec, frozen, training, cases = bound(root)
    sys.path.insert(0, spec['repository'])
    import numpy as np
    from tools.analyze_sttrack_m42_recursive import statistics
    assert sha(Path(spec['repository']) / 'tools/analyze_sttrack_m42_recursive.py') == frozen['metric_sha256']
    names = {case['sequence'] for case in cases}
    predicted = {arm: {} for arm in ['native'] + spec['variants']}
    baseline = defaultdict(list)
    for path, digest in frozen['baseline_trace_sha256'].items():
        assert sha(path) == digest
        for row in json.loads(Path(path).read_text())['rows']:
            if row['sequence'] in names:
                baseline[row['sequence']].append(row)
    for case in cases:
        rows = sorted(baseline[case['sequence']], key=lambda x: x['frame_index'])
        assert [row['frame_index'] for row in rows] == list(range(case['frames']))
        predicted['native'][case['sequence']] = [row['public_bbox'] for row in rows]
    for arm in spec['variants']:
        assert (root / (arm + '_recursive.exit')).read_text().strip() == '0'
        receipt = json.loads((root / (arm + '_recursive_receipt.json')).read_text())
        assert receipt['status'] == 'complete' and receipt['variant'] == arm
        assert receipt['recursive_spec_sha256'] == sha(root / 'recursive_spec.json')
        assert receipt['training_result_sha256'] == sha(root / 'training_result.json')
        assert receipt['head_sha256'] == training['variants'][arm]['checkpoint_sha256']
        assert {row['sequence'] for row in receipt['sequences']} == names and len(receipt['sequences']) == 22
        for item in receipt['sequences']:
            path = root / 'recursive' / arm / (item['sequence'] + '.json')
            assert sha(path) == item['sha256']
            data = json.loads(path.read_text())
            case = next(row for row in cases if row['sequence'] == item['sequence'])
            assert data['sequence'] == case['sequence']
            rows = data['rows']
            assert len(rows) == item['frames'] == case['frames']
            assert [row['frame'] for row in rows] == list(range(case['frames']))
            assert rows[0]['bbox'] == case['init_bbox']
            boxes = np.asarray([row['bbox'] for row in rows])
            assert np.isfinite(boxes).all() and (boxes[:, 2:] > 0).all()
            predicted[arm][case['sequence']] = boxes
    # Seal every independent trajectory family before interpreting subsequent GT.
    per = {arm: {} for arm in predicted}
    for case in cases:
        path = Path(spec['dataset_root']) / case['sequence'] / 'groundtruth.txt'
        assert sha(path) == frozen['development_gt_sha256'][case['sequence']]
        gt = np.loadtxt(path, delimiter=',')
        assert len(gt) == case['frames']
        for arm in predicted:
            per[arm][case['sequence']] = statistics(predicted[arm][case['sequence']], gt)
    aggregates = {}
    for arm, values in per.items():
        totals = {key: sum(row[key] for row in values.values())
                  for key in ['valid_frames', 'iou_sum', 'low_iou_frames', 'failure_episodes']}
        totals['mean_iou'] = totals['iou_sum'] / totals['valid_frames']
        totals['macro_sequence_mean_iou'] = float(np.mean([row['mean_iou'] for row in values.values()]))
        aggregates[arm] = totals
    attributes = aggregates['attributes']
    rule = spec['recursive_gate']
    broken = [name for name in sorted(names) if per['native'][name]['failure_episodes'] == 0
              and per['attributes'][name]['failure_episodes'] > 0]
    gates = { 'mean_vs_' + arm: attributes['mean_iou'] >= aggregates[arm]['mean_iou'] + rule['mean_gain_vs_' + arm]
              for arm in ['native', 'empty', 'pooled'] }
    gates['low_frames'] = all(attributes['low_iou_frames'] <= aggregates[arm]['low_iou_frames'] for arm in ['native', 'empty', 'pooled'])
    gates['H10'] = all(attributes['failure_episodes'] <= aggregates[arm]['failure_episodes'] for arm in ['native', 'empty', 'pooled'])
    gates['native_successful_sequence_protection'] = not broken
    result = dict(status='complete_recursive_development', recursive_spec_sha256=sha(root / 'recursive_spec.json'),
        training_result_sha256=sha(root / 'training_result.json'), aggregates=aggregates, per_sequence=per,
        gates=gates, primary_pass=all(gates.values()), new_failure_sequences=broken,
        receipts={arm: sha(root / (arm + '_recursive_receipt.json')) for arm in spec['variants']},
        scope='Repeatedly used 22-sequence DepthTrack Train development, 33130 frames per arm',
        metric='Continuous xywh IoU; init/invalid GT excluded; invalid GT breaks H10 low-overlap runs',
        public_evaluation=False, text_strings_updated_online=False,
        claim='Same-base equal-budget language interaction comparison; no VOT or three-dataset gain yet',
        next='Eligible for frozen low22 preparation and recursive text counterfactual checks' if all(gates.values()) else 'Stop this frozen revision; diagnose all harms')
    write(root / 'recursive_result.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'per_sequence'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arm', choices=['attributes', 'pooled', 'empty'])
    parser.add_argument('--analyze', action='store_true')
    args = parser.parse_args()
    assert args.analyze != (args.arm is not None)
    if args.analyze:
        analyze(args.root)
    else:
        run(args.root, args.arm)
