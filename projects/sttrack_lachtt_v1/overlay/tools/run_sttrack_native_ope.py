"""Same-bundle native STTrack OPE reference for DepthTrack Test and CDTB."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(root, spec):
    repository = Path(spec['repository'])
    for name, digest in spec['source_sha256'].items():
        assert sha(repository/name) == digest, name
    assert sha(spec['checkpoint']) == spec['checkpoint_sha256']
    assert sha(root/'inputs.json') == spec['inputs_sha256']
    assert sha(spec['metric_source']) == spec['metric_source_sha256']
    assert sha(root/'EXPERIMENT_PLAN.md') == spec['plan_sha256']


def track(root, spec, cases, name, contract):
    import torch
    sys.path.insert(0, spec['repository'])
    from lib.config.sttrack.config import cfg, update_config_from_file
    from lib.test.tracker.sttrack import STTrack
    from lib.train.dataset.depth_utils import get_rgbd_frame
    torch.set_num_threads(1)
    update_config_from_file(str(Path(spec['repository'])/spec['configuration']))
    params = SimpleNamespace(cfg=cfg, checkpoint=spec['checkpoint'], template_factor=2., template_size=128,
                             search_factor=4., search_size=256, save_all_boxes=False, debug=0)
    tracker = STTrack(params)
    assert tracker.num_template == 2 and tracker.update_intervals == 50 and tracker.update_threshold == .75
    expected = {}
    if contract:
        path = Path(spec['native_inputs'])
        assert sha(path) == spec['native_inputs_sha256']
        expected = {c['sequence']: c for c in json.loads(path.read_text())}
    else:
        assert (root/'contract.exit').read_text().strip() == '0'
        evidence = json.loads((root/'contract_receipt.json').read_text())
        assert evidence['status'] == 'complete' and evidence['spec_sha256'] == sha(root/'spec.json')
        assert evidence['template_updates'] > 0
    output = root/name
    output.mkdir()
    receipts = []
    started = time.time()
    for case in cases:
        folder = Path(case['root'])/case['sequence']
        boxes = [list(case['init_bbox'])]
        scores = [1.]
        updates = 0
        max_box_error = max_score_error = 0.
        for frame in range(case['frames']):
            stem = '{:08d}'.format(frame+1)
            image = get_rgbd_frame(str(folder/'color'/(stem+'.jpg')), str(folder/'depth'/(stem+'.png')),
                                   dtype='rgbcolormap', depth_clip=True)
            if frame == 0:
                tracker.initialize(image, dict(init_bbox=list(case['init_bbox'])))
                continue
            old_template = tracker.z_dict[1]
            prediction = tracker.track(image)
            boxes.append(list(prediction['target_bbox']))
            scores.append(float(prediction['best_score']))
            updates += tracker.z_dict[1] is not old_template
            if contract:
                native = expected[case['sequence']]['expected_rows'][frame]
                box_error = float(np.abs(np.asarray(boxes[-1])-native['bbox']).max())
                score_error = abs(scores[-1]-native['score'])
                assert box_error <= 1e-4 and score_error <= 1e-6
                max_box_error = max(max_box_error, box_error)
                max_score_error = max(max_score_error, score_error)
        assert len(boxes) == len(scores) == case['frames']
        assert np.isfinite(boxes).all() and np.isfinite(scores).all()
        assert (np.asarray(boxes)[:, 2:] > 0).all()
        box_path = output/(case['sequence']+'.txt')
        score_path = output/(case['sequence']+'_all_scores.txt')
        np.savetxt(box_path, boxes, fmt='%.6f', delimiter=',')
        np.savetxt(score_path, scores, fmt='%.6f')
        restored_boxes = np.loadtxt(box_path, delimiter=',')
        restored_scores = np.loadtxt(score_path)
        assert np.abs(restored_boxes-boxes).max() <= 5.01e-7
        assert np.abs(restored_scores-scores).max() <= 5.01e-7 and restored_scores[0] == 1.
        item = dict(sequence=case['sequence'], frames=len(boxes), template_updates=updates,
                    bbox_sha256=sha(box_path), confidence_sha256=sha(score_path),
                    max_native_bbox_error_px=max_box_error if contract else None,
                    max_native_score_error=max_score_error if contract else None,
                    cumulative_seconds=time.time()-started)
        receipts.append(item)
        print(json.dumps(item), flush=True)
    check(root, spec)
    result = dict(status='complete', dataset=name, spec_sha256=sha(root/'spec.json'),
                  checkpoint_sha256=spec['checkpoint_sha256'], sequences=receipts,
                  frames=sum(r['frames'] for r in receipts), template_updates=sum(r['template_updates'] for r in receipts),
                  subsequent_gt_opened=False, labels_used_for_inference=False, optimizer_steps=0,
                  elapsed_seconds=time.time()-started)
    (root/(name+'_receipt.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'sequences'}, indent=2), flush=True)


def analyze(root, spec, cases, dataset):
    assert (root/('tracking_'+dataset+'.exit')).read_text().strip() == '0'
    receipt_path = root/(dataset+'_receipt.json')
    receipt = json.loads(receipt_path.read_text())
    assert receipt['status'] == 'complete' and receipt['spec_sha256'] == sha(root/'spec.json')
    assert receipt['checkpoint_sha256'] == spec['checkpoint_sha256']
    assert [r['sequence'] for r in receipt['sequences']] == [c['sequence'] for c in cases]
    assert receipt['frames'] == sum(c['frames'] for c in cases)
    assert not receipt['subsequent_gt_opened'] and not receipt['labels_used_for_inference']
    for item, case in zip(receipt['sequences'], cases):
        assert item['frames'] == case['frames']
        assert sha(root/dataset/(case['sequence']+'.txt')) == item['bbox_sha256']
        assert sha(root/dataset/(case['sequence']+'_all_scores.txt')) == item['confidence_sha256']
    # The unchanged evaluator opens subsequent GT only after all output seals.
    module_spec = importlib.util.spec_from_file_location('native_ope_metric', spec['metric_source'])
    metric = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(metric)
    values = metric.evaluate_depthtrack_results(spec['datasets'][dataset]['root'], root/dataset,
                                                resolution=100, sequence_names=[c['sequence'] for c in cases])
    assert values['sequences'] == len(cases) and values['frames'] == receipt['frames']
    result = dict(status='complete', dataset=dataset, role='Unchanged native STTrack reference',
                  spec_sha256=sha(root/'spec.json'), checkpoint_sha256=spec['checkpoint_sha256'],
                  receipt_sha256=sha(receipt_path), metric_source_sha256=sha(spec['metric_source']), metrics=values,
                  groundtruth_sha256={c['sequence']: sha(Path(c['root'])/c['sequence']/'groundtruth.txt') for c in cases},
                  new_trained_module=False, training_gate_promotion=False)
    check(root, spec)
    (root/('metrics_'+dataset+'.json')).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(values, indent=2, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=['contract', 'track', 'analyze'], required=True)
    parser.add_argument('--dataset', choices=['depthtrack', 'cdtb'])
    args = parser.parse_args()
    spec = json.loads((args.root/'spec.json').read_text())
    check(args.root, spec)
    inputs = json.loads((args.root/'inputs.json').read_text())
    if args.mode == 'contract':
        track(args.root, spec, inputs['contract'], 'contract', True)
    else:
        assert args.dataset is not None
        cases = inputs[args.dataset]
        assert len(cases) == spec['datasets'][args.dataset]['sequences']
        assert sum(c['frames'] for c in cases) == spec['datasets'][args.dataset]['frames']
        if args.mode == 'track':
            track(args.root, spec, cases, args.dataset, False)
        else:
            analyze(args.root, spec, cases, args.dataset)


if __name__ == '__main__':
    main()
