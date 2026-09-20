"""Semantic OPE outputs with the native six-decimal box/confidence contract."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

from initialization_text import sha
from semantic_runtime import checked_plan, make_tracker, text_bank


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_predictions(output, sequence, boxes, scores):
    boxes = np.asarray(boxes)
    scores = np.asarray(scores)
    assert boxes.shape == (len(scores), 4) and scores[0] == 1.
    assert np.isfinite(boxes).all() and np.isfinite(scores).all()
    assert (boxes[:, 2:] > 0).all()
    box_path = output / (sequence + '.txt')
    score_path = output / (sequence + '_all_scores.txt')
    np.savetxt(box_path, boxes, fmt='%.6f', delimiter=',')
    np.savetxt(score_path, scores, fmt='%.6f')
    restored_boxes = np.loadtxt(box_path, delimiter=',').reshape(-1, 4)
    restored_scores = np.loadtxt(score_path).reshape(-1)
    assert np.abs(restored_boxes - boxes).max() <= 5.01e-7
    assert np.abs(restored_scores - scores).max() <= 5.01e-7
    return dict(sequence=sequence, frames=len(scores), bbox_sha256=sha(box_path),
                confidence_sha256=sha(score_path))


def track(plan_path):
    plan_digest = sha(plan_path)
    plan, bundle = checked_plan(plan_path)
    assert sha(plan['cases_path']) == plan['cases_sha256']
    cases = json.loads(Path(plan['cases_path']).read_text())
    bank = text_bank(plan, bundle)
    tracker = make_tracker(bundle)
    from lib.train.dataset.depth_utils import get_rgbd_frame
    output = Path(plan['output'])
    output.mkdir()
    receipts = []
    started = time.time()
    for case in cases:
        folder = Path(plan['dataset_root']) / case['sequence']
        boxes, scores = [list(case['init_bbox'])], [1.]
        for frame in range(case['frames']):
            stem = '%08d' % (frame + 1)
            rgb = folder / 'color' / (stem + '.jpg')
            depth = folder / 'depth' / (stem + '.png')
            image = get_rgbd_frame(str(rgb), str(depth), dtype='rgbcolormap', depth_clip=True)
            if frame == 0:
                tracker.initialize(image, bank.info(rgb, case['init_bbox']))
            else:
                prediction = tracker.track(image)
                boxes.append(list(prediction['target_bbox']))
                scores.append(float(prediction['best_score']))
        item = write_predictions(output, case['sequence'], boxes, scores)
        item['cumulative_seconds'] = time.time() - started
        receipts.append(item)
        print(json.dumps(item), flush=True)
    assert sha(plan_path) == plan_digest
    checked_plan(plan_path)
    write_json(output / 'receipt.json', dict(status='complete', plan_sha256=plan_digest,
        bundle_sha256=plan['bundle_sha256'], text_bank_sha256=plan['text_bank_sha256'],
        sequences=receipts, frames=sum(row['frames'] for row in receipts),
        subsequent_gt_opened=False, optimizer_steps=0, text_updated_online=False,
        elapsed_seconds=time.time() - started))


def analyze(plan_path):
    plan, bundle = checked_plan(plan_path)
    assert sha(plan['cases_path']) == plan['cases_sha256']
    cases = json.loads(Path(plan['cases_path']).read_text())
    output = Path(plan['output'])
    receipt = json.loads((output / 'receipt.json').read_text())
    assert receipt['status'] == 'complete' and receipt['plan_sha256'] == sha(plan_path)
    assert receipt['bundle_sha256'] == plan['bundle_sha256']
    assert receipt['text_bank_sha256'] == plan['text_bank_sha256']
    assert [row['sequence'] for row in receipt['sequences']] == [c['sequence'] for c in cases]
    assert receipt['frames'] == sum(c['frames'] for c in cases)
    for case, row in zip(cases, receipt['sequences']):
        assert row['frames'] == case['frames']
        assert sha(output / (case['sequence'] + '.txt')) == row['bbox_sha256']
        assert sha(output / (case['sequence'] + '_all_scores.txt')) == row['confidence_sha256']
    # Open labels only after the entire prediction family has been sealed.
    for case in cases:
        assert sha(Path(plan['dataset_root']) / case['sequence'] / 'groundtruth.txt') == case['gt_sha256']
    assert sha(plan['metric_source']) == plan['metric_source_sha256']
    module_spec = importlib.util.spec_from_file_location('semantic_ope_metric', plan['metric_source'])
    metric = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(metric)
    values = metric.evaluate_depthtrack_results(plan['dataset_root'], output,
        resolution=100, sequence_names=[c['sequence'] for c in cases])
    assert values['sequences'] == len(cases) and values['frames'] == receipt['frames']
    write_json(output / 'metrics.json', dict(status='complete', plan_sha256=sha(plan_path),
        bundle_sha256=plan['bundle_sha256'], receipt_sha256=sha(output / 'receipt.json'),
        metric_source_sha256=plan['metric_source_sha256'], metrics=values))
    print(json.dumps(values, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--mode', choices=['track', 'analyze'], required=True)
    args = parser.parse_args()
    track(args.plan) if args.mode == 'track' else analyze(args.plan)
