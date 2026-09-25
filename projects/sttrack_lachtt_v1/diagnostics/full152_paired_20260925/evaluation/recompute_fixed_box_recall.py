"""Measure the recall limit of sealed OPE boxes without confidence filtering."""

import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

import prepare_full as common


parser = argparse.ArgumentParser()
parser.add_argument('model', choices=('M67', 'M82'))
args = parser.parse_args()
common.checked(args.model)

datasets = {}
for dataset in ('depthtrack', 'cdtb'):
    root = common.ROOT / args.model / dataset
    plan = common.read(root / 'plan.json')
    cases = common.read(Path(plan['cases_path']))
    predictions = Path(plan['output'])
    receipt = common.read(predictions / 'receipt.json')
    report = common.read(predictions / 'metrics.json')
    assert receipt['status'] == report['status'] == 'complete'
    assert report['receipt_sha256'] == common.sha(predictions / 'receipt.json')
    assert common.sha(Path(plan['metric_source'])) == report['metric_source_sha256']
    assert len(cases) == len(receipt['sequences']) == report['metrics']['sequences']

    spec = importlib.util.spec_from_file_location('sealed_ope_metric', plan['metric_source'])
    metric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric)
    threshold = report['metrics']['threshold']
    rows = []
    for case, saved in zip(cases, receipt['sequences']):
        name = case['sequence']
        assert saved['sequence'] == name and saved['frames'] == case['frames']
        bbox_path = predictions / (name + '.txt')
        score_path = predictions / (name + '_all_scores.txt')
        gt_path = Path(plan['dataset_root']) / name / 'groundtruth.txt'
        assert common.sha(bbox_path) == saved['bbox_sha256']
        assert common.sha(score_path) == saved['confidence_sha256']
        assert common.sha(gt_path) == case['gt_sha256']
        bbox = metric._load_rows(bbox_path, 4)
        score = metric._load_rows(score_path, 1).reshape(-1)
        gt = metric._load_rows(gt_path, 4)
        assert len(bbox) == len(score) == len(gt) == case['frames']
        image = cv2.imread(str(Path(plan['dataset_root']) / name / 'color/00000001.jpg'))
        height, width = image.shape[:2]
        overlaps, visible = metric._vot_overlaps(bbox, gt, width, height)
        selected = score >= threshold
        reported = 100 * float(overlaps[selected].sum() / visible.sum())
        all_boxes = 100 * float(overlaps.sum() / visible.sum())
        rows.append(dict(sequence=name, reported_recall_percent=reported,
                         all_boxes_recall_percent=all_boxes,
                         score_selection_cost_pp=all_boxes - reported))

    reported_mean = float(np.mean([row['reported_recall_percent'] for row in rows]))
    all_boxes_mean = float(np.mean([row['all_boxes_recall_percent'] for row in rows]))
    assert abs(reported_mean - report['metrics']['recall_percent']) < 1e-8
    datasets[dataset] = dict(reported_recall_percent=reported_mean,
                             all_boxes_recall_percent=all_boxes_mean,
                             score_selection_cost_pp=all_boxes_mean - reported_mean,
                             per_sequence=rows,
                             metrics_sha256=common.sha(predictions / 'metrics.json'),
                             receipt_sha256=common.sha(predictions / 'receipt.json'))

output = common.ROOT / 'posthoc' / (args.model + '_recall_confidence_decomposition.json')
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(dict(
    scope='Fixed boxes and full recursive trajectories; only confidence filtering is removed.',
    model=args.model, datasets=datasets), indent=2) + '\n')
print(json.dumps({name: {key: value for key, value in data.items()
                         if key in ('reported_recall_percent', 'all_boxes_recall_percent',
                                    'score_selection_cost_pp')}
                  for name, data in datasets.items()}))
