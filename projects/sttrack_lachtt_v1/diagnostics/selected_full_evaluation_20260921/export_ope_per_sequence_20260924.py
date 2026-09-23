"""Describe sealed full OPE results at each dataset's official chosen threshold."""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

import prepare_full as common


parser = argparse.ArgumentParser()
parser.add_argument('model', choices=('M67', 'M82'))
parser.add_argument('dataset', choices=('depthtrack', 'cdtb'))
args = parser.parse_args()

root = common.ROOT / args.model / args.dataset
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
    image_path = Path(plan['dataset_root']) / name / 'color/00000001.jpg'
    assert common.sha(bbox_path) == saved['bbox_sha256']
    assert common.sha(score_path) == saved['confidence_sha256']
    assert common.sha(gt_path) == case['gt_sha256']
    bbox = metric._load_rows(bbox_path, 4)
    score = metric._load_rows(score_path, 1).reshape(-1)
    gt = metric._load_rows(gt_path, 4)
    assert len(bbox) == len(score) == len(gt) == case['frames']
    height, width = cv2.imread(str(image_path)).shape[:2]
    overlaps, visible = metric._vot_overlaps(bbox, gt, width, height)
    selected = score >= threshold
    precision = float(overlaps[selected].mean()) if selected.any() else 1.0
    recall = float(overlaps[selected].sum() / visible.sum())
    f_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    rows.append(dict(sequence=name, frames=len(score), visible_frames=int(visible.sum()),
                     selected_frames=int(selected.sum()), precision_percent=100 * precision,
                     recall_percent=100 * recall, f_score_percent=100 * f_score))

mean_p = np.mean([row['precision_percent'] for row in rows])
mean_r = np.mean([row['recall_percent'] for row in rows])
assert abs(mean_p - report['metrics']['precision_percent']) < 1e-8
assert abs(mean_r - report['metrics']['recall_percent']) < 1e-8

output = common.ROOT / 'posthoc' / args.model / args.dataset
output.mkdir(parents=True, exist_ok=True)
summary = dict(model=args.model, dataset=args.dataset, threshold=threshold, rows=rows,
               metrics_sha256=common.sha(predictions / 'metrics.json'),
               receipt_sha256=common.sha(predictions / 'receipt.json'),
               scope='Descriptive sequence scores at the complete dataset threshold. '
                     'Aggregate F is from macro P/R, not the mean sequence F.')
(output / 'per_sequence.json').write_text(json.dumps(summary, indent=2) + '\n')
with (output / 'per_sequence.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
