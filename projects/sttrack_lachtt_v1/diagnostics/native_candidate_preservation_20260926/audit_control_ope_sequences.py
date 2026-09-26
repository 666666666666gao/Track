"""Compare completed M89 Control OPE trajectories with sealed M82 Full152."""

import csv
import importlib.util
import json
import math
from pathlib import Path

import cv2
import numpy as np

import prepare_evaluation as common


OLD = Path('/root/autodl-tmp/sttrack_full152_evaluation_20260925')
ROOT = common.ROOT
common.checked('control')
old_recall = common.read(OLD / 'posthoc/M82_recall_confidence_decomposition.json')
summary = {'scope': 'Completed OPE, each model at its own dataset threshold; all-box recall fixes each saved trajectory.',
           'datasets': {}}

for dataset in ('depthtrack', 'cdtb'):
    root = ROOT / 'control' / dataset
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
    with (OLD / 'posthoc/M82' / dataset / 'per_sequence.csv').open(newline='') as stream:
        old_rows = {row['sequence']: row for row in csv.DictReader(stream)}
    old_all = {row['sequence']: row for row in old_recall['datasets'][dataset]['per_sequence']}
    old_plan = common.read(OLD / 'M82' / dataset / 'plan.json')
    old_receipt = common.read(Path(old_plan['output']) / 'receipt.json')
    assert len(old_receipt['sequences']) == len(cases)
    rows = []
    events = []

    for case, saved, old_saved in zip(cases, receipt['sequences'], old_receipt['sequences']):
        name = case['sequence']
        assert saved['sequence'] == name and saved['frames'] == case['frames']
        assert old_saved['sequence'] == name and old_saved['frames'] == case['frames']
        bbox_path = predictions / (name + '.txt')
        score_path = predictions / (name + '_all_scores.txt')
        old_bbox_path = Path(old_plan['output']) / (name + '.txt')
        gt_path = Path(plan['dataset_root']) / name / 'groundtruth.txt'
        assert common.sha(bbox_path) == saved['bbox_sha256']
        assert common.sha(score_path) == saved['confidence_sha256']
        assert common.sha(old_bbox_path) == old_saved['bbox_sha256']
        assert common.sha(gt_path) == case['gt_sha256']
        bbox = metric._load_rows(bbox_path, 4)
        old_bbox = metric._load_rows(old_bbox_path, 4)
        score = metric._load_rows(score_path, 1).reshape(-1)
        gt = metric._load_rows(gt_path, 4)
        assert len(bbox) == len(old_bbox) == len(score) == len(gt) == case['frames']
        image = cv2.imread(str(Path(plan['dataset_root']) / name / 'color/00000001.jpg'))
        height, width = image.shape[:2]
        overlaps, visible = metric._vot_overlaps(bbox, gt, width, height)
        old_overlaps, old_visible = metric._vot_overlaps(old_bbox, gt, width, height)
        assert np.array_equal(visible, old_visible)
        selected = score >= threshold
        precision = float(overlaps[selected].mean()) if selected.any() else 1.0
        recall = float(overlaps[selected].sum() / visible.sum())
        all_recall = float(overlaps.sum() / visible.sum())
        previous = old_rows[name]
        previous_all = old_all[name]
        new_only = np.flatnonzero((overlaps <= 0.1) & (old_overlaps >= 0.5) & visible)
        groups = np.split(new_only, np.where(np.diff(new_only) != 1)[0] + 1) if len(new_only) else []
        sequence_events = []
        for group in groups:
            if len(group) < 10:
                continue
            start = int(group[0])
            assert start > 0
            x, y, box_width, box_height = bbox[start - 1]
            side = math.ceil(math.sqrt(box_width * box_height) * 4)
            x1 = round(x + 0.5 * box_width - 0.5 * side)
            y1 = round(y + 0.5 * box_height - 0.5 * side)
            gx, gy, gw, gh = gt[start]
            cx, cy = gx + 0.5 * gw, gy + 0.5 * gh
            sequence_events.append(dict(sequence=name, start=start, end=int(group[-1]) + 1,
                                        frames=len(group), center_in_crop_at_start=bool(
                                            x1 <= cx < x1 + side and y1 <= cy < y1 + side),
                                        control_iou_at_start=float(overlaps[start]),
                                        old_m82_iou_at_start=float(old_overlaps[start])))
        events.extend(sequence_events)
        rows.append(dict(sequence=name, frames=len(score),
                         control_p=100 * precision, control_r=100 * recall,
                         control_all_box_r=100 * all_recall,
                         old_m82_p=float(previous['precision_percent']),
                         old_m82_r=float(previous['recall_percent']),
                         old_m82_all_box_r=float(previous_all['all_boxes_recall_percent']),
                         delta_r=100 * recall - float(previous['recall_percent']),
                         delta_all_box_r=100 * all_recall - float(previous_all['all_boxes_recall_percent']),
                         sustained_new_only_runs=len(sequence_events),
                         sustained_new_only_frames=sum(event['frames'] for event in sequence_events),
                         longest_new_only_run=max((event['frames'] for event in sequence_events), default=0)))

    assert set(old_rows) == set(old_all) == {row['sequence'] for row in rows}
    assert abs(float(np.mean([row['control_p'] for row in rows])) - report['metrics']['precision_percent']) < 1e-8
    assert abs(float(np.mean([row['control_r'] for row in rows])) - report['metrics']['recall_percent']) < 1e-8
    summary['datasets'][dataset] = dict(
        threshold=threshold,
        control_metrics_sha256=common.sha(predictions / 'metrics.json'),
        control_receipt_sha256=common.sha(predictions / 'receipt.json'),
        mean_control_all_box_recall=float(np.mean([row['control_all_box_r'] for row in rows])),
        mean_old_m82_all_box_recall=float(np.mean([row['old_m82_all_box_r'] for row in rows])),
        rows=rows, sustained_new_only_events=events)

out = ROOT / 'posthoc'
out.mkdir(exist_ok=True)
path = out / 'control_ope_sequence_audit.json'
path.write_text(json.dumps(summary, indent=2) + '\n')
for dataset, data in summary['datasets'].items():
    with (out / f'control_{dataset}_sequence_audit.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data['rows'][0]))
        writer.writeheader()
        writer.writerows(data['rows'])
    ordered = sorted(data['rows'], key=lambda row: row['delta_all_box_r'])
    print(dataset, 'mean_all_box_delta', data['mean_control_all_box_recall'] - data['mean_old_m82_all_box_recall'])
    print('largest_all_box_declines', [(row['sequence'], round(row['delta_all_box_r'], 3)) for row in ordered[:10]])
    print('largest_all_box_gains', [(row['sequence'], round(row['delta_all_box_r'], 3)) for row in ordered[-10:]])
    print('sustained_new_only', len(data['sustained_new_only_events']),
          'onset_inside', sum(event['center_in_crop_at_start'] for event in data['sustained_new_only_events']),
          'frames', sum(event['frames'] for event in data['sustained_new_only_events']))
print('report', path)
