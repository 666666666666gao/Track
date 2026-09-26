"""Compare sealed content-control trajectories with their Category reference."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np


MAIN = Path('/root/autodl-tmp/sttrack_full152_evaluation_20260925')
ROOT = Path('/root/autodl-tmp/sttrack_full152_content_20260926')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def family(root):
    plan_path = root / 'plan.json'
    plan = read(plan_path)
    assert sha(plan['bundle_path']) == plan['bundle_sha256']
    output = Path(plan['output'])
    receipt_path = output / 'receipt.json'
    metrics_path = output / 'metrics.json'
    receipt, report = read(receipt_path), read(metrics_path)
    assert receipt['status'] == report['status'] == 'complete'
    assert receipt['plan_sha256'] == report['plan_sha256'] == sha(plan_path)
    assert receipt['bundle_sha256'] == report['bundle_sha256'] == plan['bundle_sha256']
    assert receipt['text_bank_sha256'] == plan['text_bank_sha256']
    assert report['receipt_sha256'] == sha(receipt_path)
    assert report['metric_source_sha256'] == plan['metric_source_sha256'] == sha(plan['metric_source'])
    assert sha(plan['cases_path']) == plan['cases_sha256']
    cases = read(plan['cases_path'])
    assert [r['sequence'] for r in receipt['sequences']] == [c['sequence'] for c in cases]
    assert receipt['frames'] == report['metrics']['frames'] == sum(c['frames'] for c in cases)
    assert report['metrics']['sequences'] == len(cases)
    return plan, cases, receipt, report


def intervals(mask, minimum=10):
    edges = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    return [[int(start), int(end)] for start, end in zip(np.where(edges == 1)[0], np.where(edges == -1)[0])
            if end - start >= minimum]


def first(mask):
    found = np.flatnonzero(mask)
    return int(found[0]) if len(found) else None


def load_sequence(metric, plan, case, saved, gt, width, height):
    output = Path(plan['output'])
    box_path = output / (case['sequence'] + '.txt')
    score_path = output / (case['sequence'] + '_all_scores.txt')
    assert saved['sequence'] == case['sequence'] and saved['frames'] == case['frames']
    assert sha(box_path) == saved['bbox_sha256'] and sha(score_path) == saved['confidence_sha256']
    boxes, scores = metric._load_rows(box_path, 4), metric._load_rows(score_path, 1).reshape(-1)
    assert len(boxes) == len(scores) == len(gt) == case['frames']
    assert np.isfinite(boxes).all() and np.isfinite(scores).all() and (boxes[:, 2:] > 0).all()
    assert np.allclose(boxes[0], case['init_bbox'], rtol=0, atol=5.01e-7) and scores[0] == 1.
    overlaps, valid = metric._vot_overlaps(boxes, gt, width, height)
    return boxes, scores, overlaps, valid


def describe(boxes, scores, overlaps, valid, gt, threshold):
    measured = valid.copy()
    measured[0] = False
    low = measured & (overlaps <= .1)
    gt_center = gt[1:, :2] + gt[1:, 2:] / 2
    previous_center = boxes[:-1, :2] + boxes[:-1, 2:] / 2
    side = np.ceil(4 * np.sqrt(boxes[:-1, 2] * boxes[:-1, 3]))
    inside = np.zeros(len(gt), dtype=bool)
    inside[1:] = (np.abs(gt_center - previous_center) <= side[:, None] / 2).all(axis=1)
    runs = intervals(low)
    return dict(
        valid_noninit_frames=int(measured.sum()),
        noninit_overlap_sum=float(overlaps[measured].sum()),
        low_overlap_frames=int(low.sum()),
        h10_segments=len(runs), h10_frames=sum(b - a for a, b in runs), h10_intervals=runs,
        low_overlap_nominal_crop_outside_frames=int((low & ~inside).sum()),
        all_boxes_recall_percent=100 * float(overlaps.sum() / valid.sum()),
        reported_recall_percent=100 * float(overlaps[scores >= threshold].sum() / valid.sum()))


def main(model, dataset, variant):
    primary = family(MAIN / model / dataset)
    control = family(ROOT / model / dataset / variant)
    plan_a, cases, receipt_a, report_a = primary
    plan_b, cases_b, receipt_b, report_b = control
    assert cases == cases_b
    changed = {k for k in set(plan_a) | set(plan_b) if plan_a.get(k) != plan_b.get(k)}
    assert changed == {'text_bank_path', 'text_bank_sha256', 'output'}
    spec = importlib.util.spec_from_file_location('sealed_metric', plan_a['metric_source'])
    metric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric)
    rows = []
    for case, saved_a, saved_b in zip(cases, receipt_a['sequences'], receipt_b['sequences']):
        folder = Path(plan_a['dataset_root']) / case['sequence']
        assert sha(folder / 'groundtruth.txt') == case['gt_sha256']
        gt = metric._load_rows(folder / 'groundtruth.txt', 4)
        image = cv2.imread(str(folder / 'color/00000001.jpg'))
        height, width = image.shape[:2]
        a = load_sequence(metric, plan_a, case, saved_a, gt, width, height)
        b = load_sequence(metric, plan_b, case, saved_b, gt, width, height)
        assert np.array_equal(a[3], b[3])
        measured = a[3].copy()
        measured[0] = False
        rescue = intervals(measured & (a[2] <= .1) & (b[2] >= .5))
        damage = intervals(measured & (a[2] >= .5) & (b[2] <= .1))
        category = describe(*a, gt, report_a['metrics']['threshold'])
        alternative = describe(*b, gt, report_b['metrics']['threshold'])
        rows.append(dict(sequence=case['sequence'], category=category, alternative=alternative,
                         first_saved_bbox_difference=first((a[0] != b[0]).any(axis=1)),
                         first_saved_score_difference=first(a[1] != b[1]),
                         alternative_rescue_intervals=rescue, alternative_damage_intervals=damage,
                         alternative_rescue_frames=sum(y - x for x, y in rescue),
                         alternative_damage_frames=sum(y - x for x, y in damage),
                         all_boxes_recall_delta_pp=alternative['all_boxes_recall_percent'] - category['all_boxes_recall_percent'],
                         reported_recall_delta_pp=alternative['reported_recall_percent'] - category['reported_recall_percent']))
    aggregate = {}
    for label, report in [('category', report_a), ('alternative', report_b)]:
        entries = [r[label] for r in rows]
        valid = sum(r['valid_noninit_frames'] for r in entries)
        reported_r = float(np.mean([r['reported_recall_percent'] for r in entries]))
        assert abs(reported_r - report['metrics']['recall_percent']) < 1e-8
        aggregate[label] = dict(
            metrics=report['metrics'],
            all_boxes_recall_percent=float(np.mean([r['all_boxes_recall_percent'] for r in entries])),
            valid_noninit_frames=valid,
            valid_noninit_mean_iou=sum(r['noninit_overlap_sum'] for r in entries) / valid,
            low_overlap_frames=sum(r['low_overlap_frames'] for r in entries),
            h10_segments=sum(r['h10_segments'] for r in entries),
            h10_frames=sum(r['h10_frames'] for r in entries),
            low_overlap_nominal_crop_outside_frames=sum(r['low_overlap_nominal_crop_outside_frames'] for r in entries))
    result = dict(status='complete', model=model, dataset=dataset, alternative=variant,
                  analyzer_sha256=sha(__file__),
                  scope='Sealed full recursive tracks. GT read only after both prediction families and metrics were complete. No alternative state is submitted.',
                  caveats=['Empty still uses the trained additive adapter.',
                           'Swapped changes the category string and is not verified as semantic contradiction.',
                           'First saved difference is a trajectory observation, not a causal attribution.',
                           'Crop coverage uses the nominal factor-4 square around the preceding saved box.',
                           'H10 and rescue/damage intervals exclude initialization and break at invalid GT.'],
                  category_metrics_sha256=sha(Path(plan_a['output']) / 'metrics.json'),
                  alternative_metrics_sha256=sha(Path(plan_b['output']) / 'metrics.json'),
                  category_receipt_sha256=sha(Path(plan_a['output']) / 'receipt.json'),
                  alternative_receipt_sha256=sha(Path(plan_b['output']) / 'receipt.json'),
                  aggregate=aggregate,
                  rescue_segments=sum(len(r['alternative_rescue_intervals']) for r in rows),
                  rescue_frames=sum(r['alternative_rescue_frames'] for r in rows),
                  damage_segments=sum(len(r['alternative_damage_intervals']) for r in rows),
                  damage_frames=sum(r['alternative_damage_frames'] for r in rows),
                  per_sequence=rows)
    output = ROOT / 'analysis'
    output.mkdir(exist_ok=True)
    stem = f'{model}_{dataset}_{variant}'
    (output / (stem + '.json')).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    columns = ['sequence', 'first_saved_bbox_difference', 'first_saved_score_difference',
               'all_boxes_recall_delta_pp', 'reported_recall_delta_pp',
               'alternative_rescue_frames', 'alternative_damage_frames']
    with (output / (stem + '.csv')).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)
    print(json.dumps({key: result[key] for key in ('status', 'model', 'dataset', 'alternative',
                    'aggregate', 'rescue_segments', 'rescue_frames', 'damage_segments', 'damage_frames')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['M67', 'M82'], required=True)
    parser.add_argument('--dataset', choices=['depthtrack', 'cdtb'], required=True)
    parser.add_argument('--variant', choices=['empty', 'swapped'], required=True)
    args = parser.parse_args()
    main(args.model, args.dataset, args.variant)
