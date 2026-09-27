"""Read-only paired onset audit of completed M82 and M89 OPE trajectories."""
import importlib.util
import json
import math
from pathlib import Path

import cv2
import numpy as np

import prepare_evaluation as common


OLD = Path('/root/autodl-tmp/sttrack_full152_evaluation_20260925')
ROOT = common.ROOT


def runs(mask):
    indices = np.flatnonzero(mask)
    return np.split(indices, np.where(np.diff(indices) != 1)[0] + 1) if len(indices) else []


def crop_contains(previous, target):
    x, y, width, height = previous
    side = math.ceil(math.sqrt(width * height) * 4)
    x1 = round(x + width / 2 - side / 2)
    y1 = round(y + height / 2 - side / 2)
    gx, gy, gw, gh = target
    center = x1 <= gx + gw / 2 < x1 + side and y1 <= gy + gh / 2 < y1 + side
    full = x1 <= gx and y1 <= gy and gx + gw <= x1 + side and gy + gh <= y1 + side
    return bool(center), bool(full)


def main():
    common.checked('candidate')
    summary = dict(status='read_only_pair_onset_audit_complete',
                   definition='At least 10 consecutive GT-valid frames with one model IoU<=0.1 '
                              'and the other IoU>=0.5; onset is the first frame of that interval.',
                   datasets={})
    for dataset in ('depthtrack', 'cdtb'):
        current_plan = common.read(ROOT / 'candidate' / dataset / 'plan.json')
        old_plan = common.read(OLD / 'M82' / dataset / 'plan.json')
        current_output = Path(current_plan['output'])
        old_output = Path(old_plan['output'])
        cases = common.read(Path(current_plan['cases_path']))
        current_receipt = common.read(current_output / 'receipt.json')
        old_receipt = common.read(old_output / 'receipt.json')
        metric_spec = importlib.util.spec_from_file_location('sealed_ope_metric', current_plan['metric_source'])
        metric = importlib.util.module_from_spec(metric_spec)
        metric_spec.loader.exec_module(metric)
        events = []
        assert len(cases) == len(current_receipt['sequences']) == len(old_receipt['sequences'])
        for case, saved, old_saved in zip(cases, current_receipt['sequences'], old_receipt['sequences']):
            name = case['sequence']
            assert saved['sequence'] == old_saved['sequence'] == name
            current_path = current_output / (name + '.txt')
            old_path = old_output / (name + '.txt')
            gt_path = Path(current_plan['dataset_root']) / name / 'groundtruth.txt'
            assert common.sha(current_path) == saved['bbox_sha256']
            assert common.sha(old_path) == old_saved['bbox_sha256']
            assert common.sha(gt_path) == case['gt_sha256']
            current = metric._load_rows(current_path, 4)
            old = metric._load_rows(old_path, 4)
            gt = metric._load_rows(gt_path, 4)
            assert len(current) == len(old) == len(gt) == case['frames']
            image = cv2.imread(str(Path(current_plan['dataset_root']) / name / 'color/00000001.jpg'))
            height, width = image.shape[:2]
            new_iou, visible = metric._vot_overlaps(current, gt, width, height)
            old_iou, old_visible = metric._vot_overlaps(old, gt, width, height)
            assert np.array_equal(visible, old_visible)
            conditions = (
                ('candidate_harmed', (new_iou <= .1) & (old_iou >= .5) & visible, current, new_iou),
                ('candidate_helped', (old_iou <= .1) & (new_iou >= .5) & visible, old, old_iou),
            )
            for direction, mask, worse_boxes, worse_iou in conditions:
                for group in runs(mask):
                    if len(group) < 10:
                        continue
                    start = int(group[0])
                    assert start > 0
                    center, full = crop_contains(worse_boxes[start - 1], gt[start])
                    low_start = start
                    while low_start > 1 and visible[low_start - 1] and worse_iou[low_start - 1] <= .1:
                        low_start -= 1
                    low_center, low_full = crop_contains(worse_boxes[low_start - 1], gt[low_start])
                    events.append(dict(dataset=dataset, sequence=name, direction=direction,
                                       start=start, end=int(group[-1]) + 1, frames=len(group),
                                       center_in_worse_crop_at_onset=center,
                                       full_box_in_worse_crop_at_onset=full,
                                       severe_low_run_start=low_start,
                                       frames_after_severe_low_run_start=start - low_start,
                                       center_in_worse_crop_at_severe_low_start=low_center,
                                       full_box_in_worse_crop_at_severe_low_start=low_full,
                                       previous_gt_valid_at_severe_low_start=bool(visible[low_start - 1]),
                                       worse_iou_previous=float(worse_iou[start - 1]),
                                       candidate_iou_onset=float(new_iou[start]),
                                       old_m82_iou_onset=float(old_iou[start])))
        table = {}
        for direction in ('candidate_harmed', 'candidate_helped'):
            selected = [e for e in events if e['direction'] == direction]
            unique_low = {(e['sequence'], e['severe_low_run_start']): e for e in selected}
            table[direction] = dict(segments=len(selected), frames=sum(e['frames'] for e in selected),
                                    sequences=len({e['sequence'] for e in selected}),
                                    onset_center_in_crop=sum(e['center_in_worse_crop_at_onset'] for e in selected),
                                    onset_full_box_in_crop=sum(e['full_box_in_worse_crop_at_onset'] for e in selected),
                                    unique_severe_low_runs=len(unique_low),
                                    severe_low_start_center_in_crop=sum(e['center_in_worse_crop_at_severe_low_start'] for e in unique_low.values()),
                                    severe_low_start_full_box_in_crop=sum(e['full_box_in_worse_crop_at_severe_low_start'] for e in unique_low.values()),
                                    severe_low_start_after_valid_gt=sum(e['previous_gt_valid_at_severe_low_start'] for e in unique_low.values()),
                                    worse_previous_iou_ge_half=sum(e['worse_iou_previous'] >= .5 for e in selected),
                                    longest=sorted(selected, key=lambda e: e['frames'], reverse=True)[:8])
        summary['datasets'][dataset] = dict(sequence_count=len(cases),
                                            candidate_receipt_sha256=common.sha(current_output / 'receipt.json'),
                                            old_m82_receipt_sha256=common.sha(old_output / 'receipt.json'),
                                            counts=table, events=events)
    destination = ROOT / 'posthoc/candidate_vs_m82_ope_onsets.json'
    destination.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(dict(output=str(destination),
                          counts={name: data['counts'] for name, data in summary['datasets'].items()})))


if __name__ == '__main__':
    main()
