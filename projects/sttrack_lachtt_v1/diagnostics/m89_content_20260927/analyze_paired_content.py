"""Read-only M89 Category/Empty/Swapped trajectory decomposition."""

import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np


CATEGORY = Path('/root/autodl-tmp/sttrack_m89_evaluation_20260926/candidate')
CONTENT = Path('/root/autodl-tmp/sttrack_m89_content_20260927')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    result = dict(status='read_only_complete', datasets={},
                  no_model_training=True, no_runtime_policy_change=True,
                  scope='completed external OPE trajectory diagnosis, not a new deployable metric')
    for dataset in ('depthtrack', 'cdtb'):
        plans = dict(category=read(CATEGORY / dataset / 'plan.json'),
                     empty=read(CONTENT / dataset / 'empty/plan.json'),
                     swapped=read(CONTENT / dataset / 'swapped/plan.json'))
        assert len({plan['bundle_sha256'] for plan in plans.values()}) == 1
        assert len({plan['cases_sha256'] for plan in plans.values()}) == 1
        assert len({plan['metric_source_sha256'] for plan in plans.values()}) == 1
        cases = read(plans['category']['cases_path'])
        assert sha(plans['category']['cases_path']) == plans['category']['cases_sha256']
        metric_spec = importlib.util.spec_from_file_location('sealed_ope_metric', plans['category']['metric_source'])
        metric = importlib.util.module_from_spec(metric_spec)
        metric_spec.loader.exec_module(metric)
        receipts = {}
        for arm, plan in plans.items():
            assert sha(plan['bundle_path']) == plan['bundle_sha256']
            path = Path(plan['output']) / 'receipt.json'
            receipt = read(path)
            assert receipt['status'] == 'complete' and len(receipt['sequences']) == len(cases)
            receipts[arm] = receipt
        rows = []
        for index, case in enumerate(cases):
            name = case['sequence']
            gt_path = Path(plans['category']['dataset_root']) / name / 'groundtruth.txt'
            assert sha(gt_path) == case['gt_sha256']
            gt = metric._load_rows(gt_path, 4)
            image = cv2.imread(str(Path(plans['category']['dataset_root']) / name / 'color/00000001.jpg'))
            height, width = image.shape[:2]
            overlaps = {}
            visible = None
            for arm, plan in plans.items():
                saved = receipts[arm]['sequences'][index]
                assert saved['sequence'] == name
                path = Path(plan['output']) / f'{name}.txt'
                assert sha(path) == saved['bbox_sha256']
                boxes = metric._load_rows(path, 4)
                assert len(boxes) == len(gt) == case['frames']
                iou, mask = metric._vot_overlaps(boxes, gt, width, height)
                if visible is None:
                    visible = mask
                else:
                    assert np.array_equal(visible, mask)
                overlaps[arm] = iou
            selected = {arm: values[visible] for arm, values in overlaps.items()}
            category, empty, swapped = (selected[arm] for arm in ('category', 'empty', 'swapped'))
            row = dict(sequence=name, valid_frames=int(visible.sum()),
                       mean_iou={arm: float(values.mean()) for arm, values in selected.items()},
                       severe_low_frames={arm: int((values <= .1).sum()) for arm, values in selected.items()},
                       category_minus_empty_mean_iou=float((category - empty).mean()),
                       category_minus_swapped_mean_iou=float((category - swapped).mean()),
                       category_bad_empty_good=int(((category <= .1) & (empty >= .5)).sum()),
                       empty_bad_category_good=int(((empty <= .1) & (category >= .5)).sum()),
                       category_bad_swapped_good=int(((category <= .1) & (swapped >= .5)).sum()),
                       swapped_bad_category_good=int(((swapped <= .1) & (category >= .5)).sum()))
            rows.append(row)
        weighted = {}
        for arm in ('category', 'empty', 'swapped'):
            weighted[arm] = dict(mean_iou=sum(row['mean_iou'][arm] * row['valid_frames'] for row in rows) /
                                sum(row['valid_frames'] for row in rows),
                                 severe_low_frames=sum(row['severe_low_frames'][arm] for row in rows))
        result['datasets'][dataset] = dict(sequences=len(rows),
            total_valid_frames=sum(row['valid_frames'] for row in rows),
            bundle_sha256=plans['category']['bundle_sha256'],
            receipt_sha256={arm: sha(Path(plan['output']) / 'receipt.json') for arm, plan in plans.items()},
            weighted=weighted,
            category_vs_empty=dict(
                sequences_better=sum(row['category_minus_empty_mean_iou'] > 0 for row in rows),
                sequences_worse=sum(row['category_minus_empty_mean_iou'] < 0 for row in rows),
                category_bad_empty_good=sum(row['category_bad_empty_good'] for row in rows),
                empty_bad_category_good=sum(row['empty_bad_category_good'] for row in rows),
                worst=sorted(rows, key=lambda row: row['category_minus_empty_mean_iou'])[:8],
                best=sorted(rows, key=lambda row: row['category_minus_empty_mean_iou'], reverse=True)[:8]),
            rows=rows)
    destination = CONTENT / 'paired_content_trajectory_audit.json'
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({name: {key: value for key, value in data.items() if key != 'rows'}
                      for name, data in result['datasets'].items()}, indent=2))


if __name__ == '__main__':
    main()
