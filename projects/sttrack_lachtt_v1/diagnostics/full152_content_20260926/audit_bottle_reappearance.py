"""Read sealed M67 Category/Empty states around one known CDTB reappearance."""
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/root/autodl-tmp/sttrack_full152_content_20260926')
sys.path.insert(0, str(ROOT))
import analyze_content as audit


def main():
    sequence = 'bottle_room_occ_1'
    positions = [481, 573, 574, 575, 576, 577, 580, 600]
    result = {'status': 'complete', 'sequence': sequence,
              'scope': 'Posthoc saved trajectories, nominal factor-4 crop; no model forward or state intervention.',
              'script_sha256': audit.sha(__file__),
              'analyzer_sha256': audit.sha(audit.__file__), 'conditions': {}}
    for condition, root in [('category', audit.MAIN / 'M67/cdtb'),
                            ('empty', ROOT / 'M67/cdtb/empty')]:
        plan, cases, receipt, metrics = audit.family(root)
        case = next(c for c in cases if c['sequence'] == sequence)
        saved = next(r for r in receipt['sequences'] if r['sequence'] == sequence)
        folder = Path(plan['dataset_root']) / sequence
        assert audit.sha(folder / 'groundtruth.txt') == case['gt_sha256']
        spec = importlib.util.spec_from_file_location('sealed_metric', plan['metric_source'])
        metric = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metric)
        gt = metric._load_rows(folder / 'groundtruth.txt', 4)
        height, width = cv2.imread(str(folder / 'color/00000001.jpg')).shape[:2]
        boxes, scores, overlaps, valid = audit.load_sequence(metric, plan, case, saved, gt, width, height)
        rows = []
        for t in positions:
            previous = boxes[t - 1]
            center = previous[:2] + previous[2:] / 2
            side = float(np.ceil(4 * np.sqrt(previous[2] * previous[3])))
            row = {'frame_zero_based': t, 'previous_bbox': previous.tolist(),
                   'nominal_crop_xywh': [*(center - side / 2).tolist(), side, side],
                   'bbox': boxes[t].tolist(), 'score': float(scores[t]),
                   'gt_valid': bool(valid[t]), 'gt_bbox': None,
                   'gt_center_inside_nominal_crop': None, 'iou': None}
            if valid[t]:
                row['gt_bbox'] = gt[t].tolist()
                row['iou'] = float(overlaps[t])
                row['gt_center_inside_nominal_crop'] = bool(
                    (np.abs(gt[t, :2] + gt[t, 2:] / 2 - center) <= side / 2).all())
            rows.append(row)
        result['conditions'][condition] = {
            'plan_sha256': audit.sha(root / 'plan.json'),
            'receipt_sha256': audit.sha(Path(plan['output']) / 'receipt.json'),
            'saved_sequence': saved, 'gt_sha256': case['gt_sha256'], 'frames': rows}
    target = ROOT / 'analysis/M67_bottle_reappearance_category_empty.json'
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
