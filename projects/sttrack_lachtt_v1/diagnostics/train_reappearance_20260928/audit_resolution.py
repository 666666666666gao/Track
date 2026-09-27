"""Compare target pixel size under full-frame and last-GT local 256 inputs."""
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path('/root/autodl-tmp/sttrack_train_reappearance_20260928')
DATASET = Path('/root/autodl-tmp/depthtrack/train/sequences')


def main():
    inventory_path = ROOT / 'train_reappearance_inventory.json'
    inventory = json.loads(inventory_path.read_text())
    rows = []
    for event in inventory['events']:
        name = event['sequence']
        index = event['return_frame'] + 1
        image = cv2.imread(str(DATASET / name / 'color' / f'{index:08d}.jpg'))
        assert image is not None
        image_height, image_width = image.shape[:2]
        width, height = event['return_gt_box'][2:]
        last = event['last_gt_box']
        side = math.ceil(math.sqrt(last[2] * last[3]) * 4)
        full_min = min(width * 256 / image_width, height * 256 / image_height)
        local_min = min(width, height) * 256 / side
        rows.append(dict(sequence=name, return_frame=event['return_frame'],
                         prior_development_22=event['prior_development_22'],
                         center_outside_last_gt_crop=not event['center_in_last_gt_crop'],
                         image_width=image_width, image_height=image_height,
                         target_min_full_256=full_min,
                         target_min_local_256=local_min))
    groups = {}
    for name, selected in (
        ('all_return_events', rows),
        ('old_fit130_center_outside', [row for row in rows if not row['prior_development_22']
                                       and row['center_outside_last_gt_crop']]),
    ):
        full = np.array([row['target_min_full_256'] for row in selected])
        local = np.array([row['target_min_local_256'] for row in selected])
        groups[name] = dict(events=len(selected), full_min_median=float(np.median(full)),
                            local_min_median=float(np.median(local)),
                            full_min_below_8=int((full < 8).sum()),
                            full_min_below_16=int((full < 16).sum()),
                            local_min_below_8=int((local < 8).sum()),
                            local_min_below_16=int((local < 16).sum()))
    report = dict(status='train_only_resolution_proxy_complete',
                  inventory_sha256=hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
                  definition='Target minimum RGB pixel extent if the entire frame were resized '
                             'to 256x256 versus using the 4x last-valid-GT crop side at 256x256. '
                             'For center-outside events, the latter is a counterfactual size '
                             'only; the actual local crop contains no target center.',
                  groups=groups, events=rows,
                  limitation='This is a geometric resolution proxy. It does not run the model, '
                             'reproduce M70 preprocessing exactly, measure dense candidate '
                             'quality, or show that a target outside the local crop was recovered.')
    destination = ROOT / 'train_return_resolution.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(groups))


if __name__ == '__main__':
    main()
