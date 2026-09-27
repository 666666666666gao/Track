"""GT geometry upper bound for fixed-budget full-frame square search tiles."""
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path('/root/autodl-tmp/sttrack_train_reappearance_20260928')
DATASET = Path('/root/autodl-tmp/depthtrack/train/sequences')


def positions(length, side, count):
    return [round(value) for value in np.linspace(0, max(0, length - side), count)]


def main():
    inventory_path = ROOT / 'train_reappearance_inventory.json'
    inventory = json.loads(inventory_path.read_text())
    selected = [event for event in inventory['events']
                if not event['prior_development_22'] and not event['center_in_last_gt_crop']]
    assert len(selected) == 245
    records = []
    for event in selected:
        image_path = DATASET / event['sequence'] / 'color' / f"{event['return_frame'] + 1:08d}.jpg"
        image = cv2.imread(str(image_path))
        assert image is not None
        frame_height, frame_width = image.shape[:2]
        gx, gy, width, height = event['return_gt_box']
        image_center = 0 <= gx + width / 2 < frame_width and 0 <= gy + height / 2 < frame_height
        image_full = 0 <= gx and 0 <= gy and gx + width <= frame_width and gy + height <= frame_height
        row = dict(sequence=event['sequence'], return_frame=event['return_frame'],
                   center_inside_image=bool(image_center), full_box_inside_image=bool(image_full),
                   tiles={})
        for grid in (1, 2, 3):
            side = math.ceil(max(frame_width, frame_height) / grid)
            centers = []
            full_boxes = []
            for x1 in positions(frame_width, side, grid):
                for y1 in positions(frame_height, side, grid):
                    centers.append(image_center and x1 <= gx + width / 2 < x1 + side and
                                   y1 <= gy + height / 2 < y1 + side)
                    full_boxes.append(image_full and x1 <= gx and y1 <= gy and
                                      gx + width <= x1 + side and gy + height <= y1 + side)
            row['tiles'][str(grid)] = dict(calls=grid * grid, side=side,
                                          center_covered=bool(any(centers)),
                                          full_box_covered=bool(any(full_boxes)),
                                          target_min_256=min(width, height) * 256 / side)
        records.append(row)
    table = {}
    for grid in (1, 2, 3):
        item = [row['tiles'][str(grid)] for row in records]
        table[str(grid)] = dict(calls=grid * grid, events=len(item),
                                center_inside_image=sum(x['center_inside_image'] for x in records),
                                full_box_inside_image=sum(x['full_box_inside_image'] for x in records),
                                center_covered=sum(x['center_covered'] for x in item),
                                full_box_covered=sum(x['full_box_covered'] for x in item),
                                target_min_256_median=float(np.median([x['target_min_256'] for x in item])),
                                full_box_and_min_16=sum(x['full_box_covered'] and
                                                        x['target_min_256'] >= 16 for x in item))
    report = dict(status='train_only_tile_geometry_upper_bound',
                  inventory_sha256=hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
                  scope='Former fit130 invalid-to-valid events with target center outside the '
                        'static 4x last-GT local crop; 245 cases.',
                  construction='Square tiles of side ceil(max(image width,height)/grid), '
                               'with 1x1, 2x2, or 3x3 evenly spaced top-left positions; '
                               'each tile would be resized to 256x256.',
                  table=table, events=records,
                  limitation='GT is used only for this train-only geometric capacity audit. '
                             'No tracker generates or ranks tiles, the target may be occluded, '
                             'and full-box coverage with sufficient pixels does not imply '
                             'a correct dense prediction or safe recovery.')
    destination = ROOT / 'train_tile_capacity.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(table))


if __name__ == '__main__':
    main()
