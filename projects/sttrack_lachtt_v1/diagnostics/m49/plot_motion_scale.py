#!/usr/bin/env python3
"""Standalone scientific plots from sealed measurements and fixed image pairs."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


ROOT = Path('/root/autodl-tmp/sttrack_m49_motion_scale_v1_20260905')
M41 = Path('/root/autodl-tmp/sttrack_m41_candidate_capacity_v1_20260905')


def main():
    rows = list(csv.DictReader((ROOT/'result/failure_onsets.csv').open()))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for inside, color, label in [('True', '#2563a8', 'GT center inside factor 4 (115)'),
                                  ('False', '#db702a', 'GT center outside factor 4 (9)')]:
        subset = [r for r in rows if r['actual_center_inside'] == inside]
        axes[0].scatter([float(r['gt_displacement_norm']) for r in subset],
                        [float(r['pred_jump_gt_norm']) for r in subset], c=color, alpha=.6, s=28, label=label)
        axes[1].scatter([float(r['gt_scale_change_factor']) for r in subset],
                        [float(r['pred_scale_change_factor']) for r in subset], c=color, alpha=.6, s=28)
    axes[0].plot([0,3],[0,3], ':', color='gray', lw=1)
    axes[0].set(xlabel='GT center displacement / previous GT sqrt(area)',
                ylabel='Prediction center jump / previous GT sqrt(area)', xlim=(-.05,3), ylim=(-.05,2.05),
                title='Motion: annotation versus tracker')
    axes[0].legend(loc='lower right', fontsize=8)
    axes[1].plot([1,1.8],[1,1.8], ':', color='gray', lw=1)
    axes[1].set(xlabel='GT linear scale change factor', ylabel='Prediction linear scale change factor',
                xlim=(.98,1.8), ylim=(.98,1.8), title='Scale: max(ratio, 1 / ratio)')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle('M49: 124 native STTrack failure onsets (72 unique frame transitions)', fontsize=12)
    fig.savefig(ROOT/'motion_scale_scatter.png', dpi=180)
    fig.savefig(ROOT/'motion_scale_scatter.pdf')
    plt.close(fig)

    keys = ['cube02_indoor_1@300B','two_tennis_balls_3@0F','toy09_indoor_1@0F','yogurt_indoor_1@1000B']
    seq_root = Path(json.loads((ROOT/'spec.json').read_text())['paths']['sequence_root'])
    fig, axes = plt.subplots(4, 2, figsize=(13, 16), constrained_layout=True)
    inputs = {}
    details = []
    for i, key in enumerate(keys):
        row = next(r for r in rows if r['anchor_key'] == key)
        data = json.loads((M41/'candidates'/(key+'.json')).read_text())
        seq = row['sequence']
        gt = np.loadtxt(seq_root/seq/'groundtruth.txt', delimiter=',')
        for j, (frame, prediction) in enumerate([(int(row['previous_source_frame']),data['prior']),
                                                (int(row['source_frame']),data['public_bbox'])]):
            image_path = seq_root/seq/'color'/f'{frame+1:08d}.jpg'
            inputs[str(image_path)] = hashlib.sha256(image_path.read_bytes()).hexdigest()
            ax = axes[i,j]
            ax.imshow(Image.open(image_path))
            for box, color in [(gt[frame], '#00ff77'), (prediction, '#ff851b')]:
                ax.add_patch(Rectangle(box[:2], box[2], box[3], fill=False, color=color, lw=2))
            ax.set_title(f'{key} | {"previous" if j==0 else "onset"} source frame {frame+1}', fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 1:
                px = np.linalg.norm((np.asarray(data['public_bbox'][:2]) + .5*np.asarray(data['public_bbox'][2:])) -
                                    (np.asarray(data['prior'][:2]) + .5*np.asarray(data['prior'][2:])))
                ax.set_xlabel(f'GT center change {float(row["gt_displacement_px"]):.2f} px; prediction change {px:.2f} px')
                details.append({'key':key,'gt_center_displacement_px':float(row['gt_displacement_px']),
                                'prediction_center_displacement_px':float(px)})
    fig.suptitle('Green = GT; orange = native prediction. Pairs follow evaluation direction, including backward runs.', fontsize=12)
    fig.savefig(ROOT/'adjacent_frame_examples.png', dpi=130)
    fig.savefig(ROOT/'adjacent_frame_examples.pdf')
    plt.close(fig)
    (ROOT/'figure_inputs.json').write_text(json.dumps({'image_sha256':inputs,'examples':details},indent=2)+'\n')
    print(json.dumps(details,indent=2))


if __name__ == '__main__':
    main()
