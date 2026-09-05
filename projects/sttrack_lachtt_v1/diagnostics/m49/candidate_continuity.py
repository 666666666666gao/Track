#!/usr/bin/env python3
"""Seal GT-free continuity rankings, then label existing M41 candidates."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from vot.region.raster import calculate_overlap
from vot.region.shapes import Rectangle


ROOT = Path('/root/autodl-tmp/sttrack_m49_motion_scale_v1_20260905')
M41 = Path('/root/autodl-tmp/sttrack_m41_candidate_capacity_v1_20260905')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def center(b):
    return np.asarray(b[:2]) + .5 * np.asarray(b[2:])


def box_iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    intersection = np.prod(np.maximum(0, np.minimum(a[:2]+a[2:], b[:2]+b[2:]) - np.maximum(a[:2],b[:2])))
    return float(intersection / (np.prod(a[2:]) + np.prod(b[2:]) - intersection))


def main():
    inputs = {a['key']: a for a in json.loads((M41/'inputs.json').read_text())}
    choices, input_hashes = [], {'inputs.json': sha(M41/'inputs.json')}
    for key, info in sorted(inputs.items()):
        path = M41/'candidates'/(key+'.json')
        data = json.loads(path.read_text())
        input_hashes[str(path)] = sha(path)
        prior = np.asarray(data['prior'])
        history = [info['init_bbox']] + info['expected_boxes']
        assert info['progress'] >= 2
        assert np.max(np.abs(prior - history[info['progress']-1])) <= .001
        velocity = prior.copy()
        velocity[:2] += center(prior) - center(history[info['progress']-2])
        candidates = data['factor4']['hann']
        positions = [-float(np.linalg.norm(center(a['bbox'])-center(prior))) for a in candidates]
        overlaps = [box_iou(a['bbox'], prior) for a in candidates]
        velocity_overlaps = [box_iou(a['bbox'], velocity) for a in candidates]
        choices.append({'key': key, 'sequence': info['sequence'], 'source_frame': info['onset_frame'],
                        'native': 0, 'nearest_center': int(np.argmax(positions)),
                        'previous_box_iou': int(np.argmax(overlaps)),
                        'velocity_box_iou': int(np.argmax(velocity_overlaps)),
                        'previous_box': prior.tolist(), 'velocity_box': velocity.tolist(),
                        'candidate_boxes': [a['bbox'] for a in candidates]})
    choice_path = ROOT/'candidate_choices.json'
    with choice_path.open('x') as f:
        json.dump({'input_sha256': input_hashes, 'choices': choices}, f, indent=2, allow_nan=False)
        f.write('\n')
    choice_hash = sha(choice_path)
    # GT is first accessed only after the choices are sealed on disk.
    census = {r['anchor_key']: r for r in csv.DictReader((ROOT/'result/failure_onsets.csv').open())}
    spec = json.loads((ROOT/'spec.json').read_text())
    dimensions = json.loads(Path(spec['paths']['default_analysis']).read_text())['sequences']
    methods = ['native', 'nearest_center', 'previous_box_iou', 'velocity_box_iou']
    labelled = []
    for item in json.loads(choice_path.read_text())['choices']:
        sequence, frame = item['sequence'], item['source_frame']
        gt_path = Path(spec['paths']['sequence_root'])/sequence/'groundtruth.txt'
        gt = np.loadtxt(gt_path, delimiter=',')[frame]
        bounds = (dimensions[sequence]['width'], dimensions[sequence]['height'])
        ious = [float(calculate_overlap(Rectangle(*b), Rectangle(*gt), bounds)) for b in item['candidate_boxes']]
        labelled.append({'key': item['key'], 'sequence': sequence, 'source_frame': frame,
                         'inside_factor4': census[item['key']]['actual_center_inside'] == 'True',
                         'top10_capacity_iou_ge_0_5': max(ious) >= .5,
                         **{method+'_rank_one_based': item[method]+1 for method in methods},
                         **{method+'_iou': ious[item[method]] for method in methods}})
    assert sha(choice_path) == choice_hash
    assert len(labelled) == 124
    with (ROOT/'candidate_continuity_rows.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(labelled[0])); writer.writeheader(); writer.writerows(labelled)
    def summary(rows):
        return {'count': len(rows), 'top10_correct_capacity': sum(a['top10_capacity_iou_ge_0_5'] for a in rows),
                **{m: {'correct_iou_ge_0_5': sum(a[m+'_iou'] >= .5 for a in rows),
                       'low_iou_le_0_1': sum(a[m+'_iou'] <= .1 for a in rows),
                       'mean_iou': float(np.mean([a[m+'_iou'] for a in rows])) if rows else None}
                   for m in methods}}
    result = {'schema': 'sttrack_m49_candidate_continuity_diagnosis_v1', 'status': 'complete',
              'new_inference_frames': 0, 'optimizer_steps': 0, 'recursive_tracking_gain_measured': False,
              'choices_sealed_before_gt_sha256': choice_hash, 'source_sha256': sha(Path(__file__)),
              'census_result_sha256': sha(ROOT/'result/result.json'),
              'all': summary(labelled), 'inside_factor4': summary([a for a in labelled if a['inside_factor4']]),
              'outside_factor4': summary([a for a in labelled if not a['inside_factor4']]),
              'per_sequence': {s: summary([a for a in labelled if a['sequence']==s]) for s in sorted(dimensions)}}
    with (ROOT/'candidate_continuity_result.json').open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({k: v for k,v in result.items() if k != 'per_sequence'}, indent=2))


if __name__ == '__main__':
    main()
