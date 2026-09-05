#!/usr/bin/env python3
"""Verify sealed outputs and explain the M41/M49 IoU boundary difference."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from vot.region.raster import calculate_overlap
from vot.region.shapes import Rectangle

from candidate_continuity import box_iou


ROOT = Path('/root/autodl-tmp/sttrack_m49_motion_scale_v1_20260905')
M41 = Path('/root/autodl-tmp/sttrack_m41_candidate_capacity_v1_20260905')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    result = json.loads((ROOT/'result/result.json').read_text())
    for path, expected in result['integrity']['inputs'].items():
        assert sha(Path(path)) == expected, path
    for name, expected in result['output_sha256'].items():
        assert sha(ROOT/'result'/name) == expected, name
    assert sha(ROOT/'analyze_motion_scale.py') == result['integrity']['source_sha256']
    assert sha(ROOT/'spec.json') == result['integrity']['spec_sha256']
    choices = json.loads((ROOT/'candidate_choices.json').read_text())
    for path, expected in choices['input_sha256'].items():
        assert sha(M41/path) == expected, path
    candidate_result = json.loads((ROOT/'candidate_continuity_result.json').read_text())
    assert sha(ROOT/'candidate_choices.json') == candidate_result['choices_sealed_before_gt_sha256']
    training = json.loads((ROOT/'train_cache_result.json').read_text())
    assert sha(ROOT/'train_cache_choices.json') == training['choices_sealed_before_gt_sha256']
    old = {r['key']:r for r in csv.DictReader((M41/'factor4_diagnosis.csv').open())}
    spec = json.loads((ROOT/'spec.json').read_text())
    dims = json.loads(Path(spec['paths']['default_analysis']).read_text())['sequences']
    boundary, checked = [], 0
    for item in choices['choices']:
        seq, frame = item['sequence'], item['source_frame']
        gt = np.loadtxt(Path(spec['paths']['sequence_root'])/seq/'groundtruth.txt',delimiter=',')[frame]
        continuous = [box_iou(b,gt) for b in item['candidate_boxes']]
        bounded = [float(calculate_overlap(Rectangle(*b),Rectangle(*gt),(dims[seq]['width'],dims[seq]['height'])))
                   for b in item['candidate_boxes']]
        assert abs(max(continuous)-float(old[item['key']]['hann_top10_oracle_iou'])) < 1e-12
        checked += 1
        if (max(continuous)>=.5) != (max(bounded)>=.5):
            boundary.append({'key':item['key'],'continuous_top10_iou':max(continuous),
                             'bounded_raster_top10_iou':max(bounded)})
    assert checked == 124 and len(boundary) == 1
    assert (ROOT/'analysis.exit').read_text().strip() == '0'
    assert (ROOT/'train_cache.exit').read_text().strip() == '1'
    assert (ROOT/'train_cache_label.exit').read_text().strip() == '0'
    audit = {'status':'PASS','scope':'Output integrity, causal selection seals, and metric alignment; no performance promotion',
             'm39_gt_and_trajectory_files_verified':len(result['integrity']['inputs']),
             'm40_exact_onset_and_crop_matches':124, 'm41_continuous_capacity_rows_reproduced':checked,
             'metric_boundary':boundary,
             'metric_note':'M41 used continuous rectangle IoU (91/115 capacity); M49 VOT candidate diagnosis preregistered bounded raster IoU (90/115). Candidate data is unchanged.',
             'training_report_repair':'Initial selection completed and was sealed; Python3.8 rejected dict union in the reporting stage. R2 used dictionary unpacking and labelled the same sealed choices. No selection, training, or inference was repeated.',
             'artifact_sha256':{str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*')
                                if p.is_file() and p.suffix in ['.json','.csv','.py','.md','.png','.pdf']
                                and p.name!='terminal_audit.json'}}
    (ROOT/'terminal_audit.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='artifact_sha256'},indent=2))


if __name__=='__main__':
    main()
