#!/usr/bin/env python3
"""GT-free fixed-rule cache choices followed by DepthTrack Train labels."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path('/root/autodl-tmp/sttrack_m49_motion_scale_v1_20260905')
M44 = Path('/root/autodl-tmp/sttrack_m44_candidate_set_v1_20260905')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def center(box):
    return np.asarray(box[:2]) + .5 * np.asarray(box[2:])


def iou(a,b):
    a,b = np.asarray(a),np.asarray(b)
    intersection = np.prod(np.maximum(0,np.minimum(a[:2]+a[2:],b[:2]+b[2:])-np.maximum(a[:2],b[:2])))
    return float(intersection / (np.prod(a[2:])+np.prod(b[2:])-intersection))


def main():
    torch.set_num_threads(2)
    spec = json.loads((M44/'spec.json').read_text())
    assert sha(M44/'inference_inputs.json') == spec['inference_inputs_sha256']
    inputs = json.loads((M44/'inference_inputs.json').read_text())
    receipts = [json.loads((M44/f'shard{i}_receipt.json').read_text()) for i in [0,1]]
    expected = {r['sequence']:r['feature_sha256'] for receipt in receipts for r in receipt['sequences']}
    choices, hashes = [], {}
    methods = ['native','nearest_center','previous_box_iou','velocity_box_iou']
    for info in inputs:
        sequence = info['sequence']
        path = M44/'features'/(sequence+'.pt')
        digest = sha(path)
        assert digest == expected[sequence], sequence
        hashes[sequence] = digest
        features = torch.load(path,map_location='cpu')
        history = {r['frame_index']:np.asarray(r['bbox']) for r in info['expected_rows']}
        for index, record in enumerate(features['records']):
            frame = record['frame']
            assert record['previous_frame'] == frame-1
            prior = features['previous_public_bbox'][index].numpy().astype(float)
            assert np.max(np.abs(prior-history[frame-1])) <= .001
            boxes = features['current_boxes'][index].numpy().astype(float)
            assert np.max(np.abs(boxes[0]-features['public_bbox'][index].numpy())) <= .001
            velocity = prior.copy()
            velocity[:2] += center(prior)-center(history[frame-2])
            ranks = {'native':0, 'nearest_center':int(np.argmin([np.linalg.norm(center(b)-center(prior)) for b in boxes])),
                     'previous_box_iou':int(np.argmax([iou(b,prior) for b in boxes])),
                     'velocity_box_iou':int(np.argmax([iou(b,velocity) for b in boxes]))}
            choices.append({'key':record['key'],'sequence':sequence,'split':info['split'],
                            'rank':ranks,'box':{m:boxes[ranks[m]].tolist() for m in methods}})
    choice_path = ROOT/'train_cache_choices.json'
    with choice_path.open('x') as f:
        json.dump({'feature_sha256':hashes,'choices':choices},f,indent=2,allow_nan=False); f.write('\n')
    digest = sha(choice_path)
    # No current/previous GT label has been accessed above.
    assert sha(M44/'training_labels.json') == spec['labels_sha256']
    labels = json.loads((M44/'training_labels.json').read_text())
    rows = []
    for row in json.loads(choice_path.read_text())['choices']:
        target = labels[row['key']]['current']
        rows.append({k:row[k] for k in ['key','sequence','split']} | {
            'strata':labels[row['key']]['strata'], 'rank':row['rank'],
            'iou':{m:iou(row['box'][m],target) if target is not None else None for m in methods}})
    assert sha(choice_path) == digest
    def summarize(subset):
        valid = [r for r in subset if r['iou']['native'] is not None]
        return {'pairs':len(subset), 'valid_gt_pairs':len(valid),
                'unavailable_gt_pairs':len(subset)-len(valid),
                **{m:{'correct':sum(r['iou'][m]>=.5 for r in valid),
                      'mean_iou':float(np.mean([r['iou'][m] for r in valid])),
                      'rescued_native_wrong':sum(r['iou']['native']<.5 and r['iou'][m]>=.5 for r in valid),
                      'broke_native_correct':sum(r['iou']['native']>=.5 and r['iou'][m]<.5 for r in valid),
                      'broke_native_correct_to_low':sum(r['iou']['native']>=.5 and r['iou'][m]<=.1 for r in valid)}
                   for m in methods}}
    result = {'schema':'sttrack_m49_train_cache_continuity_v1','status':'complete',
              'scope':'Existing DepthTrack Train event cache, no new optimization or recursion',
              'metric':'continuous box IoU, matching M44 cache labels; not bounded VOT raster IoU',
              'verified_feature_files':len(hashes),'pairs':len(rows),
              'choices_sealed_before_gt_sha256':digest,'labels_sha256':sha(M44/'training_labels.json'),
              'source_sha256':sha(Path(__file__)),
              'by_split':{s:summarize([r for r in rows if r['split']==s]) for s in ['fit','development']},
              'per_sequence':{s:summarize([r for r in rows if r['sequence']==s]) for s in sorted(expected)}}
    assert len(rows)==2101 and len(hashes)==85
    for name,data in [('train_cache_result.json',result),('train_cache_rows.json',rows)]:
        with (ROOT/name).open('x') as f: json.dump(data,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='per_sequence'},indent=2))


if __name__=='__main__':
    main()
