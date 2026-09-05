"""Finalize sealed M50 trajectories using complete SHA-bound default traces."""
import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    plan = json.loads((root/'spec.json').read_text())
    parent = Path(plan['source_root'])
    spec = json.loads((parent/'spec.json').read_text())
    repo = Path(spec['repository']); sys.path.insert(0, str(repo))
    from tools.train_sttrack_m44 import sha, check_binding
    from tools.audit_sttrack_m43 import independent_overlap
    check_binding(parent, spec)
    assert sha(parent/'spec.json') == plan['source_spec_sha256']
    assert sha(parent/'inference_inputs.json') == spec['inference_inputs_sha256']
    for name, digest in plan['source_sha256'].items():
        assert sha(repo/name) == digest, name
    cases = {c['sequence']: c for c in json.loads((parent/'inference_inputs.json').read_text())
             if c['split'] == 'development'}
    assert len(cases) == 22 and sorted(cases) == plan['sequences']
    assert (root/'recursive.exit').read_text().strip() == '0'
    receipt = json.loads((root/'recursive_receipt.json').read_text())
    assert receipt['status'] == 'complete' and receipt['spec_sha256'] == sha(root/'spec.json')
    assert {r['sequence'] for r in receipt['sequences']} == set(cases)
    assert sha(parent/'recursive_result.json') == plan['default_result_sha256']
    control = json.loads((parent/'recursive_result.json').read_text())
    table, per, checks = [], {}, []
    baseline_rows = defaultdict(list)
    for path, digest in spec['baseline_trace_sha256'].items():
        assert sha(path) == digest
        for row in json.loads(Path(path).read_text())['rows']:
            if row['sequence'] in cases:
                baseline_rows[row['sequence']].append(row)
    for item in receipt['sequences']:
        name = item['sequence']; case = cases[name]
        path = root/'recursive'/(name+'.json')
        assert sha(path) == item['sha256']
        data = json.loads(path.read_text()); rows = data['rows']
        assert [r['frame'] for r in rows] == list(range(case['frames']))
        boxes = np.asarray([r['bbox'] for r in rows]); assert np.isfinite(boxes).all()
        assert (boxes[:,2:] > 0).all() and np.array_equal(boxes[0], case['init_bbox'])
        gt = np.loadtxt(Path(spec['dataset_root'])/name/'groundtruth.txt', delimiter=',')[:len(rows)]
        values, measured = independent_overlap(boxes, gt)
        metrics = dict(valid_frames=int(measured['valid_frames']), iou_sum=float(measured['iou_sum']),
                       mean_iou=float(measured['mean_iou']), low_iou_frames=int(measured['low_iou_frames']),
                       failure_episodes=int(measured['failure_episodes']))
        per[name] = metrics
        base_rows = sorted(baseline_rows[name], key=lambda r: r['frame_index'])
        assert [r['frame_index'] for r in base_rows] == list(range(case['frames']))
        baseline = np.asarray([r['public_bbox'] for r in base_rows])
        assert baseline.shape == boxes.shape
        _, base = independent_overlap(baseline, gt)
        for key, value in base.items():
            assert math.isclose(value, control['per_sequence']['default'][name][key], rel_tol=1e-12, abs_tol=1e-10)
        template_scale = math.sqrt(float(np.prod(boxes[0,2:])))
        native_writes, extra_writes = 0, []
        for r in rows[1:]:
            assert math.isfinite(r['score'])
            scale = math.sqrt(r['bbox'][2]*r['bbox'][3])
            ratio = max(scale/template_scale, template_scale/scale)
            assert r['template_reference_scale'] == template_scale and r['template_scale_ratio'] == ratio
            native = r['frame'] % 50 == 0 and r['score'] > .75
            expected = 'native' if native else ('scale' if r['score'] > .75 and ratio >= plan['scale_change'] else None)
            assert r['template_update'] == expected
            if native: native_writes += 1
            if expected == 'scale': extra_writes.append(r['frame'])
            if expected is not None: template_scale = scale
        prefix_end = extra_writes[0]+1 if extra_writes else len(rows)
        assert np.array_equal(boxes[:prefix_end], baseline[:prefix_end]), name
        assert all(r['score'] == base_rows[r['frame']]['public_score'] for r in rows[1:prefix_end])
        assert native_writes == item['native_updates'] and len(extra_writes) == item['scale_updates']
        checks.append(dict(sequence=name, exact_default_prefix_through_first_write=True,
                           first_scale_write=extra_writes[0] if extra_writes else None))
        table.append(dict(sequence=name, **metrics, native_updates=native_writes, scale_updates=len(extra_writes),
                          mean_iou_gain=metrics['mean_iou']-base['mean_iou'],
                          low_frame_delta=metrics['low_iou_frames']-int(base['low_iou_frames']),
                          episode_delta=metrics['failure_episodes']-int(base['failure_episodes'])))
    total = {k:sum(x[k] for x in per.values()) for k in ['valid_frames','iou_sum','low_iou_frames','failure_episodes']}
    total['mean_iou'] = total['iou_sum']/total['valid_frames']
    default = control['aggregates']['default']
    positive = sum(r['mean_iou_gain']>0 for r in table)
    broken = sorted(n for n,r in per.items() if control['per_sequence']['default'][n]['failure_episodes']==0 and r['failure_episodes']>0)
    gates = dict(mean_iou=total['mean_iou'] >= default['mean_iou']+.01,
                 fewer_low_frames=total['low_iou_frames']<default['low_iou_frames'],
                 no_episode_increase=total['failure_episodes']<=default['failure_episodes'],
                 positive_sequences=positive>=3, successful_sequence_protection=not broken)
    result = dict(status='complete', integrity_pass=True, primary_pass=all(gates.values()), gates=gates,
                  aggregates=dict(default=default,m50=total), per_sequence=per, positive_sequences=positive,
                  new_failure_sequences=broken, native_updates=sum(r['native_updates'] for r in table),
                  scale_updates=sum(r['scale_updates'] for r in table), prefix_checks=checks,
                  frames=sum(c['frames'] for c in cases.values()), spec_sha256=sha(root/'spec.json'),
                  checkpoint_sha256=spec['checkpoint_sha256'], source_sha256=sha(__file__),
                  optimizer_steps=0, scope='Repeatedly used DepthTrack Train development sequences. Fixed template policy only; no new weights or public metrics.')
    (root/'recursive_result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (root/'per_sequence.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    print(json.dumps({k:v for k,v in result.items() if k not in ['per_sequence','prefix_checks']},indent=2))
    return


if __name__ == '__main__': main()
