"""Diagnostic intervention on one sealed template write, not a deployable rule."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch


root = Path('/root/autodl-tmp/sttrack_m48_native_continuity_v1_20260905')
plan = json.loads((root / 'spec.json').read_text())
source = Path(plan['source_root'])
spec = json.loads((source / 'spec.json').read_text())
repo = Path(spec['repository'])
sys.path.insert(0, str(repo))
from tools.train_sttrack_m44 import sha, check_binding
from tools.audit_sttrack_m43 import independent_overlap
from lib.config.sttrack.config import cfg, update_config_from_file
from lib.test.tracker.sttrack_candidate_continuity import STTrackCandidateContinuity
from lib.train.dataset.depth_utils import get_rgbd_frame

check_binding(source, spec)
assert sha(root / 'recursive_result.json') == 'e3934bd736e9f14b6009cad4c5c05090a2527c3410aa54bb179eab2d84feeb7b'
diagnosis = json.loads((root / 'mobilephone_diagnosis.json').read_text())
assert diagnosis['only_one_override'] and diagnosis['first_override'] == 383
name = 'mobilephone02_indoor'
intervention = 450
case = next(c for c in json.loads((source / 'inference_inputs.json').read_text()) if c['sequence'] == name)
sealed_path = root / 'recursive' / (name + '.json')
assert sha(sealed_path) == diagnosis['trajectory_sha256']
sealed = json.loads(sealed_path.read_text())['rows']
assert sealed[intervention]['none'] and sealed[intervention]['score'] > .75
output = root / 'phone_skip_write450'
assert not output.exists()
output.mkdir()
protocol = dict(sequence=name, intervention_frame=intervention, intervention='Suppress only this template write; native threshold restored immediately afterward.',
    selection='This frame is selected from a sealed failed development trajectory, using its NONE output and template-write log. Timing is diagnostic and must not be deployed as a rule.',
    control_result_sha256=sha(root / 'recursive_result.json'), control_trajectory_sha256=sha(sealed_path),
    source_sha256=sha(__file__), no_training=True, no_public_evaluation=True)
(output / 'spec.json').write_text(json.dumps(protocol, indent=2) + '\n')
torch.set_num_threads(1)
update_config_from_file(str(repo / 'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
checkpoint = Path(plan['control_root']) / 'geometry_final.pth'
assert sha(checkpoint) == plan['checkpoint_sha256']
params = SimpleNamespace(cfg=cfg, checkpoint=spec['checkpoint'], template_factor=2., template_size=128,
    search_factor=4., search_size=256, save_all_boxes=False, debug=0)
tracker = STTrackCandidateContinuity(params, checkpoint)
folder = Path(spec['dataset_root']) / name


def image_at(frame):
    return get_rgbd_frame(str(folder / 'color' / f'{frame+1:08d}.jpg'),
        str(folder / 'depth' / f'{frame+1:08d}.png'), dtype='rgbcolormap', depth_clip=True)


tracker.initialize(image_at(0), dict(init_bbox=case['init_bbox']))
threshold = tracker.update_threshold
rows = [dict(frame=0, bbox=case['init_bbox'], score=1., choice=0, none=False, template_written=False)]
for frame in range(1, case['frames']):
    if frame == intervention:
        tracker.update_threshold = float('inf')
    old = tracker.z_dict[1]
    out = tracker.track(image_at(frame))
    tracker.update_threshold = threshold
    written = old is not tracker.z_dict[1]
    if frame <= intervention:
        assert out['target_bbox'] == sealed[frame]['bbox'] and float(out['best_score']) == sealed[frame]['score']
        assert out['association_candidate'] == sealed[frame]['choice']
    if frame == intervention:
        assert not written
    rows.append(dict(frame=frame, bbox=out['target_bbox'], score=float(out['best_score']),
        choice=out['association_candidate'], none=out['association_none'], template_written=written))
trajectory = output / 'trajectory.json'
trajectory.write_text(json.dumps(dict(sequence=name, rows=rows), allow_nan=False) + '\n')
# All predictions are sealed before labels are opened.
gt = np.loadtxt(folder / 'groundtruth.txt', delimiter=',')
values, metrics = independent_overlap(np.asarray([r['bbox'] for r in rows]), gt)
metrics['low_iou_frames'] = int(metrics['low_iou_frames'])
control = json.loads((root / 'recursive_result.json').read_text())['per_sequence'][name]
report = dict(status='complete', sequence=name, tracked_frames=case['frames']-1, metrics=metrics, control_metrics=control,
    mean_gain=metrics['mean_iou']-control['mean_iou'], low_frame_delta=metrics['low_iou_frames']-control['low_iou_frames'],
    episode_delta=metrics['failure_episodes']-control['failure_episodes'],
    exact_control_bbox_score_through_intervention=True, write450_suppressed=True,
    later_template_writes=[dict(frame=r['frame'], score=r['score'], gt_iou=float(values[r['frame']]) if np.isfinite(values[r['frame']]) else None)
        for r in rows if r['frame'] > intervention and r['template_written']],
    changes=[r['frame'] for r in rows if r['choice'] != 0], trajectory_sha256=sha(trajectory),
    spec_sha256=sha(output / 'spec.json'), source_sha256=sha(__file__), checkpoint_sha256=sha(checkpoint),
    scope='One post hoc timing intervention on a known development regression. This estimates this write\'s effect under the fixed tracker; not a deployable trigger, new full22 metric, formal score or gate pass.')
(output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps(report, indent=2))
