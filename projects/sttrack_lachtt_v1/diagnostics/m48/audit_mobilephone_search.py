"""Read-only native crop geometry at the confirmed reappearance failure."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root = Path('/root/autodl-tmp/sttrack_m48_native_continuity_v1_20260905')
source = Path('/root/autodl-tmp/sttrack_m44_candidate_set_v1_20260905')
spec = json.loads((source / 'spec.json').read_text())
diagnosis = json.loads((root / 'mobilephone_diagnosis.json').read_text())
assert sha(root / 'recursive/mobilephone02_indoor.json') == diagnosis['trajectory_sha256']
frame = diagnosis['failure_episodes'][0]['start']
assert frame == 497
control = json.loads((root / 'recursive/mobilephone02_indoor.json').read_text())['rows']
ablation_root = root / 'phone_skip_write450'
ablation = json.loads((ablation_root / 'result.json').read_text())
assert sha(ablation_root / 'trajectory.json') == ablation['trajectory_sha256']
changed = json.loads((ablation_root / 'trajectory.json').read_text())['rows']
baseline = {}
for path, digest in spec['baseline_trace_sha256'].items():
    assert sha(path) == digest
    for row in json.loads(Path(path).read_text())['rows']:
        if row['sequence'] == 'mobilephone02_indoor':
            baseline[row['frame_index']] = row
gt_path = Path(spec['dataset_root']) / 'mobilephone02_indoor/groundtruth.txt'
gt_all = np.loadtxt(gt_path, delimiter=',')
gt = gt_all[frame].tolist()
assert np.isfinite(gt).all() and gt[2] > 0 and gt[3] > 0
assert not (np.isfinite(gt_all[frame-1]).all() and (gt_all[frame-1, 2:] > 0).all())
priors = dict(native_default=baseline[frame-1]['public_bbox'], m48=control[frame-1]['bbox'], skip_write450=changed[frame-1]['bbox'])
rows = {}
for name, prior in priors.items():
    x, y, w, h = prior
    size = math.ceil(math.sqrt(w*h)*4)
    left = round(x + .5*w - .5*size)
    top = round(y + .5*h - .5*size)
    right, bottom = left+size, top+size
    gx, gy, gw, gh = gt
    area = max(0., min(right, gx+gw)-max(left, gx))*max(0., min(bottom, gy+gh)-max(top, gy))
    rows[name] = dict(previous_bbox=prior, factor=4, crop_xyxy=[left, top, right, bottom],
        gt_center_inside=left <= gx+.5*gw < right and top <= gy+.5*gh < bottom,
        full_gt_inside=left <= gx and top <= gy and gx+gw <= right and gy+gh <= bottom,
        gt_area_covered_fraction=area/(gw*gh))
report = dict(status='complete', frame_zero_based=frame, previous_frame_has_no_valid_gt=True, current_gt=gt,
    rows=rows, source_sha256=sha(__file__), groundtruth_sha256=sha(gt_path),
    sampling_source_sha256=sha(Path(spec['repository']) / 'lib/train/data/processing_utils.py'),
    control_trajectory_sha256=diagnosis['trajectory_sha256'], ablation_trajectory_sha256=ablation['trajectory_sha256'],
    scope='Diagnostic geometry only, using native ceil(sqrt(w*h)*4) and rounded crop origin from sample_target. GT is never used to recenter a tracker or create a deployable trigger.')
(root / 'mobilephone_search_geometry.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
