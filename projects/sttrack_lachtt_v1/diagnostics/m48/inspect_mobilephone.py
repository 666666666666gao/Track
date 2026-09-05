"""Replay the only default-zero-episode protection failure in sealed M48."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


root = Path('/root/autodl-tmp/sttrack_m48_native_continuity_v1_20260905')
plan = json.loads((root / 'spec.json').read_text())
source = Path(plan['source_root'])
spec = json.loads((source / 'spec.json').read_text())
repo = Path(spec['repository'])
sys.path.insert(0, str(repo))
from tools.train_sttrack_m44 import sha, check_binding
from tools.train_sttrack_m42 import overlaps
from tools.audit_sttrack_m43 import independent_overlap
from lib.config.sttrack.config import cfg, update_config_from_file
from lib.test.tracker.sttrack_candidate_continuity import STTrackCandidateContinuity
from lib.train.dataset.depth_utils import get_rgbd_frame

check_binding(source, spec)
assert sha(root / 'recursive_result.json') == 'e3934bd736e9f14b6009cad4c5c05090a2527c3410aa54bb179eab2d84feeb7b'
result = json.loads((root / 'recursive_result.json').read_text())
assert result['integrity_pass'] and result['new_failure_sequences'] == ['mobilephone02_indoor']
name = result['new_failure_sequences'][0]
event = next(r for r in result['first_overrides'] if r['sequence'] == name)
assert event['changes'] == 1
stop = event['first_override']
case = next(c for c in json.loads((source / 'inference_inputs.json').read_text()) if c['sequence'] == name)
folder = Path(spec['dataset_root']) / name
path = root / 'recursive' / (name + '.json')
assert sha(path) == event['trajectory_sha256']
sealed = json.loads(path.read_text())['rows']
checkpoint = Path(plan['control_root']) / 'geometry_final.pth'
assert sha(checkpoint) == plan['checkpoint_sha256']
torch.set_num_threads(1)
update_config_from_file(str(repo / 'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
params = SimpleNamespace(cfg=cfg, checkpoint=spec['checkpoint'], template_factor=2., template_size=128,
    search_factor=4., search_size=256, save_all_boxes=False, debug=0)
tracker = STTrackCandidateContinuity(params, checkpoint)
forward = tracker.association.head.forward
capture = {}


def observed(*args):
    output = forward(*args)
    if tracker.frame_id == stop:
        capture['inputs'] = [x.detach().clone() for x in args]
        capture['logits'] = output[0].detach().clone()
    return output


def image_at(frame):
    return get_rgbd_frame(str(folder / 'color' / f'{frame+1:08d}.jpg'),
        str(folder / 'depth' / f'{frame+1:08d}.png'), dtype='rgbcolormap', depth_clip=True)


tracker.association.head.forward = observed
tracker.initialize(image_at(0), dict(init_bbox=case['init_bbox']))
updates = []
for frame in range(1, case['frames']):
    old = tracker.z_dict[1]
    out = tracker.track(image_at(frame))
    assert out['target_bbox'] == sealed[frame]['bbox'] and float(out['best_score']) == sealed[frame]['score']
    assert out['association_candidate'] == sealed[frame]['choice']
    assert out['association_proposal'] == sealed[frame]['proposal'] and out['association_vetoed'] == sealed[frame]['vetoed']
    assert out['continuity_delta'] == sealed[frame]['continuity_delta']
    if old is not tracker.z_dict[1]:
        updates.append(dict(frame=frame, bbox=list(out['target_bbox']), score=float(out['best_score'])))
    if frame == stop:
        capture['candidates'] = tracker.previous_set['candidates']
        capture['chosen'] = out['association_candidate']

# Labels are opened only after the complete replay has matched its sealed path.
gt = np.loadtxt(folder / 'groundtruth.txt', delimiter=',')
boxes = np.asarray([r['bbox'] for r in sealed], dtype=np.float64)
values, metrics = independent_overlap(boxes, gt)
metrics['low_iou_frames'] = int(metrics['low_iou_frames'])
assert metrics == result['per_sequence'][name]
current, previous, refs, geometry, scores, previous_choice = capture['inputs']
past = int(previous_choice[0])
similarities = {}
for ref_name, reference in [('previous', previous[0, past]), ('initial_template', refs[0, 0]), ('dynamic_template', refs[0, 1])]:
    similarities[ref_name] = {label: [float(F.cosine_similarity(current[0, index, k].flatten(), reference[k].flatten(), dim=0)) for k in [0, 1]]
        for label, index in [('default', 0), ('chosen', capture['chosen'])]}
candidate_boxes = torch.tensor([c['bbox'] for c in capture['candidates']])
candidate_iou = overlaps(candidate_boxes, torch.tensor(gt[stop], dtype=torch.float32))
episodes = []
start = None
for frame in range(len(values) + 1):
    low = frame < len(values) and np.isfinite(values[frame]) and values[frame] <= .1
    if low and start is None:
        start = frame
    if not low and start is not None:
        if frame - start >= 10:
            episodes.append(dict(start=start, end_exclusive=frame, length=frame-start))
        start = None
assert len(episodes) == metrics['failure_episodes']
report = dict(status='complete', replay_frames=case['frames']-1, full_bbox_confidence_proposals_and_deltas_exact=True,
    sequence=name, first_override=stop, only_one_override=True, chosen=capture['chosen'], previous_choice=past,
    first_default_iou=event['first_default_iou'], first_selected_iou=event['first_selected_iou'],
    candidate_boxes=candidate_boxes.tolist(), candidate_iou=candidate_iou.tolist(),
    candidate_scores=[c['score'] for c in capture['candidates']], raw_logits=capture['logits'][0].tolist(),
    native_cosines=similarities, admitted_delta=sealed[stop]['continuity_delta'],
    default_chosen_box_iou=float(overlaps(candidate_boxes[0], candidate_boxes[capture['chosen']])),
    template_updates=[dict(**r, gt_iou=float(values[r['frame']]) if np.isfinite(values[r['frame']]) else None) for r in updates], failure_episodes=episodes,
    first_local_window=[dict(frame=i, iou=float(values[i]) if np.isfinite(values[i]) else None,
        bbox=sealed[i]['bbox'], score=sealed[i]['score'], proposal=sealed[i]['proposal'], choice=sealed[i]['choice']) for i in range(stop-1, stop+11)],
    source_sha256=sha(__file__), trajectory_sha256=sha(path), result_sha256=sha(root / 'recursive_result.json'),
    scope='Diagnostic replay of the sole failed protection sequence after all outputs were sealed. No threshold change, fitting, new policy or public result.')
(root / 'mobilephone_diagnosis.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['candidate_boxes','first_local_window','raw_logits']}, indent=2))
