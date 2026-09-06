from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

root = Path('/root/autodl-tmp/sttrack_m58_semantic_spatial_v1_20260906')
previous = Path('/root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906')
integration = json.loads((root / 'integration.json').read_text())
repo = Path(integration['repository'])
sys.path.insert(0, str(repo))
from lib.config.sttrack.config import cfg, update_config_from_file
from lib.models.sttrack.semantic_spatial_adapter import SemanticSpatialAdapter
from lib.test.tracker.sttrack import STTrack
from lib.test.tracker.sttrack_semantic import STTrackSemantic
from lib.train.dataset.depth_utils import get_rgbd_frame

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name, digest in integration['source_sha256'].items():
    assert sha(repo / name) == digest
torch.set_num_threads(1)
torch.manual_seed(2026)
torch.cuda.manual_seed_all(2026)
update_config_from_file(str(repo / 'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
assert cfg.MODEL.TSG.FIX_QUERY_WINDOW and cfg.TEST.UPDATE_INTERVALS == 50 and cfg.TEST.UPDATE_THRESHOLD == .75
checkpoint = Path('/root/autodl-tmp/sttrack_checkpoints/STTrack_Vot22.pth.tar')
assert sha(checkpoint) == integration['native_checkpoint_sha256']
params = SimpleNamespace(cfg=cfg, checkpoint=str(checkpoint), base_checkpoint_sha256=sha(checkpoint),
    template_factor=2., template_size=128, search_factor=4., search_size=256, save_all_boxes=False, debug=0)
output = root / 'native_parity'
output.mkdir()
adapter = SemanticSpatialAdapter()
for mode, use_text in [('text', True), ('visual', False)]:
    torch.save(dict(architecture='semantic_spatial_v1', model=adapter.state_dict(), use_text=use_text,
        base_checkpoint_sha256=params.base_checkpoint_sha256, actual_dataset_optimizer_steps=0), output / (mode + '_zero.pth'))
del adapter
trackers = {'off': STTrack(params), 'text': STTrackSemantic(params, str(output / 'text_zero.pth')),
    'visual': STTrackSemantic(params, str(output / 'visual_zero.pth'))}
banks = [torch.load(previous / name, map_location='cpu') for name in ['text_fit.pt', 'text_development.pt']]
inventory = json.loads((root / 'data_inventory.json').read_text())['sequences_detail']
cases = []
for name in ['chair01_indoor', 'bag05_indoor', 'cup08_indoor']:
    row = next(r for r in inventory if r['sequence'] == name)
    assert row['first_box_valid'] and row['rgb_frames'] >= 102
    cases.append(dict(sequence=name, init_bbox=row['first_box']))
frozen = json.loads((previous / 'recursive_spec.json').read_text())
baseline = {r['sequence']: {} for r in cases}
for path, digest in frozen['baseline_trace_sha256'].items():
    assert sha(path) == digest
    for row in json.loads(Path(path).read_text())['rows']:
        if row['sequence'] in baseline and row['frame_index'] < 102:
            baseline[row['sequence']][row['frame_index']] = row['public_bbox']
assert all(sorted(rows) == list(range(102)) for rows in baseline.values())
started = time.time()
records = []
for case in cases:
    name = case['sequence']
    folder = Path('/root/autodl-tmp/depthtrack/train/sequences') / name
    bank = next(b for b in banks if name in b['sequences'])
    index = bank['sequences'].index(name)
    info = dict(init_bbox=list(case['init_bbox']), text_tokens=bank['tokens'][index],
        text_mask=bank['mask'][index], empty_text=bank['empty'])
    updates = []
    for frame in range(102):
        image = get_rgbd_frame(str(folder / 'color' / ('%08d.jpg' % (frame + 1))),
            str(folder / 'depth' / ('%08d.png' % (frame + 1))), dtype='rgbcolormap', depth_clip=True)
        if frame == 0:
            for tracker in trackers.values():
                tracker.initialize(image, dict(info))
            assert trackers['off'].state == baseline[name][0]
            assert trackers['off'].semantic_context is None
            assert trackers['text'].track_query_before is trackers['visual'].track_query_before is None
            assert torch.equal(trackers['text'].semantic_context['initial'], trackers['visual'].semantic_context['initial'])
            continue
        old_template = id(trackers['off'].z_dict[1])
        outputs = {mode: tracker.track(image) for mode, tracker in trackers.items()}
        reference = outputs['off']
        assert reference['target_bbox'] == baseline[name][frame], (name, frame, 'legacy_bbox')
        for mode in ['text', 'visual']:
            other = trackers[mode]
            assert outputs[mode]['target_bbox'] == reference['target_bbox'], (name, frame, mode, 'bbox')
            assert float(outputs[mode]['best_score']) == float(reference['best_score']), (name, frame, mode, 'score')
            assert other.frame_id == trackers['off'].frame_id == frame
            assert all(torch.equal(a, b) for a, b in zip(other.track_query_before, trackers['off'].track_query_before))
            assert all(torch.equal(a, b) for a, b in zip(other.z_dict, trackers['off'].z_dict))
            assert np.array_equal(other.z_patch_arr, trackers['off'].z_patch_arr)
        if id(trackers['off'].z_dict[1]) != old_template:
            updates.append(frame)
    row = dict(sequence=name, frames=102, tracked_frames=101, template_writes=updates,
        post_frame100_write_read_checked=100 in updates, legacy_native_bbox_prefix_exact=True,
        text_and_visual_zero_bbox_score_query_templates_exact=True)
    records.append(row)
    print(json.dumps(row), flush=True)
assert records[0]['sequence'] == 'chair01_indoor' and records[0]['post_frame100_write_read_checked']
for name, digest in integration['source_sha256'].items():
    assert sha(repo / name) == digest
result = dict(status='three_path_native_parity_complete', observed_utc=datetime.now(timezone.utc).isoformat(),
    native_checkpoint_sha256=sha(checkpoint), integration_sha256=sha(root / 'integration.json'),
    checker_sha256=sha(__file__), source_files_verified=len(integration['source_sha256']),
    parameters=289154, cases=records, image_frames=306, tracker_frame_states=918, track_calls=909,
    elapsed_seconds=time.time() - started, actual_dataset_optimizer_steps=0,
    subsequent_gt_used_for_tracking=False, baseline_comparison_uses_public_bboxes_only=True,
    learned_tracker_performance_tested=False, public_evaluation=False)
(root / 'native_parity_result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
