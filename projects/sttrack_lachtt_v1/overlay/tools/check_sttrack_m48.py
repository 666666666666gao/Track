"""Check the fixed predicate and actual admitted/vetoed tracker state paths."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    root = p.parse_args().root
    plan = json.loads((root / 'spec.json').read_text())
    source = Path(plan['source_root'])
    spec = json.loads((source / 'spec.json').read_text())
    repo = Path(spec['repository'])
    sys.path.insert(0, str(repo))
    from tools.train_sttrack_m44 import sha, check_binding
    from lib.config.sttrack.config import cfg, update_config_from_file
    from lib.test.tracker.sttrack import STTrack
    from lib.test.tracker.sttrack_candidate_set import STTrackCandidateSet
    from lib.test.tracker.sttrack_candidate_continuity import STTrackCandidateContinuity, native_continuity
    from lib.train.dataset.depth_utils import get_rgbd_frame
    torch.set_num_threads(1)
    check_binding(source, spec)
    for name, digest in plan['source_sha256'].items():
        assert sha(repo / name) == digest
    current = torch.zeros(4, 10, 2, 16, 768, device='cuda')
    previous = torch.zeros_like(current)
    current[..., 0] = 1
    previous[..., 0] = 1
    previous[:, 1] = 0
    previous[:, 1, ..., 1] = 1
    current[:3, 1] = 0
    current[:3, 1, ..., 1] = 1
    current[2, 0, 1] = 0
    current[2, 0, 1, :, 1] = 1
    current[2, 1, 1] = 0
    current[2, 1, 1, :, 0] = 1
    accepted, delta = native_continuity(current, previous, torch.tensor([0, 1, 1, 0], device='cuda'), torch.ones(4, dtype=torch.long, device='cuda'))
    assert accepted.tolist() == [False, True, False, True]
    assert delta[2, 0] > 0 and delta[2, 1] < 0

    audit_path = Path('/root/autodl-tmp/sttrack_m48_native_continuity_audit_v1_20260905/native_continuity_audit.json')
    assert sha(audit_path) == plan['fitting_audit_sha256']
    audited = {r['key']: r for r in json.loads(audit_path.read_text())['rows']}
    cases = {r['sequence'] for r in audited.values()}
    receipts = [r for shard in [0, 1] for r in json.loads((source / f'shard{shard}_receipt.json').read_text())['sequences'] if r['sequence'] in cases]
    compared = 0
    for receipt in receipts:
        path = source / 'features' / (receipt['sequence'] + '.pt')
        assert sha(path) == receipt['feature_sha256']
        data = torch.load(path, map_location='cpu')
        proposed = torch.tensor([audited[r['key']]['proposal'] for r in data['records']], device='cuda')
        allowed, _ = native_continuity(data['current'].cuda().float(), data['previous'].cuda().float(), torch.zeros(len(proposed), dtype=torch.long, device='cuda'), proposed)
        assert allowed.tolist() == [audited[r['key']]['accepted'] for r in data['records']]
        compared += len(proposed)
    assert compared == 1511
    del data, current, previous

    update_config_from_file(str(repo / 'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    checkpoint = Path(plan['control_root']) / 'geometry_final.pth'
    assert sha(checkpoint) == plan['checkpoint_sha256']
    params = SimpleNamespace(cfg=cfg, checkpoint=spec['checkpoint'], template_factor=2., template_size=128,
        search_factor=4., search_size=256, save_all_boxes=False, debug=0)
    baseline = STTrack(params)
    candidate = STTrackCandidateContinuity(params, checkpoint)
    assert sum(v.numel() for v in candidate.association.parameters()) == plan['parameters'] == 448739
    cases = [c for c in json.loads((source / 'inference_inputs.json').read_text()) if c['split'] == 'fit'][:2]

    def image_at(case, frame):
        folder = Path(spec['dataset_root']) / case['sequence']
        return get_rgbd_frame(str(folder / 'color' / f'{frame+1:08d}.jpg'),
            str(folder / 'depth' / f'{frame+1:08d}.png'), dtype='rgbcolormap', depth_clip=True)

    def force_proposal(*args):
        logits = torch.full((1, 11), -20., device='cuda')
        logits[:, 1] = 20.
        return logits, torch.zeros(1, 11, 11, device='cuda')

    def constant_admission(accepted):
        return lambda current, previous, previous_choice, proposal: (
            torch.full((len(current),), accepted, device='cuda'), torch.zeros(len(current), 2, device='cuda'))

    def same_state(a, b):
        assert a.state == b.state
        assert a.frame_id == b.frame_id
        assert len(a.z_dict) == len(b.z_dict)
        assert all(torch.equal(x, y) for x, y in zip(a.z_dict, b.z_dict))
        assert len(a.track_query_before) == len(b.track_query_before)
        assert all(torch.equal(x, y) for x, y in zip(a.track_query_before, b.track_query_before))
        assert a.box_mask_z is b.box_mask_z is None

    candidate.association.head.forward = force_proposal
    tracked = updates = vetoes = 0
    with patch('lib.test.tracker.sttrack_candidate_continuity.native_continuity', constant_admission(False)):
        for case in cases:
            image = image_at(case, 0)
            info = dict(init_bbox=list(case['init_bbox']))
            baseline.initialize(image, info)
            candidate.initialize(image, info)
            for frame in range(1, 61):
                image = image_at(case, frame)
                old = baseline.z_dict[1]
                a = baseline.track(image)
                b = candidate.track(image)
                assert a['target_bbox'] == b['target_bbox'] and a['best_score'] == b['best_score']
                assert b['association_candidate'] == candidate.previous_choice == 0
                assert b['association_vetoed'] == (frame >= 2)
                same_state(baseline, candidate)
                updates += int(old is not baseline.z_dict[1])
                vetoes += int(b['association_vetoed'])
                tracked += 1
    assert tracked == 120 and vetoes == 118 and updates > 0
    del baseline
    control = STTrackCandidateSet(params, checkpoint)
    control.association.forward = force_proposal
    case = cases[0]
    image = image_at(case, 0)
    info = dict(init_bbox=list(case['init_bbox']))
    control.initialize(image, info)
    candidate.initialize(image, info)
    choices = []
    with patch('lib.test.tracker.sttrack_candidate_continuity.native_continuity', constant_admission(True)):
        for frame in range(1, 4):
            image = image_at(case, frame)
            a = control.track(image)
            b = candidate.track(image)
            assert a['target_bbox'] == b['target_bbox'] and a['best_score'] == b['best_score']
            same_state(control, candidate)
            assert control.previous_choice == candidate.previous_choice
            if frame >= 2:
                assert b['association_candidate'] == 1 and not b['association_vetoed']
                choices.append(b['previous_choice_input'])
    assert choices == [0, 1]
    assert sha(checkpoint) == plan['checkpoint_sha256']
    for name, digest in plan['source_sha256'].items():
        assert sha(repo / name) == digest
    result = dict(status='PASS', spec_sha256=sha(root / 'spec.json'), checkpoint_sha256=sha(checkpoint),
        source_sha256=plan['source_sha256'], synthetic_predicate_cases=4, fitting_cpu_gpu_decisions_equal=1511,
        forced_veto_frames=118, default_parity_frames=120, native_template_updates=updates,
        bbox_confidence_query_template_mask_exact=True, forced_admitted_frames=2, admitted_matches_m45=True,
        preceding_selected_index_propagates=True, additional_parameters=0, additional_optimizer_steps=0,
        scope='Predicate and state integration only; no development/public performance and no optimization.')
    (root / 'runtime_contract.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
