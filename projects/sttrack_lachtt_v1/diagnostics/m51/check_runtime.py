"""Verify M51 training/runtime parity and the unchanged native initialization."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


ROOT = Path('/root/autodl-tmp/sttrack_m51_relative_geometry_v1_20260905')
plan = json.loads((ROOT/'spec.json').read_text())
parent = Path(plan['source_root'])
spec = json.loads((parent/'spec.json').read_text())
repo = Path(spec['repository'])
sys.path.insert(0, str(repo))

from tools.train_sttrack_m44 import sha, check_binding
from lib.config.sttrack.config import cfg, update_config_from_file
from lib.models.sttrack.lachtt_candidate_set import CandidateSetAssociation
from lib.models.sttrack.lachtt_relative_geometry import RelativeCandidateSetAssociation, RelativeGeometryInference
from lib.test.tracker.sttrack import STTrack
from lib.test.tracker.sttrack_relative_candidate_set import STTrackRelativeCandidateSet
from lib.train.dataset.depth_utils import get_rgbd_frame


def main():
    check_binding(parent, spec)
    for name, digest in plan['source_sha256'].items():
        assert sha(repo/name) == digest, name
    training = json.loads((ROOT/'geometry_result.json').read_text())
    checkpoint = ROOT/'geometry_final.pth'
    assert training['status'] == 'complete' and sha(checkpoint) == training['checkpoint_sha256']
    assert training['m51_spec_sha256'] == sha(ROOT/'spec.json')
    saved = torch.load(checkpoint, map_location='cpu')
    assert saved['geometry_encoding'] == 'previous_selected_box_relative'
    torch.set_num_threads(1)
    update_config_from_file(str(repo/'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    params = SimpleNamespace(cfg=cfg, checkpoint=spec['checkpoint'], template_factor=2., template_size=128,
                             search_factor=4., search_size=256, save_all_boxes=False, debug=0)
    tracker = STTrackRelativeCandidateSet(params, checkpoint)
    trained = RelativeCandidateSetAssociation(True).cuda().eval()
    trained.load_state_dict(saved['model'], strict=True)
    receipt = json.loads((parent/'shard0_receipt.json').read_text())['sequences'][0]
    feature = parent/'features'/(receipt['sequence']+'.pt')
    assert sha(feature) == receipt['feature_sha256']
    data = torch.load(feature, map_location='cpu')
    indices = torch.arange(10) % len(data['geometry'])
    inputs = [data[key][indices].cuda().float() for key in ['current', 'previous', 'references', 'geometry', 'scores']]
    choices = torch.arange(10, device='cuda')
    with torch.no_grad():
        expected = trained(*inputs, choices)
        actual = tracker.association(*inputs, choices)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    del trained, data, inputs, expected, actual

    # The fresh head adds zero identity logits, so its full recursive path must
    # reproduce native STTrack, including the existing dynamic template update.
    torch.manual_seed(2026)
    tracker.association = RelativeGeometryInference(CandidateSetAssociation(True).cuda().eval()).eval()
    native = STTrack(params)
    cases = json.loads((parent/'inference_inputs.json').read_text())
    case = next(c for c in cases if c['sequence'] == 'chair01_indoor')
    assert case['split'] == 'fit'
    folder = Path(spec['dataset_root'])/case['sequence']
    def image_at(i):
        return get_rgbd_frame(str(folder/'color'/f'{i+1:08d}.jpg'), str(folder/'depth'/f'{i+1:08d}.png'),
                              dtype='rgbcolormap', depth_clip=True)
    image = image_at(0)
    tracker.initialize(image, dict(init_bbox=list(case['init_bbox'])))
    native.initialize(image, dict(init_bbox=list(case['init_bbox'])))
    writes = 0
    for frame in range(1, 121):
        image = image_at(frame)
        old_template = native.z_dict[1]
        reference, result = native.track(image), tracker.track(image)
        assert reference['target_bbox'] == result['target_bbox'] and reference['best_score'] == result['best_score']
        assert result['association_candidate'] == 0
        assert all(torch.equal(a, b) for a, b in zip(native.z_dict, tracker.z_dict))
        assert all(torch.equal(a, b) for a, b in zip(native.track_query_before, tracker.track_query_before))
        writes += native.z_dict[1] is not old_template
    assert writes >= 1
    out = dict(status='PASS', trained_head_runtime_logits_and_affinity_exact=True,
               previous_choices_checked=list(range(10)), fresh_native_parity_frames=120,
               native_template_writes=writes, sequence=case['sequence'], gt_files_read=False,
               checkpoint_sha256=sha(checkpoint), spec_sha256=sha(ROOT/'spec.json'), source_sha256=sha(Path(__file__)))
    (ROOT/'runtime_contract.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__': main()
