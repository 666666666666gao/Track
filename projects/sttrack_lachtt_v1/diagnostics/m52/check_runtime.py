"""Check both trained M52 heads through the actual tracker loader."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


ROOT = Path('/root/autodl-tmp/sttrack_m52_policy_state_augmentation_v1_20260905')
PLAN = json.loads((ROOT/'spec.json').read_text())
PARENT = Path(PLAN['source_root'])
SPEC = json.loads((PARENT/'spec.json').read_text())
REPO = Path(SPEC['repository'])
sys.path.insert(0, str(REPO))

from tools.train_sttrack_m52 import sha, check_sources, FIELDS
from lib.models.sttrack.lachtt_candidate_set import CandidateSetAssociation
from lib.test.tracker.sttrack_candidate_set import STTrackCandidateSet
from lib.config.sttrack.config import cfg, update_config_from_file


def main():
    check_sources(ROOT)
    audit = json.loads((ROOT/'data_audit.json').read_text())
    assert audit['status'] == 'PASS'
    item = next(r for r in audit['state_rows'] if r['nonzero_previous_choice_events'] > 0)
    path = ROOT/'features'/(item['sequence']+'.pt')
    assert sha(path) == item['feature_sha256']
    data = torch.load(path, map_location='cpu')
    indices = torch.tensor([0]+[i for i, r in enumerate(data['records']) if r['previous_choice'] != 0][:4])
    inputs = [data[key][indices].cuda().float() for key in FIELDS]
    choices = torch.tensor([data['records'][i]['previous_choice'] for i in indices], device='cuda')
    assert (choices != 0).any()
    torch.set_num_threads(1)
    update_config_from_file(str(REPO/'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    params = SimpleNamespace(cfg=cfg, checkpoint=SPEC['checkpoint'], template_factor=2., template_size=128,
                             search_factor=4., search_size=256, save_all_boxes=False, debug=0)
    checked = {}
    for arm in ['control', 'mixed']:
        checkpoint = ROOT/arm/'geometry_final.pth'
        training = json.loads((ROOT/arm/'training_result.json').read_text())
        assert sha(checkpoint) == training['checkpoint_sha256']
        tracker = STTrackCandidateSet(params, checkpoint)
        reference = CandidateSetAssociation(True).cuda().eval()
        reference.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'], strict=True)
        with torch.no_grad():
            a = tracker.association(*inputs, choices)
            b = reference(*inputs, choices)
        assert all(torch.equal(x, y) for x, y in zip(a, b))
        checked[arm] = dict(checkpoint_sha256=sha(checkpoint), logits_and_affinity_exact=True)
        del tracker, reference
    check_sources(ROOT)
    result = dict(status='PASS', arms=checked, spec_sha256=sha(ROOT/'spec.json'), source_sha256=sha(Path(__file__)),
                  sequence=item['sequence'], actual_previous_choices=choices.cpu().tolist(), labels_opened=False)
    (ROOT/'runtime_contract.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
