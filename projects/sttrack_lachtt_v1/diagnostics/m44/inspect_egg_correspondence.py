"""Inspect sealed static egg errors; do not change or run recursive tracking."""
import hashlib
import json
from pathlib import Path
import sys
import torch

root = Path('/root/autodl-tmp/sttrack_m44_candidate_set_v1_20260905')
spec = json.loads((root / 'spec.json').read_text())
repo = Path(spec['repository'])
sys.path.insert(0, str(repo))
from lib.models.sttrack.lachtt_candidate_set import CandidateSetAssociation, select_candidate

torch.set_num_threads(1)
training = json.loads((root / 'training_result.json').read_text())
assert training['status'] == 'complete'
name = 'egg_indoor'
path = root / 'features' / (name + '.pt')
receipts = [json.loads((root / f'shard{i}_receipt.json').read_text()) for i in [0, 1]]
expected = [r for s in receipts for r in s['sequences'] if r['sequence'] == name]
assert len(expected) == 1 and hashlib.sha256(path.read_bytes()).hexdigest() == expected[0]['feature_sha256']
data = torch.load(path, map_location='cpu')
observations = []
for arm in spec['variants']:
    checkpoint = root / (arm + '_final.pth')
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == training['variants'][arm]['checkpoint_sha256']
    model = CandidateSetAssociation(arm == 'geometry')
    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'], strict=True)
    model.eval()
    targets = {r['key']: r for r in training['variants'][arm]['development']['rows']}
    for i, record in enumerate(data['records']):
        frozen = targets[record['key']]
        if not (frozen['default_iou'] >= .5 and frozen['selected_iou'] <= .1):
            continue
        inputs = [data[k][i:i+1].float() for k in ['current', 'previous', 'references', 'geometry', 'scores']]
        inputs.append(torch.tensor([record['previous_choice']]))
        with torch.no_grad():
            logits, affinity = model(*inputs)
        choice = int(select_candidate(logits)[0])
        assert choice == frozen['chosen']
        column = affinity[0, :, record['previous_choice']]
        match_choice = int(column.argmax())
        observations.append(dict(arm=arm, **frozen, previous_selected=record['previous_choice'],
                                 identity_logits=logits[0].tolist(),
                                 previous_selected_affinity_column=column.tolist(),
                                 correspondence_argmax=match_choice,
                                 current_boxes=data['current_boxes'][i].tolist(),
                                 previous_selected_bbox=data['previous_boxes'][i, record['previous_choice']].tolist(),
                                 current_log_response=data['scores'][i, :10].tolist()))
result = dict(status='complete', scope='All sealed static egg severe regressions in both arms. CPU inspection only; no new training, recursive action, threshold selection or public metric.',
              training_result_sha256=hashlib.sha256((root / 'training_result.json').read_bytes()).hexdigest(),
              inspector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=observations)
(root / 'egg_correspondence_diagnosis.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps([dict(arm=r['arm'], key=r['key'], identity_choice=r['chosen'], match_choice=r['correspondence_argmax'],
                       target=r['target'], previous_target=r['previous_target'], affinity_column=r['previous_selected_affinity_column']) for r in observations], indent=2))
