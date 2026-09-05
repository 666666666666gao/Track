"""Check actual M46 tensors against the frozen paired training record on CPU."""
import hashlib
import json
import math
from pathlib import Path
import sys

import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root = Path('/root/autodl-tmp/sttrack_m46_initial_frame_v1_20260905')
plan = json.loads((root / 'spec.json').read_text())
parent = Path(plan['source_root'])
spec = json.loads((parent / 'spec.json').read_text())
sys.path.insert(0, spec['repository'])
from lib.models.sttrack.lachtt_candidate_set import CandidateSetAssociation
from tools.train_sttrack_m44 import check_binding, tensor_sha

torch.set_num_threads(1)
check_binding(parent, spec)
assert sha(parent / 'spec.json') == plan['source_spec_sha256']
control_root = Path(plan['control_root'])
assert sha(control_root / 'geometry_result.json') == plan['control_training_result_sha256']
assert sha(control_root / 'recursive_result.json') == plan['control_recursive_result_sha256']
for name, digest in plan['source_sha256'].items():
    assert sha(Path(spec['repository']) / name) == digest, name
training = json.loads((root / 'geometry_result.json').read_text())
control = json.loads((control_root / 'geometry_result.json').read_text())
checkpoint = root / 'geometry_final.pth'
assert sha(checkpoint) == training['checkpoint_sha256']
saved = torch.load(checkpoint, map_location='cpu')
assert saved['m46_spec_sha256'] == training['m46_spec_sha256'] == sha(root / 'spec.json')
assert saved['spec_sha256'] == training['spec_sha256'] == sha(parent / 'spec.json')
assert saved['binding_sha256'] == training['source_binding_sha256'] == sha(parent / 'training_binding.json')
assert saved['base_checkpoint_sha256'] == spec['checkpoint_sha256']
assert saved['target_rule'] == training['target_rule'] == 'default_if_iou_at_least_half'
assert saved['variant'] == training['variant'] == 'geometry'
assert saved['optimization_rounds'] == training['optimization_rounds'] == plan['optimization']['rounds'] == 20
steps = 20 * math.ceil(plan['optimization']['examples_per_round'] / plan['optimization']['batch_size'])
assert saved['optimizer_steps'] == training['optimizer_steps'] == control['optimizer_steps'] == steps == 960
assert [r['round'] for r in training['losses']] == list(range(1, 21))
assert all(math.isfinite(r[k]) for r in training['losses'] for k in ['loss', 'identity_loss', 'matching_loss'])
assert training['fit']['events'] == plan['fit_pairs'] == 2015
assert training['development']['events'] == plan['development_pairs'] == 590
assert training['matched_control_initialization_and_optimizer_budget'] and training['reload_logits_exact']
assert training['distinct_fit_inputs_seen'] == 2015 and training['early_inputs_seen'] == 504
torch.manual_seed(plan['optimization']['seed'])
model = CandidateSetAssociation(True)
initial = {key: tensor_sha(value) for key, value in model.state_dict().items()}
assert initial == training['initial_state_sha256'] == control['initial_state_sha256']
model.load_state_dict(saved['model'], strict=True)
assert all(torch.isfinite(value).all() for value in saved['model'].values())
changed = [key for key, value in saved['model'].items() if tensor_sha(value) != initial[key]]
assert changed == training['changed_tensors'] and 'identity.weight' in changed
assert sum(value.numel() for value in model.parameters()) == training['parameters'] == plan['parameters'] == 448739
report = dict(status='pass', scope='Actual checkpoint tensors and paired training provenance; no recursive performance claim.',
              checkpoint_sha256=sha(checkpoint), training_result_sha256=sha(root / 'geometry_result.json'),
              spec_sha256=sha(root / 'spec.json'), audit_source_sha256=sha(__file__),
              parameters=448739, optimizer_steps=steps, optimization_rounds=20, changed_tensors=changed,
              initial_state_exact=True, paired_optimizer_budget_exact=True, sample_order_sha256=training['sample_order_sha256'], strict_load=True,
              all_weights_finite=True, base_and_source_unchanged=True)
(root / 'weight_audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
