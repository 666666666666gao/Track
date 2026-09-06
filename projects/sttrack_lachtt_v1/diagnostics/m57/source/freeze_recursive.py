"""Seal four fixed final heads and bind complete M57 recursive evaluation."""
import ast
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from train import binding, sha, write


root = Path('/root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906')
prior = Path('/root/autodl-tmp/sttrack_m56_attribute_alignment_v1_20260906')
spec = binding(root)
assert not (root / 'recursive_spec.json').exists()
assert (root / 'training.exit').read_text().strip() == '0'
training = json.loads((root / 'training_result.json').read_text())
assert training['status'] == 'complete_train' and training['equal_initialization_order_budget_parameters']
assert training['spec_sha256'] == sha(root / 'spec.json')
assert list(training['variants']) == spec['variants']
results = {}
for arm in spec['variants']:
    result_path = root / 'training' / (arm + '_result.json')
    assert sha(result_path) == training['variants'][arm]['result_sha256']
    result = json.loads(result_path.read_text())
    assert result['status'] == 'complete_train' and result['optimizer_steps'] == 960 and result['epochs'] == 20
    assert result['spec_sha256'] == sha(root / 'spec.json')
    assert result['fit_sequences'] == 63 and result['fit_events'] == 1511
    assert result['parameters'] == 484547 and not result['development_targets_loaded']
    path = root / 'training' / (arm + '_final.pth')
    assert sha(path) == result['checkpoint_sha256'] == training['variants'][arm]['checkpoint_sha256']
    saved = torch.load(path, map_location='cpu')
    assert saved['m57_spec_sha256'] == sha(root / 'spec.json') and saved['variant'] == arm
    assert saved['reference_mode'] == arm.split('_')[0] and saved['use_text'] == arm.endswith('_text')
    assert saved['epochs'] == 20 and saved['optimizer_steps'] == 960
    assert saved['base_checkpoint_sha256'] == spec['base_checkpoint_sha256']
    assert saved['text_bank_sha256'] == sha(root / 'text_fit.pt')
    assert saved['initial_fit_sha256'] == sha(root / 'initial_references/initial_fit.pt')
    assert sum(value.numel() for value in saved['model'].values()) == spec['parameters']
    results[arm] = result
for field in ['sample_order_sha256', 'ordered_event_keys_sha256', 'initial_state_sha256', 'optimizer_steps', 'parameters']:
    assert all(value[field] == results[spec['variants'][0]][field] for value in results.values()), field
old = json.loads((prior / 'recursive_spec.json').read_text())
inputs = (prior / 'recursive_inputs.json').read_bytes()
assert sha(prior / 'recursive_inputs.json') == old['inputs_sha256']
cases = json.loads(inputs)
initialization = json.loads((root / 'initialization_inputs.json').read_text())
assert [{key: row[key] for key in ['sequence', 'frames', 'init_bbox']}
        for row in initialization if row['split'] == 'development'] == cases
assert len(cases) == 22 and sum(case['frames'] for case in cases) == 33130
(root / 'recursive_inputs.json').write_bytes(inputs)
collection = json.loads((root / 'collection_spec.json').read_text())
receipt = json.loads((root / 'initial_references/receipt.json').read_text())
assert sha(root / 'text_development.pt') == collection['text_files']['development']['sha256']
assert sha(root / 'initial_references/initial_development.pt') == receipt['files']['development']['sha256']
assert sha(Path(spec['repository']) / 'tools/analyze_sttrack_m42_recursive.py') == old['metric_sha256']
ast.parse((root / 'run_recursive.py').read_text(), feature_version=(3, 8))
frozen = dict(schema='sttrack_m57_same_slot_initial_binding_recursive_v1',
    frozen_utc=datetime.now(timezone.utc).isoformat(),
    training_spec_sha256=sha(root / 'spec.json'), training_result_sha256=sha(root / 'training_result.json'),
    runner_sha256=sha(root / 'run_recursive.py'), preparer_sha256=sha(__file__),
    inputs_sha256=sha(root / 'recursive_inputs.json'), metric_sha256=old['metric_sha256'],
    inference_input_sha256={str(root / name): sha(root / name) for name in [
        'text_development.pt', 'initial_references/initial_development.pt', 'training_result.json']},
    baseline_trace_sha256=old['baseline_trace_sha256'], development_gt_sha256=old['development_gt_sha256'],
    prior_recursive_spec_sha256=sha(prior / 'recursive_spec.json'),
    arms=spec['variants'], sequences=22, frames_per_arm=33130,
    native_default_templates=True, inference_uses_subsequent_gt=False,
    default_trajectory_is_independent=True, same_slot_mask_across_arms=True,
    t0_reference_recomputed_causally_and_verified=True,
    primary_variant='initial_text', public_automatic_launch=False,
    weight_selection='All four fixed final epoch20/step960 heads sealed before any M57 development numerical analysis',
    analysis_rule='Seal all four complete trajectory families before reading recursive GT; use preregistered training-spec gates',
    scope='Previously reused DepthTrack Train development; no VOT/DepthTrack Test/CDTB performance claim')
write(root / 'recursive_spec.json', frozen)
write(root / 'training_seal.json', dict(status='four_fixed_heads_sealed',
    observed_utc=datetime.now(timezone.utc).isoformat(), training_spec_sha256=sha(root / 'spec.json'),
    training_result_sha256=sha(root / 'training_result.json'), recursive_spec_sha256=sha(root / 'recursive_spec.json'),
    checkpoint_sha256={arm: value['checkpoint_sha256'] for arm, value in results.items()},
    same_initialization_order_budget_parameters=True, numerical_development_analysis_started=False))
print(json.dumps(dict(status='frozen_recursive_spec', recursive_spec_sha256=sha(root / 'recursive_spec.json'),
    runner_sha256=frozen['runner_sha256'], heads=4, frames_per_arm=33130, same_initialization_order_budget_parameters=True)))
