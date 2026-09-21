"""Isolated M88 preparation from sealed M84; reuse banks and initial tensors."""
from pathlib import Path
import hashlib, json, shutil, torch

R = Path(__file__).resolve().parent
P = Path('/root/autodl-tmp/sttrack_m84_centered_20260920')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
def save(p, v):
    Path(p).write_text(json.dumps(v, indent=2, allow_nan=False) + '\n')

assert not (R/'training_spec.json').exists()
assert sha(P/'training_spec.json') == '0f9bb841edd3e2c5171cd78ce9d1030d29a243561006d111d2c98eebfc74abd5'
assert sha(P/'recursive_result.json') == 'a9281e0833b8ba1c01220374f3d6a93dbc9d22e6e8648c3495bb009083f2d5f7'
parent = read(P/'training_spec.json')
integration = read(P/'integration.json')
for n,h in integration['source_sha256'].items():
    assert sha(P/'code'/n) == h
shutil.copytree(P/'code', R/'code', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for n in ['causal_training.py','support_loss.py','window_competition.py','native_preservation.py','recursive_metric.py','data_inventory.json']:
    shutil.copyfile(P/n, R/n)
model = 'lib/models/sttrack/semantic_spatial_adapter.py'
loader = 'lib/test/tracker/sttrack_semantic.py'
shutil.copyfile(R/'semantic_spatial_adapter.py', R/'code'/model)
p = R/'code'/loader
original = p.read_text()
assert original.count("'semantic_spatial_centered_v1'") == 1
p.write_text(original.replace("'semantic_spatial_centered_v1'", "'semantic_spatial_local_reference_v1'"))
integration.update(status='M88_search_conditioned_local_reference', parent_integration_sha256=sha(P/'integration.json'), changed_or_added_paths=[model,loader])
for n in [model,loader]:
    integration['source_sha256'][n] = sha(R/'code'/n)
save(R/'integration.json', integration)
(R/'native_parity').mkdir()
zero = torch.load(P/'native_parity/category_zero.pth', map_location='cpu')
assert sha(P/'native_parity/category_zero.pth') == parent['initial_checkpoint_sha256']['category']
zero['architecture'] = 'semantic_spatial_local_reference_v1'
torch.save(zero, R/'native_parity/category_zero.pth')
loaded = torch.load(R/'native_parity/category_zero.pth', map_location='cpu')
assert all(torch.equal(v,loaded['model'][k]) for k,v in zero['model'].items())
for split, names in [('fit', ['category','empty']), ('development', ['category','empty','swapped'])]:
    for name in names:
        item = parent['banks'][split][name]
        assert sha(item['path']) == item['sha256']
spec = dict(parent)
spec.update(status='prepared_not_frozen', primary_control='M84 same initial tensors, original text banks, loss, fit order and budget; spatially constant initial binding',
    parent_training_spec_sha256=sha(P/'training_spec.json'), parent_result_path=str(P/'recursive_result.json'), parent_result_sha256=sha(P/'recursive_result.json'),
    plan_sha256=sha(R/'EXPERIMENT_PLAN.md'), training_script_sha256=sha(R/'train_causal.py'), run_queue_sha256=sha(R/'run_m88.sh'),
    preflight_source_sha256=sha(R/'preflight_m88.py'), integration_sha256=sha(R/'integration.json'),
    initial_checkpoint_sha256={'category':sha(R/'native_parity/category_zero.pth')},
    architecture='semantic_spatial_local_reference_v1', local_reference_equation='softmax_j((q_k+r_p) dot z_j/sqrt(64))',
    acceptance_definition='4 M84 increment + 5 native including zero-H10 protection + 4 Swapped content + 1 Empty/native parity; all14 required')
for k in ['banks','sequence_order','seed','learning_rate','weight_decay','gradient_clip','gradient_accumulation_frames','preservation_weight','total_training_track_calls','expected_optimizer_steps']:
    assert spec[k] == parent[k], k
save(R/'training_spec.json', spec)
old = read(P/'recursive_spec.json')
rec = dict(old, training_spec_sha256=sha(R/'training_spec.json'), runner_sha256=sha(R/'run_recursive.py'), queue_sha256=sha(R/'run_m88.sh'))
save(R/'recursive_spec.json', rec)
save(R/'preparation_receipt.json', dict(status='prepared', initial_parameter_tensors_equal=True, original_M84_banks_reused=True,
    changed_model_paths=[model,loader], source_sha256=sha(__file__), training_spec_sha256=sha(R/'training_spec.json'), recursive_spec_sha256=sha(R/'recursive_spec.json')))
print(json.dumps(read(R/'preparation_receipt.json')))
