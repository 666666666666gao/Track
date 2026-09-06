"""Freeze M57 after the completed, forced-native runtime input contract."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


root = Path('/root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906')
prior = Path('/root/autodl-tmp/sttrack_m56_attribute_alignment_v1_20260906')
collection = json.loads((root / 'collection_spec.json').read_text())
receipt = json.loads((root / 'initial_references/receipt.json').read_text())
contract_spec = json.loads((root / 'runtime_contract_spec.json').read_text())
contract = json.loads((root / 'runtime_contract/contract.json').read_text())
parent = json.loads((prior / 'spec.json').read_text())
variants = ['candidate_text', 'candidate_visual', 'initial_text', 'initial_visual']
assert not (root / 'spec.json').exists() and not (root / 'training').exists()
assert (root / 'collection.exit').read_text().strip() == '0'
assert (root / 'runtime_contract.exit').read_text().strip() == '0'
assert receipt['status'] == 'complete_initial_reference_collection'
assert receipt['spec_sha256'] == sha(root / 'collection_spec.json')
assert contract['status'] == 'complete_runtime_input_contract'
assert contract['spec_sha256'] == sha(root / 'runtime_contract_spec.json')
assert [row['variant'] for row in contract['arms']] == variants
assert contract['native_choices_forced'] and contract['formal_training_steps'] == 0
for row in contract['arms']:
    assert row['frames'] == 1002 and len(row['checked_events']) == 22
    assert row['runtime_input_tensors_exactly_equal'] and row['event_public_boxes_exactly_equal']
    assert row['initial_state_sha256'] == '234b6c97a3f53bd0832679794c9e9f37b749555daa36b818e550dafb3557395a'
code = Path(collection['code_root'])
for name, digest in collection['source_sha256'].items():
    assert sha(code / name) == digest, name
assert len(collection['source_sha256']) == 172
ast.parse((root / 'train.py').read_text(), feature_version=(3, 8))
ast.parse((root / 'runtime.py').read_text(), feature_version=(3, 8))
assert sha(root / 'runtime.py') == collection['source_sha256']['lib/test/tracker/sttrack_initial_instance_candidate_set.py']
inputs = [root / name for name in [
    'base_decision.json', 'EXPERIMENT_PLAN.md', 'collection_spec.json',
    'initial_references/receipt.json', 'runtime_contract_spec.json',
    'runtime_contract/contract.json', 'check_runtime_inputs.py',
    'text_fit.pt', 'initial_references/initial_fit.pt', 'fit_labels.json',
    'collect_initial.py', 'freeze_training.py']]
inputs += [prior / 'spec.json', Path(parent['source_root']) / 'spec.json',
           Path(parent['base_checkpoint'])]
assert sha(root / 'text_fit.pt') == collection['text_files']['fit']['sha256']
assert sha(root / 'initial_references/initial_fit.pt') == receipt['files']['fit']['sha256']
assert sha(root / 'fit_labels.json') == collection['target_file_sha256']
assert sha(parent['base_checkpoint']) == collection['checkpoint_sha256']
assert len(parent['fit_features']) == 63 and sum(x['events'] for x in parent['fit_features']) == 1511
spec = dict(
    schema='sttrack_m57_initial_instance_binding_v1',
    frozen_utc=datetime.now(timezone.utc).isoformat(),
    repository=str(code), source_sha256=collection['source_sha256'],
    source_root=parent['source_root'], dataset_root=parent['dataset_root'],
    parent_spec_sha256=parent['parent_spec_sha256'],
    prior_m56_spec_sha256=sha(prior / 'spec.json'),
    trainer_sha256=sha(root / 'train.py'),
    input_sha256={str(path): sha(path) for path in inputs},
    base_checkpoint=parent['base_checkpoint'], base_checkpoint_sha256=collection['checkpoint_sha256'],
    fit_features=parent['fit_features'], parameters=484547, variants=variants,
    optimization=dict(seed=2026, epochs=20, batch_size=32, lr=0.0003, weight_decay=0.01,
        grad_clip=1.0, optimizer='AdamW', optimizer_steps=960, device='cuda', cpu_threads=4,
        precision='float32; no AMP; all four arms sequential on physical GPU0', checkpoint='fixed final epoch only'),
    model=dict(phrase_dimension=768, phrase_slots=5, local_channels=32, learned_slot_parameters=160,
        same_slot_mask_across_arms=True, all_arms_use_same_t0_reference=True,
        visual_control_retains_slot_metadata=True, initial_binding_object_index=20,
        backbone_frozen=True, text_encoder_frozen=True, native_default_template_updates=True,
        candidate_generation_uses_language=False, native_TSG_aliasing_retained=True),
    training_target=parent['training_target'], state_distribution=parent['state_distribution'],
    annotation_scope=parent['annotation_scope'],
    training_data_scope='Only 63 fitting sequences, 1511 labels, separate fitting text and t0 banks; no development tensor or numeric label files are opened by fitting',
    recursive_gate=dict(primary_variant='initial_text', mean_gain_vs_native=0.002,
        mean_gain_vs_initial_visual=0.001, mean_gain_vs_candidate_text=0.001,
        semantic_difference_of_differences_strictly_positive=True,
        low_frames_no_increase_vs_native=True, H10_no_increase_vs_native=True,
        protect_native_zero_H10=True, all_four_arms_all_22_sequences_required=True),
    low22_gate=parent['low22_gate'],
    evaluation_order='All four final weights sealed before development numeric analysis; all four complete recursive predictions sealed before recursive GT analysis; primary gate then fixed-weight content controls then low22 then the same model on three complete datasets',
    independent_review='Astra/max partial prior review retained; service quota prevents a new completed independent review; no fabricated PASS',
    claim='Frozen hypothesis only: native-state cache remains, and neither language benefit nor full identity association or memory innovation is established')
(root / 'spec.json').write_text(json.dumps(spec, indent=2, allow_nan=False) + '\n')
print(json.dumps(dict(status='frozen_training_spec', spec_sha256=sha(root / 'spec.json'),
    variants=variants, parameters=spec['parameters'], source_files=len(spec['source_sha256']),
    trainer_sha256=spec['trainer_sha256'], contract_sha256=sha(root / 'runtime_contract/contract.json'))))
