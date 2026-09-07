"""Isolate the support-architecture evaluation entry; leave historical interfaces intact."""
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import py_compile
import shutil
import sys

BASE = Path('/root/autodl-tmp')
ROOT = BASE / 'sttrack_m67_supervised_semantic_support_20260907'
OLD = BASE / 'sttrack_m58_evaluation_preparation_20260906'
OUT = ROOT / 'evaluation_interface'
SHA = {
    'initialization_text.py': '5ffd459fe13242537571809f8a77b1a57c2b5dea97291221f4c5b4738eedf941',
    'semantic_runtime.py': '7d71bfd7d0c3deb657d45657c491d7aa5d8b6544c2a6ac1626cf7940d6f6c261',
    'run_semantic_ope.py': 'e307b62197df87d4626a5693350fb14461b1ae397499ebd6fd92d122cba98758',
    'run_semantic_vot.py': '1ae1f2246aaa943f7d896a4086937b4680bc4b4c6d9dd61d7829ca1a7f5d7272',
    'm39_vot_bridge.py': '230acf10f378a6babfacf9979ea07a1ce89c34952cc0c6b9568376249e265316',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    import torch
    torch.set_num_threads(1)
    assert sha(ROOT / 'training_spec.json') == '2027b1893343e5e1520c447bd82a5c45468542cecaab0f59b3ec72a345e48b2e'
    training = read(ROOT / 'training_spec.json')
    assert sha(ROOT / 'integration.json') == training['integration_sha256']
    integration = read(ROOT / 'integration.json')
    for name, digest in integration['source_sha256'].items():
        assert sha(ROOT / 'code' / name) == digest
    for name, digest in SHA.items():
        assert sha(OLD / name) == digest
    assert sha(OLD / 'text_protocol.json') == 'd08acfb068ac5f7c428d5decb5f17af4655383543eeb4b4277e65cfda14bcac1'
    assert not (ROOT / 'recursive_result.json').exists()
    OUT.mkdir()
    for name in SHA:
        shutil.copyfile(OLD / name, OUT / name)
    path = OUT / 'semantic_runtime.py'; original = path.read_text()
    before = "    assert bundle['architecture'] == 'semantic_spatial_v1'"
    after = "    assert bundle['architecture'] == 'semantic_spatial_support_v1'\n    assert bundle['null_support'] is True"
    assert original.count(before) == 1
    updated = original.replace(before, after)
    before = "    assert saved['use_text'] == bundle['use_text']"
    after = before + "\n    assert saved['architecture'] == bundle['architecture']\n    assert saved['null_support'] == bundle['null_support']\n    assert saved['arm'] == bundle['arm']\n    assert saved['support_loss_weight'] == bundle['support_loss_weight']"
    assert updated.count(before) == 1
    updated = updated.replace(before, after)
    path.write_text(updated)
    (OUT / 'runtime_delta.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True), updated.splitlines(True),
        fromfile='M58/semantic_runtime.py', tofile='M67/semantic_runtime.py')))
    for name in SHA:
        py_compile.compile(str(OUT / name), doraise=True)
        if name != 'semantic_runtime.py':
            assert sha(OUT / name) == SHA[name]
    old_category_protocol = BASE / 'sttrack_m64_category_candidate_20260907/text_protocol.json'
    assert sha(old_category_protocol) == '20d3e4ee754822053239e6c7073b7819c8928ce4904620596ef4596e51912846'
    prior = read(old_category_protocol)
    protocol = {k: v for k, v in prior.items() if k != 'selection_provenance'}
    protocol['selection_provenance'] = ('M67 uses the category-only lexical policy during both paired training and inference, '
        'following M65. Both M67 heads are newly trained from zero residual initialization. '
        'This is not deployment-only removal of attributes from the M58 full-attribute head.')
    protocol['training_spec_sha256'] = sha(ROOT / 'training_spec.json')
    write(OUT / 'text_protocol.json', protocol)
    sys.path.insert(0, str(OUT))
    from initialization_text import InitializationTextBank, initialization_key, vot_wire_bbox
    from run_semantic_ope import write_predictions
    import semantic_runtime
    import numpy as np
    import trax
    assert Path(semantic_runtime.__file__).resolve() == OUT / 'semantic_runtime.py'
    assert sha(ROOT / 'data_inventory.json') == training['inventory_sha256']
    inventory = read(ROOT / 'data_inventory.json')
    banks = {}
    for split, key in [('fit', 'text_fit_sha256'), ('development', 'text_development_sha256')]:
        p = ROOT / ('text_' + split + '.pt'); assert sha(p) == training[key]
        banks[split] = torch.load(p, map_location='cpu')
    assert torch.equal(banks['fit']['empty'], banks['development']['empty'])
    keys, tokens, masks, sources = [], [], [], []
    for row in inventory['sequences_detail']:
        bank = banks[row['split']]; index = bank['sequences'].index(row['sequence'])
        text = bank['tokens'][index]; mask = bank['mask'][index]
        assert text.shape == (5, 768) and mask.shape == (5,)
        assert all(torch.equal(text[k], bank['empty']) for k in range(1, 5) if mask[k])
        rgb_sha = sha(row['initial_rgb']); box = row['first_box']
        # All actual Train initializations are unchanged by the real TraX rectangle conversion.
        assert box == list(trax.Rectangle.create(*box).bounds()) == vot_wire_bbox(box)
        keys.append(initialization_key(rgb_sha, box)); tokens.append(text); masks.append(mask)
        sources.append(dict(sequence=row['sequence'], split=row['split'], initial_rgb_sha256=rgb_sha, init_bbox=box, original_index=index))
    assert len(keys) == len(set(keys)) == 152
    fixture = dict(format='initialization_observation_v1', protocol_sha256=sha(OUT / 'text_protocol.json'), keys=keys,
        tokens=torch.stack(tokens), mask=torch.stack(masks), empty=banks['fit']['empty'], sources=sources,
        source_text_bank_sha256={split: sha(ROOT / ('text_' + split + '.pt')) for split in banks},
        scope='CPU Train initialization routing fixture; no candidate head or public evaluation bundle.')
    torch.save(fixture, OUT / 'train_initialization_fixture.pt')
    router = InitializationTextBank(OUT / 'train_initialization_fixture.pt', sha(OUT / 'train_initialization_fixture.pt'), sha(OUT / 'text_protocol.json'))
    for row in inventory['sequences_detail']:
        bank = banks[row['split']]; index = bank['sequences'].index(row['sequence'])
        info = router.info(row['initial_rgb'], row['first_box'])
        assert info['init_bbox'] == row['first_box']
        assert torch.equal(info['text_tokens'], bank['tokens'][index])
        assert torch.equal(info['text_mask'], bank['mask'][index])
        assert torch.equal(info['empty_text'], bank['empty'])
    # Reuse sealed historical predictions to exercise only the actual six-decimal writer.
    historic = BASE / 'sttrack_m65_category_null_support_20260907'
    receipt = read(historic / 'null_recursive_receipt.json')
    expected = next(x for x in receipt['sequences'] if x['sequence'] == 'bag05_indoor')
    p = historic / 'recursive/null/bag05_indoor.json'; assert sha(p) == expected['sha256']
    rows = read(p)['rows']
    serialized = OUT / 'serialization_reference'; serialized.mkdir()
    result = write_predictions(serialized, 'historical_bag05', [r['bbox'] for r in rows], [1.] + [r['score'] for r in rows[1:]])
    assert len(rows) == 889 and result['frames'] == 889
    for name, digest in SHA.items():
        assert sha(OLD / name) == digest
    assert sha(ROOT / 'training_spec.json') == protocol['training_spec_sha256']
    for name, digest in integration['source_sha256'].items():
        assert sha(ROOT / 'code' / name) == digest
    assert not torch.cuda.is_initialized()
    spec = dict(status='support_architecture_interface_prepared_before_M67_final', observed_utc=datetime.now(timezone.utc).isoformat(),
        preparer_sha256=sha(__file__), parent_interface_source_sha256=SHA,
        interface_sha256={name: sha(OUT / name) for name in SHA}, runtime_delta_sha256=sha(OUT / 'runtime_delta.patch'),
        architecture='semantic_spatial_support_v1', null_support=True,
        checkpoint_bound_metadata=['architecture', 'null_support', 'arm', 'support_loss_weight', 'training_spec_sha256', 'use_text'],
        training_spec_sha256=sha(ROOT / 'training_spec.json'), integration_sha256=sha(ROOT / 'integration.json'),
        text_protocol_sha256=sha(OUT / 'text_protocol.json'), train_fixture_sha256=sha(OUT / 'train_initialization_fixture.pt'),
        allowed_use='Prepare only. Fixed final bundle and actual OPE/TraX replay parity require M67 completion and content checks; no promotion from CPU routing.',
        original_interfaces_modified=False, training_or_model_code_modified=False, public_evaluation_allowed=False)
    write(OUT / 'spec.json', spec)
    check = dict(status='cpu_support_interface_and_category_routing_checked', observed_utc=datetime.now(timezone.utc).isoformat(),
        spec_sha256=sha(OUT / 'spec.json'), actual_Train_initializations_checked=152, exact_token_mask_bbox_matches=152,
        unchanged_interface_files=4, changed_interface_files=['semantic_runtime.py'],
        generated_caption_calls=0, new_embedding_calls=0, new_tracking_calls=0, new_optimizer_steps=0,
        subsequent_GT_opened=False, public_images_opened=0, cuda_initialized=False,
        historical_serialization_reference=result, actual_completed_M67_checkpoint_loading_checked=False,
        actual_gpu_wrapper_parity_checked=False, actual_TraX_server_exchange_checked=False,
        final_bundle_created=False, independent_model_review_pass=False)
    write(OUT / 'cpu_contract_result.json', check)
    print(json.dumps(dict(spec_sha256=sha(OUT / 'spec.json'), result=check), indent=2), flush=True)


if __name__ == '__main__':
    main()
