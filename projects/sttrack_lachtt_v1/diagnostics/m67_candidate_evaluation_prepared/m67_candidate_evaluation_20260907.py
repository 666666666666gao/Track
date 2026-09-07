"""Prepare inputs now; bind the M67 Support final only after both frozen gates pass."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

BASE = Path('/root/autodl-tmp')
M67 = BASE / 'sttrack_m67_supervised_semantic_support_20260907'
ROOT = M67 / 'candidate_evaluation'
INTERFACE = M67 / 'evaluation_interface'
OLD = BASE / 'sttrack_m64_category_candidate_20260907'
TRAIN_SHA = '2027b1893343e5e1520c447bd82a5c45468542cecaab0f59b3ec72a345e48b2e'
INTERFACE_SHA = '323620eab82b163a3a9e6de54ea8e17da5dbc0cf7336b3717ef443082b59e07b'
PROTOCOL_SHA = '61814694e51d9747d42a86d2c36888dba91406213e912368be3600272ff20bcf'
SOURCE_FILES = ['m67_candidate_evaluation_20260907.py', 'm67_learned_entry_parity_20260907.py', 'm67_vot_low22_20260907.py',
                'm67_candidate_entry_20260907.sh', 'm67_candidate_low22_20260907.sh']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def inputs():
    assert sha(M67 / 'training_spec.json') == TRAIN_SHA
    assert sha(M67 / 'recursive_spec.json') == 'd4dd7ca74b8f4b318f8e32f961bc21a10b95aa3802137dd293b49948d83411c5'
    assert sha(INTERFACE / 'spec.json') == INTERFACE_SHA
    interface = read(INTERFACE / 'spec.json'); training = read(M67 / 'training_spec.json')
    assert sha(INTERFACE / 'text_protocol.json') == PROTOCOL_SHA == interface['text_protocol_sha256']
    assert sha(M67 / 'integration.json') == training['integration_sha256']
    integration = read(M67 / 'integration.json')
    for name, digest in integration['source_sha256'].items(): assert sha(M67 / 'code' / name) == digest, name
    for name, digest in interface['interface_sha256'].items(): assert sha(INTERFACE / name) == digest, name
    return training, interface, integration


def prepare():
    import torch
    torch.set_num_threads(1)
    training, interface, _ = inputs()
    assert not (M67 / 'recursive_result.json').exists(), 'Freeze this preparation before M67 outcomes.'
    assert sha(INTERFACE / 'train_initialization_fixture.pt') == interface['train_fixture_sha256']
    assert sha(M67 / 'text_development.pt') == training['text_development_sha256']
    assert sha(OLD / 'low22_category.pt') == '332d92cafa874bfc34a4354af0e49969503c50fbfd09e632224c498ece6c8496'
    assert sha(OLD / 'low22_captions/plan.json') == '5835a31ff987f9f9f1cd89f346843b3830491f8a966585880e631d022bad0ecf'
    sys.path.insert(0, str(INTERFACE))
    from initialization_text import InitializationTextBank, initialization_key, vot_wire_bbox
    fixture = torch.load(INTERFACE / 'train_initialization_fixture.pt', map_location='cpu')
    native = torch.load(M67 / 'text_development.pt', map_location='cpu')
    cases = read(M67 / 'recursive_spec.json')['cases']
    keys, indices = [], []
    for case in cases:
        image = Path(training['dataset_root']) / case['sequence'] / 'color/00000001.jpg'
        key = initialization_key(sha(image), case['init_bbox'])
        keys.append(key); indices.append(fixture['keys'].index(key))
    assert len(keys) == len(set(keys)) == 22
    dev = dict(format=fixture['format'], protocol_sha256=PROTOCOL_SHA, keys=keys,
        tokens=fixture['tokens'][indices].clone(), mask=fixture['mask'][indices].clone(), empty=fixture['empty'].clone())
    low = torch.load(OLD / 'low22_category.pt', map_location='cpu')
    assert set(low) == set(dev)
    low['protocol_sha256'] = PROTOCOL_SHA
    ROOT.mkdir()
    torch.save(dev, ROOT / 'development_category.pt'); torch.save(low, ROOT / 'low22_category.pt')
    routers = {name: InitializationTextBank(ROOT / (name + '_category.pt'), sha(ROOT / (name + '_category.pt')), PROTOCOL_SHA)
               for name in ['development', 'low22']}
    for case in cases:
        image = Path(training['dataset_root']) / case['sequence'] / 'color/00000001.jpg'
        info = routers['development'].info(image, case['init_bbox']); i = native['sequences'].index(case['sequence'])
        assert info['init_bbox'] == case['init_bbox'] == vot_wire_bbox(case['init_bbox'])
        for a, b in [('text_tokens', 'tokens'), ('text_mask', 'mask')]: assert torch.equal(info[a], native[b][i])
        assert torch.equal(info['empty_text'], native['empty'])
    observations = read(OLD / 'low22_captions/plan.json')
    assert len(observations['rows']) == len(observations['cases']) == len(low['keys']) == 303
    for row in observations['rows']:
        assert sha(row['image']) == row['image_sha256']
        assert row['key'] == initialization_key(row['image_sha256'], row['init_bbox'])
        assert row['init_bbox'] == vot_wire_bbox(row['init_bbox'])
        routers['low22'].info(row['image'], row['init_bbox'])
    old = torch.load(OLD / 'low22_category.pt', map_location='cpu')
    assert low['keys'] == old['keys']
    for name in ['tokens', 'mask', 'empty']: assert torch.equal(low[name], old[name])
    assert all(torch.equal(low['tokens'][i, k], low['empty']) for i in range(303) for k in range(1, 5) if low['mask'][i, k])
    spec = dict(status='conditional_M67_evaluation_prepared_before_final_outcomes', observed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={name: sha(BASE / name) for name in SOURCE_FILES}, training_spec_sha256=TRAIN_SHA,
        interface_spec_sha256=INTERFACE_SHA, text_protocol_sha256=PROTOCOL_SHA,
        banks={name: dict(path=str(ROOT / (name + '_category.pt')), sha256=sha(ROOT / (name + '_category.pt')))
               for name in ['development', 'low22']},
        parent_low22_bank_sha256=sha(OLD / 'low22_category.pt'), parent_low22_caption_plan_sha256=sha(OLD / 'low22_captions/plan.json'),
        final_head=str(M67 / 'training/support/final.pth'), architecture='semantic_spatial_support_v1', null_support=True, arm='support', support_loss_weight=.1,
        reported_confidence='Original Hann maximum, identical to M67 recursive predictions; M68 alternative confidence is not used.',
        template_control='Original interval50 and raw Hann confidence strictly greater than0.75.',
        same_bundle_across_three_datasets=True, reused_development_sets_not_unseen=True,
        full_evaluation_gate=dict(EAO_min=57.635993, ACC_min=75.719622, ROB_min=73.022401, confirmed_failures_max=118,
            preserve_all_seven_native_zero_failure_sequences=True, all303_anchors_complete=True, identical_bundle_and_text_protocol=True),
        gate_provenance='Carry forward the predeclared M64 low22 gate unchanged before any M67 final outcome; no new threshold selection.',
        required_order=['M67 completed evidence audit and original11 development gates', 'M67 fixed-Support-head content gates',
                        'Bind this one final head', 'Actual OPE and TraX parity on3 fixed Train prefixes', 'All303 low22 anchors',
                        'Full three datasets only after the low22 gate passes'],
        independent_model_review_pass=False, final_bundle_created=False, public_evaluation_allowed=False)
    write(ROOT / 'spec.json', spec)
    assert not torch.cuda.is_initialized()
    result = dict(status='cpu_M67_evaluation_inputs_prepared', observed_utc=datetime.now(timezone.utc).isoformat(), spec_sha256=sha(ROOT / 'spec.json'),
        actual_Train_initializations_checked=22, actual_existing_low22_initializations_checked=303,
        low22_lexical_tensors_unchanged=True, generated_caption_calls=0, new_embedding_calls=0, new_tracking_calls=0,
        final_checkpoint_loaded=False, cuda_initialized=False, final_bundle_created=False, public_evaluation_started=False,
        subsequent_GT_opened=False, actual_gpu_entry_parity_checked=False, independent_model_review_pass=False)
    write(ROOT / 'cpu_preparation_result.json', result); print(json.dumps(result, indent=2))


def checked_preparation():
    spec = read(ROOT / 'spec.json'); inputs()
    for name, digest in spec['source_sha256'].items(): assert sha(BASE / name) == digest, name
    for bank in spec['banks'].values(): assert sha(bank['path']) == bank['sha256']
    assert spec['training_spec_sha256'] == TRAIN_SHA and spec['text_protocol_sha256'] == PROTOCOL_SHA
    return spec


def gates():
    spec = checked_preparation()
    assert sha(BASE / 'audit_m67_completed_20260907.py') == '1773887deb3be27aa11a4409aa77bd7581f32b8d891ecab2a03454d277e769bf'
    audit = read(M67 / 'completed_evidence_audit.json')
    assert audit['status'] == 'completed_M67_artifacts_and_development_audited'
    assert audit['auditor_sha256'] == sha(BASE / 'audit_m67_completed_20260907.py')
    assert audit['training_spec_sha256'] == TRAIN_SHA
    assert audit['result_sha256'] == sha(M67 / 'recursive_result.json')
    assert audit['paired_development_gate_pass'] and all(audit['recomputed_frozen_gates'].values()), 'Original M67 development gate failed.'
    for name in ['controller', 'training_control', 'training_support', 'control_recursive', 'support_recursive', 'recursive_analysis']:
        assert (M67 / (name + '.exit')).read_text().strip() == '0'
    content = M67 / 'content_counterfactuals'; r = read(content / 'result.json')
    assert r['status'] == 'completed_fixed_head_category_content_controls'
    assert r['source_sha256'] == sha(BASE / 'm67_content_counterfactuals_20260907.py') == '40b765b22c398bc18ed4ac3f5ef33edc7b9611cf1c8d3172062f518d435f8285'
    assert r['spec_sha256'] == sha(content / 'spec.json') == 'c107a0b08a6825803f6a57f0c7ab5f469e06288c3df203dfbb5ce880a4f6c400'
    assert r['M67_completed_audit_sha256'] == sha(M67 / 'completed_evidence_audit.json')
    assert r['content_gate_pass'] and r['low22_candidate_preparation_allowed']
    assert all(value for group in r['gates'].values() for value in group.values()), 'Original M67 content gate failed.'
    for name in ['prefix', 'empty', 'swapped', 'analysis', 'controller']:
        assert (content / (name + '.exit')).read_text().strip() == '0'
    head_sha = sha(spec['final_head'])
    assert head_sha == r['head_sha256'] == audit['training']['support']['final_checkpoint_sha256']
    assert r['receipts']['category'] == sha(M67 / 'support_recursive_receipt.json')
    for name in ['empty', 'swapped']: assert r['receipts'][name] == sha(content / name / 'receipt.json')
    return spec, audit, r


def build():
    import torch
    spec, audit, content = gates()
    assert not (ROOT / 'bundle.json').exists()
    training, interface, integration = inputs()
    head = torch.load(spec['final_head'], map_location='cpu')
    assert head['status'] == 'complete' and head['completed_sequences'] == 130
    for key in ['architecture', 'null_support', 'arm', 'support_loss_weight']: assert head[key] == spec[key]
    assert head['use_text'] and head['training_spec_sha256'] == TRAIN_SHA
    assert head['frame_count'] == 186694 and head['optimizer_steps'] == 5798
    assert head['base_checkpoint_sha256'] == training['native_checkpoint_sha256'] == sha(training['native_checkpoint'])
    bundle = dict(repository=str(M67 / 'code'), configuration='experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml',
        architecture=spec['architecture'], null_support=True, arm='support', support_loss_weight=.1, use_text=True, seed=training['seed'],
        source_sha256=integration['source_sha256'], interface_sha256=interface['interface_sha256'],
        base_checkpoint=training['native_checkpoint'], base_checkpoint_sha256=training['native_checkpoint_sha256'],
        adapter_checkpoint=spec['final_head'], adapter_checkpoint_sha256=sha(spec['final_head']), training_spec_sha256=TRAIN_SHA,
        text_protocol_path=str(INTERFACE / 'text_protocol.json'), text_protocol_sha256=PROTOCOL_SHA,
        reported_confidence=spec['reported_confidence'], template_control=spec['template_control'])
    write(ROOT / 'bundle.json', bundle)
    for name, bank in spec['banks'].items():
        write(ROOT / (name + '_plan.json'), dict(bundle_path=str(ROOT / 'bundle.json'), bundle_sha256=sha(ROOT / 'bundle.json'),
            text_bank_path=bank['path'], text_bank_sha256=bank['sha256']))
    sys.path.insert(0, str(INTERFACE)); from semantic_runtime import checked_plan
    for name in spec['banks']: checked_plan(ROOT / (name + '_plan.json'))
    result = dict(status='one_M67_Support_final_bundle_bound_after_development_and_content_gates',
        observed_utc=datetime.now(timezone.utc).isoformat(), preparation_spec_sha256=sha(ROOT / 'spec.json'),
        bundle_sha256=sha(ROOT / 'bundle.json'), text_protocol_sha256=PROTOCOL_SHA,
        M67_completed_audit_sha256=sha(M67 / 'completed_evidence_audit.json'), content_result_sha256=sha(M67 / 'content_counterfactuals/result.json'),
        head_sha256=sha(spec['final_head']), actual_gpu_entry_parity_checked=False, public_evaluation_allowed=False)
    write(ROOT / 'binding_result.json', result); print(json.dumps(result, indent=2))


def checked():
    spec, _, _ = gates(); binding = read(ROOT / 'binding_result.json')
    assert binding['preparation_spec_sha256'] == sha(ROOT / 'spec.json')
    assert binding['M67_completed_audit_sha256'] == sha(M67 / 'completed_evidence_audit.json')
    assert binding['content_result_sha256'] == sha(M67 / 'content_counterfactuals/result.json')
    assert binding['bundle_sha256'] == sha(ROOT / 'bundle.json')
    sys.path.insert(0, str(INTERFACE)); from semantic_runtime import checked_plan
    _, bundle = checked_plan(ROOT / 'development_plan.json')
    assert bundle['adapter_checkpoint_sha256'] == binding['head_sha256'] == sha(spec['final_head'])
    return dict(spec, bundle_sha256=binding['bundle_sha256'], binding_result_sha256=sha(ROOT / 'binding_result.json'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['prepare', 'check', 'build']); args = p.parse_args()
    {'prepare': prepare, 'check': checked_preparation, 'build': build}[args.action]()
