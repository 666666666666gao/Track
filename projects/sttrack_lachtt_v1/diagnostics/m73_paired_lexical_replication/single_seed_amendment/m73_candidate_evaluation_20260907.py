"""CPU entry preparation; bind only the predetermined M73 final after content passes."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

B = Path('/root/autodl-tmp')
M73 = B / 'sttrack_m73_paired_lexical_replication_20260907'
MODEL = M73 / 'seed2027'
ROOT = M73 / 'candidate_evaluation'
INTERFACE = ROOT / 'interface'
OLD = B / 'sttrack_m67_supervised_semantic_support_20260907/evaluation_interface'
OLD_LOW = B / 'sttrack_m64_category_candidate_20260907'
CONTENT = M73 / 'content_counterfactuals'
CONTENT_SCRIPT = B / 'm73_content_counterfactuals_20260907.py'
CONTENT_SHA = '28a6c668146a8c4b90c5b588d3aa670593bb1a295c59497b2ddfa5a690202341'
CONTENT_SPEC_SHA = '151d1bf496af7b10d1af2e9e1d454b6afdd481a65f2bead30ed729271038822d'
INTERFACE_SHA = '323620eab82b163a3a9e6de54ea8e17da5dbc0cf7336b3717ef443082b59e07b'
SOURCES = ['m73_candidate_evaluation_20260907.py', 'm73_learned_entry_parity_20260907.py', 'm73_vot_low22_20260907.py',
    'm73_candidate_entry_20260907.sh', 'm73_candidate_low22_20260907.sh']


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def read(p): return json.loads(Path(p).read_text())
def write(p, v): Path(p).write_text(json.dumps(v, indent=2, allow_nan=False) + '\n')
def now(): return datetime.now(timezone.utc).isoformat()


def inputs():
    assert sha(CONTENT_SCRIPT) == CONTENT_SHA and sha(CONTENT / 'spec.json') == CONTENT_SPEC_SHA
    loader = importlib.util.spec_from_file_location('m73_content_entry_prerequisite', str(CONTENT_SCRIPT))
    module = importlib.util.module_from_spec(loader); loader.loader.exec_module(module)
    _, training = module.checked()
    assert sha(OLD / 'spec.json') == INTERFACE_SHA
    interface = read(OLD / 'spec.json')
    for n, h in interface['interface_sha256'].items(): assert sha(OLD / n) == h
    assert sha(OLD / 'text_protocol.json') == interface['text_protocol_sha256']
    integration = read(MODEL / 'integration.json')
    assert sha(MODEL / 'integration.json') == training['integration_sha256']
    return training, interface, integration, module


def prepare():
    import torch
    torch.set_num_threads(1)
    training, parent, _, _ = inputs()
    assert not (M73 / 'result.json').exists() and not (MODEL / 'training/category/final.pth').exists()
    assert not (CONTENT / 'activation.json').exists()
    assert sha(OLD / 'train_initialization_fixture.pt') == parent['train_fixture_sha256']
    assert sha(OLD_LOW / 'low22_category.pt') == '332d92cafa874bfc34a4354af0e49969503c50fbfd09e632224c498ece6c8496'
    assert sha(OLD_LOW / 'low22_captions/plan.json') == '5835a31ff987f9f9f1cd89f346843b3830491f8a966585880e631d022bad0ecf'
    ROOT.mkdir(); INTERFACE.mkdir()
    for n in parent['interface_sha256']:
        shutil.copyfile(OLD / n, INTERFACE / n); compile((INTERFACE / n).read_text(), n, 'exec')
    protocol = read(OLD / 'text_protocol.json')
    protocol['training_spec_sha256'] = sha(MODEL / 'training_spec.json')
    protocol['selection_provenance'] = ('M73 compares category and empty training in fixed seed2027; seed2028 was cancelled by the user; '
        'seed2027 Category final is the only candidate, conditional on unchanged single-seed development and same-head content checks. '
        'Category retention is used in this arm during both training and deployment.')
    write(INTERFACE / 'text_protocol.json', protocol)
    protocol_sha = sha(INTERFACE / 'text_protocol.json')
    sys.path.insert(0, str(INTERFACE))
    from initialization_text import InitializationTextBank, initialization_key, vot_wire_bbox
    fixture = torch.load(OLD / 'train_initialization_fixture.pt', map_location='cpu')
    banks = {split: torch.load(training['banks'][split]['category']['path'], map_location='cpu') for split in ['fit', 'development']}
    assert len(fixture['keys']) == len(fixture['sources']) == 152
    for i, row in enumerate(fixture['sources']):
        bank = banks[row['split']]; j = bank['sequences'].index(row['sequence'])
        image = Path(training['dataset_root']) / row['sequence'] / 'color/00000001.jpg'
        assert sha(image) == row['initial_rgb_sha256']
        assert row['init_bbox'] == vot_wire_bbox(row['init_bbox'])
        assert fixture['keys'][i] == initialization_key(sha(image), row['init_bbox'])
        assert torch.equal(fixture['tokens'][i], bank['tokens'][j]) and torch.equal(fixture['mask'][i], bank['mask'][j])
        assert torch.equal(fixture['empty'], bank['empty'])
    cases = read(MODEL / 'recursive_spec.json')['cases']; indices = []
    for c in cases:
        image = Path(training['dataset_root']) / c['sequence'] / 'color/00000001.jpg'
        indices.append(fixture['keys'].index(initialization_key(sha(image), c['init_bbox'])))
    dev = dict(format=fixture['format'], protocol_sha256=protocol_sha, keys=[fixture['keys'][i] for i in indices],
        tokens=fixture['tokens'][indices].clone(), mask=fixture['mask'][indices].clone(), empty=fixture['empty'].clone())
    low = torch.load(OLD_LOW / 'low22_category.pt', map_location='cpu'); assert set(low) == set(dev)
    low['protocol_sha256'] = protocol_sha
    torch.save(dev, ROOT / 'development_category.pt'); torch.save(low, ROOT / 'low22_category.pt')
    routers = {n: InitializationTextBank(ROOT / (n+'_category.pt'), sha(ROOT / (n+'_category.pt')), protocol_sha) for n in ['development', 'low22']}
    for c in cases:
        image = Path(training['dataset_root']) / c['sequence'] / 'color/00000001.jpg'
        info = routers['development'].info(image, c['init_bbox']); j = banks['development']['sequences'].index(c['sequence'])
        assert info['init_bbox'] == c['init_bbox'] == vot_wire_bbox(c['init_bbox'])
        assert torch.equal(info['text_tokens'], banks['development']['tokens'][j]) and torch.equal(info['text_mask'], banks['development']['mask'][j])
        assert torch.equal(info['empty_text'], banks['development']['empty'])
    plan = read(OLD_LOW / 'low22_captions/plan.json'); assert len(plan['rows']) == len(plan['cases']) == len(low['keys']) == 303
    for row in plan['rows']:
        assert sha(row['image']) == row['image_sha256'] and row['key'] == initialization_key(row['image_sha256'], row['init_bbox'])
        assert row['init_bbox'] == vot_wire_bbox(row['init_bbox'])
        routers['low22'].info(row['image'], row['init_bbox'])
    old = torch.load(OLD_LOW / 'low22_category.pt', map_location='cpu')
    assert low['keys'] == old['keys']
    for n in ['tokens', 'mask', 'empty']: assert torch.equal(low[n], old[n])
    assert all(torch.equal(low['tokens'][i,k], low['empty']) for i in range(303) for k in range(1,5) if low['mask'][i,k])
    for name in SOURCES:
        if name.endswith('.py'): compile((B / name).read_text(), name, 'exec')
        else: subprocess.run(['bash', '-n', str(B / name)], check=True)
    spec = dict(status='M73_candidate_entry_CPU_prepared_before_final_results', observed_utc=now(),
        source_sha256={n:sha(B/n) for n in SOURCES}, training_spec_sha256=sha(MODEL / 'training_spec.json'),
        M73_spec_sha256=sha(M73 / 'spec.json'), M73_frozen_sha256=sha(M73 / 'frozen.json'),
        content_source_sha256=CONTENT_SHA, content_spec_sha256=CONTENT_SPEC_SHA,
        parent_interface_spec_sha256=INTERFACE_SHA, interface_sha256={n:sha(INTERFACE/n) for n in parent['interface_sha256']},
        text_protocol_sha256=protocol_sha, initial_fixture_sha256=parent['train_fixture_sha256'],
        banks={n:dict(path=str(ROOT/(n+'_category.pt')),sha256=sha(ROOT/(n+'_category.pt'))) for n in ['development','low22']},
        parent_low22_bank_sha256=sha(OLD_LOW/'low22_category.pt'), parent_low22_caption_plan_sha256=sha(OLD_LOW/'low22_captions/plan.json'),
        final_head=str(MODEL/'training/category/final.pth'), candidate_seed=2027,
        architecture='semantic_spatial_support_v1', null_support=True, arm='category', support_loss_weight=0.,
        reported_confidence='Original Hann maximum; no Null product, normalization, or confidence rescaling.',
        template_control='Original interval50 and raw Hann maximum strictly greater than0.75.',
        same_bundle_across_three_datasets=True, reused_development_sets_not_unseen=True,
        full_evaluation_gate=dict(EAO_min=57.635993,ACC_min=75.719622,ROB_min=73.022401,confirmed_failures_max=118,
            preserve_all_seven_native_zero_failure_sequences=True,all303_anchors_complete=True,identical_bundle_and_text_protocol=True),
        gate_provenance='Carry forward the M64/M67 low22 gate unchanged before M73 final outcomes.',
        entry_prefixes=['bag05_indoor','container01_indoor','mobilephone02_indoor'],frames_per_entry_prefix=202,
        required_order=['Retained M73 seed2027 development gates and completed audit','Fixed seed2027 Category head content gates',
            'Bind that one final','Real OPE/TraX parity on three Train prefixes','All303 low22 anchors','Full three datasets only after low22 gate passes'],
        actual_gpu_entry_parity_checked=False,final_bundle_created=False,public_evaluation_allowed=False,independent_model_review_pass=False)
    write(ROOT/'spec.json',spec)
    result = dict(status='M73_CPU_initialization_routing_and_inherited_interface_verified',observed_utc=now(),
        spec_sha256=sha(ROOT/'spec.json'),actual_Train_initializations_checked=152,development_router_matches=22,
        existing_low22_initializations_checked=303,low22_lexical_tensors_unchanged=True,interface_files_byte_identical_to_M67=5,
        actual_gpu_entry_parity_checked=False,new_tracking_calls=0,new_optimizer_steps=0,subsequent_GT_opened=False,
        new_caption_calls=0,new_embedding_calls=0,final_checkpoint_loaded=False,final_bundle_created=False,
        cuda_initialized=torch.cuda.is_initialized(),public_evaluation_started=False,independent_model_review_pass=False)
    assert not result['cuda_initialized'];write(ROOT/'cpu_preparation_result.json',result);print(json.dumps(result))


def checked_preparation():
    training, _, _, _ = inputs(); spec = read(ROOT/'spec.json')
    for n,h in spec['source_sha256'].items(): assert sha(B/n)==h
    for n,h in spec['interface_sha256'].items(): assert sha(INTERFACE/n)==h
    for bank in spec['banks'].values(): assert sha(bank['path'])==bank['sha256']
    assert sha(INTERFACE/'text_protocol.json')==spec['text_protocol_sha256']
    assert sha(MODEL/'training_spec.json')==spec['training_spec_sha256']
    assert spec['candidate_seed']==2027 and spec['final_head']==str(MODEL/'training/category/final.pth')
    return spec


def gates():
    spec=checked_preparation(); _,_,_,module=inputs()
    _,_,audit=module.eligible()
    assert (M73/'single_seed_amendment/controller/controller.exit').read_text().strip()=='0'
    follower=read(M73/'single_seed_amendment/controller/result.json')
    assert follower['status']=='completed_M73_audit_and_conditional_content' and follower['content_started']
    result=read(CONTENT/'result.json')
    assert follower['content_result_sha256']==sha(CONTENT/'result.json')
    assert result['status']=='completed_M73_predetermined_final_fixed_head_content'
    assert result['source_sha256']==CONTENT_SHA and result['spec_sha256']==CONTENT_SPEC_SHA
    assert result['M73_audit_sha256']==sha(M73/'completion_tools/completed.json')
    assert result['lexical_criteria_pass'] and result['candidate_entry_parity_preparation_allowed']
    assert all(v for g in result['lexical_criteria'].values() for v in g.values())
    for n in ['activate','idle','prefix','empty','swapped','analyze','controller']:
        assert (CONTENT/(n+'.exit')).read_text().strip()=='0'
    assert result['candidate_seed']==2027 and sha(spec['final_head'])==result['head_sha256']==audit['candidate_head_sha256']
    assert result['receipts']['category']==sha(MODEL/'category_recursive_receipt.json')
    for n in ['empty','swapped']: assert result['receipts'][n]==sha(CONTENT/n/'receipt.json')
    return spec,audit,result


def build():
    import torch
    spec,audit,content=gates(); assert not (ROOT/'bundle.json').exists()
    training,_,integration,_=inputs();head=torch.load(spec['final_head'],map_location='cpu')
    assert head['status']=='complete' and head['completed_sequences']==130
    for k in ['architecture','null_support','arm','support_loss_weight']:assert head[k]==spec[k]
    assert head['use_text'] and head['seed']==2027 and head['training_spec_sha256']==spec['training_spec_sha256']
    assert head['frame_count']==186694 and head['optimizer_steps']==5798
    assert head['base_checkpoint_sha256']==training['native_checkpoint_sha256']==sha(training['native_checkpoint'])
    bundle=dict(repository=str(MODEL/'code'),configuration='experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml',
        architecture=spec['architecture'],null_support=True,arm='category',support_loss_weight=0.,use_text=True,seed=2027,
        source_sha256=integration['source_sha256'],interface_sha256=spec['interface_sha256'],
        base_checkpoint=training['native_checkpoint'],base_checkpoint_sha256=training['native_checkpoint_sha256'],
        adapter_checkpoint=spec['final_head'],adapter_checkpoint_sha256=sha(spec['final_head']),training_spec_sha256=spec['training_spec_sha256'],
        text_protocol_path=str(INTERFACE/'text_protocol.json'),text_protocol_sha256=spec['text_protocol_sha256'],
        reported_confidence=spec['reported_confidence'],template_control=spec['template_control'])
    write(ROOT/'bundle.json',bundle)
    for n,bank in spec['banks'].items():write(ROOT/(n+'_plan.json'),dict(bundle_path=str(ROOT/'bundle.json'),bundle_sha256=sha(ROOT/'bundle.json'),
        text_bank_path=bank['path'],text_bank_sha256=bank['sha256']))
    sys.path.insert(0,str(INTERFACE));from semantic_runtime import checked_plan
    for n in spec['banks']:checked_plan(ROOT/(n+'_plan.json'))
    result=dict(status='M73_predetermined_Category_final_bound_after_development_and_content',observed_utc=now(),
        preparation_spec_sha256=sha(ROOT/'spec.json'),bundle_sha256=sha(ROOT/'bundle.json'),
        M73_audit_sha256=sha(M73/'completion_tools/completed.json'),content_result_sha256=sha(CONTENT/'result.json'),
        head_sha256=sha(spec['final_head']),actual_gpu_entry_parity_checked=False,public_evaluation_allowed=False)
    write(ROOT/'binding_result.json',result);print(json.dumps(result))


def checked():
    spec,_,_=gates();bound=read(ROOT/'binding_result.json')
    assert bound['preparation_spec_sha256']==sha(ROOT/'spec.json')
    assert bound['M73_audit_sha256']==sha(M73/'completion_tools/completed.json')
    assert bound['content_result_sha256']==sha(CONTENT/'result.json')
    assert bound['bundle_sha256']==sha(ROOT/'bundle.json')
    sys.path.insert(0,str(INTERFACE));from semantic_runtime import checked_plan
    _,bundle=checked_plan(ROOT/'development_plan.json')
    assert bundle['adapter_checkpoint_sha256']==bound['head_sha256']==sha(spec['final_head'])
    return dict(spec,bundle_sha256=bound['bundle_sha256'],binding_result_sha256=sha(ROOT/'binding_result.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','check','build']);a=p.parse_args()
    {'prepare':prepare,'check':checked_preparation,'build':build}[a.action]()
