"""Conditional full-dataset preparation for the unchanged M64 bundle."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT=Path('/root/autodl-tmp/sttrack_m64_category_candidate_20260907')
FULL=ROOT/'full_evaluation'
INTERFACE=Path('/root/autodl-tmp/sttrack_m58_evaluation_preparation_20260906')
GENERATOR=Path('/root/autodl-tmp/sttrack_m58_initialization_generator_20260906')
EXPORTER=Path('/root/autodl-tmp/sttrack_m58_vot_initialization_export_20260906/export_initializations.py')
NATIVE=Path('/root/autodl-tmp/sttrack_default_rgbd_ope_v1_20260906')
MANIFEST=Path('/root/autodl-tmp/sttrack_default_full127_v1_20260905/run/shard_manifest.json')
FINGERPRINTS={NATIVE/'spec.json':'e623ef63de89da423fc12b877f58d56219f88ef9f1232f20de9e42fb6c8665e1',
 NATIVE/'inputs.json':'61541e35f7b9e3c40427df79067fc0be20b8622cf275e93025e4a1547bf68601',
 NATIVE/'metrics_depthtrack.json':'bd89f02b8be95699cc845dae1a2473cc553a02771ee94093b84504878fa3892d',
 NATIVE/'metrics_cdtb.json':'a00f5db6e853dd9ac59ef2e174ed7a06e991701e61bfd85badf29ae273c1b731',
 EXPORTER:'3459b8fb2274dc79aa6132172fca2c20ba817867f5a028f80186711029694159',
 MANIFEST:'8e76256f1c7c135a65a1b262506356769b59557db60eb83bc21ef1392890dd01'}


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def module(name,path):
    s=importlib.util.spec_from_file_location(name,str(path));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def sources():
    spec=module('candidate64',Path('/root/autodl-tmp/m64_category_candidate_20260907.py')).checked()
    for p,h in FINGERPRINTS.items():assert sha(p)==h,p
    sys.path.insert(0,str(INTERFACE))
    from semantic_runtime import checked_plan
    _,bundle=checked_plan(ROOT/'low22_plan.json')
    inputs=json.loads((NATIVE/'inputs.json').read_text());native=json.loads((NATIVE/'spec.json').read_text())
    for name,count,frames in [('depthtrack',50,76373),('cdtb',80,101956)]:
        assert len(inputs[name])==count and sum(r['frames'] for r in inputs[name])==frames
        assert all(r['root']==native['datasets'][name]['root'] for r in inputs[name])
        gt=json.loads((NATIVE/('metrics_'+name+'.json')).read_text())['groundtruth_sha256']
        assert set(gt)=={r['sequence'] for r in inputs[name]}
    full=json.loads(MANIFEST.read_text());assert len(full['sequences'])==127 and full['total_anchor_count']==1765
    assert full['total_estimated_frames']==1327004
    return spec,bundle,inputs,native


def eligible():
    spec,bundle,inputs,native=sources()
    assert (ROOT/'low22_controller.exit').read_text().strip()=='0'
    result=json.loads((ROOT/'low22_result.json').read_text())
    assert result['full_three_dataset_evaluation_allowed'] and all(result['gate_checks'].values())
    assert result['bundle_sha256']==spec['bundle_sha256'] and result['text_protocol_sha256']==spec['text_protocol_sha256']
    assert result['execution_sha256']==sha(ROOT/'low22_execution.json')
    assert result['merge_sha256']==sha(ROOT/'low22_run/merge_result.json')
    gate=spec['full_evaluation_gate'];metrics=result['metrics_percent']
    assert all(metrics[k]>=gate[k+'_min'] for k in ['EAO','ACC','ROB'])
    assert result['confirmed_failures']<=gate['confirmed_failures_max']
    assert len(result['failure_outcomes'])==303 and len(result['protected_native_sequences'])==7
    assert all(result['per_sequence_failures'][n]['confirmed_failures']==0 for n in result['protected_native_sequences'])
    return spec,bundle,inputs,native


def check_sources():
    spec,bundle,inputs,native=sources()
    p=ROOT/'full_preparation_readiness.json'
    result=dict(status='source_and_metadata_contract_verified_only',observed_utc=datetime.now(timezone.utc).isoformat(),
        preparation_source_sha256=sha(__file__),bundle_sha256=spec['bundle_sha256'],text_protocol_sha256=spec['text_protocol_sha256'],
        source_sha256={str(p):h for p,h in FINGERPRINTS.items()},
        OPE_datasets=native['datasets'],VOT_sequences=127,VOT_anchors=1765,VOT_planned_frame_positions=1327004,
        public_images_decoded=0,new_caption_calls=0,new_tracking_calls=0,new_training_steps=0,
        full_low22_gate_required=True,full_evaluation_started=False,
        next='Only after all M64 low22 checks pass: prepare first-frame observations, generate and encode with the shared protocol, bind the same bundle, then run full inference. This check does not qualify full evaluation or claim an execution test.')
    write(p,result);print(json.dumps(result,indent=2))


def prepare_ope(name):
    spec,bundle,inputs,native=eligible()
    root=FULL/name;root.mkdir(parents=True)
    gt=json.loads((NATIVE/('metrics_'+name+'.json')).read_text())['groundtruth_sha256']
    cases=[];observations=[]
    for row in inputs[name]:
        folder=Path(row['root'])/row['sequence']
        assert sha(folder/'groundtruth.txt')==gt[row['sequence']]
        cases.append(dict(sequence=row['sequence'],frames=row['frames'],init_bbox=row['init_bbox'],gt_sha256=gt[row['sequence']]))
        observations.append(dict(id=row['sequence'],image=str(folder/'color/00000001.jpg'),bbox=row['init_bbox']))
    write(root/'cases.json',cases)
    write(root/'caption_inputs.json',dict(coordinate_convention='ope_raw_xywh',cases=observations))
    sys.path.insert(0,str(GENERATOR));import initialization_captions as app
    app.prepare(root/'caption_inputs.json',root/'captions')
    write(root/'preparation.json',dict(status='full_OPE_observations_frozen_after_low22_pass',dataset=name,
        source_sha256=sha(__file__),low22_result_sha256=sha(ROOT/'low22_result.json'),bundle_sha256=spec['bundle_sha256'],
        text_protocol_sha256=spec['text_protocol_sha256'],cases_sha256=sha(root/'cases.json'),
        caption_plan_sha256=sha(root/'captions/plan.json'),sequences=len(cases),frames=sum(c['frames'] for c in cases),
        actual_caption_calls=0,actual_tracking_calls=0))
    print((root/'preparation.json').read_text())


def bank_ope(name):
    import torch
    spec,bundle,inputs,native=eligible();root=FULL/name
    prepared=json.loads((root/'preparation.json').read_text());assert prepared['source_sha256']==sha(__file__)
    encoded=json.loads((root/'captions/encoding_result.json').read_text())
    assert encoded['bank_sha256']==sha(root/'captions/text_bank.pt')
    c=module('candidate64bank',Path('/root/autodl-tmp/m64_category_candidate_20260907.py'))
    raw=torch.load(root/'captions/text_bank.pt',map_location='cpu')
    bank=c.retain_category(raw,spec['text_protocol_sha256']);torch.save(bank,root/'category.pt')
    plan=dict(bundle_path=str(ROOT/'bundle.json'),bundle_sha256=spec['bundle_sha256'],
        text_bank_path=str(root/'category.pt'),text_bank_sha256=sha(root/'category.pt'),
        cases_path=str(root/'cases.json'),cases_sha256=prepared['cases_sha256'],
        dataset_root=native['datasets'][name]['root'],output=str(root/'predictions'),
        metric_source=native['metric_source'],metric_source_sha256=native['metric_source_sha256'])
    write(root/'plan.json',plan)
    sys.path.insert(0,str(INTERFACE));from semantic_runtime import checked_plan,text_bank
    checked,bound=checked_plan(root/'plan.json');lookup=text_bank(checked,bound)
    for case in json.loads((root/'cases.json').read_text()):lookup.info(Path(plan['dataset_root'])/case['sequence']/'color/00000001.jpg',case['init_bbox'])
    write(root/'bank_binding.json',dict(status='all_first_frame_OPE_inputs_bound',dataset=name,plan_sha256=sha(root/'plan.json'),
        bundle_sha256=spec['bundle_sha256'],text_protocol_sha256=spec['text_protocol_sha256'],sequences=len(bank['keys']),
        encoding_result_sha256=sha(root/'captions/encoding_result.json'),actual_tracking_calls=0))
    print((root/'bank_binding.json').read_text())


def export_vot():
    spec,_,_,_=eligible()
    root=FULL/'vot';root.mkdir(parents=True)
    exporter=module('official_full_initializations',EXPORTER)
    result=exporter.collect(MANIFEST,FINGERPRINTS[MANIFEST],root/'initializations')
    assert result['anchors']==1765 and result['sequences']==127
    write(root/'preparation.json',dict(status='full_VOT_initializations_exported_after_low22_pass',source_sha256=sha(__file__),
        low22_result_sha256=sha(ROOT/'low22_result.json'),bundle_sha256=spec['bundle_sha256'],
        text_protocol_sha256=spec['text_protocol_sha256'],export_result_sha256=sha(root/'initializations/export_result.json'),
        new_caption_calls=0,new_tracking_calls=0,next='Prepare remaining initialization observations; preserve all303 existing low22 caption/tensor records exactly before binding the full bank and sharded run.'))
    print((root/'preparation.json').read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['check_sources','prepare_ope','bank_ope','export_vot']);p.add_argument('--dataset',choices=['depthtrack','cdtb']);a=p.parse_args()
    if a.action=='check_sources':check_sources()
    elif a.action=='export_vot':export_vot()
    else:
        assert a.dataset is not None
        {'prepare_ope':prepare_ope,'bank_ope':bank_ope}[a.action](a.dataset)
