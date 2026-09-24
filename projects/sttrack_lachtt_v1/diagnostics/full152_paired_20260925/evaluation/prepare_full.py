"""Bind paired Full152 final models to shared initialization-only full inputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT = Path('/root/autodl-tmp/sttrack_full152_evaluation_20260925')
BASE = Path('/root/autodl-tmp')
INTERFACE = ROOT / 'interface'
GENERATOR = ROOT / 'generator'
NATIVE = BASE / 'sttrack_default_rgbd_ope_v1_20260906'
MANIFEST = BASE / 'sttrack_default_full127_v1_20260905/run/shard_manifest.json'
MODELS = {
    'M67': ('sttrack_full152_paired_20260925/M67', 'control', 2027),
    'M82': ('sttrack_full152_paired_20260925/M82', 'category', 2027),
}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def read(p): return json.loads(Path(p).read_text())


def write(p, v):
    Path(p).write_text(json.dumps(v, indent=2, allow_nan=False) + '\n')


def module(name, path):
    s = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def bind_models():
    import torch
    assert sha(NATIVE/'inputs.json') == '61541e35f7b9e3c40427df79067fc0be20b8622cf275e93025e4a1547bf68601'
    assert sha(MANIFEST) == '8e76256f1c7c135a65a1b262506356769b59557db60eb83bc21ef1392890dd01'
    bindings = {}
    for name, (folder, arm, seed) in MODELS.items():
        r = BASE/folder; spec = read(r/'training_spec.json'); target=ROOT/name; target.mkdir()
        final=r/'training'/arm/'final.pth'; digest=sha(final)
        result=read(r/'training'/arm/'result.json')
        assert result['final_checkpoint_sha256']==digest
        saved=torch.load(final,map_location='cpu')
        assert saved['status']=='complete' and saved['completed_sequences']==152
        assert saved['frame_count']==219802 and saved['optimizer_steps']==result['optimizer_steps']
        assert saved['training_spec_sha256']==sha(r/'training_spec.json')
        assert spec['seed']==seed and saved['arm']==arm and saved['support_loss_weight']==0
        assert saved['use_text'] and saved['null_support'] and saved['architecture']=='semantic_spatial_support_v1'
        assert sha(spec['native_checkpoint'])==spec['native_checkpoint_sha256']==saved['base_checkpoint_sha256']
        assert sha(r/'integration.json')==spec['integration_sha256']
        sources=read(r/'integration.json')['source_sha256']
        for f,h in sources.items(): assert sha(r/'code'/f)==h,f
        b=dict(repository=str(r/'code'),configuration='experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml',
            architecture=saved['architecture'],null_support=True,arm=arm,support_loss_weight=0.,use_text=True,seed=seed,
            source_sha256=sources,interface_sha256={p.name:sha(p) for p in sorted(INTERFACE.glob('*.py'))},
            base_checkpoint=spec['native_checkpoint'],base_checkpoint_sha256=spec['native_checkpoint_sha256'],
            adapter_checkpoint=str(final),adapter_checkpoint_sha256=digest,training_spec_sha256=sha(r/'training_spec.json'),
            text_protocol_path=str(ROOT/'text_protocol.json'),text_protocol_sha256=sha(ROOT/'text_protocol.json'),
            reported_confidence='Unmodified raw Hann maximum',template_control='interval50, raw Hann score strictly greater than0.75')
        write(target/'bundle.json',b)
        bindings[name]=dict(bundle_sha256=sha(target/'bundle.json'),checkpoint_sha256=digest,training_seed=seed,
            training_result_sha256=sha(r/'training'/arm/'result.json'),optimizer_steps=result['optimizer_steps'])
    write(ROOT/'selection.json',dict(status='two_Full152_final_models_frozen_before_external_metrics',
        observed_utc=datetime.now(timezone.utc).isoformat(),models=bindings,order=['M67','M82'],
        external_metric_checkpoint_selection=False,
        plan_sha256=sha(ROOT/'EXPERIMENT_PLAN.md'),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in
            list(ROOT.glob('*.py'))+list(ROOT.glob('run_full*.sh'))+list((ROOT/'generator').glob('*')) if p.is_file()},
        old_130_sequence_results_unchanged=True,user_requested_Full152_training_and_full_evaluation=True))
    print(json.dumps(bindings))


def checked(name):
    sys.path.insert(0,str(INTERFACE))
    selection=read(ROOT/'selection.json'); bpath=ROOT/name/'bundle.json'
    assert sha(bpath)==selection['models'][name]['bundle_sha256']
    assert sha(ROOT/'EXPERIMENT_PLAN.md')==selection['plan_sha256']
    for n,h in selection['source_sha256'].items():assert sha(ROOT/n)==h,n
    return read(bpath)


def retain(bank):
    out=dict(bank);out['tokens']=bank['tokens'].clone()
    for i in range(len(out['keys'])):
        for k in range(1,5):
            if bool(out['mask'][i,k]):out['tokens'][i,k]=out['empty']
    out['protocol_sha256']=sha(ROOT/'text_protocol.json')
    return out


def prepare_ope(name,dataset):
    import torch
    checked(name); shared=ROOT/'inputs_bfloat16'/dataset; target=ROOT/name/dataset; target.mkdir()
    assert (shared/'generate.exit').read_text().strip()=='0'
    assert (shared/'encode.exit').read_text().strip()=='0'
    encoded=read(shared/'captions/encoding_result.json')
    assert sha(shared/'captions/text_bank.pt')==encoded['bank_sha256']
    bank=retain(torch.load(shared/'captions/text_bank.pt',map_location='cpu'))
    # Both model-specific banks serialize exactly the same tensors and protocol.
    torch.save(bank,target/'category.pt')
    rows=read(NATIVE/'inputs.json')[dataset]; native=read(NATIVE/'spec.json')
    hashes=read(NATIVE/('metrics_'+dataset+'.json'))['groundtruth_sha256']
    cases=[dict(sequence=x['sequence'],frames=x['frames'],init_bbox=x['init_bbox'],gt_sha256=hashes[x['sequence']]) for x in rows]
    write(target/'cases.json',cases)
    plan=dict(bundle_path=str(ROOT/name/'bundle.json'),bundle_sha256=sha(ROOT/name/'bundle.json'),
        text_bank_path=str(target/'category.pt'),text_bank_sha256=sha(target/'category.pt'),
        cases_path=str(target/'cases.json'),cases_sha256=sha(target/'cases.json'),
        dataset_root=native['datasets'][dataset]['root'],output=str(target/'predictions'),
        metric_source=native['metric_source'],metric_source_sha256=native['metric_source_sha256'])
    write(target/'plan.json',plan)
    from semantic_runtime import checked_plan,text_bank
    p,b=checked_plan(target/'plan.json');reader=text_bank(p,b)
    for c in cases:reader.info(Path(p['dataset_root'])/c['sequence']/'color/00000001.jpg',c['init_bbox'])
    write(target/'input_binding.json',dict(status='complete',plan_sha256=sha(target/'plan.json'),
        generation_result_sha256=sha(shared/'captions/generation_result.json'),encoding_result_sha256=sha(shared/'captions/encoding_result.json'),
        sequences=len(cases),frames=sum(c['frames'] for c in cases),shared_text_observations=True,optimizer_steps=0))


def prepare_text():
    root=ROOT/'inputs_bfloat16';root.mkdir()
    assert sha(NATIVE/'inputs.json')=='61541e35f7b9e3c40427df79067fc0be20b8622cf275e93025e4a1547bf68601'
    sys.path.insert(0,str(GENERATOR));import initialization_captions as app
    inputs=read(NATIVE/'inputs.json')
    for name,count,frames in [('depthtrack',50,76373),('cdtb',80,101956)]:
        rows=inputs[name];assert len(rows)==count and sum(r['frames'] for r in rows)==frames
        dest=root/name;dest.mkdir()
        cases=[dict(id=x['sequence'],image=str(Path(x['root'])/x['sequence']/'color/00000001.jpg'),bbox=x['init_bbox']) for x in rows]
        write(dest/'caption_inputs.json',dict(coordinate_convention='ope_raw_xywh',cases=cases))
        app.prepare(dest/'caption_inputs.json',dest/'captions')
    probe=root/'numeric_replay';probe.mkdir()
    original=read(ROOT/'cdtb/captions/plan.json');row=original['rows'][15]
    assert row['key']=='6a47b8f52932ed8190ee0eb59cffd61aa0c97cca51094bba1b82b8d6dcfee8d1'
    write(probe/'inputs.json',dict(coordinate_convention='ope_raw_xywh',cases=[dict(id='saved_numeric_failure',image=row['image'],bbox=row['init_bbox'])]))
    app.prepare(probe/'inputs.json',probe/'captions')


def prepare_vot_text():
    root=ROOT/'vot_inputs';root.mkdir()
    exporter_path=BASE/'sttrack_m58_vot_initialization_export_20260906/export_initializations.py'
    assert sha(exporter_path)=='3459b8fb2274dc79aa6132172fca2c20ba817867f5a028f80186711029694159'
    assert sha(MANIFEST)=='8e76256f1c7c135a65a1b262506356769b59557db60eb83bc21ef1392890dd01'
    ex=module('export_initializations',exporter_path)
    result=ex.collect(MANIFEST,sha(MANIFEST),root/'initializations')
    assert result['anchors']==1765 and result['sequences']==127
    sys.path.insert(0,str(GENERATOR));import initialization_captions as app
    plan=app.prepare(root/'initializations/caption_inputs.json',root/'all_initializations')
    write(root/'partition.json',dict(status='shared_bfloat16_inputs_frozen',
        all_plan_sha256=sha(root/'all_initializations/plan.json'),
        observations=len(plan['rows']),new_observations=len(plan['rows']),anchors=1765,sequences=127,
        reuse_caption_cases=0,reused_prediction_anchors=0,
        reason='Uniform bfloat16 generation after demonstrated float16 NaN; no mixed precision caption bank.'))



def bind_vot(name):
    import torch
    checked(name); shared=ROOT/'vot_inputs';target=ROOT/name/'vot';target.mkdir()
    partition=read(shared/'partition.json')
    assert sha(shared/'all_initializations/plan.json')==partition['all_plan_sha256']
    encoded=read(shared/'all_initializations/encoding_result.json')
    assert sha(shared/'all_initializations/text_bank.pt')==encoded['bank_sha256']
    bank=retain(torch.load(shared/'all_initializations/text_bank.pt',map_location='cpu'))
    plan=read(shared/'all_initializations/plan.json')
    assert bank['keys']==[r['key'] for r in plan['rows']]
    torch.save(bank,target/'category.pt')
    write(target/'plan.json',dict(bundle_path=str(ROOT/name/'bundle.json'),bundle_sha256=sha(ROOT/name/'bundle.json'),
        text_bank_path=str(target/'category.pt'),text_bank_sha256=sha(target/'category.pt')))
    from semantic_runtime import checked_plan,text_bank
    p,b=checked_plan(target/'plan.json');reader=text_bank(p,b)
    for row in plan['rows']:reader.info(row['image'],row['init_bbox'])
    frozen=read(MANIFEST);tracker='sttrack_full152_'+name.lower()+'_full127';run=target/'run';run.mkdir()
    wrapper=target/'selected_semantic_vot.py'
    wrapper.write_text('import sys\nsys.path.insert(0,'+repr(str(INTERFACE))+')\nfrom run_semantic_vot import run\nrun('+repr(str(target/'plan.json'))+')\n')
    shards=[]
    for s in frozen['shards']:
        src=Path(s['root']);dest=run/('shard-%02d'%s['index']);gpu=s['index']%2
        assert sha(src/'config.yaml')==s['config_sha256'] and sha(src/'sequences/list.txt')==s['list_sha256']
        shutil.copytree(src/'sequences',dest/'sequences',symlinks=True)
        shutil.copyfile(src/'config.yaml',dest/'config.yaml')
        ini=(f'[{tracker}]\nlabel = {name} Full152 full127\nprotocol = traxpython\ncommand = selected_semantic_vot\npaths = {target}\n'
             f'python = /root/autodl-tmp/envs/sttrack/bin/python\nenv_CUDA_VISIBLE_DEVICES = {gpu}\nenv_PYTHONPATH = {target}\n'
             'env_TOKENIZERS_PARALLELISM = false\nenv_PYTHONDONTWRITEBYTECODE = 1\ntimeout = 600\nrestart = false\n')
        (dest/'trackers.ini').write_text(ini)
        shards.append(dict(s,root=str(dest),gpu=gpu,trackers_sha256=sha(dest/'trackers.ini')))
    write(run/'shard_manifest.json',dict(frozen,schema='selected_full127_v1',tracker=tracker,gpu_count=2,shards=shards))
    files=[ROOT/name/'bundle.json',target/'plan.json',target/'category.pt',wrapper,run/'shard_manifest.json',MANIFEST,
        ROOT/'run_vot_failure_family_shards.py',ROOT/'text_protocol.json',shared/'partition.json']
    files += [Path(s['root'])/n for s in shards for n in ['config.yaml','trackers.ini','sequences/list.txt']]
    metadata=read(shared/'initializations/metadata_sha256.json')
    for f,h in metadata.items():assert sha(f)==h,f
    files += [Path(f) for f in metadata]
    write(target/'execution.json',dict(status='frozen_before_tracking',model=name,bundle_sha256=sha(ROOT/name/'bundle.json'),
        source_sha256={str(f):sha(f) for f in files},anchors=1765,sequences=127,reused_prediction_anchors=0,
        planned_frame_positions=1327004,poll_seconds=3600,workers=len(shards),
        training_steps_before_external_evaluation=read(ROOT/'selection.json')['models'][name]['optimizer_steps']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['bind_models','prepare_text','prepare_ope','prepare_vot_text','bind_vot'])
    p.add_argument('--model',choices=list(MODELS));p.add_argument('--dataset',choices=['depthtrack','cdtb']);a=p.parse_args()
    if a.action=='bind_models':bind_models()
    elif a.action=='prepare_text':prepare_text()
    elif a.action=='prepare_vot_text':prepare_vot_text()
    elif a.action=='prepare_ope':prepare_ope(a.model,a.dataset)
    else:bind_vot(a.model)
