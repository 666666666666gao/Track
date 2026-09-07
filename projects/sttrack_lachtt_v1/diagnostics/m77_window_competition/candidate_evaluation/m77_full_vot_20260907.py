"""Conditional full VOT preparation preserving the exact M77 low22 inputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path('/root/autodl-tmp/sttrack_m77_window_competition_20260907/candidate_evaluation')
FULL=ROOT/'full_evaluation/vot'
INTERFACE=Path('/root/autodl-tmp/sttrack_m77_window_competition_20260907/candidate_evaluation/interface')
LOW_CAPTIONS=Path('/root/autodl-tmp/sttrack_m64_category_candidate_20260907/low22_captions')
GENERATOR=Path('/root/autodl-tmp/sttrack_m58_initialization_generator_20260906')
TRACKER='sttrack_m77_category_full127'
PYTHON='/root/miniconda3/envs/mplt/bin/python'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def module(name,p):
    s=importlib.util.spec_from_file_location(name,str(p));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def eligible():
    p=Path('/root/autodl-tmp/m77_full_preparation_20260907.py')
    proof=json.loads((ROOT/'full_preparation_readiness.json').read_text())
    assert sha(p)==proof['preparation_source_sha256']
    helper=module('full_prep64',p)
    spec,bundle,_,_=helper.eligible()
    return helper,spec,bundle


def prepare_captions():
    helper,spec,bundle=eligible()
    exported=json.loads((FULL/'initializations/export_result.json').read_text())
    prepared=json.loads((FULL/'preparation.json').read_text())
    assert sha(FULL/'initializations/export_result.json')==prepared['export_result_sha256']
    for n,h in exported['files_sha256'].items():assert sha(FULL/'initializations'/n)==h
    assert exported['anchors']==1765 and exported['sequences']==127
    sys.path.insert(0,str(GENERATOR));import initialization_captions as app
    plan=app.prepare(FULL/'initializations/caption_inputs.json',FULL/'all_initializations')
    old_plan=json.loads((LOW_CAPTIONS/'plan.json').read_text())
    old_ids={c['id']:c['key'] for c in old_plan['cases']};all_ids={c['id']:c['key'] for c in plan['cases']}
    assert len(old_ids)==303 and len(all_ids)==1765 and all(all_ids[k]==v for k,v in old_ids.items())
    old_anchors=json.loads(Path('/root/autodl-tmp/sttrack_m58_vot_initialization_export_20260906/low22_inputs/anchors.json').read_text())
    all_anchors={r['id']:r for r in json.loads((FULL/'initializations/anchors.json').read_text())}
    for old in old_anchors:
        new=all_anchors[old['id']]
        for k in ['sequence','index','direction','image','depth','bbox','frames','frame_paths_sha256']:assert old[k]==new[k],(old['id'],k)
    rows={r['key']:r for r in plan['rows']};old_keys=set(old_ids.values())
    cases=[dict(id=c['id'],image=rows[c['key']]['image'],bbox=rows[c['key']]['init_bbox']) for c in plan['cases'] if c['key'] not in old_keys]
    # Coordinates are already the received wire values; do not convert twice.
    write(FULL/'remaining_caption_inputs.json',dict(coordinate_convention='ope_raw_xywh',cases=cases))
    remaining=app.prepare(FULL/'remaining_caption_inputs.json',FULL/'remaining_captions')
    assert set(r['key'] for r in remaining['rows'])==set(rows)-old_keys
    summary=dict(status='full_initializations_partitioned_without_regenerating_low22',observed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__),bundle_sha256=spec['bundle_sha256'],text_protocol_sha256=spec['text_protocol_sha256'],
        old_caption_plan_sha256=sha(LOW_CAPTIONS/'plan.json'),old_caption_records_sha256=sha(LOW_CAPTIONS/'records.jsonl'),
        old_category_bank_sha256=sha(ROOT/'low22_category.pt'),all_plan_sha256=sha(FULL/'all_initializations/plan.json'),
        remaining_plan_sha256=sha(FULL/'remaining_captions/plan.json'),all_cases=1765,reused_low22_cases=303,
        all_unique_observations=len(plan['rows']),new_caption_observations=len(remaining['rows']),
        identical_low22_initializations_and_frame_orders=True,new_caption_calls=0,new_tracking_calls=0)
    write(FULL/'caption_partition.json',summary);print(json.dumps(summary,indent=2))


def bind_and_prepare_run():
    import torch
    helper,spec,bundle=eligible()
    partition=json.loads((FULL/'caption_partition.json').read_text());assert partition['source_sha256']==sha(__file__)
    for p,k in [(ROOT/'low22_category.pt','old_category_bank_sha256'),(LOW_CAPTIONS/'records.jsonl','old_caption_records_sha256'),(FULL/'all_initializations/plan.json','all_plan_sha256'),(FULL/'remaining_captions/plan.json','remaining_plan_sha256')]:assert sha(p)==partition[k]
    encoded=json.loads((FULL/'remaining_captions/encoding_result.json').read_text())
    assert encoded['bank_sha256']==sha(FULL/'remaining_captions/text_bank.pt')
    old=torch.load(ROOT/'low22_category.pt',map_location='cpu');raw=torch.load(FULL/'remaining_captions/text_bank.pt',map_location='cpu')
    assert torch.equal(old['empty'],raw['empty'])
    new=helper.retain_category(raw,spec['text_protocol_sha256'])
    assert not set(old['keys'])&set(new['keys'])
    plan=json.loads((FULL/'all_initializations/plan.json').read_text());keys=[r['key'] for r in plan['rows']]
    lookup={key:(b,i) for b in [old,new] for i,key in enumerate(b['keys'])};assert set(lookup)==set(keys)
    bank=dict(format='initialization_observation_v1',protocol_sha256=spec['text_protocol_sha256'],keys=keys,
        tokens=torch.stack([lookup[k][0]['tokens'][lookup[k][1]] for k in keys]),
        mask=torch.stack([lookup[k][0]['mask'][lookup[k][1]] for k in keys]),empty=old['empty'].clone())
    for i,k in enumerate(old['keys']):
        j=keys.index(k)
        assert torch.equal(bank['tokens'][j],old['tokens'][i]) and torch.equal(bank['mask'][j],old['mask'][i])
    torch.save(bank,FULL/'category.pt')
    tracker_plan=dict(bundle_path=str(ROOT/'bundle.json'),bundle_sha256=spec['bundle_sha256'],text_bank_path=str(FULL/'category.pt'),text_bank_sha256=sha(FULL/'category.pt'))
    write(FULL/'plan.json',tracker_plan)
    sys.path.insert(0,str(INTERFACE));from semantic_runtime import checked_plan,text_bank
    checked,bound=checked_plan(FULL/'plan.json');reader=text_bank(checked,bound)
    for row in plan['rows']:reader.info(row['image'],row['init_bbox'])
    metadata=json.loads((FULL/'initializations/metadata_sha256.json').read_text())
    for p,h in metadata.items():assert sha(p)==h,p
    frozen=json.loads(helper.MANIFEST.read_text());run=FULL/'run';run.mkdir()
    wrapper=FULL/'m77_full_semantic_vot.py'
    wrapper.write_text("import sys\nsys.path.insert(0,'"+str(INTERFACE)+"')\nfrom run_semantic_vot import run\nrun('"+str(FULL/'plan.json')+"')\n")
    ini=(f'[{TRACKER}]\nlabel = M77 category-retention full127\nprotocol = traxpython\ncommand = m77_full_semantic_vot\npaths = {FULL}\n'
         'python = /root/autodl-tmp/envs/sttrack/bin/python\nenv_CUDA_VISIBLE_DEVICES = 1\n'
         f'env_PYTHONPATH = {FULL}\nenv_TOKENIZERS_PARALLELISM = false\nenv_PYTHONDONTWRITEBYTECODE = 1\ntimeout = 600\nrestart = false\n')
    low_manifest=json.loads((ROOT/'low22_run/shard_manifest.json').read_text())
    old_ids={n for s in low_manifest['shards'] for n in s['expected_trajectories']};assert len(old_ids)==303
    low_merge=json.loads((ROOT/'low22_run/merge_result.json').read_text())
    low_base=ROOT/'low22_run/master';seeded={};shards=[]
    for s in frozen['shards']:
        src=Path(s['root']);dest=run/('shard-%02d'%s['index'])
        assert sha(src/'config.yaml')==s['config_sha256'] and sha(src/'sequences/list.txt')==s['list_sha256']
        shutil.copytree(src/'sequences',dest/'sequences',symlinks=True)
        shutil.copyfile(src/'config.yaml',dest/'config.yaml');(dest/'trackers.ini').write_text(ini)
        for n in s['expected_trajectories']:
            if n not in old_ids:continue
            seq=n.rsplit('_',1)[0];target=dest/'results'/TRACKER/'baseline'/seq;target.mkdir(parents=True,exist_ok=True)
            for suffix in ['.bin','_confidence.value','_time.value']:
                rel='results/'+low_merge['tracker']+'/baseline/'+seq+'/'+n+suffix
                source=low_base/rel;assert sha(source)==low_merge['result_sha256'][rel]
                out=target/source.name;shutil.copyfile(source,out)
                assert sha(out)==sha(source);seeded[str(out.relative_to(FULL))]=dict(source=str(source),sha256=sha(out))
        shards.append(dict(s,root=str(dest),gpu=1,trackers_sha256=sha(dest/'trackers.ini')))
    assert len(seeded)==909
    manifest=dict(frozen,schema='m77_same_bundle_full127_v1',tracker=TRACKER,gpu_count=1,shards=shards,
        native_source='Only native dataset/shard metadata reused; all predictions belong to the M77 bundle.')
    write(run/'shard_manifest.json',manifest)
    runner=FULL/'run_vot_failure_family_shards.py';shutil.copyfile(ROOT/'run_vot_failure_family_shards.py',runner)
    write(FULL/'preseed_receipt.json',dict(status='same_bundle_low22_results_reused',source_merge_sha256=sha(ROOT/'low22_run/merge_result.json'),
        anchors=303,files=seeded,identical_low22_tokens_masks_and_initializations=True,native_predictions_reused=0))
    files=[Path(__file__),wrapper,runner,FULL/'category.pt',FULL/'plan.json',run/'shard_manifest.json',FULL/'preseed_receipt.json',ROOT/'bundle.json',INTERFACE/'text_protocol.json']
    files += [Path(s['root'])/n for s in shards for n in ['trackers.ini','config.yaml','sequences/list.txt']]
    files += [INTERFACE/n for n in bundle['interface_sha256']]
    files += [Path(p) for p in metadata]
    files += [FULL/'initializations/metadata_sha256.json',FULL/'caption_partition.json']
    execution=dict(status='full127_frozen_before_new_tracking',observed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(p):sha(p) for p in files},low22_result_sha256=sha(ROOT/'low22_result.json'),
        bundle_sha256=spec['bundle_sha256'],text_protocol_sha256=spec['text_protocol_sha256'],
        sequences=127,anchors=1765,reused_anchors=303,new_anchors=1462,planned_frame_positions=1327004,
        new_planned_frame_positions=1327004-220483,workers=4,gpu=1,poll_seconds=240,
        target_strictly_greater_than=dict(EAO=77.9,ACC=82.1,ROB=93.7),new_training_steps=0,independent_review_pass=False,
        timing_note='No uniform-hardware FPS claim from reused low22 and newly executed trajectories.')
    write(FULL/'execution.json',execution);print(json.dumps({k:v for k,v in execution.items() if k!='source_sha256'},indent=2))


def analyze():
    helper,spec,bundle=eligible();execution=json.loads((FULL/'execution.json').read_text())
    for p,h in execution['source_sha256'].items():assert sha(p)==h,p
    assert (FULL/'tracking.exit').read_text().strip()=='0'
    run=FULL/'run';merge=json.loads((run/'merge_result.json').read_text())
    assert merge['anchor_count']==1765 and merge['result_file_count']==5295
    assert merge['source_manifest_sha256']==sha(run/'shard_manifest.json')
    for rel,h in merge['result_sha256'].items():assert sha(run/'master'/rel)==h
    name='m77_category_full127_analysis'
    with (FULL/'toolkit_analysis.log').open('w') as log:
        subprocess.run([PYTHON,'-m','vot','analysis','--workspace',str(run/'master'),'--format','json','--name',name,TRACKER],
            env=dict(os.environ,PYTHONPATH='/home/SUTrack_RGBD_L'),stdout=log,stderr=subprocess.STDOUT,check=True)
    path=run/'master/analysis'/(name+'.json');v=json.loads(path.read_text())['results']['baseline']['results']
    metrics=dict(EAO=float(v[0][0][0])*100,ACC=float(v[2][0][0])*100,ROB=float(v[2][0][1])*100)
    sys.path.insert(0,'/home/SUTrack_RGBD_L');from tools.finalize_vot_transaction_low22 import collect_confirmed_failure_outcomes
    outcomes,failures,per,settings=collect_confirmed_failure_outcomes(run/'master',TRACKER,expected_anchors=1765)
    assert len(outcomes)==1765 and len(per)==127
    result=dict(status='complete_same_bundle_full127',observed_utc=datetime.now(timezone.utc).isoformat(),
        bundle_sha256=spec['bundle_sha256'],text_protocol_sha256=spec['text_protocol_sha256'],execution_sha256=sha(FULL/'execution.json'),
        merge_sha256=sha(run/'merge_result.json'),analysis_sha256=sha(path),metrics_percent=metrics,
        confirmed_failures=failures,per_sequence_failures=per,failure_outcomes=outcomes,failure_settings=settings,
        VOT_target_checks={k:metrics[k]>v for k,v in execution['target_strictly_greater_than'].items()},
        full_three_dataset_goal_complete=False,independent_review_pass=False)
    write(FULL/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['per_sequence_failures','failure_outcomes']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare_captions','bind_and_prepare_run','analyze']);a=p.parse_args()
    {'prepare_captions':prepare_captions,'bind_and_prepare_run':bind_and_prepare_run,'analyze':analyze}[a.action]()
