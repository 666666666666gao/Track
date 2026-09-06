"""Evaluate one frozen category-retention bundle on all303 VOT low22 anchors."""
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

ROOT=Path('/root/autodl-tmp/sttrack_m64_category_candidate_20260907')
INTERFACE=Path('/root/autodl-tmp/sttrack_m58_evaluation_preparation_20260906')
M39=Path('/root/autodl-tmp/sttrack_lachtt_m39_vot_low22_template_ablation_v1_20260902')
FROZEN=Path('/root/autodl-tmp/sutrack_vot_low22_anchor_identity_v1/run/shard_manifest.json')
TRACKER='sttrack_m64_category_low22'
PYTHON='/root/miniconda3/envs/mplt/bin/python'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def candidate():
    path=Path('/root/autodl-tmp/m64_category_candidate_20260907.py')
    spec=importlib.util.spec_from_file_location('candidate_protocol',str(path))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m.checked()


def prepare():
    spec=candidate()
    for name in ['train_generate','train_verify','low22_prepare','low22_generate','low22_encode','low22_bank_fixed']:
        assert (ROOT/(name+'.exit')).read_text().strip()=='0'
    bank=json.loads((ROOT/'low22_bank_result.json').read_text())
    assert bank['cases']==303 and bank['bundle_sha256']==spec['bundle_sha256']
    sys.path.insert(0,str(INTERFACE))
    from semantic_runtime import checked_plan
    plan,bundle=checked_plan(ROOT/'low22_plan.json')
    assert sha(FROZEN)=='600b1ebb8b0c2f69b831f954e907e63709fd69afb7ea94c5b58e8c7408a29eed'
    assert sha(M39/'m39_result.json')=='cf953c0d3c69609bcd83c11cb24ba57f37e30b38d3b3bcad32860b3a9ba9c1b5'
    frozen=json.loads(FROZEN.read_text()); run=ROOT/'low22_run';run.mkdir()
    wrapper=ROOT/'m64_semantic_vot.py'
    wrapper.write_text("import sys\nsys.path.insert(0, '"+str(INTERFACE)+"')\nfrom run_semantic_vot import run\nrun('"+str(ROOT/'low22_plan.json')+"')\n")
    ini=(f'[{TRACKER}]\nlabel = M64 category-retention RGB-D-L\nprotocol = traxpython\n'
         f'command = m64_semantic_vot\npaths = {ROOT}\npython = /root/autodl-tmp/envs/sttrack/bin/python\n'
         f'env_CUDA_VISIBLE_DEVICES = 1\nenv_PYTHONPATH = {ROOT}\nenv_TOKENIZERS_PARALLELISM = false\n'
         'env_PYTHONDONTWRITEBYTECODE = 1\ntimeout = 600\nrestart = false\n')
    shards=[]
    for old in frozen['shards']:
        source=Path(old['root']); dest=run/('shard-%02d'%old['index'])
        assert sha(source/'config.yaml')==old['config_sha256']
        assert sha(source/'sequences/list.txt')==old['list_sha256']
        shutil.copytree(source/'sequences',dest/'sequences',symlinks=True)
        shutil.copyfile(source/'config.yaml',dest/'config.yaml')
        (dest/'trackers.ini').write_text(ini)
        for name in frozen['sequences']:
            a=source/'sequences'/name/'anchor.value';b=dest/'sequences'/name/'anchor.value'
            if a.exists():assert sha(a)==sha(b)
        shards.append(dict(old,root=str(dest),gpu=1,trackers_sha256=sha(dest/'trackers.ini')))
    manifest=dict(frozen,schema='m64_same_bundle_category_low22_v1',tracker=TRACKER,gpu_count=1,shards=shards)
    write(run/'shard_manifest.json',manifest)
    runner=ROOT/'run_vot_failure_family_shards.py'
    shutil.copyfile('/home/SUTrack_RGBD_L/tools/run_vot_failure_family_shards.py',runner)
    files=[Path(__file__),wrapper,runner,ROOT/'low22_plan.json',ROOT/'low22_category.pt',ROOT/'bundle.json',ROOT/'text_protocol.json',ROOT/'spec.json',M39/'m39_result.json',run/'shard_manifest.json']
    files += [Path(s['root'])/n for s in shards for n in ['trackers.ini','config.yaml','sequences/list.txt']]
    files += [INTERFACE/n for n in bundle['interface_sha256']]
    for n in ['tools/finalize_vot_transaction_low22.py']:
        files.append(Path('/home/SUTrack_RGBD_L')/n)
    execution=dict(status='frozen_before_low22_tracking',observed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(p):sha(p) for p in files},candidate_spec_sha256=sha(ROOT/'spec.json'),
        bundle_sha256=spec['bundle_sha256'],text_protocol_sha256=spec['text_protocol_sha256'],
        text_bank_sha256=plan['text_bank_sha256'],manifest_sha256=sha(run/'shard_manifest.json'),
        sequences=22,anchors=303,planned_frame_positions=220483,workers=4,gpu=1,poll_seconds=240,
        result_files_preseeded=0,new_weights=0,new_optimizer_steps=0,gate=spec['full_evaluation_gate'],
        reused_development_benchmark=True,independent_review_pass=False)
    write(ROOT/'low22_execution.json',execution)
    print(json.dumps({k:v for k,v in execution.items() if k!='source_sha256'},indent=2))


def checked_execution():
    candidate()
    ex=json.loads((ROOT/'low22_execution.json').read_text())
    for p,h in ex['source_sha256'].items():assert sha(p)==h,p
    return ex


def analyze():
    ex=checked_execution()
    assert (ROOT/'low22_tracking.exit').read_text().strip()=='0'
    run=ROOT/'low22_run'; merge=json.loads((run/'merge_result.json').read_text())
    assert merge['anchor_count']==303 and merge['result_file_count']==909
    assert merge['source_manifest_sha256']==ex['manifest_sha256']
    for rel,h in merge['result_sha256'].items():assert sha(run/'master'/rel)==h
    name='m64_category_low22_analysis'
    env=dict(os.environ,PYTHONPATH='/home/SUTrack_RGBD_L')
    with (ROOT/'toolkit_analysis.log').open('w') as log:
        subprocess.run([PYTHON,'-m','vot','analysis','--workspace',str(run/'master'),'--format','json','--name',name,TRACKER],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    path=run/'master/analysis'/(name+'.json');data=json.loads(path.read_text())['results']['baseline']['results']
    metrics=dict(EAO=float(data[0][0][0])*100,ACC=float(data[2][0][0])*100,ROB=float(data[2][0][1])*100)
    sys.path.insert(0,'/home/SUTrack_RGBD_L')
    from tools.finalize_vot_transaction_low22 import collect_confirmed_failure_outcomes
    outcomes,failures,per,settings=collect_confirmed_failure_outcomes(run/'master',TRACKER,expected_anchors=303)
    old=json.loads((M39/'m39_result.json').read_text())['arms']['default']
    assert set(outcomes)==set(old['failure_outcomes'])
    protected=[n for n,r in old['per_sequence_failures'].items() if r['confirmed_failures']==0];assert len(protected)==7
    gate=ex['gate']
    checks={k+'_minimum':metrics[k]>=gate[k+'_min'] for k in metrics}
    checks['failure_reduction']=failures<=gate['confirmed_failures_max']
    checks['native_zero_failure_protection']=all(per[n]['confirmed_failures']==0 for n in protected)
    checks['all303_bound_and_complete']=len(outcomes)==303
    result=dict(status='completed_low22_same_bundle_evaluation',observed_utc=datetime.now(timezone.utc).isoformat(),
        execution_sha256=sha(ROOT/'low22_execution.json'),bundle_sha256=ex['bundle_sha256'],text_protocol_sha256=ex['text_protocol_sha256'],
        metrics_percent=metrics,confirmed_failures=failures,per_sequence_failures=per,failure_outcomes=outcomes,failure_settings=settings,
        rescued_anchors=[k for k,v in outcomes.items() if not v['failed'] and old['failure_outcomes'][k]['failed']],
        newly_failed_anchors=[k for k,v in outcomes.items() if v['failed'] and not old['failure_outcomes'][k]['failed']],
        protected_native_sequences=protected,gate_checks=checks,full_three_dataset_evaluation_allowed=all(checks.values()),
        analysis_sha256=sha(path),merge_sha256=sha(run/'merge_result.json'),independent_review_pass=False,
        interpretation='One pre-frozen candidate on reused low22 development benchmark. This is not a full127 result; failure does not authorize threshold or caption changes.')
    write(ROOT/'low22_result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['failure_outcomes','per_sequence_failures','rescued_anchors','newly_failed_anchors']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','check','analyze']);a=p.parse_args()
    {'prepare':prepare,'check':checked_execution,'analyze':analyze}[a.action]()
