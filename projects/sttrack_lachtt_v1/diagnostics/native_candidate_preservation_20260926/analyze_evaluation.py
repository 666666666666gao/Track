"""Analyze sealed Full152 VOT and collect both models' six full metric results."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import prepare_evaluation as common

ROOT=common.ROOT


def vot(name):
    b=common.checked(name);root=ROOT/name/'vot';execution=common.read(root/'execution.json')
    for p,h in execution['source_sha256'].items():assert common.sha(p)==h,p
    assert (ROOT/name/'vot_tracking.exit').read_text().strip()=='0'
    run=root/'run';merge=common.read(run/'merge_result.json');tracker=merge['tracker']
    assert merge['anchor_count']==1765 and merge['result_file_count']==5295
    assert merge['source_manifest_sha256']==common.sha(run/'shard_manifest.json')
    for rel,h in merge['result_sha256'].items():assert common.sha(run/'master'/rel)==h,rel
    analysis_name='m89_'+name.lower()+'_full127'
    with (root/'analysis.log').open('w') as log:
        subprocess.run(['/root/miniconda3/envs/mplt/bin/python','-m','vot','analysis','--workspace',str(run/'master'),
            '--format','json','--name',analysis_name,tracker],env=dict(os.environ,PYTHONPATH='/home/SUTrack_RGBD_L'),
            stdout=log,stderr=subprocess.STDOUT,check=True)
    path=run/'master/analysis'/(analysis_name+'.json');v=common.read(path)['results']['baseline']['results']
    metrics=dict(EAO=float(v[0][0][0])*100,ACC=float(v[2][0][0])*100,ROB=float(v[2][0][1])*100)
    sys.path.insert(0,'/home/SUTrack_RGBD_L')
    from tools.finalize_vot_transaction_low22 import collect_confirmed_failure_outcomes
    outcomes,failures,per,settings=collect_confirmed_failure_outcomes(run/'master',tracker,expected_anchors=1765)
    assert len(outcomes)==1765 and len(per)==127
    assert common.sha(b['base_checkpoint'])==b['base_checkpoint_sha256']
    common.write(root/'result.json',dict(status='complete_full127',model=name,metrics_percent=metrics,
        bundle_sha256=common.sha(ROOT/name/'bundle.json'),checkpoint_sha256=b['adapter_checkpoint_sha256'],
        merge_sha256=common.sha(run/'merge_result.json'),analysis_sha256=common.sha(path),
        confirmed_failures=failures,per_sequence_failures=per,failure_outcomes=outcomes,failure_settings=settings,
        external_training_steps=common.read(ROOT/'selection.json')['models'][name]['optimizer_steps'],
        previous_results_unchanged=True))
    print(json.dumps(metrics),flush=True)


def collect():
    rows=[];bindings={}
    for name in common.MODELS:
        b=common.checked(name);bindings[name]=b['adapter_checkpoint_sha256']
        for dataset,n,frames in [('depthtrack',50,76373),('cdtb',80,101956)]:
            root=ROOT/name/dataset/'predictions';r=common.read(root/'metrics.json');receipt=common.read(root/'receipt.json')
            assert r['status']=='complete' and r['bundle_sha256']==common.sha(ROOT/name/'bundle.json')
            assert common.sha(root/'receipt.json')==r['receipt_sha256']
            assert len(receipt['sequences'])==n and receipt['frames']==frames
            rows.append(dict(model=name,dataset=dataset,checkpoint_sha256=bindings[name],result_path=str(root/'metrics.json'),
                result_sha256=common.sha(root/'metrics.json'),metrics=r['metrics']))
        r=common.read(ROOT/name/'vot/result.json');assert r['status']=='complete_full127'
        rows.append(dict(model=name,dataset='vot',checkpoint_sha256=bindings[name],
            result_path=str(ROOT/name/'vot/result.json'),result_sha256=common.sha(ROOT/name/'vot/result.json'),metrics=r['metrics_percent']))
    selection=common.read(ROOT/'selection.json')
    common.write(ROOT/'all_results.json',dict(status='six_Full152_evaluations_complete',models=bindings,results=rows,
        checkpoint_selection_from_external_metrics=False,
        new_training_steps={name:selection['models'][name]['optimizer_steps'] for name in common.MODELS},
        independent_completed_audit=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['vot','collect']);p.add_argument('--model',choices=list(common.MODELS));a=p.parse_args()
    vot(a.model) if a.action=='vot' else collect()
