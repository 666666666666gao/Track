"""Verify complete saved predictions and numeric targets for one M78 bundle."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path('/root/autodl-tmp/sttrack_m78_raw_competition_20260908/candidate_evaluation')
FULL=ROOT/'full_evaluation'
INTERFACE=Path('/root/autodl-tmp/sttrack_m78_raw_competition_20260908/candidate_evaluation/interface')
TARGETS={'depthtrack':{'precision_percent':65.2,'recall_percent':64.9,'f_score_percent':65.1},
         'cdtb':{'precision_percent':72.9,'recall_percent':75.6,'f_score_percent':74.2},
         'vot':{'EAO':77.9,'ACC':82.1,'ROB':93.7}}


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def module(name,p):
    s=importlib.util.spec_from_file_location(name,str(p));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def main():
    helper=module('final_helper',Path('/root/autodl-tmp/m78_full_preparation_20260908.py'))
    spec,bundle,_,_=helper.eligible()
    sys.path.insert(0,str(INTERFACE));from semantic_runtime import checked_plan
    vot=json.loads((FULL/'vot/result.json').read_text())
    assert vot['status']=='complete_same_bundle_full127'
    saved=torch.load(bundle['adapter_checkpoint'],map_location='cpu')
    assert saved['status']=='complete' and saved['completed_sequences']==130 and saved['use_text']
    assert saved['training_spec_sha256']==bundle['training_spec_sha256']
    values={};evidence={};checks={}
    for name,count,frames in [('depthtrack',50,76373),('cdtb',80,101956)]:
        root=FULL/name;plan,bound=checked_plan(root/'plan.json')
        assert plan['bundle_sha256']==spec['bundle_sha256'] and bound['text_protocol_sha256']==spec['text_protocol_sha256']
        cases=json.loads(Path(plan['cases_path']).read_text());assert sha(plan['cases_path'])==plan['cases_sha256']
        assert len(cases)==count and sum(c['frames'] for c in cases)==frames
        out=Path(plan['output']);receipt=json.loads((out/'receipt.json').read_text());metrics=json.loads((out/'metrics.json').read_text())
        assert receipt['status']==metrics['status']=='complete'
        assert receipt['plan_sha256']==metrics['plan_sha256']==sha(root/'plan.json')
        assert receipt['bundle_sha256']==metrics['bundle_sha256']==spec['bundle_sha256']
        assert receipt['text_bank_sha256']==plan['text_bank_sha256']
        assert metrics['receipt_sha256']==sha(out/'receipt.json')
        assert [r['sequence'] for r in receipt['sequences']]==[c['sequence'] for c in cases]
        for case,row in zip(cases,receipt['sequences']):
            name_seq=case['sequence'];assert row['frames']==case['frames']
            box=out/(name_seq+'.txt');score=out/(name_seq+'_all_scores.txt')
            assert sha(box)==row['bbox_sha256'] and sha(score)==row['confidence_sha256']
            boxes=np.loadtxt(box,delimiter=',').reshape(-1,4);scores=np.loadtxt(score).reshape(-1)
            assert boxes.shape==(case['frames'],4) and scores.shape==(case['frames'],)
            assert np.isfinite(boxes).all() and (boxes[:,2:]>0).all() and np.isfinite(scores).all()
            assert scores[0]==1. and np.abs(boxes[0]-case['init_bbox']).max()<=5.01e-7
            assert sha(Path(plan['dataset_root'])/name_seq/'groundtruth.txt')==case['gt_sha256']
        assert sha(plan['metric_source'])==plan['metric_source_sha256']==metrics['metric_source_sha256']
        metric=module('official_metric_'+name,plan['metric_source'])
        repeated=metric.evaluate_depthtrack_results(plan['dataset_root'],out,resolution=100,sequence_names=[c['sequence'] for c in cases])
        assert repeated['sequences']==count and repeated['frames']==receipt['frames']==frames
        for key in ['precision','recall','f_score','precision_percent','recall_percent','f_score_percent']:
            assert abs(repeated[key]-metrics['metrics'][key])<1e-10,(name,key)
        assert repeated['threshold']==metrics['metrics']['threshold']
        values[name]=repeated;checks[name]={k:repeated[k]>=v for k,v in TARGETS[name].items()}
        evidence[name]=dict(plan_sha256=sha(root/'plan.json'),bank_sha256=plan['text_bank_sha256'],receipt_sha256=sha(out/'receipt.json'),metrics_sha256=sha(out/'metrics.json'),official_PRF_recomputed_from_saved_outputs=True)
    ex=json.loads((FULL/'vot/execution.json').read_text())
    assert sha(FULL/'vot/execution.json')==vot['execution_sha256']
    for p,h in ex['source_sha256'].items():assert sha(p)==h,p
    for record in [ex,vot]:
        assert record['bundle_sha256']==spec['bundle_sha256'] and record['text_protocol_sha256']==spec['text_protocol_sha256']
    master=FULL/'vot/run/master';merge_path=FULL/'vot/run/merge_result.json';merge=json.loads(merge_path.read_text())
    assert sha(merge_path)==vot['merge_sha256'] and merge['anchor_count']==1765 and merge['result_file_count']==5295
    for rel,h in merge['result_sha256'].items():assert sha(master/rel)==h
    analysis=master/'analysis/m78_category_full127_analysis.json';assert sha(analysis)==vot['analysis_sha256']
    data=json.loads(analysis.read_text())['results']['baseline']['results']
    v=dict(EAO=float(data[0][0][0])*100,ACC=float(data[2][0][0])*100,ROB=float(data[2][0][1])*100)
    assert v==vot['metrics_percent']
    sys.path.insert(0,'/home/SUTrack_RGBD_L');from tools.finalize_vot_transaction_low22 import collect_confirmed_failure_outcomes
    outcomes,failures,per,settings=collect_confirmed_failure_outcomes(master,merge['tracker'],expected_anchors=1765)
    assert outcomes==vot['failure_outcomes'] and failures==vot['confirmed_failures'] and per==vot['per_sequence_failures']
    assert len(per)==127 and len(outcomes)==1765
    values['vot']=v;checks['vot']={k:v[k]>minimum for k,minimum in TARGETS['vot'].items()}
    evidence['vot']=dict(result_sha256=sha(FULL/'vot/result.json'),analysis_sha256=sha(analysis),merge_sha256=sha(merge_path),all5295_output_hashes_verified=True,all1765_failure_outcomes_recomputed=True)
    report=dict(status='complete_saved_output_and_same_bundle_verification',observed_utc=datetime.now(timezone.utc).isoformat(),
        verifier_sha256=sha(__file__),bundle_sha256=spec['bundle_sha256'],text_protocol_sha256=spec['text_protocol_sha256'],
        base_checkpoint_sha256=bundle['base_checkpoint_sha256'],adapter_checkpoint_sha256=bundle['adapter_checkpoint_sha256'],
        completed_DepthTrack_Train_sequences=130,training_spec_sha256=bundle['training_spec_sha256'],
        metrics=values,targets=TARGETS,target_checks=checks,all_numeric_targets_pass=all(all(c.values()) for c in checks.values()),
        evidence=evidence,independent_model_review_pass=False,goal_completion_claimed=False,
        remaining='Executor must inspect results and remaining protocol requirements, synchronize architecture/plan/metrics handoff and publication, and perform the thread completion audit before marking the goal complete.')
    (FULL/'same_bundle_verification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
