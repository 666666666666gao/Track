"""Read-only completion audit; scalar metric code is reused from verified M63."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib,importlib.util,json,math
from pathlib import Path
import numpy as np

BASE=Path('/root/autodl-tmp')
ROOT=BASE/'sttrack_m67_supervised_semantic_support_20260907'
PARENT=BASE/'sttrack_m65_category_null_support_20260907'
TRAIN_SHA='2027b1893343e5e1520c447bd82a5c45468542cecaab0f59b3ec72a345e48b2e'
RECURSIVE_SHA='d4dd7ca74b8f4b318f8e32f961bc21a10b95aa3802137dd293b49948d83411c5'
M63=BASE/'m63_evidence_audit_20260907.py'
M63_SHA='2c611c8dae3c226c5c625301a98ff1f74f4ff03e727c5599060dea1650cac66d'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def near(a,b,where):assert abs(a-b)<1e-8,(where,a,b)
def scalar_metric():
    assert sha(M63)==M63_SHA
    s=importlib.util.spec_from_file_location('verified_m63_scalar_metric',str(M63))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return m.overlaps_and_statistics

def sealed_family(root,arm,spec,training):
    assert (root/(arm+'_recursive.exit')).read_text().strip()=='0'
    receipt_path=root/(arm+'_recursive_receipt.json');receipt=read(receipt_path)
    assert receipt['status']=='complete' and receipt['arm']==arm
    assert receipt['recursive_spec_sha256']==sha(root/'recursive_spec.json')
    assert not receipt['subsequent_gt_opened'] and not receipt['text_updated_online']
    assert receipt['training_result_sha256']==sha(root/'training'/arm/'result.json')
    assert receipt['head_sha256']==sha(root/'training'/arm/'final.pth')
    assert [x['sequence'] for x in receipt['sequences']]==[x['sequence'] for x in spec['cases']]
    data={};writes={}
    for item,case in zip(receipt['sequences'],spec['cases']):
        seq=case['sequence'];p=root/'recursive'/arm/(seq+'.json');assert sha(p)==item['sha256']
        x=read(p);rows=x['rows'];assert x['sequence']==seq and x['arm']==arm
        assert len(rows)==case['frames']==item['frames']
        assert [r['frame'] for r in rows]==list(range(case['frames']))
        assert rows[0]==dict(frame=0,bbox=case['init_bbox'],score=None)
        boxes=np.asarray([r['bbox'] for r in rows]);scores=np.asarray([r['score'] for r in rows[1:]])
        assert np.isfinite(boxes).all() and (boxes[:,2:]>0).all()
        assert np.isfinite(scores).all() and (scores>=0).all() and (scores<=1).all()
        data[seq]=rows;writes[seq]=sum(r['frame']%50==0 and r['score']>.75 for r in rows[1:])
    assert receipt['total_frames']==sum(len(rows) for rows in data.values())==33130
    return data,dict(receipt_sha256=sha(receipt_path),head_sha256=receipt['head_sha256'],
        sequences=len(data),image_frames=33130,track_calls=33108,reconstructed_template_writes=writes)

def sealed_native(spec,training):
    path=Path(spec['native_result_path']);assert sha(path)==training['native_result_sha256']
    previous=read(path.parent/'recursive_spec.json');names={c['sequence'] for c in spec['cases']}
    rows={n:[] for n in names}
    for filename,digest in previous['baseline_trace_sha256'].items():
        p=Path(filename);assert sha(p)==digest
        saved=read(p);assert saved['complete'] and not saved['ground_truth_used_after_initialization']
        for x in saved['rows']:
            if x['sequence'] in rows:rows[x['sequence']].append(x)
    data={}
    for case in spec['cases']:
        seq=case['sequence'];source=sorted(rows[seq],key=lambda x:x['frame_index'])
        assert len(source)==case['frames'] and [x['frame_index'] for x in source]==list(range(case['frames']))
        assert source[0]['public_bbox']==case['init_bbox']
        data[seq]=[dict(frame=x['frame_index'],bbox=x['public_bbox']) for x in source]
    return data,dict(result_sha256=sha(path),trace_sha256=previous['baseline_trace_sha256'])

def recompute(root,arms,result):
    spec=read(root/'recursive_spec.json');training=read(root/'training_spec.json')
    assert sha(root/'training_spec.json')==spec['training_spec_sha256']
    assert sha(root/'run_recursive.py')==spec['runner_sha256']
    assert sha(root/'recursive_metric.py')==spec['metric_sha256']
    data={};families={}
    for arm in arms:data[arm],families[arm]=sealed_family(root,arm,spec,training)
    data['native'],families['native']=sealed_native(spec,training)
    # Every complete predicted family and native trace is sealed before metric GT opens.
    scalar=scalar_metric();per={a:{} for a in data};overlap={a:{} for a in data};intervals={a:{} for a in data}
    for case in spec['cases']:
        seq=case['sequence'];p=Path(training['dataset_root'])/seq/'groundtruth.txt';assert sha(p)==case['gt_sha256']
        gt=np.loadtxt(p,delimiter=',').reshape(-1,4);assert len(gt)==case['frames']
        for arm in data:
            v,s,e=scalar(data[arm][seq],gt)
            for k,value in s.items():near(value,result['per_sequence'][arm][seq][k],(arm,seq,k))
            per[arm][seq]=s;overlap[arm][seq]=v;intervals[arm][seq]=e
    aggregates={}
    for arm,values in per.items():
        x={k:sum(r[k] for r in values.values()) for k in ['valid_frames','iou_sum','low_iou_frames','failure_episodes']}
        x.update(mean_iou=x['iou_sum']/x['valid_frames'],macro_sequence_mean_iou=float(np.mean([r['mean_iou'] for r in values.values()])))
        assert x['valid_frames']==28897
        for k,v in x.items():near(v,result['aggregates'][arm][k],(arm,k))
        aggregates[arm]=x
    return spec,training,per,aggregates,overlap,intervals,families

def reference():
    assert sha(PARENT/'recursive_result.json')=='0101fa8185855044dda7462df2f9606b6e58c88248e8a7f1cf87497fb055b150'
    _,_,_,aggregate,_,_,families=recompute(PARENT,['control','null'],read(PARENT/'recursive_result.json'))
    report=dict(status='historical_sealed_family_and_scalar_metric_reference_pass',observed_utc=datetime.now(timezone.utc).isoformat(),
        auditor_sha256=sha(__file__),scalar_metric_source_sha256=M63_SHA,reference_result_sha256=sha(PARENT/'recursive_result.json'),
        aggregates=aggregate,families=families,reference_trajectory_families=3,reference_sequences_each=22,
        new_M67_prediction_files_opened=False,M67_completion_audited=False,new_training_steps=0,new_tracking_calls=0,
        independent_model_review_pass=False)
    write(ROOT/'completion_auditor_reference.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='families'},indent=2))

def training_artifacts(training):
    import torch
    report={};initial_states={};base_states={};final_states={}
    for arm in ['control','support']:
        assert (ROOT/('training_'+arm+'.exit')).read_text().strip()=='0'
        path=ROOT/'training'/arm;r=read(path/'result.json')
        assert r['status']=='one_full_causal_fit_pass_complete' and r['arm']==arm and r['null_support']
        assert r['sequences']==130 and r['total_track_calls']==186694 and r['learned_parameters']==289154
        assert r['training_spec_sha256']==TRAIN_SHA and r['base_parameters_and_buffers_unchanged']
        assert r['support_loss_weight']==training['support_loss_weights'][arm]
        assert r['gt_after_prediction_for_loss_only'] and not r['gt_reinitialization_after_first_frame'] and not r['backward_through_crops_or_time']
        assert sha(path/'sequence_log.jsonl')==r['sequence_log_sha256'] and sha(path/'sampled_state_trace.jsonl')==r['sampled_trace_sha256']
        assert sha(path/'final.pth')==r['final_checkpoint_sha256']
        initial=ROOT/'native_parity'/(arm+'_zero.pth');assert sha(initial)==training['initial_checkpoint_sha256'][arm]
        initial=torch.load(initial,map_location='cpu');final=torch.load(path/'final.pth',map_location='cpu');latest=torch.load(path/'latest.pth',map_location='cpu')
        assert initial['architecture']==final['architecture']==latest['architecture']=='semantic_spatial_support_v1'
        assert all(x['null_support'] and x['use_text'] for x in [initial,final,latest])
        assert final['status']=='complete' and final['completed_sequences']==130
        assert final['frame_count']==r['total_track_calls'] and final['optimizer_steps']==r['optimizer_steps']
        assert final['actual_dataset_optimizer_steps']==r['optimizer_steps'] and final['training_spec_sha256']==TRAIN_SHA
        assert final['base_checkpoint_sha256']==training['native_checkpoint_sha256']
        assert final['support_loss_weight']==latest['support_loss_weight']==training['support_loss_weights'][arm]
        assert initial['model'].keys()==final['model'].keys()==latest['model'].keys()
        assert sum(x.numel() for x in final['model'].values())==289154
        assert all(torch.isfinite(x).all() for x in final['model'].values())
        assert all(torch.equal(final['model'][k],latest['model'][k]) for k in final['model'])
        assert any(not torch.equal(final['model'][k],initial['model'][k]) for k in final['model'])
        assert len(final['optimizer']['state'])==len(final['model'])
        assert {int(x['step']) for x in final['optimizer']['state'].values()}=={r['optimizer_steps']}
        records=[json.loads(x) for x in (path/'sequence_log.jsonl').read_text().splitlines()]
        traces=[json.loads(x) for x in (path/'sampled_state_trace.jsonl').read_text().splitlines()]
        assert [x['sequence'] for x in records]==[x['sequence'] for x in training['sequence_order']]
        by_seq={x['sequence']:[] for x in records}
        for row in traces:by_seq[row['sequence']].append(row)
        calls=steps=0;labels=Counter();scheduled_writes=0;support_rows=[]
        for index,(case,record) in enumerate(zip(training['sequence_order'],records)):
            seq=case['sequence'];n=case['rgb_frames'];p=Path(training['dataset_root'])/seq/'groundtruth.txt'
            assert sha(p)==case['groundtruth_sha256'];gt=np.loadtxt(p,delimiter=',').reshape(-1,4)
            if seq=='toy07_indoor_320':
                assert len(gt)==1406 and n==1367 and case['groundtruth_sha256']=='683e8ae7ae401b71b8d10e9bb489c3956a150163606f5bac925a911f395444e2'
                gt=gt[:n]
            assert len(gt)==n
            valid=np.isfinite(gt).all(1)&(gt[:,2]>0)&(gt[:,3]>0);valid[0]=False
            expected_steps=sum(bool(valid[start:min(start+32,n)].any()) for start in range(1,n,32))
            calls+=n-1;steps+=expected_steps;labels.update(record['label_counts'])
            assert record['sequence_index']==index and record['total_sequences']==130 and record['frames']==n
            assert record['track_calls']==n-1 and record['total_track_calls']==calls
            assert record['total_optimizer_steps']==steps and record['supervised_frames']==int(valid.sum())
            assert record['label_counts'].get('invalid',0)==int((~valid[1:]).sum())
            assert sum(record['label_counts'].values())==n-1
            assert math.isfinite(record['mean_training_loss']) and math.isfinite(record['maximum_preclip_gradient_norm'])
            sampled=by_seq[seq];expected=sorted({1,n-1}|set(range(50,n,50)))
            assert [x['frame_index'] for x in sampled]==expected
            count=0
            for x in sampled:
                f=x['frame_index'];assert math.isfinite(x['best_score']) and 0<=x['best_score']<=1
                assert x['template_write']==(f%50==0 and x['best_score']>.75)
                count+=x['template_write']
                assert (x['label']=='invalid')==(not valid[f])
                for k in ['previous_bbox','bbox']:assert np.isfinite(x[k]).all() and np.asarray(x[k])[2:].min()>0
                if arm=='support' and valid[f]:
                    assert x['support_weight']==.1 and math.isfinite(x['support_loss']) and x['support_loss']>=0
                    assert x['support_positive_cells']==int(x['label']=='centre_inside')
                    assert 0<=x['support_negative_cells']<=256-x['support_positive_cells']
                    assert 0<=x['positive_null_mass']<=1 and 0<=x['negative_null_mass']<=1
                    support_rows.append(x)
                else:
                    assert 'support_loss' not in x and 'support_weight' not in x

            assert record['template_writes']==count
            scheduled_writes+=count
        assert steps==r['optimizer_steps'] and calls==186694 and dict(labels)==r['training_label_counts']
        initial_states[arm]=initial['model'];base_states[arm]=r['base_state_before_sha256'];final_states[arm]=final['model']
        report[arm]=dict(result_sha256=sha(path/'result.json'),final_checkpoint_sha256=sha(path/'final.pth'),
            sequences=130,track_calls=calls,optimizer_steps=steps,supervised_frames=sum(x['supervised_frames'] for x in records),
            sampled_state_rows=len(traces),all_scheduled_write_counts_checked=scheduled_writes,
            final_optimizer_state_steps_verified=True,base_unchanged_runtime_assertion_present=True,
            initial_adapter_state_sha256=r['initial_adapter_state_sha256'],support_loss_weight=training['support_loss_weights'][arm],support_sampled_rows_checked=len(support_rows))
    assert all(torch.equal(initial_states['control'][k],initial_states['support'][k]) for k in initial_states['control'])
    assert base_states['control']==base_states['support']
    assert report['control']['initial_adapter_state_sha256']==report['support']['initial_adapter_state_sha256']
    for key in ['optimizer_steps','track_calls','supervised_frames']:assert report['control'][key]==report['support'][key]
    assert any(not torch.equal(final_states['control'][k],final_states['support'][k]) for k in final_states['control'])
    return report

def completed():
    assert sha(ROOT/'training_spec.json')==TRAIN_SHA and sha(ROOT/'recursive_spec.json')==RECURSIVE_SHA
    ref=read(ROOT/'completion_auditor_reference.json');assert ref['auditor_sha256']==sha(__file__)
    assert ref['status']=='historical_sealed_family_and_scalar_metric_reference_pass'
    for n in ['controller.exit','recursive_analysis.exit','training_control.exit','training_support.exit','control_recursive.exit','support_recursive.exit']:
        assert (ROOT/n).read_text().strip()=='0',n
    training=read(ROOT/'training_spec.json');integration=read(ROOT/'integration.json')
    assert sha(ROOT/'integration.json')==training['integration_sha256']
    for n,h in integration['source_sha256'].items():assert sha(ROOT/'code'/n)==h
    for n,k in [('train_causal.py','training_script_sha256'),('causal_training.py','causal_script_sha256'),
        ('support_loss.py','support_loss_sha256'),('run_m67.sh','run_queue_sha256'),('text_fit.pt','text_fit_sha256'),('text_development.pt','text_development_sha256')]:assert sha(ROOT/n)==training[k]
    assert sha(training['native_checkpoint'])==training['native_checkpoint_sha256']
    result=read(ROOT/'recursive_result.json');assert result['status']=='complete_recursive_development'
    assert result['training_spec_sha256']==TRAIN_SHA and result['recursive_spec_sha256']==RECURSIVE_SHA
    spec,training,per,aggregate,overlap,intervals,families=recompute(ROOT,['control','support'],result)
    train=training_artifacts(training)
    primary=aggregate['support'];control=aggregate['control'];native=aggregate['native'];rule=training['promotion_gates']
    broken={a:[seq for seq in per[a] if per[a][seq]['failure_episodes']==0 and per['support'][seq]['failure_episodes']>0] for a in ['native','control']}
    parent_result=read(PARENT/'recursive_result.json')
    assert sha(PARENT/'recursive_result.json')=='0101fa8185855044dda7462df2f9606b6e58c88248e8a7f1cf87497fb055b150'
    protected=[seq for seq,x in parent_result['per_sequence']['control'].items() if x['failure_episodes']==0]
    assert protected==training['protected_prior_control_sequences']
    broken['prior_control']=[seq for seq in protected if per['support'][seq]['failure_episodes']>0]
    gates=dict(prior_control_success_protection=not broken['prior_control'],mean_vs_native=primary['mean_iou']>=native['mean_iou']+rule['support_pooled_mean_vs_native_minimum'],
      mean_vs_control=primary['mean_iou']>=control['mean_iou']+rule['support_pooled_mean_vs_control_minimum'],
      macro_vs_native=primary['macro_sequence_mean_iou']>=native['macro_sequence_mean_iou'],
      macro_vs_control=primary['macro_sequence_mean_iou']>=control['macro_sequence_mean_iou'],
      low_frames_vs_native=primary['low_iou_frames']<=native['low_iou_frames'],low_frames_vs_control=primary['low_iou_frames']<=control['low_iou_frames'],
      H10_vs_native=primary['failure_episodes']<=native['failure_episodes'],H10_vs_control=primary['failure_episodes']<=control['failure_episodes'],
      native_success_protection=not broken['native'],control_success_protection=not broken['control'])
    assert gates==result['gates'] and all(gates.values())==result['primary_pass']
    assert broken['native']==result['new_failure_sequences'] and broken['control']==result['broken_control_success_sequences']
    assert broken['prior_control']==result['broken_prior_control_success_sequences']
    historical_control_delta={k:control[k]-parent_result['aggregates']['null'][k] for k in ['mean_iou','macro_sequence_mean_iou','low_iou_frames','failure_episodes']}
    harms=[]
    for reference_arm in ['native','control']:
        for seq,runs in intervals['support'].items():
            for start,end in runs:
                if np.all(overlap[reference_arm][seq][start:end]>=.5):
                    harms.append(dict(sequence=seq,reference=reference_arm,start=start,end_exclusive=end,frames=end-start))
    report=dict(status='completed_M67_artifacts_and_development_audited',observed_utc=datetime.now(timezone.utc).isoformat(),
      auditor_sha256=sha(__file__),scalar_metric_source_sha256=M63_SHA,reference_check_sha256=sha(ROOT/'completion_auditor_reference.json'),
      training_spec_sha256=TRAIN_SHA,recursive_spec_sha256=RECURSIVE_SHA,result_sha256=sha(ROOT/'recursive_result.json'),
      training=train,families=families,matched_control_minus_historical_M65_Null=historical_control_delta,recomputed_aggregates=aggregate,recomputed_frozen_gates=gates,
      broken_success_sequences=broken,sustained_support_H10_with_reference_correct_every_frame=harms,
      paired_development_gate_pass=all(gates.values()),content_counterfactuals_allowed=all(gates.values()),
      public_evaluation_allowed=False,formal_three_dataset_metrics_exist=False,
      independent_model_review_pass=False,new_tracking_calls=0,new_optimizer_steps=0,new_caption_calls=0,
      scope='Read-only executor audit. No gate changes or checkpoint selection. Content attribution is still required before low22; no same-bundle formal metrics.')
    write(ROOT/'completed_evidence_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['training','families']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['reference','completed']);a=p.parse_args()
    if a.action=='reference':reference()
    else:completed()
