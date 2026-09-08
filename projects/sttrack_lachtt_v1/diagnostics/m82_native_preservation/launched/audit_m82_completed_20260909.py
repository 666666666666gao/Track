"""Independent scalar development recomputation; no training or inference."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math
R=Path('/root/autodl-tmp/sttrack_m82_native_preservation_20260909')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def equal(a,b):
    if isinstance(a,float): assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-10),(a,b)
    else: assert a==b,(a,b)
assert not (R/'saved_development_audit.json').exists()
for n in ['controller','recursive_analysis','training_category','training_empty','category_recursive','empty_recursive','category_empty_recursive','category_swapped_recursive']:
    assert (R/(n+'.exit')).read_text().strip()=='0',n
s=read(R/'training_spec.json');e=read(R/'recursive_spec.json');f=read(R/'frozen.json');result=read(R/'recursive_result.json')
assert sha(R/'training_spec.json')==f['training_spec_sha256']==e['training_spec_sha256']==result['training_spec_sha256']
assert sha(R/'recursive_spec.json')==f['recursive_spec_sha256']==result['recursive_spec_sha256']
for n,h in f['source_sha256'].items():assert sha(R/n)==h
assert sha(R/'run_pair.sh')==f['queue_sha256']
training={}
for arm in ['category','empty']:
    t=read(R/'training'/arm/'result.json');training[arm]=t
    assert t['status']=='one_full_causal_fit_pass_complete' and t['sequences']==130
    assert t['total_track_calls']==186694 and t['optimizer_steps']==5798
    assert t['base_parameters_and_buffers_unchanged'] and t['learned_parameters']==289154
    assert t['training_spec_sha256']==sha(R/'training_spec.json')
    assert sha(R/'training'/arm/'final.pth')==t['final_checkpoint_sha256']
    assert t['preservation_weight']==s['preservation_weight']==1.
    assert sha(R/'native_preservation.py')==t['preservation_loss_sha256']==s['preservation_loss_sha256']
    for name,key in [('sequence_log.jsonl','sequence_log_sha256'),('sampled_state_trace.jsonl','sampled_trace_sha256')]:assert sha(R/'training'/arm/name)==t[key]
    rows=[json.loads(x) for x in (R/'training'/arm/'sequence_log.jsonl').read_text().splitlines()]
    assert [x['sequence'] for x in rows]==[x['sequence'] for x in s['sequence_order']]
    assert sum(x['track_calls'] for x in rows)==186694 and rows[-1]['total_optimizer_steps']==5798
    assert rows[-1]['cumulative_native_eligible_frames']==t['native_eligible_frames']
    equal(rows[-1]['cumulative_preservation_kl_sum'],t['preservation_kl_sum'])
    assert 0<t['native_eligible_frames']<=sum(x['supervised_frames'] for x in rows)
    for line in (R/'training'/arm/'sampled_state_trace.jsonl').read_text().splitlines():
        v=json.loads(line)
        assert v['native_eligible']==(v['label']=='centre_inside' and v['native_iou']>=.5)
        if v['native_eligible']:
            assert math.isfinite(v['preservation_kl']) and v['preservation_kl']>=-1e-6
            equal(v['preservation_weighted'],v['preservation_kl'])
        else: assert v['preservation_weighted']==0.
for key in ['initial_adapter_state_sha256','base_state_before_sha256','optimizer_steps','total_track_calls']:
    assert training['category'][key]==training['empty'][key]

families=['category','empty','category_empty','category_swapped']
predicted={a:{} for a in families}
# Validate and seal all family receipts before opening any development GT.
for arm in families:
    receipt=read(R/(arm+'_recursive_receipt.json'))
    assert sha(R/(arm+'_recursive_receipt.json'))==result['receipts'][arm]
    assert receipt['status']=='complete' and receipt['total_frames']==33130
    assert receipt['recursive_spec_sha256']==sha(R/'recursive_spec.json')
    train_arm='empty' if arm=='empty' else 'category'
    assert receipt['head_sha256']==training[train_arm]['final_checkpoint_sha256']
    for case,row in zip(e['cases'],receipt['sequences']):
        assert case['sequence']==row['sequence']
        p=R/'recursive'/arm/(row['sequence']+'.json');assert sha(p)==row['sha256']
        data=read(p);assert data['arm']==arm
        a=data['rows'];assert len(a)==row['frames']==case['frames']
        assert [x['frame'] for x in a]==list(range(case['frames'])) and a[0]['bbox']==case['init_bbox']
        assert all(all(math.isfinite(v) for v in x['bbox']) and x['bbox'][2]>0 and x['bbox'][3]>0 for x in a)
        predicted[arm][case['sequence']]=a
    assert set(predicted[arm])==set(s['development_sequences'])

def statistics(rows,gt):
    valid=low=count=invalid=run=0;values=[]
    for i,(row,g) in enumerate(zip(rows,gt)):
        if i==0:continue
        if not all(math.isfinite(x) for x in g) or min(g[2:])<=0:
            count+=int(run>=10);run=0;invalid+=1;continue
        a=row['bbox'];ix=max(0.,min(a[0]+a[2],g[0]+g[2])-max(a[0],g[0]));iy=max(0.,min(a[1]+a[3],g[1]+g[3])-max(a[1],g[1]));inter=ix*iy
        value=inter/(a[2]*a[3]+g[2]*g[3]-inter);values.append(value);valid+=1
        if value<=.1:low+=1;run+=1
        else:count+=int(run>=10);run=0
    count+=int(run>=10)
    return dict(valid_frames=valid,iou_sum=math.fsum(values),mean_iou=math.fsum(values)/valid,low_iou_frames=low,failure_episodes=count,invalid_gt_frames=invalid)

per={a:{} for a in families}
for case in e['cases']:
    p=Path(s['dataset_root'])/case['sequence']/'groundtruth.txt';assert sha(p)==case['gt_sha256']
    gt=[[float(v) for v in line.split(',')] for line in p.read_text().splitlines() if line.strip()]
    assert len(gt)==case['frames'] and all(len(g)==4 for g in gt)
    for arm in families:
        row=statistics(predicted[arm][case['sequence']],gt);per[arm][case['sequence']]=row
        for k,v in row.items():equal(v,result['per_sequence'][arm][case['sequence']][k])
aggregates={}
for arm,rows in per.items():
    a={k:sum(v[k] for v in rows.values()) for k in ['valid_frames','iou_sum','low_iou_frames','failure_episodes']}
    a.update(mean_iou=a['iou_sum']/a['valid_frames'],macro_sequence_mean_iou=math.fsum(v['mean_iou'] for v in rows.values())/22)
    assert a['valid_frames']==28897
    for k,v in a.items():equal(v,result['aggregates'][arm][k])
    aggregates[arm]=a
native_path=Path(e['native_result_path']);assert sha(native_path)==s['native_result_sha256']
native=read(native_path);n=native['aggregates']['native'];a=aggregates['category'];b=aggregates['empty'];g=s['promotion_gates']
broken=lambda ref:[name for name,v in ref.items() if v['failure_episodes']==0 and per['category'][name]['failure_episodes']>0]
gates=dict(mean_vs_native=a['mean_iou']>=n['mean_iou']+g['category_pooled_mean_vs_native_minimum'],mean_vs_control=a['mean_iou']>=b['mean_iou']+g['category_pooled_mean_vs_control_minimum'],macro_vs_native=a['macro_sequence_mean_iou']>=n['macro_sequence_mean_iou'],macro_vs_control=a['macro_sequence_mean_iou']>=b['macro_sequence_mean_iou'],low_frames_vs_native=a['low_iou_frames']<=n['low_iou_frames'],low_frames_vs_control=a['low_iou_frames']<=b['low_iou_frames'],H10_vs_native=a['failure_episodes']<=n['failure_episodes'],H10_vs_control=a['failure_episodes']<=b['failure_episodes'],native_success_protection=not broken(native['per_sequence']['native']),control_success_protection=not broken(per['empty']))
assert gates==result['gates'] and all(gates.values())==result['primary_pass']
m=Path(s['matched_M78_result_path']);assert sha(m)==s['matched_M78_result_sha256'];m78=read(m)
increment={};content={}
for arm in ['category','empty']:
    a=aggregates[arm];b=m78['aggregates'][arm]
    increment[arm]=dict(mean=a['mean_iou']>=b['mean_iou'],macro=a['macro_sequence_mean_iou']>=b['macro_sequence_mean_iou'],low=a['low_iou_frames']<=b['low_iou_frames'],H10=a['failure_episodes']<=b['failure_episodes'])
for arm in ['category_empty','category_swapped']:
    a=aggregates['category'];b=aggregates[arm]
    content[arm]=dict(mean=a['mean_iou']>b['mean_iou'],macro=a['macro_sequence_mean_iou']>=b['macro_sequence_mean_iou'],low=a['low_iou_frames']<=b['low_iou_frames'],H10=a['failure_episodes']<=b['failure_episodes'])
assert increment==result['preservation_incremental_gates'] and content==result['content_gates']
audit=dict(status='complete_independent_scalar_M82_training_and_development_verification',observed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),result_sha256=sha(R/'recursive_result.json'),families=4,sequences=88,positions=132520,valid_positions_per_family=28897,training_artifacts_verified=True,scalar_metrics_and_gates_verified=True,primary_pass_count=sum(gates.values()),content_pass_count=sum(sum(g.values()) for g in content.values()),preservation_category_pass_count=sum(increment['category'].values()),independent_model_review_pass=False)
(R/'saved_development_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit))
