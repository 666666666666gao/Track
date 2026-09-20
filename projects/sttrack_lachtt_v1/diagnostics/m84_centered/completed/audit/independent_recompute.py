"""Read-only M84 audit; stdlib scalar IoU and explicit run boundaries, no author metric import."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math
import tarfile
import zipfile
import pickletools
import pickle
import io
import struct
from collections import OrderedDict

ROOT = Path(r'D:\Program Files\UserCache\gb\codex\tmp\sttrack_m84_centered_20260920\completed')
WORK = ROOT.parent
M82 = WORK.parent / 'm82_complete_20260920'
OUT = Path(__file__).parent
def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    return hashlib.file_digest(Path(p).open('rb'), 'sha256').hexdigest()
errors=[]
checks=Counter()
hash_checks=[]
max_diffs=Counter()
def check(ok, label, group):
    checks[group]+=1
    if not ok: errors.append(label)
def hash_check(p, expected, label):
    p=Path(p)
    actual=sha(p) if p.is_file() else None
    row=dict(path=str(p),expected=expected,actual=actual,match=actual==expected,label=label)
    hash_checks.append(row)
    check(row['match'],label,'hash_binding')
    return row['match']
def num_check(actual,expected,label,tol=1e-8):
    difference=abs(actual-expected)
    max_diffs[label.split(':')[0]]=max(max_diffs[label.split(':')[0]],difference)
    check(math.isfinite(actual) and difference<=tol,label,'numeric_value')
def scalar_stats(rows,gt):
    values=[]; valid=low=invalid=0; run=0; runs=[]; run_start=None; lows=[]; per_frame=[]
    for i,(row,g) in enumerate(zip(rows,gt)):
        ok=i>0 and all(math.isfinite(x) for x in g) and g[2]>0 and g[3]>0
        if i>0 and not ok: invalid+=1
        v=None
        if ok:
            x,y,w,h=row['bbox']; gx,gy,gw,gh=g
            intersection=max(0., min(x+w,gx+gw)-max(x,gx))*max(0.,min(y+h,gy+gh)-max(y,gy))
            v=intersection/(w*h+gw*gh-intersection)
            valid+=1; values.append(v)
        is_low=ok and v<=0.1
        if is_low:
            if run==0:run_start=i
            run+=1; low+=1
        else:
            if run>=10:runs.append([run_start,i])
            run=0
        lows.append(bool(is_low)); per_frame.append(v)
    if run>=10:runs.append([run_start,len(rows)])
    total=math.fsum(values)
    return dict(valid_frames=valid,iou_sum=total,mean_iou=total/valid,low_iou_frames=low,failure_episodes=len(runs),invalid_gt_frames=invalid),runs,per_frame
def aggregate(per):
    keys=['valid_frames','iou_sum','low_iou_frames','failure_episodes']
    sums={k:sum(x[k] for x in per.values()) for k in keys}
    sums['mean_iou']=sums['iou_sum']/sums['valid_frames']
    sums['macro_sequence_mean_iou']=math.fsum(x['mean_iou'] for x in per.values())/len(per)
    return sums
def gt_read(p):
    return [[float(x) for x in line.strip().split(',')] for line in p.read_text().splitlines() if line.strip()]
def compare_dict(a,b,label):
    check(set(a)==set(b),label+':keys','structure')
    for key in a.keys() & b.keys():num_check(a[key],b[key],label+':'+key)

manifest=read(ROOT/'evidence_manifest.json')
for rel,item in manifest.items():
    p=ROOT/rel
    hash_check(p,item['sha256'],'manifest:'+rel)
    check(p.stat().st_size==item['bytes'],'manifest bytes:'+rel,'byte_size')
spec=read(ROOT/'recursive_spec.json'); training=read(ROOT/'training_spec.json'); frozen=read(ROOT/'frozen.json')
integration=read(ROOT/'integration.json'); result=read(ROOT/'recursive_result.json'); trained=read(ROOT/'training/category/result.json')
hash_check(ROOT/'training_spec.json',spec['training_spec_sha256'],'recursive training spec')
for rel,key in [('run_recursive.py','runner_sha256'),('recursive_metric.py','metric_sha256'),('run_m84.sh','queue_sha256')]:
    hash_check(ROOT/rel,spec[key],'recursive spec:'+key)
for rel,key in [('training_spec.json','training_spec_sha256'),('recursive_spec.json','recursive_spec_sha256')]:
    hash_check(ROOT/rel,frozen[key],'frozen:'+key)
    hash_check(ROOT/rel,result[key],'result:'+key)
for rel,key in [('preflight_result.json','preflight_sha256'),('code_review_receipt.json','review_receipt_sha256')]:
    hash_check(WORK/rel,frozen[key],'frozen:'+key)
for rel,key in [('EXPERIMENT_PLAN.md','plan_sha256'),('integration.json','integration_sha256'),('run_m84.sh','run_queue_sha256')]:
    hash_check(ROOT/rel,training[key],'training spec:'+key)
for rel,key in [('data_inventory.json','inventory_sha256'),('train_causal.py','training_script_sha256'),('causal_training.py','causal_script_sha256'),('support_loss.py','support_loss_sha256'),('window_competition.py','window_loss_sha256'),('native_preservation.py','preservation_loss_sha256'),('preflight_m84.py','preflight_source_sha256')]:
    hash_check(WORK/rel,training[key],'training spec sibling:'+key)
for rel,digest in integration['source_sha256'].items():hash_check(ROOT/'code'/rel,digest,'integration:'+rel)
hash_check(M82/'integration.json',integration['parent_M82_integration_sha256'],'parent M82 integration')
hash_check(M82/'training_spec.json',training['parent_training_spec_sha256'],'parent M82 training spec')
for rel,key in [('final.pth','final_checkpoint_sha256'),('sequence_log.jsonl','sequence_log_sha256')]:
    hash_check(ROOT/'training/category'/rel,trained[key],'trained:'+key)
sampled_path=WORK/'training_completed/training/category/sampled_state_trace.jsonl'
hash_check(sampled_path,trained['sampled_trace_sha256'],'trained sampled trace sibling')
for key in ['window_loss_sha256','preservation_loss_sha256']:
    check(training[key]==trained[key],'training-result:'+key,'binding_equality')
for rel,key in [('verify_training_complete.py','verifier_sha256'),('training/category/result.json','result_sha256'),('training/category/final.pth','final_checkpoint_sha256')]:
    hash_check(ROOT/rel,read(ROOT/'training_saved_verification.json')[key],'saved verification:'+key)
hash_check(ROOT/'controls/native_result.json',training['native_result_sha256'],'native control')
hash_check(ROOT/'controls/M82_recursive_result.json',training['parent_result_sha256'],'M82 control')
hash_check(M82/'native_reference.json',training['native_result_sha256'],'independent native metric copy')
hash_check(M82/'recursive_result.json',training['parent_result_sha256'],'independent M82 result copy')
hash_check(ROOT/'training/category/final.pth',result['head_sha256'],'result head')
hash_check(ROOT/'training_spec.json',trained['training_spec_sha256'],'trained spec')

native=read(ROOT/'controls/native_result.json')['per_sequence']['native']
parent=read(ROOT/'controls/M82_recursive_result.json')
per={'native':native}
h10={}; frame_values={}; receipts={}; gtmap={}; prediction_rows={}; receipt_checks=0
cases=spec['cases']; names=[c['sequence'] for c in cases]
check(len(names)==len(set(names))==22,'22 unique cases','structure')
check(set(names)==set(training['development_sequences']),'development set agreement','structure')
for c in cases:
    name=c['sequence']; p=ROOT/'dataset_gt'/(name+'.txt')
    hash_check(p,c['gt_sha256'],'development GT:'+name)
    gt=gt_read(p); gtmap[name]=gt
    check(len(gt)==c['frames'],'GT frame count:'+name,'frame_contract')
    check(gt[0]==c['init_bbox'],'GT init box:'+name,'frame_contract')
    other=M82/'dataset_gt'/name/'groundtruth.txt'
    hash_check(other,c['gt_sha256'],'independent M82 GT copy:'+name)

arms=['category','category_empty','category_swapped']
for arm in arms:
    p=ROOT/(arm+'_recursive_receipt.json'); receipt=read(p); receipts[arm]=receipt
    hash_check(p,result['receipts'][arm],'result receipt:'+arm)
    check(receipt['status']=='complete' and receipt['arm']==arm,arm+' receipt complete','receipt')
    check(receipt['subsequent_gt_opened'] is False and receipt['text_updated_online'] is False,arm+' receipt causal flags','receipt')
    check(receipt['total_frames']==33130 and len(receipt['sequences'])==22,arm+' receipt counts','receipt')
    for filename,key in [('recursive_spec.json','recursive_spec_sha256'),('training/category/final.pth','head_sha256'),('training/category/result.json','training_result_sha256')]:
        hash_check(ROOT/filename,receipt[key],arm+' '+key)
    per[arm]={};h10[arm]={};frame_values[arm]={};prediction_rows[arm]={}
    for c,item in zip(cases,receipt['sequences']):
        name=c['sequence'];p=ROOT/'recursive'/arm/(name+'.json')
        check(item['sequence']==name and item['frames']==c['frames'],arm+' receipt sequence:'+name,'receipt')
        hash_check(p,item['sha256'],arm+' prediction:'+name);receipt_checks+=1
        data=read(p);rows=data['rows'];prediction_rows[arm][name]=rows
        check(data['sequence']==name and data['arm']==arm,arm+' identity:'+name,'frame_contract')
        check(len(rows)==c['frames'] and [r['frame'] for r in rows]==list(range(c['frames'])),arm+' chronology:'+name,'frame_contract')
        check(rows[0]['bbox']==c['init_bbox'] and rows[0]['score'] is None,arm+' init:'+name,'frame_contract')
        check(all(len(r['bbox'])==4 and all(math.isfinite(x) for x in r['bbox']) and min(r['bbox'][2:])>0 for r in rows),arm+' boxes:'+name,'frame_contract')
        check(all(isinstance(r['score'],(int,float)) and math.isfinite(r['score']) for r in rows[1:]),arm+' scores:'+name,'frame_contract')
        metrics,runs,values=scalar_stats(rows,gtmap[name]);per[arm][name]=metrics;h10[arm][name]=runs;frame_values[arm][name]=values
        compare_dict(metrics,result['per_sequence'][arm][name],'raw per-sequence '+arm+'/'+name)

# Independently recompute the reusable M82 category control from its raw predictions.
mr=read(M82/'category_recursive_receipt.json'); per['M82_mask_control']={};h10['M82_mask_control']={}
check(mr['status']=='complete' and mr['total_frames']==33130 and len(mr['sequences'])==22,'M82 category receipt counts','receipt')
for c,item in zip(cases,mr['sequences']):
    name=c['sequence'];p=M82/'recursive/category'/(name+'.json')
    check(item['sequence']==name and item['frames']==c['frames'],'M82 receipt sequence:'+name,'receipt')
    hash_check(p,item['sha256'],'M82 prediction:'+name);receipt_checks+=1
    data=read(p);rows=data['rows']
    check(len(rows)==c['frames'] and [r['frame'] for r in rows]==list(range(c['frames'])),'M82 chronology:'+name,'frame_contract')
    metrics,runs,values=scalar_stats(rows,gtmap[name]);per['M82_mask_control'][name]=metrics;h10['M82_mask_control'][name]=runs
    compare_dict(metrics,parent['per_sequence']['category'][name],'raw M82 '+name)
    compare_dict(metrics,result['per_sequence']['M82_mask_control'][name],'M84 imported M82 '+name)

# New independent native reference extracted from the frozen baseline trace shards.
NATIVE=ROOT/'native_reference'
native_manifest=read(NATIVE/'manifest.json')
for rel,item in native_manifest.items():
    hash_check(NATIVE/rel,item['sha256'],'native extraction manifest:'+rel)
    check((NATIVE/rel).stat().st_size==item['bytes'],'native extraction bytes:'+rel,'byte_size')
prov=read(NATIVE/'reference_provenance.json');source_spec=read(NATIVE/'source_recursive_spec.json')
hash_check(NATIVE/'source_recursive_spec.json',prov['source_spec_sha256'],'native extraction source spec')
check(prov['source_trace_sha256']==source_spec['baseline_trace_sha256'],'native trace SHA identity','binding_equality')
check(sha(NATIVE/'source_recursive_spec.json')==read(ROOT/'controls/native_result.json')['recursive_spec_sha256'],'native original result spec identity','binding_equality')
hash_check(WORK/'native_reference.tar.gz','9b62820ccea97a5593768d6e04e085fac51b4e6856d2db651364c73296c5a1ea','native reference transferred archive')
for path,header in prov['headers'].items():
    check(header['checkpoint']['sha256']==training['native_checkpoint_sha256'],'native source checkpoint identity:'+path,'binding_equality')
    check(header['complete'] and not header['ground_truth_used_after_initialization'],'native source receipt flags:'+path,'receipt')
native_raw={};native_parity={};h10['native']={}
for c in cases:
    name=c['sequence'];data=read(NATIVE/(name+'.json'));rows=data['rows']
    check(data['sequence']==name and len(rows)==c['frames'],'native raw sequence count:'+name,'frame_contract')
    check([r['frame'] for r in rows]==list(range(c['frames'])),'native raw chronology:'+name,'frame_contract')
    hash_check(ROOT/'dataset_gt'/(name+'.txt'),source_spec['development_gt_sha256'][name],'native original GT binding:'+name)
    metrics,runs,values=scalar_stats(rows,gtmap[name]);native_raw[name]=metrics;h10['native'][name]=runs
    compare_dict(metrics,native[name],'raw native '+name)
    compare_dict(metrics,result['per_sequence']['native'][name],'M84 imported native '+name)
    empty=prediction_rows['category_empty'][name]
    bbox_mismatches=[i for i,(x,y) in enumerate(zip(rows,empty)) if x['bbox']!=y['bbox']]
    score_mismatches=[i for i,(x,y) in enumerate(zip(rows,empty)) if i>0 and x['score']!=y['score']]
    native_parity[name]=dict(frames=len(rows),all_bbox_exact=not bbox_mismatches,noninitial_score_exact=not score_mismatches,bbox_mismatches=bbox_mismatches,score_mismatches=score_mismatches)
    check(not bbox_mismatches and not score_mismatches,'native raw exact parity:'+name,'native_raw_parity')
    check(all(native_parity[name][k]==v for k,v in prov['parity'][name].items()),'native provenance parity match:'+name,'native_raw_parity')
per['native']=native_raw

# Independently derived aggregates and all frozen conditions.
aggregates={arm:aggregate(per[arm]) for arm in ['native','M82_mask_control']+arms}
for arm,agg in aggregates.items():compare_dict(agg,result['aggregates'][arm],'aggregate '+arm)
current=aggregates['category']
def compare(other):
    return dict(mean=current['mean_iou']>other['mean_iou'],macro=current['macro_sequence_mean_iou']>=other['macro_sequence_mean_iou'],low=current['low_iou_frames']<=other['low_iou_frames'],H10=current['failure_episodes']<=other['failure_episodes'])
native_damage=[s for s in names if native[s]['failure_episodes']==0 and per['category'][s]['failure_episodes']>0]
parity={s:all(per['category_empty'][s][k]==native[s][k] for k in ['valid_frames','low_iou_frames','failure_episodes']) and abs(per['category_empty'][s]['iou_sum']-native[s]['iou_sum'])<=1e-8 for s in names}
native_gates=compare(aggregates['native']);native_gates['native_zero_H10_protection']=not native_damage
gates=dict(M82_increment=compare(aggregates['M82_mask_control']),native=native_gates,swapped_content=compare(aggregates['category_swapped']),empty_native_parity={'all_sequences':all(parity.values())})
for group,g in gates.items():
    for key,v in g.items():check(v==result['gates'][group][key],group+'/'+key,'frozen_gate_recompute')
check(result['native_success_damage']==native_damage,'native success damage list','derived_result')
check(result['empty_native_parity']==parity,'empty native parity list','derived_result')
gate_count=sum(len(g) for g in gates.values());gate_pass=sum(sum(g.values()) for g in gates.values())
check(result['gate_count']==gate_count==14,'14 frozen gates','derived_result')
check(result['all_gates_pass']==(gate_pass==gate_count),'all gates result','derived_result')
loo={}
for arm in ['category_empty','category_swapped']:
    loo[arm]={}
    for s in names:
        keep=[n for n in names if n!=s]
        left=math.fsum(per['category'][n]['iou_sum'] for n in keep)/sum(per['category'][n]['valid_frames'] for n in keep)
        right=math.fsum(per[arm][n]['iou_sum'] for n in keep)/sum(per[arm][n]['valid_frames'] for n in keep)
        loo[arm][s]=left-right
        num_check(left-right,result['content_leave_one_out'][arm][s],'LOO:'+arm+'/'+s,1e-12)

# Validate local human-readable derivatives without relying on them for the recomputation.
csv_counts={}
with (ROOT/'aggregate_metrics.csv').open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
csv_counts['aggregate_metrics.csv']=len(rows)
check(len(rows)==5 and {r['arm'] for r in rows}==set(aggregates),'aggregate CSV completeness','CSV')
for row in rows:
    for k,v in aggregates[row['arm']].items():num_check(float(row[k]),v,'CSV aggregate:'+row['arm']+'/'+k)
with (ROOT/'per_sequence_metrics.csv').open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
csv_counts['per_sequence_metrics.csv']=len(rows)
check(len(rows)==110 and len({(r['arm'],r['sequence']) for r in rows})==110,'per-sequence CSV completeness','CSV')
for row in rows:
    for k,v in per[row['arm']][row['sequence']].items():num_check(float(row[k]),v,'CSV per-sequence:'+row['arm']+'/'+row['sequence']+'/'+k)
with (ROOT/'frozen_gates.csv').open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
csv_counts['frozen_gates.csv']=len(rows)
check(len(rows)==14 and len({(r['group'],r['criterion']) for r in rows})==14,'frozen gate CSV completeness','CSV')
for row in rows:check((row['pass']=='True')==gates[row['group']][row['criterion']],'CSV gate:'+row['group']+'/'+row['criterion'],'CSV')
with (ROOT/'leave_one_sequence_out.csv').open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
csv_counts['leave_one_sequence_out.csv']=len(rows)
check(len(rows)==44 and len({(r['comparison'],r['removed_sequence']) for r in rows})==44,'LOO CSV completeness','CSV')
for row in rows:num_check(float(row['mean_iou_delta']),loo[row['comparison']][row['removed_sequence']],'CSV LOO:'+row['comparison']+'/'+row['removed_sequence'],1e-12)

# Saved damage intervals, recomputed from raw IoU values without importing the diagnostic.
def mask_intervals(mask):
    active=[];result=[]
    for i,on in enumerate(mask):
        if on:active.append(i)
        else:
            if len(active)>=10:result.append([active[0],active[-1]+1])
            active=[]
    if len(active)>=10:result.append([active[0],active[-1]+1])
    return result
damage_report=read(ROOT/'saved_damage_diagnostic.json');damage_rows=[]
for name in names:
    cv=frame_values['category'][name];ev=frame_values['category_empty'][name]
    bad=mask_intervals([c is not None and e is not None and c<=.1 and e>=.5 for c,e in zip(cv,ev)])
    good=mask_intervals([c is not None and e is not None and e<=.1 and c>=.5 for c,e in zip(cv,ev)])
    first={a:next((i for i,(c,o) in enumerate(zip(prediction_rows['category'][name],prediction_rows[a][name])) if any(abs(x-y)>1e-6 for x,y in zip(c['bbox'],o['bbox']))),None) for a in ['category_empty','category_swapped']}
    row=dict(sequence=name,first_bbox_difference_gt1e_6=first,H10={a:h10[a][name] for a in arms},sustained_damage_vs_empty=bad,sustained_rescue_vs_empty=good)
    damage_rows.append(row)
for a,b in zip(damage_rows,damage_report['sequences']):check(a==b,'saved damage diagnostic:'+a['sequence'],'posthoc_diagnostic')
damage_totals={}
for term in ['damage','rescue']:
    ranges=[r for x in damage_rows for r in x['sustained_'+term+'_vs_empty']]
    damage_totals['sustained_'+term+'_runs']=len(ranges)
    damage_totals['sustained_'+term+'_frames']=sum(b-a for a,b in ranges)
for k,v in damage_totals.items():check(v==damage_report[k],'saved damage total:'+k,'posthoc_diagnostic')

# Saved training sequence totals and sampled chronology; no GPU replay claimed.
seqrows=[json.loads(x) for x in (ROOT/'training/category/sequence_log.jsonl').read_text().splitlines()]
check(len(seqrows)==130,'130 training rows','training')
counts=Counter();calls=0;previous_steps=0
for i,(row,plan) in enumerate(zip(seqrows,training['sequence_order'])):
    check(row['sequence']==plan['sequence'] and row['sequence_index']==i,'training sequence order:'+str(i),'training')
    check(row['track_calls']==plan['rgb_frames']-1 and row['frames']==plan['rgb_frames'],'training frame count:'+str(i),'training')
    check(sum(row['label_counts'].values())==row['track_calls'],'training label sum:'+str(i),'training')
    check(row['supervised_frames']==row['track_calls']-row['label_counts'].get('invalid',0),'training supervised:'+str(i),'training')
    check(math.isfinite(row['maximum_preclip_gradient_norm']) and (row['mean_training_loss'] is None or math.isfinite(row['mean_training_loss'])),'training finite:'+str(i),'training')
    calls+=row['track_calls'];counts.update(row['label_counts'])
    check(row['total_track_calls']==calls and row['total_optimizer_steps']>=previous_steps,'training cumulative:'+str(i),'training')
    previous_steps=row['total_optimizer_steps']
check(calls==trained['total_track_calls']==training['total_training_track_calls']==186694,'training calls final','training')
check(previous_steps==trained['optimizer_steps']==training['expected_optimizer_steps']==5798,'training steps final','training')
check(dict(counts)==trained['training_label_counts'],'training label totals','training')
for key in ['competition_frames','competition_loss_sum','competition_negatives','native_eligible_frames','preservation_kl_sum']:
    num_check(seqrows[-1]['cumulative_'+key],trained[key],'training cumulative:'+key)
fitnames=[x['sequence'] for x in training['sequence_order']]
check(len(set(fitnames))==130 and not set(fitnames)&set(names),'fit-development disjoint sequences','training')
trace=[json.loads(x) for x in sampled_path.read_text().splitlines()]
expected=[(p['sequence'],i) for p in training['sequence_order'] for i in range(1,p['rgb_frames']) if i%50==0 or i==1 or i==p['rgb_frames']-1]
check([(r['sequence'],r['frame_index']) for r in trace]==expected,'sampled trace chronological coverage','training')
trace_flags=Counter()
for row in trace:
    for k,v in row.items():
        if isinstance(v,bool):trace_flags[k+':'+str(v)]+=1

# Source, seed, data order and fixed bank identity across the M82 control and M84 method.
m82_training=read(M82/'training_spec.json')
shared_keys=['seed','sequence_order','development_sequences','native_checkpoint_sha256','banks','learning_rate','weight_decay','gradient_accumulation_frames','gradient_clip','native_update_interval','native_update_threshold','window_loss_weight','window_hard_negative_count','window_negative_maximum_iou','preservation_weight','preservation_teacher_minimum_iou','epochs']
shared_identity={k:training[k]==m82_training[k] for k in shared_keys}
for k,v in shared_identity.items():check(v,'shared control identity:'+k,'control_identity')
source_diff=[]
m82_int=read(M82/'integration.json')
for name,h in integration['source_sha256'].items():
    if m82_int['source_sha256'].get(name)!=h:source_diff.append(name)

# Exit and log receipts.
exit_codes={p.name:p.read_text().strip() for p in ROOT.glob('*.exit')}
for name,value in exit_codes.items():check(value=='0','exit:'+name,'terminal')
logs={}
for p in ROOT.glob('*.log'):
    txt=p.read_text();logs[p.name]={'bytes':p.stat().st_size,'lines':len(txt.splitlines()),'exception_markers':[x for x in ['Traceback (most recent call last):','AssertionError','RuntimeError','CUDA out of memory'] if x in txt]}
    check(not logs[p.name]['exception_markers'],'log exceptions:'+p.name,'terminal')
for arm in arms:
    lines=(ROOT/(arm+'_recursive.log')).read_text().splitlines()
    parsed=[json.loads(x) for x in lines if x.startswith('{')]
    check(len(parsed)==23,'recursive log receipts:'+arm,'terminal')
    check(parsed[:22]==receipts[arm]['sequences'],'recursive log per-sequence match:'+arm,'terminal')
    check(parsed[-1]=={k:v for k,v in receipts[arm].items() if k!='sequences'},'recursive log end receipt:'+arm,'terminal')
analysis_log=read(ROOT/'recursive_analysis.log')
check(analysis_log=={k:v for k,v in result.items() if k!='per_sequence'},'analysis log exact result','terminal')
train_log=(ROOT/'training_category.log').read_text()
check(all(json.dumps(r) in train_log for r in seqrows),'training log contains all 130 sequence receipts','terminal')

archive_native_candidates={}
for name in ['review_evidence.tar.gz','review_inputs.tar.gz']:
    with tarfile.open(M82/name,'r:gz') as t:
        members=t.getnames()
    archive_native_candidates[name]=[x for x in members if any(k in x.lower() for k in ['native','default','baseline','shard'])]

cp=ROOT/'training/category/final.pth'
def rebuild_tensor(storage,offset,size,stride,requires_grad,hooks):
    return dict(storage=storage,offset=offset,size=size,stride=stride,requires_grad=requires_grad)
class StorageTag:pass
class Reader(pickle.Unpickler):
    def find_class(self,module,name):
        permitted={('collections','OrderedDict'):OrderedDict,('torch','FloatStorage'):StorageTag,('torch._utils','_rebuild_tensor_v2'):rebuild_tensor}
        return permitted[(module,name)]
    def persistent_load(self,ident):
        assert ident[0]=='storage' and ident[1] is StorageTag
        return dict(key=ident[2],device=ident[3],count=ident[4])
with zipfile.ZipFile(cp) as z:
    pk=next(n for n in z.namelist() if n.endswith('/data.pkl'))
    raw_pickle=z.read(pk);opcodes=list(pickletools.genops(raw_pickle))
    checkpoint_pickle_globals=sorted(set(arg for op,arg,pos in opcodes if op.name=='GLOBAL'))
    checkpoint=Reader(io.BytesIO(raw_pickle)).load()
    cp_meta={k:v for k,v in checkpoint.items() if k not in ['model','optimizer']}
    model_numel={k:math.prod(v['size']) for k,v in checkpoint['model'].items()}
    storages=[n for n in z.namelist() if '/data/' in n]
    finite_count=0
    for name in storages:
        data=z.read(name);values=list(struct.iter_unpack('<f',data))
        check(all(math.isfinite(v[0]) for v in values),'checkpoint finite storage:'+name,'checkpoint')
        finite_count+=len(values)
    optimizer=checkpoint['optimizer']
    check(len(optimizer['param_groups'])==1,'one checkpoint optimizer group','checkpoint')
    group=optimizer['param_groups'][0]
    check(group['lr']==training['learning_rate'] and group['weight_decay']==training['weight_decay'],'checkpoint optimizer LR/decay','checkpoint')
    check(len(group['params'])==len(optimizer['state'])==len(checkpoint['model'])-1,'checkpoint model/optimizer tensor count','checkpoint')
    for param,state in optimizer['state'].items():
        step_tensor=state['step']; step_name=pk.removesuffix('data.pkl')+'data/'+step_tensor['storage']['key']
        check(struct.unpack('<f',z.read(step_name))[0]==5798,'checkpoint optimizer step:'+str(param),'checkpoint')
check(cp_meta['status']=='complete' and cp_meta['architecture']==training['architecture'],'checkpoint status/architecture','checkpoint')
check(cp_meta['seed']==2027 and cp_meta['completed_sequences']==130 and cp_meta['frame_count']==186694 and cp_meta['optimizer_steps']==cp_meta['actual_dataset_optimizer_steps']==5798,'checkpoint training totals','checkpoint')
check(cp_meta['training_spec_sha256']==sha(ROOT/'training_spec.json') and cp_meta['base_checkpoint_sha256']==training['native_checkpoint_sha256'],'checkpoint training/base bindings','checkpoint')
check(sum(model_numel.values())-model_numel['empty_text']==training['stored_parameters']==289154,'checkpoint trainable parameter count','checkpoint')
checkpoint_report=dict(metadata=cp_meta,model_tensor_count=len(model_numel),model_numel=model_numel,trainable_parameter_count=sum(model_numel.values())-model_numel['empty_text'],float_storage_files=len(storages),finite_float_values=finite_count,optimizer_state_count=len(optimizer['state']),load_method='Restricted stdlib pickle reader permitting only FloatStorage tag, OrderedDict and inert tensor reconstruction; tensor bytes checked directly with struct; no torch or runtime execution.')

out=dict(
    checks=dict(checks),check_count=sum(checks.values()),errors=errors,error_count=len(errors),
    independent_metric='Python stdlib scalar continuous xywh intersection / union; math.fsum; initialize excluded; invalid GT breaks consecutive low runs; H10 counts maximal low runs with length >=10',
    manifest_files=len(manifest),integration_source_files=len(integration['source_sha256']),
    hash_binding_checks=len(hash_checks),hash_bindings=hash_checks,max_absolute_numeric_differences=dict(max_diffs),
    raw_prediction_files_checked=receipt_checks+len(cases),m84_raw_frames=3*sum(c['frames'] for c in cases),
    m84_raw_noninitial_positions=3*sum(c['frames']-1 for c in cases),
    m84_valid_metric_positions=sum(aggregates[a]['valid_frames'] for a in arms),
    additional_M82_raw_frames=sum(c['frames'] for c in cases),
    gt_frames=sum(c['frames'] for c in cases),gt_invalid_noninitial=sum(per['category'][s]['invalid_gt_frames'] for s in names),
    per_sequence=per,aggregates=aggregates,H10_intervals_zero_based_half_open=h10,
    gates=gates,gate_count=gate_count,gate_pass_count=gate_pass,native_success_damage=native_damage,
    empty_native_metric_parity=parity,content_leave_one_out=loo,
    loo_positive_counts={a:sum(v>0 for v in d.values()) for a,d in loo.items()},
    loo_nonpositive={a:{s:v for s,v in d.items() if v<=0} for a,d in loo.items()},
    CSV_rows=csv_counts,training_sequence_rows=len(seqrows),training_calls=calls,training_steps=previous_steps,
    saved_damage_diagnostic=dict(sequences=damage_rows,**damage_totals),
    training_label_counts=dict(counts),sampled_training_trace_rows=len(trace),sampled_trace_flags=dict(trace_flags),
    control_identity=shared_identity,M82_M84_changed_integration_sources=source_diff,
    exit_codes=exit_codes,logs=logs,archive_native_candidates=archive_native_candidates,
    checkpoint_pickle_globals=checkpoint_pickle_globals,checkpoint=checkpoint_report,
    native_raw_independent_recursion_available=True,native_raw_parity=native_parity,
    native_reference_boundary='All 22 locally extracted raw independent native sequences independently recomputed and compared with Empty: all 33130 bbox arrays and 33108 noninitial scores exact. The extraction source and local manifests verified. Original two remote shards and their original runtime/checkpoint bytes were not available to this reviewer; original shard hashes remain extraction-run provenance assertions, not locally rehashed source bytes. M83 same-state native readouts were not used.',
    input_hashes={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and not p.name.startswith('M84_COMPLETED_AUDIT')},
)
(OUT/'independent_recompute.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['hash_bindings','per_sequence','H10_intervals_zero_based_half_open','input_hashes','content_leave_one_out','empty_native_metric_parity','logs','sampled_trace_flags','max_absolute_numeric_differences','saved_damage_diagnostic','native_raw_parity']},indent=2))
