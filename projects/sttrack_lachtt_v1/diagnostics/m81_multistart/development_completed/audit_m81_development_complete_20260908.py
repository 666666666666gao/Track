"""Independent scalar M81 metrics; all nine families must be sealed first."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, math

R=Path('/root/autodl-tmp/sttrack_m81_multistart_20260908')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())

def scalar_statistics(rows, gt):
    values=[];low=count=invalid=run=0
    for i,(row,g) in enumerate(zip(rows,gt)):
        if i==0:continue
        valid=all(math.isfinite(x) for x in g) and g[2]>0 and g[3]>0
        if not valid:
            invalid+=1;count+=run>=10;run=0;continue
        b=row['bbox']
        w=max(0.,min(b[0]+b[2],g[0]+g[2])-max(b[0],g[0]))
        h=max(0.,min(b[1]+b[3],g[1]+g[3])-max(b[1],g[1]))
        inter=w*h;v=inter/(b[2]*b[3]+g[2]*g[3]-inter);values.append(v)
        if v<=.1:low+=1;run+=1
        else:count+=run>=10;run=0
    count+=run>=10
    assert values
    return dict(valid_frames=len(values),iou_sum=math.fsum(values),mean_iou=math.fsum(values)/len(values),
                low_iou_frames=low,failure_episodes=count,invalid_gt_frames=invalid)

def compare(actual, expected):
    assert set(actual)==set(expected)
    for key,value in actual.items():
        assert math.isclose(value,expected[key],rel_tol=0,abs_tol=1e-8),(key,value,expected[key])

def main():
    out=R/'saved_development_audit.json';assert not out.exists()
    t=read(R/'training_spec.json');e=read(R/'evaluation_spec.json');f=read(R/'frozen.json');result=read(R/'result.json')
    assert result['status']=='complete_M81_t0_and_multistart_development'
    assert sha(R/'training_spec.json')==f['training_spec_sha256']==result['training_spec_sha256']
    assert sha(R/'evaluation_spec.json')==f['evaluation_spec_sha256']==result['evaluation_spec_sha256']
    assert sha(R/'development_initializations.json')==t['development_initializations_sha256']
    assert sha(R/'evaluate.py')==e['evaluator_sha256'] and sha(R/'recursive_metric.py')==e['metric_sha256']
    families=['t0_category','t0_empty_trained','t0_empty_content','t0_swapped','multi_category','multi_empty_trained','multi_native','multi_M78_category','multi_M78_empty_trained']
    assert set(e['families'])==set(families) and len(e['families'])==9
    for name in ['training_category','training_empty','analysis','controller']+['eval_'+name for name in families]:
        assert (R/(name+'.exit')).read_text().strip()=='0'
    heads={}
    for arm in ['category','empty']:
        training=read(R/'training'/arm/'result.json')
        assert training['total_track_calls']==186694 and training['optimizer_steps']==5798 and training['initializations']==426
        heads[arm]=sha(R/'training'/arm/'final.pth');assert heads[arm]==training['final_checkpoint_sha256']
        assert sha(e['parent_heads'][arm]['path'])==e['parent_heads'][arm]['sha256']
    assert sha(t['native_checkpoint'])==t['native_checkpoint_sha256']
    cases={c['sequence']:c for c in e['cases']};assert len(cases)==22
    multi=read(R/'development_initializations.json')['episodes'];assert len(multi)==74
    t0=[dict(id=c['sequence']+':0',sequence=c['sequence'],start_frame=0,end_frame_inclusive=c['frames']-1,init_bbox=c['init_bbox']) for c in e['cases']]
    sealed={};plans={};position_count=episode_count=0
    for name in families:
        is_multi=name.startswith('multi_');eps=multi if is_multi else t0;plans[name]=eps
        empty=name.endswith('empty_trained') or name.endswith('empty_content');condition='empty' if empty else 'category'
        bank=e['multi_banks'][condition] if is_multi else t['banks']['development'][condition]
        if name=='t0_swapped':bank=e['swapped_bank']
        head=t['native_checkpoint_sha256'] if name=='multi_native' else e['parent_heads'][condition]['sha256'] if '_M78_' in name else heads['empty' if name.endswith('empty_trained') else 'category']
        receipt_path=R/('receipt_'+name+'.json');receipt=read(receipt_path)
        assert sha(receipt_path)==result['receipt_sha256'][name]
        assert receipt['status']=='complete' and receipt['family']==name
        assert receipt['evaluation_spec_sha256']==sha(R/'evaluation_spec.json') and receipt['head_sha256']==head
        assert receipt['bank_sha256']==bank['sha256']==sha(bank['path'])
        assert receipt['metric_gt_opened'] is False and receipt['text_updated_online'] is False
        assert receipt['total_track_calls']==33108 and receipt['total_frames']==33108+len(eps)
        assert len(receipt['episodes'])==len(eps)
        sealed[name]={};coverage={seq:[] for seq in cases}
        for ep,entry in zip(eps,receipt['episodes']):
            filename=hashlib.sha256(ep['id'].encode()).hexdigest()+'.json'
            assert entry['id']==ep['id'] and entry['sequence']==ep['sequence'] and entry['path']==filename
            p=R/'recursive'/name/filename;assert sha(p)==entry['sha256'];data=read(p);rows=data['rows']
            assert data['id']==ep['id'] and data['family']==name and data['sequence']==ep['sequence']
            assert len(rows)==entry['frames']==ep['end_frame_inclusive']-ep['start_frame']+1
            assert entry['track_calls']==len(rows)-1
            assert rows[0]['bbox']==ep['init_bbox'] and rows[0]['score'] is None
            for i,row in enumerate(rows):
                assert row['frame']==ep['start_frame']+i
                b=row['bbox'];assert len(b)==4 and all(math.isfinite(x) for x in b) and b[2]>0 and b[3]>0
                if i:assert math.isfinite(row['score'])
            coverage[ep['sequence']].extend(row['frame'] for row in rows[1:])
            assert ep['id'] not in sealed[name];sealed[name][ep['id']]=rows
            position_count+=len(rows);episode_count+=1
        for seq,indices in coverage.items():assert indices==list(range(1,cases[seq]['frames'])),(name,seq)
    assert position_count==298430 and episode_count==458

    # No metric GT is read until every output family above is complete and verified.
    gt={}
    for seq,c in cases.items():
        p=Path(t['dataset_root'])/seq/'groundtruth.txt';assert sha(p)==c['gt_sha256']
        gt[seq]=[[float(v) for v in line.split(',')] for line in p.read_text().splitlines() if line.strip()]
        assert len(gt[seq])==c['frames']
    per={};aggregates={};keys=['valid_frames','iou_sum','low_iou_frames','failure_episodes']
    for name in families:
        per[name]={}
        for ep in plans[name]:
            stat=scalar_statistics(sealed[name][ep['id']],gt[ep['sequence']][ep['start_frame']:ep['end_frame_inclusive']+1])
            compare(stat,result['episode_metrics'][name][ep['id']])
            seq=per[name].setdefault(ep['sequence'],{k:0 for k in keys})
            for k in keys:seq[k]+=stat[k]
        for seq,s in per[name].items():
            s['mean_iou']=s['iou_sum']/s['valid_frames'];compare(s,result['per_sequence'][name][seq])
        a={k:sum(s[k] for s in per[name].values()) for k in keys}
        assert a['valid_frames']==28897
        a.update(mean_iou=a['iou_sum']/a['valid_frames'],macro_sequence_mean_iou=math.fsum(s['mean_iou'] for s in per[name].values())/22)
        compare(a,result['aggregates'][name]);aggregates[name]=a
    assert sha(e['native_result_path'])==e['native_result_sha256'] and sha(e['parent_result_path'])==e['parent_result_sha256']
    native=read(e['native_result_path']);parent=read(e['parent_result_path'])
    assert result['per_sequence']['t0_native']==native['per_sequence']['native']
    assert result['aggregates']['t0_native']==native['aggregates']['native'] and result['reference_M78_t0']==parent['aggregates']
    aggregates['t0_native']=native['aggregates']['native'];per['t0_native']=native['per_sequence']['native']
    gates={};broken={}
    for protocol in ['t0','multi']:
        a=aggregates[protocol+'_category']
        for ref,margin in [(protocol+'_native',.002),(protocol+'_empty_trained',.001)]:
            b=aggregates[ref];key=protocol+'_vs_'+ref
            broken[ref]=[seq for seq in per[ref] if per[ref][seq]['failure_episodes']==0 and per[protocol+'_category'][seq]['failure_episodes']>0]
            gates.update({key+'_pooled':a['mean_iou']>=b['mean_iou']+margin,key+'_macro':a['macro_sequence_mean_iou']>=b['macro_sequence_mean_iou'],key+'_low':a['low_iou_frames']<=b['low_iou_frames'],key+'_H10':a['failure_episodes']<=b['failure_episodes'],key+'_protect':not broken[ref]})
    assert gates==result['primary_gates'] and broken==result['broken_success_sequences']
    content={};a=aggregates['t0_category']
    for ref in ['t0_empty_content','t0_swapped']:
        b=aggregates[ref]
        content.update({ref+'_pooled':a['mean_iou']>b['mean_iou'],ref+'_macro':a['macro_sequence_mean_iou']>b['macro_sequence_mean_iou'],ref+'_low':a['low_iou_frames']<=b['low_iou_frames'],ref+'_H10':a['failure_episodes']<=b['failure_episodes']})
    assert content==result['content_gates']
    a=aggregates['multi_category'];b=aggregates['multi_M78_category']
    coverage=dict(pooled=a['mean_iou']>=b['mean_iou']+.001,macro=a['macro_sequence_mean_iou']>=b['macro_sequence_mean_iou'],low=a['low_iou_frames']<=b['low_iou_frames'],H10=a['failure_episodes']<=b['failure_episodes'])
    assert coverage==result['initialization_coverage_gates']
    for flag,g in [('primary_pass',gates),('content_pass',content),('coverage_pass',coverage)]:assert result[flag]==all(g.values())
    assert result['all_checks_pass']==all(result[k] for k in ['primary_pass','content_pass','coverage_pass'])
    audit=dict(status='complete_independent_scalar_M81_metric_and_receipt_verification',observed_utc=datetime.now(timezone.utc).isoformat(),
               source_sha256=sha(__file__),result_sha256=sha(R/'result.json'),families=9,episodes=episode_count,positions=position_count,
               valid_positions_per_family=28897,noninitialization_frame_coverage_exactly_once=True,aggregates=aggregates,
               primary_pass_count=sum(gates.values()),content_pass_count=sum(content.values()),coverage_pass_count=sum(coverage.values()),
               independent_model_review_pass=False,gpu_inference_calls=0,scope='Reused DepthTrack Train development22; t0 and multistart separate; common H10 reset boundaries')
    out.write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n');print(json.dumps(audit))

if __name__=='__main__':main()
