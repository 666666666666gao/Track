"""Offline M77 content analysis recovery; original failing controller remains sealed."""
from pathlib import Path
import sys,json,hashlib
from m77_content_followup_20260907 import R,O,AUDITOR,checked,read,sha,module,now,write
ORIGINAL_SHA='58e52b047f428857255e56ebd18a966f81f74fcb787217d5d52eebfdc6453a7e'
Q=O/'analysis_recovery_20260908'
assert sha('/root/autodl-tmp/m77_content_followup_20260907.py')==ORIGINAL_SHA
assert (O/'content_analysis.exit').read_text().strip()==(O/'controller.exit').read_text().strip()=='1'
assert not (Q/'result.json').exists()

def analyze():
    import numpy as np
    s,t,rs,a,audit=checked();data={};receipts={}
    for name in ['category','empty','swapped']:
        if name=='category':directory=R/'recursive/category';rp=R/'category_recursive_receipt.json'
        else:
            assert (O/(name+'.exit')).read_text().strip()=='0';directory=O/name;rp=directory/'receipt.json'
        receipt=read(rp);assert receipt['status']=='complete' and receipt['head_sha256']==a['head_sha256'] and receipt['total_frames']==33130
        assert [v['sequence'] for v in receipt['sequences']]==[v['sequence'] for v in s['cases']]
        if name!='category':
            assert receipt['variant']==name and not receipt['prefix_only'] and not receipt['subsequent_GT_opened']
            assert receipt['spec_sha256']==sha(O/'spec.json') and receipt['bank_sha256']==s['banks'][name]['sha256']
            assert receipt['activation_sha256']==sha(O/'activation.json')
        data[name]={}
        for case,v in zip(s['cases'],receipt['sequences']):
            p=directory/(case['sequence']+'.json');assert sha(p)==v['sha256'];x=read(p);rows=x['rows']
            assert x['arm']==name and x['sequence']==case['sequence']
            assert len(rows)==case['frames'] and [r['frame'] for r in rows]==list(range(case['frames']))
            assert rows[0]==dict(frame=0,bbox=case['init_bbox'],score=None)
            boxes=np.asarray([r['bbox'] for r in rows]);scores=np.asarray([r['score'] for r in rows[1:]])
            assert np.isfinite(boxes).all() and (boxes[:,2:]>0).all() and np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
            data[name][case['sequence']]=rows
        receipts[name]=sha(rp)
    # Open later-frame metric GT only after every prediction family is fully sealed.
    sys.path.insert(0,str(R));from recursive_metric import statistics
    helper=module('m77_audit_for_content',AUDITOR);_,independent=helper.helpers();scalar=independent.scalar_metric()
    per={n:{} for n in data};writes={n:0 for n in data};overlaps={n:{} for n in data};segments={}
    for case in s['cases']:
        seq=case['sequence'];p=Path(t['dataset_root'])/seq/'groundtruth.txt';assert sha(p)==case['gt_sha256'];gt=np.loadtxt(p,delimiter=',').reshape(-1,4);assert len(gt)==case['frames']
        for n in data:
            rows=data[n][seq];boxes=np.asarray([r['bbox'] for r in rows]);v=statistics(boxes,gt);iou,recomputed,spans=scalar(rows,gt)
            for k in v:assert abs(v[k]-recomputed[k])<1e-8,(n,seq,k)
            per[n][seq]=v;overlaps[n][seq]=iou
            if n=='category':segments[seq]=spans
            writes[n]+=sum(r['frame']%50==0 and r['score']>.75 for r in rows[1:])
    aggregates={}
    for n,values in per.items():
        v={k:sum(r[k] for r in values.values()) for k in ['valid_frames','iou_sum','low_iou_frames','failure_episodes']}
        v.update(mean_iou=v['iou_sum']/v['valid_frames'],macro_sequence_mean_iou=float(np.mean([r['mean_iou'] for r in values.values()])))
        assert v['valid_frames']==28897;aggregates[n]=v
    for k,v in aggregates['category'].items():assert abs(v-audit['aggregates']['category'][k])<1e-8
    c=aggregates['category'];criteria={};harms=[]
    for n in ['empty','swapped']:
        v=aggregates[n];criteria[n]=dict(pooled=c['mean_iou']>=v['mean_iou']+.001,macro=c['macro_sequence_mean_iou']>=v['macro_sequence_mean_iou'],low=c['low_iou_frames']<=v['low_iou_frames'],H10=c['failure_episodes']<=v['failure_episodes'])
        for seq,spans in segments.items():
            for start,end in spans:
                if bool((overlaps[n][seq][start:end]>=.5).all()):harms.append(dict(reference=n,sequence=seq,start=int(start),end_exclusive=int(end),frames=int(end-start)))
    result=dict(status='completed_M77_fixed_Category_head_content_diagnostic',observed_utc=now(),source_sha256=sha(__file__),spec_sha256=sha(O/'spec.json'),activation_sha256=sha(O/'activation.json'),head_sha256=a['head_sha256'],
        receipts=receipts,aggregates=aggregates,per_sequence=per,reconstructed_template_writes=writes,descriptive_criteria=criteria,descriptive_criteria_pass=all(v for x in criteria.values() for v in x.values()),
        development_gate_pass=audit['development_gates_pass'],strict_category_H10_reference_correct_every_frame=harms,independent_scalar_recomputation=True,
        seed=2027,new_full_tracking_calls=66216,new_optimizer_steps=0,additional_seeds=[],public_evaluation_allowed=False,independent_model_review_pass=False,
        scope='One fixed Category head, reused Train development22, original/empty/swapped word content. Automatic category truth and same-class identity are not established. Failed development conditions remain failed regardless of content diagnostics.')
    result.update(original_source_sha256=ORIGINAL_SHA, original_analysis_exit=1, recovery_scope='Only scalar input corrected from bbox array to frame records; no tracking, optimization or prediction changes', new_full_tracking_calls=0, reused_control_tracking_calls=66216, original_controller_exit_preserved=True)
    write(Q/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['per_sequence','strict_category_H10_reference_correct_every_frame']},indent=2))

analyze()
