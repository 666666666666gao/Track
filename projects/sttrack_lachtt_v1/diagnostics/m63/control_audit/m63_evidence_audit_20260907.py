"""Audit saved M63 controls and, after completion, all factorial outcomes."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

BASE=Path('/root/autodl-tmp')
ROOT=BASE/'sttrack_m63_location_write_factorial_20260907'
PARENT=BASE/'sttrack_m58_semantic_spatial_v2_20260906'
SOURCE=BASE/'m63_location_write_factorial_20260907.py'
SOURCE_SHA='866e213132efa4e94538b0547afc98624992859126c8b7ea45d3e9b3241bf44e'
SPEC_SHA='903c48d5efc83f727b3e5553cedd3063613f379f4e679f74260e61e1654a4132'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def source():
    assert sha(SOURCE)==SOURCE_SHA and sha(ROOT/'spec.json')==SPEC_SHA
    s=importlib.util.spec_from_file_location('m63_verified_source',str(SOURCE))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    _,spec,training=m.checked()
    return m,spec,training


def family(m,spec,mode,prefix=False):
    name=('prefix_' if prefix else 'full_')+mode
    assert (ROOT/(name+'.exit')).read_text().strip()=='0'
    receipt_path=ROOT/name/'receipt.json';receipt=json.loads(receipt_path.read_text())
    assert receipt['status']=='complete' and receipt['mode']==mode and receipt['prefix_only']==prefix
    assert receipt['spec_sha256']==SPEC_SHA and receipt['source_sha256']==SOURCE_SHA
    assert not receipt['subsequent_GT_opened'] and receipt['optimizer_steps']==receipt['new_captions']==0
    cases=[c for c in spec['cases'] if not prefix or c['sequence'] in spec['prefix_sequences']]
    assert [r['sequence'] for r in receipt['sequences']]==[c['sequence'] for c in cases]
    ref_root=m.M59 if mode[0]=='C' else m.M60
    ref_receipt=ref_root/('category_receipt.json' if mode[0]=='C' else 'receipt.json')
    assert sha(ref_receipt)==spec['reference_receipt_sha256'][mode[0]]
    references={r['sequence']:r for r in json.loads(ref_receipt.read_text())['sequences']}
    data={};checked={}
    for case,item in zip(cases,receipt['sequences']):
        seq=case['sequence'];count=spec['prefix_frames'] if prefix else case['frames']
        path=ROOT/name/(seq+'.json');assert sha(path)==item['sha256'] and item['frames']==count
        saved=json.loads(path.read_text());rows=saved['rows']
        assert saved['mode']==mode and saved['sequence']==seq and len(rows)==count
        assert [r['frame'] for r in rows]==list(range(count))
        assert rows[0]==dict(frame=0,bbox=case['init_bbox'],score=None,write_score=None,template_write=False)
        boxes=np.asarray([r['bbox'] for r in rows]);scores=np.asarray([r['score'] for r in rows[1:]])
        assert np.isfinite(boxes).all() and (boxes[:,2:]>0).all() and np.isfinite(scores).all()
        ref_path=ref_root/('predictions/category' if mode[0]=='C' else 'predictions')/(seq+'.json')
        assert sha(ref_path)==references[seq]['sha256']
        ref=json.loads(ref_path.read_text())['rows'];assert len(ref)==case['frames']
        first=None;max_box=max_score=0.;writes=0
        for row in rows[1:]:
            f=row['frame'];scheduled=f%50==0
            if scheduled:
                assert np.isfinite(row['write_score'])
                assert row['template_write']==(row['write_score']>.75)
                if mode[0]==mode[1]:assert row['write_score']==row['score']
            else:assert row['write_score'] is None and not row['template_write']
            writes+=row['template_write']
            if first is None:
                max_box=max(max_box,float(np.abs(np.asarray(row['bbox'])-ref[f]['bbox']).max()))
                max_score=max(max_score,abs(row['score']-ref[f]['score']))
                default_write=scheduled and ref[f]['score']>.75
                if row['template_write']!=default_write:first=f
        assert max_box<=1e-4 and max_score<=1e-6
        assert first==item['first_write_branch_frame'] and writes==item['template_writes']
        assert abs(max_box-item['max_pre_branch_bbox_error'])<1e-12
        assert abs(max_score-item['max_pre_branch_score_error'])<1e-12
        if mode[0]==mode[1]:assert first is None
        checked[seq]=dict(frames=count,sha256=sha(path),template_writes=writes,first_write_branch_frame=first,
            max_pre_branch_bbox_error=max_box,max_pre_branch_score_error=max_score)
        data[seq]=rows
    assert receipt['frames']==sum(r['frames'] for r in checked.values())
    assert receipt['all_frame_reference_parity']==(mode[0]==mode[1])
    return data,dict(receipt_sha256=sha(receipt_path),frames=receipt['frames'],track_calls=receipt['frames']-len(cases),
        sequences=checked,template_writes=sum(r['template_writes'] for r in checked.values()))


def controls():
    m,spec,_=source();checked={}
    for mode,prefix in [('CC',False),('SS',False),('CS',True),('SC',True)]:
        _,report=family(m,spec,mode,prefix);checked[('prefix_' if prefix else 'full_')+mode]=report
    report=dict(status='sealed_controls_and_crossed_prefixes_audited',observed_utc=datetime.now(timezone.utc).isoformat(),
        auditor_sha256=sha(__file__),source_sha256=SOURCE_SHA,spec_sha256=SPEC_SHA,families=checked,
        full_control_images=66260,crossed_prefix_images=1212,new_GT_files_opened=False,new_tracking_calls=0,
        incomplete_crossed_full_predictions_read=False,independent_model_review_pass=False,
        scope='Executor checks of sealed prediction files, write rules and pre-branch reference equality only; no new performance analysis.')
    write(ROOT/'control_evidence_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='families'},indent=2))


def overlaps_and_statistics(rows,gt):
    values=np.full(len(rows),np.nan)
    for f in range(1,len(rows)):
        g=gt[f]
        if not (np.isfinite(g).all() and g[2]>0 and g[3]>0):continue
        x,y,w,h=map(float,rows[f]['bbox']);gx,gy,gw,gh=map(float,g)
        intersection=max(0.,min(x+w,gx+gw)-max(x,gx))*max(0.,min(y+h,gy+gh)-max(y,gy))
        values[f]=intersection/(w*h+gw*gh-intersection)
    intervals=[];start=None
    for f in range(len(values)+1):
        low=f<len(values) and values[f]<=.1
        if low and start is None:start=f
        if not low and start is not None:
            if f-start>=10:intervals.append([start,f])
            start=None
    valid=np.isfinite(values)
    stats=dict(valid_frames=int(valid.sum()),iou_sum=float(values[valid].sum()),mean_iou=float(values[valid].mean()),
        low_iou_frames=int((values<=.1).sum()),failure_episodes=len(intervals),invalid_gt_frames=int((~valid[1:]).sum()))
    return values,stats,intervals


def completed():
    m,spec,training=source()
    prior=json.loads((ROOT/'control_evidence_audit.json').read_text());assert prior['auditor_sha256']==sha(__file__)
    assert (ROOT/'job.exit').read_text().strip()==(ROOT/'analysis.exit').read_text().strip()=='0'
    result=json.loads((ROOT/'result.json').read_text())
    assert result['status']=='completed_fixed_weight_location_write_factorial' and result['spec_sha256']==SPEC_SHA
    data={};checks={}
    for mode in m.MODES:
        data[mode],checks[mode]=family(m,spec,mode)
        assert checks[mode]['receipt_sha256']==result['receipts'][mode]
    # No new GT file opens until all four complete families have verified.
    per={mode:{} for mode in m.MODES};overlaps={mode:{} for mode in m.MODES};intervals={mode:{} for mode in m.MODES}
    for case in spec['cases']:
        seq=case['sequence'];path=Path(training['dataset_root'])/seq/'groundtruth.txt'
        assert sha(path)==case['gt_sha256']
        gt=np.loadtxt(path,delimiter=',').reshape(-1,4);assert len(gt)==case['frames']
        for mode in m.MODES:
            v,s,e=overlaps_and_statistics(data[mode][seq],gt)
            for k,x in s.items():assert abs(x-result['per_sequence'][mode][seq][k])<1e-8,(mode,seq,k)
            overlaps[mode][seq]=v;per[mode][seq]=s;intervals[mode][seq]=e
    aggregates={}
    for mode in m.MODES:
        rows=list(per[mode].values())
        a={k:sum(r[k] for r in rows) for k in ['valid_frames','iou_sum','low_iou_frames','failure_episodes']}
        a.update(mean_iou=a['iou_sum']/a['valid_frames'],macro_sequence_mean_iou=float(np.mean([r['mean_iou'] for r in rows])),sequences=len(rows))
        for k,v in a.items():assert abs(v-result['aggregates'][mode][k])<1e-8,(mode,k)
        assert checks[mode]['template_writes']==result['template_writes'][mode]
        aggregates[mode]=a
    for name,a,b in [('localization_given_C_write','CC','SC'),('localization_given_S_write','CS','SS'),
        ('write_given_C_localization','CC','CS'),('write_given_S_localization','SC','SS')]:
        for k,v in result['contrasts'][name].items():assert abs(aggregates[a][k]-aggregates[b][k]-v)<1e-10
    parent_path=PARENT/'recursive_result.json';parent=json.loads(parent_path.read_text())
    assert sha(parent_path)=='54fd8445297e1eef761fa36e8a17def01afa027f71371984998f08229f6f5ee9'
    native_protected=[s for s,r in parent['per_sequence']['native'].items() if r['failure_episodes']==0]
    category_protected=[s for s,r in per['CC'].items() if r['failure_episodes']==0]
    assert category_protected==result['category_zero_H10_sequences']
    sustained=[]
    for mode,ref_mode in [('CS','CC'),('SC','SS')]:
        for seq in per[mode]:
            branch=checks[mode]['sequences'][seq]['first_write_branch_frame']
            for begin,end in intervals[mode][seq]:
                if np.all(overlaps[ref_mode][seq][begin:end]>=.5):
                    assert branch is not None and begin>branch
                    sustained.append(dict(mode=mode,reference_mode=ref_mode,sequence=seq,start=begin,end_exclusive=end,
                        frames=end-begin,first_write_branch_frame=branch,frames_after_first_write_branch=begin-branch))
    report=dict(status='complete_saved_factorial_evidence_audit',observed_utc=datetime.now(timezone.utc).isoformat(),
        auditor_sha256=sha(__file__),source_sha256=SOURCE_SHA,spec_sha256=SPEC_SHA,result_sha256=sha(ROOT/'result.json'),
        control_audit_sha256=sha(ROOT/'control_evidence_audit.json'),parent_result_sha256=sha(parent_path),families=checks,
        recomputed_aggregates=aggregates,all_four_contrasts_recomputed=True,
        native_zero_H10_sequences=native_protected,category_zero_H10_sequences=category_protected,
        new_failures_on_native_success={mode:[s for s in native_protected if per[mode][s]['failure_episodes']>0] for mode in m.MODES},
        new_failures_on_category_success={mode:[s for s in category_protected if per[mode][s]['failure_episodes']>0] for mode in m.MODES},
        sustained_crossed_H10_with_same_localization_reference_correct_every_frame=sustained,
        scalar_continuous_IoU_and_H10_recomputed=True,new_tracking_calls=0,new_caption_calls=0,
        new_GT_opened_only_after_all_four_families_verified=True,public_evaluation_allowed=False,independent_model_review_pass=False,
        scope='DepthTrack Train development evidence; crossed donor category is diagnostic input, not a deployment protocol. No formal three-dataset metric or promotion claim.')
    write(ROOT/'completed_evidence_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='families'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['controls','completed']);a=p.parse_args()
    {'controls':controls,'completed':completed}[a.action]()
