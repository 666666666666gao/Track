"""Post-sealing Train diagnosis; no GT participates in replay or candidates."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,math
import numpy as np

BASE=Path('/root/autodl-tmp');ROOT=BASE/'sttrack_m66_same_state_diagnostic_20260907'
PARENT=BASE/'sttrack_m65_category_null_support_20260907'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def many_iou(boxes,gt):
    b=np.asarray(boxes,dtype=np.float64);g=np.asarray(gt,dtype=np.float64)
    inter=np.maximum(0.,np.minimum(b[:,0]+b[:,2],g[0]+g[2])-np.maximum(b[:,0],g[0]))*np.maximum(0.,np.minimum(b[:,1]+b[:,3],g[1]+g[3])-np.maximum(b[:,1],g[1]))
    return inter/(b[:,2]*b[:,3]+g[2]*g[3]-inter)

def main():
    spec=read(ROOT/'spec.json');assert sha(ROOT/'spec.json')=='762f9397c324b11a5cace2c622c4d286a70d205542d27a6820778029c1bd4e19'
    assert sha(BASE/'m66_same_state_diagnostic_20260907.py')==spec['runner_sha256']
    assert sha(PARENT/'recursive_result.json')==spec['parent_result_sha256']
    train=read(PARENT/'training_spec.json');data={};receipts={}
    for arm in ['control','null']:
        assert (ROOT/(arm+'.exit')).read_text().strip()=='0'
        r=read(ROOT/(arm+'_receipt.json'))
        assert r['status']=='complete_exact_replay_and_noncommitting_head_probes' and r['all_public_boxes_and_scores_exact'] and r['probe_state_unchanged']
        assert r['spec_sha256']==sha(ROOT/'spec.json') and r['head_sha256']==spec['head_sha256'][arm]
        assert not r['subsequent_gt_opened'] and not r['new_text_content_controls']
        assert [x['sequence'] for x in r['sequences']]==[x['sequence'] for x in spec['cases']]
        data[arm]={};receipts[arm]=sha(ROOT/(arm+'_receipt.json'))
        for row,case in zip(r['sequences'],spec['cases']):
            p=ROOT/arm/(row['sequence']+'.json');assert sha(p)==row['sha256']
            x=read(p);assert x['all_public_boxes_and_scores_exact'] and x['image_frames']==case['frames']
            assert len(x['geometry'])==case['frames']-1 and len(x['probes'])==row['probe_frames']
            assert [g['frame'] for g in x['geometry']]==list(range(1,case['frames']))
            original=PARENT/'recursive'/arm/(case['sequence']+'.json');assert sha(original)==x['reference_sha256']
            previous=read(original)['rows']
            assert all(g['bbox']==previous[g['frame']]['bbox'] and g['score']==previous[g['frame']]['score'] and g['previous_bbox']==previous[g['frame']-1]['bbox'] for g in x['geometry'])
            data[arm][case['sequence']]=x
    assert (ROOT/'controller.exit').read_text().strip()=='0'
    # Both complete public replays and candidate files are sealed before subsequent GT.
    mpath=BASE/'m63_evidence_audit_20260907.py'
    assert sha(mpath)=='2c611c8dae3c226c5c625301a98ff1f74f4ff03e727c5599060dea1650cac66d'
    s=importlib.util.spec_from_file_location('metric',str(mpath));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    parent=read(PARENT/'recursive_result.json');summary={};evaluated={};onsets=[]
    for case in spec['cases']:
        seq=case['sequence'];p=Path(train['dataset_root'])/seq/'groundtruth.txt';assert sha(p)==case['gt_sha256']
        gt=np.loadtxt(p,delimiter=',').reshape(-1,4);assert len(gt)==case['frames']
        valid=np.isfinite(gt).all(1)&(gt[:,2:]>0).all(1)
        summary[seq]={};evaluated[seq]={}
        for arm in ['control','null']:
            x=data[arm][seq];rows=[dict(frame=0,bbox=case['init_bbox'])]+[dict(frame=g['frame'],bbox=g['bbox']) for g in x['geometry']]
            overlap,stats,intervals=m.overlaps_and_statistics(rows,gt)
            for k,v in stats.items():assert abs(v-parent['per_sequence'][arm][seq][k])<1e-8,(arm,seq,k)
            ev=[]
            for rec in x['probes']:
                f=rec['frame'];geom=x['geometry'][f-1];heads=rec['heads']
                base=dict(frame=f,valid_gt=bool(valid[f]),actual_score=geom['score'],scheduled=f%50==0,actual_write=geom['template_write'])
                sem=heads['actual_semantic'];nat=heads['unadapted_head_same_current_features']
                ni=nat['selected_index'];si=sem['selected_index']
                base.update(semantic_selected_index=si,unadapted_selected_index=ni,peak_changed=si!=ni,
                    unadapted_same_semantic_peak_score=nat['hann_scores'][si],unadapted_own_peak_score=nat['hann_scores'][ni],
                    write_flip_same_selected_peak=(f%50==0 and ((nat['hann_scores'][si]>.75)!=geom['template_write'])),
                    zero_slot_mass_at_semantic_peak=rec['zero_slot_softmax_mass'][si],zero_slot_mass_mean=float(np.mean(rec['zero_slot_softmax_mass'])))
                if valid[f]:
                    g=gt[f];crop=geom['search_rectangle'];cx,cy=g[:2]+g[2:]/2
                    centre_in=crop[0]<=cx<crop[0]+crop[2] and crop[1]<=cy<crop[1]+crop[3]
                    full_in=crop[0]<=g[0] and crop[1]<=g[1] and g[0]+g[2]<=crop[0]+crop[2] and g[1]+g[3]<=crop[1]+crop[3]
                    prior=geom['previous_bbox'];sz=crop[2]
                    normx=.5+(cx-(prior[0]+prior[2]/2))/sz;normy=.5+(cy-(prior[1]+prior[3]/2))/sz
                    base.update(gt=g.tolist(),public_iou=float(overlap[f]),search_centre_inside=bool(centre_in),search_full_box_inside=bool(full_in))
                    hs={}
                    for key,h in heads.items():
                        v=many_iou(h['dense_boxes'],g);nv=many_iou([c['bbox'] for c in h['nms']],g)
                        hit=np.flatnonzero(nv>=.5);idx=h['selected_index']
                        hs[key]=dict(selected_iou=float(v[idx]),raw_selected_iou=float(v[h['raw_selected_index']]),
                            dense_best_iou=float(v.max()),top10_best_iou=float(nv.max()),first_correct_rank=int(hit[0])+1 if len(hit) else None)
                    assert abs(hs['actual_semantic']['selected_iou']-overlap[f])<1e-5
                    base['heads']=hs
                    base['semantic_peak_native_regression_iou']=float(many_iou([nat['dense_boxes'][si]],g)[0])
                    base['native_peak_semantic_regression_iou']=float(many_iou([sem['dense_boxes'][ni]],g)[0])
                    if 0<=normx<1 and 0<=normy<1:
                        index=min(15,int(normy*16))*16+min(15,int(normx*16))
                        base['zero_slot_mass_at_gt_centre_cell']=rec['zero_slot_softmax_mass'][index]
                    if valid[f-1]:
                        before=gt[f-1];scale=math.sqrt(before[2]*before[3])
                        base['adjacent_gt_motion_scale']=float(np.linalg.norm(g[:2]+g[2:]/2-before[:2]-before[2:]/2)/scale)
                        base['adjacent_gt_linear_scale_factor']=float(math.sqrt(g[2]*g[3]/(before[2]*before[3])))
                ev.append(base)
            lookup={row['frame']:row for row in ev}
            for start,end in intervals:
                if start in lookup:
                    onset=dict(sequence=seq,arm=arm,start=start,end_exclusive=end,frames=end-start,probe=lookup[start])
                    prev=start-1
                    while prev>=0 and not valid[prev]:prev-=1
                    onset['preceding_invalid_gt_frames']=start-1-prev
                    onset['last_valid_frame_before']=prev
                    onset['last_public_iou_before_gap']=float(overlap[prev]) if prev>=1 else None
                    onset['last_template_write_frame']=max([0]+[g['frame'] for g in x['geometry'][:start-1] if g['template_write']])
                    onsets.append(onset)
            good=[r for r in ev if r['valid_gt']]
            bad=[r for r in good if r['public_iou']<=.1]
            summary[seq][arm]=dict(statistics=stats,H10_intervals=intervals,probe_frames=len(ev),valid_probe_frames=len(good),low_probe_frames=len(bad),
                low_probes_gt_center_inside=sum(r['search_centre_inside'] for r in bad),
                low_probes_semantic_top10_correct=sum(r['heads']['actual_semantic']['top10_best_iou']>=.5 for r in bad),
                low_probes_semantic_dense_correct=sum(r['heads']['actual_semantic']['dense_best_iou']>=.5 for r in bad),
                low_probes_unadapted_top1_correct=sum(r['heads']['unadapted_head_same_current_features']['selected_iou']>=.5 for r in bad),
                good_probes_unadapted_top1_severe=sum(r['public_iou']>=.5 and r['heads']['unadapted_head_same_current_features']['selected_iou']<=.1 for r in good),
                scheduled_write_flips_same_peak=sum(r['write_flip_same_selected_peak'] for r in ev),
                actual_template_writes=sum(g['template_write'] for g in x['geometry']))
            evaluated[seq][arm]=ev
    result=dict(status='complete_post_sealing_same_state_diagnostic',observed_utc=datetime.now(timezone.utc).isoformat(),
        spec_sha256=sha(ROOT/'spec.json'),analyzer_sha256=sha(__file__),receipts=receipts,summary=summary,onsets=onsets,
        new_tracking_calls=0,new_optimizer_steps=0,new_text_content_controls=False,public_evaluation_allowed=False,
        scope='Biased Train development failure diagnosis. Unadapted probe is on the learned path state, not an independent native STTrack trajectory. No new policy or language attribution.',
        zero_slot_mass_scope='Actual normalization mass for Null; hypothetical appended-zero mass for Control, whose real normalization has no zero slot.',
        independent_model_review_pass=False)
    write(ROOT/'result.json',result);write(ROOT/'evaluated_probes.json',evaluated)
    print(json.dumps(dict(status=result['status'],result_sha256=sha(ROOT/'result.json'),summary=summary,onsets=onsets),indent=2),flush=True)

if __name__=='__main__':main()
