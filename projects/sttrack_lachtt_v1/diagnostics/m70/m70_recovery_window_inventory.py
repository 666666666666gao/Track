"""Train-only recovery observation inventory; geometry is not candidate recognition."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib,importlib.util,json,math
from pathlib import Path

BASE=Path('/root/autodl-tmp');PARENT=BASE/'sttrack_m65_category_null_support_20260907'
ROOT=BASE/'sttrack_m70_recovery_window_inventory_20260907'
AUDITOR=BASE/'audit_m65_completed_20260907.py'
AUDITOR_SHA='ef6a3f5f9f7b8635497fc993fded0924e7f37f5b67d14c3fab46fc27b7e596db'
SCALAR=BASE/'m63_evidence_audit_20260907.py'
SCALAR_SHA='2c611c8dae3c226c5c625301a98ff1f74f4ff03e727c5599060dea1650cac66d'
AUDIT_SHA='26a79a424ca8b0dec01ff3c12f8cda6653b4a156c12e5c3b297112f52cf69f88'
TRAIN_SHA='fc04a897f3d3e246203981a2cb3b83ea50075392e30c0c960c8908eaaeabb93b'
RECURSIVE_SHA='09f3f193de81f9cf91518b2c05497bfea30e8ab88db812e3585f6d3618767723'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def module(name,p):
    s=importlib.util.spec_from_file_location(name,str(p));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def parents():
    for p,h in [(AUDITOR,AUDITOR_SHA),(SCALAR,SCALAR_SHA),(PARENT/'completed_evidence_audit.json',AUDIT_SHA),
                (PARENT/'training_spec.json',TRAIN_SHA),(PARENT/'recursive_spec.json',RECURSIVE_SHA)]:assert sha(p)==h
    a=read(PARENT/'completed_evidence_audit.json');assert not a['paired_development_gate_pass']
    assert sha(PARENT/'recursive_result.json')==a['result_sha256']
    t=read(PARENT/'training_spec.json');r=read(PARENT/'recursive_spec.json')
    source=PARENT/'code/lib/train/data/processing_utils.py'
    integration=read(PARENT/'integration.json')
    assert sha(source)==integration['source_sha256']['lib/train/data/processing_utils.py']
    return t,r,sha(source)

def prepare():
    t,r,crop_sha=parents();ROOT.mkdir()
    s=dict(status='frozen_before_all22_recovery_geometry_census',observed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__),completed_M65_audit_sha256=AUDIT_SHA,training_spec_sha256=TRAIN_SHA,
        recursive_spec_sha256=RECURSIVE_SHA,crop_source_sha256=crop_sha,cases=r['cases'],primary='M65 Null',
        references=['M65 Control','native STTrack'],dataset_root=t['dataset_root'],
        strata=dict(H10='Every M65 Null continuous-IoU<=0.1 run lasting at least10 valid frames; invalid GT breaks runs. Take its first frame.',
            valid_after_invalid='Every first valid GT frame after a consecutive invalid GT run of at least10 frames. No absence or reappearance label is inferred.',
            healthy='Earliest start within each of four equal temporal quarters with Null AND Control IoU>=0.5 at the prior frame and the next10-frame window, all GT valid. At most4 per sequence.'),
        GT_scope='GT used only after complete historical predictions are sealed, for offline diagnostic selection and geometry. Healthy selection uses future labels and is not a causal deployment trigger.',
        geometry=dict(local_factor=4,wide_factor=7,centre_rule='Native ceil(sqrt(w*h)*factor), round(previous centre-side/2). Nominal square; image intersection reported separately.',
            dense_observation_grid='Same side length as factor4 crop, 50 percent overlap via max(1,floor(side/2)); include image edges. Centres derived only from image size and previous predicted scale.',
            grid_interpretation='Window counts and geometric coverage only. No image features, decoded candidate boxes or recognized target are produced.'),
        new_tracking_calls=0,new_optimizer_steps=0,public_evaluation_allowed=False,
        M65_failed_gate_unchanged=True,M67_training_unchanged=True,independent_model_review_pass=False)
    write(ROOT/'spec.json',s);print(json.dumps(dict(spec_sha256=sha(ROOT/'spec.json'),source_sha256=sha(__file__))))

def crop(previous,factor):
    x,y,w,h=map(float,previous);side=math.ceil(math.sqrt(w*h)*factor)
    return [round(x+.5*w-.5*side),round(y+.5*h-.5*side),side,side]

def coverage(rect,g):
    x,y,w,h=rect;gx,gy,gw,gh=map(float,g);cx,cy=gx+gw/2,gy+gh/2
    return dict(centre_inside=bool(x<=cx<x+w and y<=cy<y+h),full_box_inside=bool(x<=gx and y<=gy and gx+gw<=x+w and gy+gh<=y+h))

def axis_starts(length,side):
    if side>=length:return [round((length-side)/2)]
    starts=list(range(0,length-side+1,max(1,side//2)))
    if starts[-1]!=length-side:starts.append(length-side)
    return starts

def grid_geometry(previous,g,width,height):
    side=crop(previous,4)[2];xs=axis_starts(width,side);ys=axis_starts(height,side)
    gx,gy,gw,gh=map(float,g);cx,cy=gx+gw/2,gy+gh/2
    return dict(side=side,x_windows=len(xs),y_windows=len(ys),windows=len(xs)*len(ys),
        centre_covered=bool(any(x<=cx<x+side for x in xs) and any(y<=cy<y+side for y in ys)),
        full_box_covered=bool(any(x<=gx and gx+gw<=x+side for x in xs) and any(y<=gy and gy+gh<=y+side for y in ys)))

def run():
    import numpy as np
    from PIL import Image
    s=read(ROOT/'spec.json');assert s['source_sha256']==sha(__file__)
    t,r,_=parents();audit=module('m70_sealed_audit',AUDITOR);scalar=module('m70_scalar',SCALAR)
    data={};receipts={}
    for arm in ['control','null']:data[arm],receipts[arm]=audit.sealed_family(PARENT,arm,r,t)
    data['native'],receipts['native']=audit.sealed_native(r,t)
    # No GT has been opened above. All three historical families are sealed now.
    events=[];statistics={};valid_after_invalid_count=0
    for case in s['cases']:
        seq=case['sequence'];folder=Path(t['dataset_root'])/seq;gp=folder/'groundtruth.txt';assert sha(gp)==case['gt_sha256']
        gt=np.loadtxt(gp,delimiter=',').reshape(-1,4);assert len(gt)==case['frames']
        valid=np.isfinite(gt).all(1)&(gt[:,2]>0)&(gt[:,3]>0)
        values={};stats={};intervals={}
        for arm in data:values[arm],stats[arm],intervals[arm]=scalar.overlaps_and_statistics(data[arm][seq],gt)
        statistics[seq]=stats;chosen={}
        def add(frame,tag,details):
            entry=chosen.setdefault(frame,dict(sequence=seq,frame=frame,tags=[],selection={}))
            entry['tags'].append(tag);entry['selection'][tag]=details
        for start,end in intervals['null']:add(start,'H10',dict(interval=[start,end],length=end-start))
        invalid_length=0
        for f in range(1,len(gt)):
            if not valid[f]:invalid_length+=1
            else:
                if invalid_length>=10:add(f,'valid_after_invalid',dict(preceding_invalid_frames=invalid_length));valid_after_invalid_count+=1
                invalid_length=0
        for q in range(4):
            begin=max(1,len(gt)*q//4);end=len(gt)*(q+1)//4
            for f in range(begin,min(end,len(gt)-9)):
                if all(bool(np.all(values[a][f-1:f+10]>=.5)) for a in ['null','control']):
                    add(f,'healthy',dict(quarter=q,verified_interval=[f-1,f+10]));break
        for f,e in sorted(chosen.items()):
            with Image.open(folder/'color'/('%08d.jpg'%(f+1))) as im:width,height=im.size
            g=gt[f];assert valid[f]
            e.update(image_width=width,image_height=height,GT_bbox=list(map(float,g)),arms={})
            image_cover=coverage([0,0,width,height],g);e['GT_vs_image']=image_cover
            for arm in data:
                prev=data[arm][seq][f-1]['bbox'];local=crop(prev,4);wide=crop(prev,7)
                e['arms'][arm]=dict(previous_bbox=prev,current_iou=float(values[arm][f]),
                    local=dict(rectangle=local,**coverage(local,g)),wide=dict(rectangle=wide,**coverage(wide,g)))
            e['null_scale_grid']=grid_geometry(data['null'][seq][f-1]['bbox'],g,width,height)
            events.append(e)
    assert sum(x['null']['failure_episodes'] for x in statistics.values())==68
    for arm in data:
        aggregate=read(PARENT/'completed_evidence_audit.json')['recomputed_aggregates'][arm]
        for key in ['valid_frames','low_iou_frames','failure_episodes']:
            assert sum(x[arm][key] for x in statistics.values())==aggregate[key]
    summary={}
    for tag in ['H10','valid_after_invalid','healthy']:
        rows=[e for e in events if tag in e['tags']];counts={}
        for arm in data:
            counts[arm]=dict(local_centre_inside=sum(e['arms'][arm]['local']['centre_inside'] for e in rows),
                local_full_box_inside=sum(e['arms'][arm]['local']['full_box_inside'] for e in rows),
                wide_centre_inside=sum(e['arms'][arm]['wide']['centre_inside'] for e in rows))
        outside=[e for e in rows if not e['arms']['null']['local']['centre_inside']]
        windows=[e['null_scale_grid']['windows'] for e in rows]
        summary[tag]=dict(events=len(rows),sequences=len({e['sequence'] for e in rows}),arms=counts,
            null_local_outside=len(outside),null_outside_wide_covers=sum(e['arms']['null']['wide']['centre_inside'] for e in outside),
            null_outside_grid_centre_covers=sum(e['null_scale_grid']['centre_covered'] for e in outside),
            null_outside_grid_full_box_covers=sum(e['null_scale_grid']['full_box_covered'] for e in outside),
            grid_windows_total=sum(windows),grid_windows_median=float(np.median(windows)),grid_windows_max=max(windows))
    cases=[dict(sequence=e['sequence'],frame=e['frame']) for e in events]
    assert len(cases)==len({(e['sequence'],e['frame']) for e in cases})
    write(ROOT/'events_for_replay.json',dict(scope='GT-selected offline Train diagnostic; coordinates, tags and GT are not provided to the future branch.',cases=cases))
    write(ROOT/'geometry_events.json',dict(source_sha256=sha(__file__),spec_sha256=sha(ROOT/'spec.json'),events=events))
    result=dict(status='completed_all22_historical_recovery_geometry_inventory',observed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__),spec_sha256=sha(ROOT/'spec.json'),receipts=receipts,summary=summary,
        unique_events=len(events),events_sha256=sha(ROOT/'geometry_events.json'),replay_cases_sha256=sha(ROOT/'events_for_replay.json'),
        total_grid_windows=sum(e['null_scale_grid']['windows'] for e in events),statistics=statistics,
        all68_Null_H10_included=True,GT_invalid_not_interpreted_as_absence=True,
        geometry_is_not_recognition=True,new_tracking_calls=0,new_optimizer_steps=0,
        public_evaluation_allowed=False,independent_model_review_pass=False,M65_failed_gate_unchanged=True,M67_training_unchanged=True)
    write(ROOT/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['receipts','statistics']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);a=p.parse_args()
    prepare() if a.action=='prepare' else run()
