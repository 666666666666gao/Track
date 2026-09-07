"""Fixed-state cross-region recognition diagnosis; shadow outputs never control tracking."""
import argparse,copy,hashlib,importlib.util,json,math,sys,time
from datetime import datetime,timezone
from pathlib import Path

BASE=Path('/root/autodl-tmp');PARENT=BASE/'sttrack_m65_category_null_support_20260907'
INVENTORY=BASE/'sttrack_m70_recovery_window_inventory_20260907';ROOT=INVENTORY/'candidate_capacity'
GEOMETRY=BASE/'m70_recovery_window_inventory_20260907.py'
GEOMETRY_SHA='26a1be55df55b83fb4028bb7f86faa33788dd3434debd82f87f4950a514e05d9'
INVENTORY_SHA='8822782394e73154a034e71d783b8eb4e8b27c3302b37ad94e06d9f029b9f17a'
AUDIT_SHA='26a79a424ca8b0dec01ff3c12f8cda6653b4a156c12e5c3b297112f52cf69f88'
HEAD_SHA='5caf9c65c850cdd79d1aee75a002ffd0b74d5876292680349b9387318780e758'

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
    assert sha(GEOMETRY)==GEOMETRY_SHA and sha(INVENTORY/'result.json')==INVENTORY_SHA
    assert sha(PARENT/'completed_evidence_audit.json')==AUDIT_SHA and sha(PARENT/'training/null/final.pth')==HEAD_SHA
    g=module('m70_capacity_geometry',GEOMETRY);t,r,_=g.parents()
    assert sha(INVENTORY/'events_for_replay.json')==read(INVENTORY/'result.json')['replay_cases_sha256']
    assert sha(PARENT/'integration.json')==t['integration_sha256']
    for name,h in read(PARENT/'integration.json')['source_sha256'].items():assert sha(PARENT/'code'/name)==h
    assert sha(t['native_checkpoint'])==t['native_checkpoint_sha256']
    assert sha(PARENT/'text_development.pt')==t['text_development_sha256']
    return g,t,r

def windows(previous,width,height,g):
    local=g.crop(previous,4);wide=g.crop(previous,7)
    out=[dict(kind='local',prior=list(previous),factor=4.,rectangle=local),
         dict(kind='wide',prior=list(previous),factor=7.,rectangle=wide)]
    side=max(width,height);coarse=[round((width-side)/2),round((height-side)/2),side,side]
    out.append(dict(kind='coarse',prior=coarse,factor=1.,rectangle=coarse))
    side=local[2]
    for y in g.axis_starts(height,side):
        for x in g.axis_starts(width,side):
            box=[x,y,side,side];out.append(dict(kind='grid',prior=box,factor=1.,rectangle=box))
    assert all(g.crop(w['prior'],w['factor'])==w['rectangle'] for w in out)
    return out

def prepare():
    g,t,r=parents();ROOT.mkdir()
    events=read(INVENTORY/'events_for_replay.json')['cases'];assert len(events)==216
    assert sha(INVENTORY/'geometry_events.json')==read(INVENTORY/'result.json')['events_sha256']
    geometry=read(INVENTORY/'geometry_events.json')['events'];costs={c['sequence']:c['frames']-1 for c in r['cases']}
    count=0
    for e in geometry:
        ws=windows(e['arms']['null']['previous_bbox'],e['image_width'],e['image_height'],g)
        assert len(ws)-3==e['null_scale_grid']['windows'];count+=len(ws)
        costs[e['sequence']]+=2*len(ws)
    assert count==11555
    queue=ROOT/'run_capacity.sh'
    queue.write_text('''#!/bin/bash
set -u
root=/root/autodl-tmp/sttrack_m70_recovery_window_inventory_20260907/candidate_capacity
python=/root/autodl-tmp/envs/sttrack/bin/python
script=/root/autodl-tmp/m70_candidate_capacity_20260907.py
cd "$root" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CUDA_VISIBLE_DEVICES='' "$python" -u "$script" idle > device_check.log 2>&1
status=$?
printf '%s\\n' "$status" > device_check.exit
if [ "$status" -ne 0 ]; then printf '1\\n' > controller.exit; exit 1; fi
CUDA_VISIBLE_DEVICES=0 "$python" -u "$script" smoke > smoke.log 2>&1
status=$?
printf '%s\\n' "$status" > smoke.exit
if [ "$status" -ne 0 ]; then printf '1\\n' > controller.exit; exit 1; fi
run_shard() {
    CUDA_VISIBLE_DEVICES="$1" "$python" -u "$script" shard --shard "$1" > "shard$1.log" 2>&1
    status=$?
    printf '%s\\n' "$status" > "shard$1.exit"
    return "$status"
}
run_shard 0 & first=$!
run_shard 1 & second=$!
wait "$first"; first_status=$?
wait "$second"; second_status=$?
if [ "$first_status" -ne 0 ] || [ "$second_status" -ne 0 ]; then printf '1\\n' > controller.exit; exit 1; fi
CUDA_VISIBLE_DEVICES='' "$python" -u "$script" analyze > analysis.log 2>&1
status=$?
printf '%s\\n' "$status" > analysis.exit
printf '%s\\n' "$status" > controller.exit
exit "$status"
''')
    spec=dict(status='frozen_before_cross_region_model_outputs',observed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__),queue_sha256=sha(queue),inventory_result_sha256=INVENTORY_SHA,
        geometry_source_sha256=GEOMETRY_SHA,replay_cases_sha256=sha(INVENTORY/'events_for_replay.json'),
        completed_M65_audit_sha256=AUDIT_SHA,head_sha256=HEAD_SHA,cases=r['cases'],events=events,
        text_conditions=['category','empty'],text_bank_sha256=t['text_development_sha256'],
        base_sha256=t['native_checkpoint_sha256'],training_spec_sha256=sha(PARENT/'training_spec.json'),
        integration_sha256=sha(PARENT/'integration.json'),
        public_path='M65 Null category protocol on its original complete recursive path; every public bbox/score must exactly equal sealed history.',
        shadow_state='Clone template/query/mask/semantic inputs before direct network.forward; the network mutates its query-list input. Discard shadow queries and never call tracker.track for shadow observations.',
        observations=['Original factor4','Original-centre factor7','Full-image square resized to256','Factor4-size grid with50-percent overlap and image edges'],
        dense_protocol='Decode all256 positions with original cal_bbox and mapping. Existing decode_nms_candidates selects top10 positions separately on Hann and raw response; every reported box uses the same native dense decoding, not a second float64 size/offset formula. Boxes float64; scores float32 in compressed NPZ.',
        coarse_route='For each text condition, map each of the top10 raw coarse decoded-box centres to its nearest grid-window centre; take the first3 distinct windows, with no padding if fewer.',
        comparisons='Local/wide/all-grid capacity; selected3-window capacity. Cross coarse-route text with fine-window text in a2x2 fixed-state readout. Score-only argmax is diagnostic and never committed.',
        scope='All216 GT-selected Train development positions from the sealed inventory, including healthy controls. Not a deployment trigger or unseen test.',
        inference_GT_policy='Only original t0 initialization box; run reads frame-only schedule and sealed predictions, never geometry_events or subsequent GT. GT and event tags are read in analyze after both shard receipts seal.',
        smoke=dict(sequence='bag05_indoor',frames=102,probe_frames=[1,50,100],exact_public_reference_required=True,local_maps_exact_to_public=True,state_equality_after_each_probe=True),
        shard_sequences=[[c['sequence'] for c in r['cases'][i::2]] for i in range(2)],
        estimated_calls_by_sequence=costs,public_track_calls=33108,shadow_calls_per_text=count,total_shadow_calls=2*count,
        expected_total_formal_forward_calls=33108+2*count,smoke_calls_extra=True,
        new_parameters=0,optimizer_steps=0,new_caption_calls=0,new_embeddings=0,
        public_evaluation_allowed=False,M65_failed_gate_unchanged=True,M67_training_unchanged=True,
        independent_model_review_pass=False)
    write(ROOT/'spec.json',spec)
    print(json.dumps(dict(status=spec['status'],spec_sha256=sha(ROOT/'spec.json'),source_sha256=sha(__file__),
        formal_forward_calls=spec['expected_total_formal_forward_calls']),indent=2))

def checked():
    g,t,r=parents();s=read(ROOT/'spec.json')
    assert sha(__file__)==s['source_sha256'] and sha(ROOT/'run_capacity.sh')==s['queue_sha256']
    assert sha(INVENTORY/'events_for_replay.json')==s['replay_cases_sha256']
    assert s['events']==read(INVENTORY/'events_for_replay.json')['cases']
    return g,t,s

def idle():
    import subprocess,shutil
    used=[int(x.strip()) for x in subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()]
    assert len(used)==2 and max(used)<500,used
    free=shutil.disk_usage(BASE).free;assert free>1_000_000_000,free
    print(json.dumps(dict(gpu_memory_MiB=used,disk_free_bytes=free)))

def run(shard=None,smoke=False):
    g,t,s=checked()
    if not smoke:
        assert (ROOT/'smoke.exit').read_text().strip()=='0'
        receipt=read(ROOT/'smoke/receipt.json');assert receipt['source_sha256']==sha(__file__) and receipt['all_public_boxes_and_scores_exact']
        assert receipt['state_unchanged_after_each_shadow'] and receipt['local_maps_exact_to_public']
    import numpy as np
    import torch
    from types import SimpleNamespace
    sys.path.insert(0,str(PARENT/'code'))
    from lib.config.sttrack.config import cfg,update_config_from_file
    from lib.test.tracker.sttrack_semantic import STTrackSemantic
    from lib.test.tracker.sttrack_lachtt_observation import decode_nms_candidates,_map_local_box
    from lib.train.dataset.depth_utils import get_rgbd_frame
    from lib.train.data.processing_utils import sample_target
    torch.set_num_threads(1);torch.manual_seed(t['seed']);torch.cuda.manual_seed_all(t['seed'])
    update_config_from_file(str(PARENT/'code/experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    params=SimpleNamespace(cfg=cfg,checkpoint=t['native_checkpoint'],base_checkpoint_sha256=t['native_checkpoint_sha256'],
        template_factor=2.,template_size=128,search_factor=4.,search_size=256,save_all_boxes=False,debug=0)
    tracker=STTrackSemantic(params,str(PARENT/'training/null/final.pth'))
    assert tracker.use_text and tracker.network.semantic_adapter.null_support and not tracker.network.training
    bank=torch.load(PARENT/'text_development.pt',map_location='cpu')
    outdir=ROOT/('smoke' if smoke else 'shard'+str(shard));outdir.mkdir()
    real_forward=tracker.network.forward;capture={'active':False}
    def public_forward(*args,**kwargs):
        out=real_forward(*args,**kwargs)
        if capture['active']:capture['maps']={k:out[0][k].detach().clone() for k in ['score_map','size_map','offset_map']}
        return out
    tracker.network.forward=public_forward
    identity=torch.eye(256,device='cuda').reshape(256,1,16,16)
    def state():
        return dict(bbox=tracker.state,query=tracker.track_query_before,templates=tracker.z_dict,
            mask=tracker.box_mask_z,frame=tracker.frame_id,semantic=tracker.semantic_context,z_patch=tracker.z_patch_arr)
    def equal(a,b):
        if torch.is_tensor(a):assert torch.equal(a,b)
        elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:equal(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b)
            for x,y in zip(a,b):equal(x,y)
        else:assert a==b
    def probe(image,window,condition):
        patch,resize,_=sample_target(image,window['prior'],window['factor'],output_sz=256)
        kwargs=copy.deepcopy(dict(template=tracker.z_dict,ce_template_mask=tracker.box_mask_z,
            track_query_before=tracker.track_query_before,keep_rate=tracker.keep_rate,semantic_context=tracker.semantic_context))
        if condition=='empty':
            context=kwargs['semantic_context'];text=context['text'];text[context['mask']]=bank['empty'].to(device=text.device,dtype=text.dtype)
        kwargs['search']=[tracker.preprocessor.process(patch)]
        with torch.no_grad():
            out=real_forward(**kwargs)[0]
            response=tracker.output_window*out['score_map']
            local=tracker.network.box_head.cal_bbox(identity,out['size_map'].expand(256,-1,-1,-1),out['offset_map'].expand(256,-1,-1,-1))
            dense=np.asarray([_map_local_box(b,window['prior'],256,resize,image.shape[0],image.shape[1]) for b in (local*256/resize).cpu().tolist()],dtype=np.float64)
            peaks={}
            for name,scores in [('hann',response),('raw',out['score_map'])]:
                candidates=decode_nms_candidates(scores,out['size_map'],out['offset_map'],[window['prior']],[resize],image.shape,256,10)
                indexes=[x['grid_row']*16+x['grid_column'] for x in candidates]
                peaks[name]=np.asarray(indexes,dtype=np.int16)
        maps={k:out[k].detach().clone() for k in ['score_map','size_map','offset_map']} if window['kind']=='local' and condition=='category' else None
        return dict(boxes=dense,hann=response.detach().cpu().numpy().reshape(256),raw=out['score_map'].detach().cpu().numpy().reshape(256),
            nms_hann=peaks['hann'],nms_raw=peaks['raw']),maps
    started=time.time();records=[];public_calls=0;shadow_calls=0
    names=[s['smoke']['sequence']] if smoke else s['shard_sequences'][shard]
    original=read(PARENT/'null_recursive_receipt.json');refs={x['sequence']:x for x in original['sequences']}
    for case in s['cases']:
        seq=case['sequence']
        if seq not in names:continue
        ref=PARENT/'recursive/null'/(seq+'.json');assert sha(ref)==refs[seq]['sha256'];reference=read(ref)['rows']
        folder=Path(t['dataset_root'])/seq
        def frame(f):return get_rgbd_frame(str(folder/'color'/('%08d.jpg'%(f+1))),str(folder/'depth'/('%08d.png'%(f+1))),dtype='rgbcolormap',depth_clip=True)
        j=bank['sequences'].index(seq)
        tracker.initialize(frame(0),dict(init_bbox=case['init_bbox'],text_tokens=bank['tokens'][j],text_mask=bank['mask'][j],empty_text=bank['empty']))
        count=s['smoke']['frames'] if smoke else case['frames']
        selected=set(s['smoke']['probe_frames']) if smoke else {e['frame'] for e in s['events'] if e['sequence']==seq}
        assert selected and max(selected)<count
        event_records=[]
        for f in range(1,count):
            image=frame(f);capture['active']=False;local_maps=None
            if f in selected:
                frozen=copy.deepcopy(state());ws=windows(tracker.state,image.shape[1],image.shape[0],g);arrays={}
                for condition in s['text_conditions']:
                    items=[]
                    for w in ws:
                        item,maps=probe(image,w,condition);equal(state(),frozen);items.append(item);shadow_calls+=1
                        if maps is not None:local_maps=maps
                    for key in items[0]:arrays[condition+'_'+key]=np.stack([x[key] for x in items])
                centres=np.asarray([[w['rectangle'][0]+w['rectangle'][2]/2,w['rectangle'][1]+w['rectangle'][3]/2] for w in ws[3:]])
                routes={}
                for condition in s['text_conditions']:
                    route=[]
                    for peak in arrays[condition+'_nms_raw'][2]:
                        box=arrays[condition+'_boxes'][2,int(peak)];centre=box[:2]+box[2:]/2
                        index=int(np.argmin(np.square(centres-centre).sum(1)))+3
                        if index not in route:route.append(index)
                        if len(route)==3:break
                    assert 1<=len(route)<=3;routes[condition]=route
                npz=outdir/(seq+'__'+str(f)+'.npz');np.savez_compressed(npz,**arrays)
                meta=outdir/(seq+'__'+str(f)+'.json')
                write(meta,dict(sequence=seq,frame=f,image_shape=list(image.shape),previous_bbox=list(tracker.state),
                    windows=ws,routes=routes,array_sha256=sha(npz),spec_sha256=sha(ROOT/'spec.json'),GT_opened=False))
                event_records.append(dict(frame=f,metadata_sha256=sha(meta),array_sha256=sha(npz),windows=len(ws)))
                equal(state(),frozen);capture['active']=True
            prediction=tracker.track(image);public_calls+=1;capture['active']=False
            assert list(prediction['target_bbox'])==reference[f]['bbox'] and float(prediction['best_score'])==reference[f]['score'],(seq,f)
            if local_maps is not None:
                for key in local_maps:assert torch.equal(local_maps[key],capture['maps'][key]),(seq,f,key)
                del arrays,items,local_maps,frozen
        item=dict(sequence=seq,image_frames=count,public_track_calls=count-1,reference_sha256=sha(ref),events=event_records,
            elapsed_seconds=time.time()-started);records.append(item)
        print(json.dumps({k:v for k,v in item.items() if k!='events'}),flush=True)
    checked()
    receipt=dict(status='completed_exact_public_replay_with_noncommitting_cross_region_probes',source_sha256=sha(__file__),
        spec_sha256=sha(ROOT/'spec.json'),head_sha256=HEAD_SHA,smoke_only=smoke,shard=shard,sequences=records,
        public_track_calls=public_calls,shadow_forward_calls=shadow_calls,event_count=sum(len(x['events']) for x in records),
        all_public_boxes_and_scores_exact=True,state_unchanged_after_each_shadow=True,local_maps_exact_to_public=True,
        subsequent_GT_opened=False,optimizer_steps=0,public_evaluation_allowed=False,elapsed_seconds=time.time()-started)
    write(outdir/'receipt.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='sequences'},indent=2),flush=True)

def overlaps(boxes,gt,np):
    left=np.maximum(boxes[...,:2],gt[:2]);right=np.minimum(boxes[...,:2]+boxes[...,2:],gt[:2]+gt[2:])
    inter=np.maximum(0,right-left).prod(-1);return inter/(boxes[...,2:].prod(-1)+gt[2]*gt[3]-inter)

def analyze():
    import numpy as np
    g,t,s=checked();sealed={};receipts={};total_public=0;total_shadow=0
    for shard in range(2):
        assert (ROOT/('shard'+str(shard)+'.exit')).read_text().strip()=='0'
        folder=ROOT/('shard'+str(shard));r=read(folder/'receipt.json');receipts[str(shard)]=sha(folder/'receipt.json')
        assert r['source_sha256']==sha(__file__) and r['spec_sha256']==sha(ROOT/'spec.json') and r['head_sha256']==HEAD_SHA
        assert not r['smoke_only'] and r['shard']==shard and not r['subsequent_GT_opened']
        assert r['all_public_boxes_and_scores_exact'] and r['state_unchanged_after_each_shadow'] and r['local_maps_exact_to_public']
        assert [x['sequence'] for x in r['sequences']]==s['shard_sequences'][shard]
        total_public+=r['public_track_calls'];total_shadow+=r['shadow_forward_calls']
        for case in r['sequences']:
            for item in case['events']:
                key=(case['sequence'],item['frame']);assert key not in sealed
                stem=case['sequence']+'__'+str(item['frame']);meta=folder/(stem+'.json');array=folder/(stem+'.npz')
                assert sha(meta)==item['metadata_sha256'] and sha(array)==item['array_sha256']
                value=read(meta);assert value['array_sha256']==sha(array) and not value['GT_opened']
                assert len(value['windows'])==item['windows'];sealed[key]=(value,array)
    assert total_public==s['public_track_calls'] and total_shadow==s['total_shadow_calls']
    assert set(sealed)=={(e['sequence'],e['frame']) for e in s['events']}
    # GT and event-selection labels are first accessed only after every model output is sealed above.
    inventory=read(INVENTORY/'result.json');assert sha(INVENTORY/'geometry_events.json')==inventory['events_sha256']
    labels={(e['sequence'],e['frame']):e for e in read(INVENTORY/'geometry_events.json')['events']}
    groundtruth={}
    for case in s['cases']:
        p=Path(t['dataset_root'])/case['sequence']/'groundtruth.txt';assert sha(p)==case['gt_sha256']
        groundtruth[case['sequence']]=np.loadtxt(p,delimiter=',').reshape(-1,4)
    metrics=[]
    for (seq,f),(meta,array) in sorted(sealed.items()):
        gt=groundtruth[seq][f];assert np.isfinite(gt).all() and gt[2]>0 and gt[3]>0
        entry=dict(sequence=seq,frame=f,tags=labels[(seq,f)]['tags'],
            null_local_centre_inside=labels[(seq,f)]['arms']['null']['local']['centre_inside'],readouts={})
        with np.load(array) as a:
            n=len(meta['windows'])
            for text in s['text_conditions']:
                boxes=a[text+'_boxes'];assert boxes.shape==(n,256,4) and np.isfinite(boxes).all()
                iou=overlaps(boxes,gt,np)
                groups=dict(local=[0],wide=[1],coarse=[2],all_grid=list(range(3,n)))
                groups.update({route+'_route3':meta['routes'][route] for route in s['text_conditions']})
                for group,ids in groups.items():
                    dense=float(iou[ids].max());entry['readouts'][text+'/'+group+'/dense']=dict(best_iou=dense,correct_candidate_exists=dense>=.5,windows=len(ids))
                    for response in ['hann','raw']:
                        indexes=a[text+'_nms_'+response][ids];selected_iou=iou[np.asarray(ids)[:,None],indexes]
                        scores=a[text+'_'+response][np.asarray(ids)[:,None],indexes];flat=int(scores.argmax())
                        top=float(selected_iou.reshape(-1)[flat]);best=float(selected_iou.max())
                        entry['readouts'][text+'/'+group+'/'+response]=dict(best_iou=best,correct_candidate_exists=best>=.5,
                            score_argmax_iou=top,score_argmax_correct=top>=.5,score_argmax_severely_wrong=top<=.1,windows=len(ids))
        metrics.append(entry)
    summaries={}
    selectors={'all':lambda x:True,'H10':lambda x:'H10' in x['tags'],
        'H10_local_inside':lambda x:'H10' in x['tags'] and x['null_local_centre_inside'],
        'H10_local_outside':lambda x:'H10' in x['tags'] and not x['null_local_centre_inside'],
        'valid_after_invalid':lambda x:'valid_after_invalid' in x['tags'],'healthy':lambda x:'healthy' in x['tags']}
    for name,choose in selectors.items():
        rows=[x for x in metrics if choose(x)];values={}
        for key in metrics[0]['readouts']:
            points=[x['readouts'][key] for x in rows]
            value=dict(correct_candidate_events=sum(x['correct_candidate_exists'] for x in points),mean_best_iou=float(np.mean([x['best_iou'] for x in points])))
            if 'score_argmax_iou' in points[0]:value.update(score_argmax_correct_events=sum(x['score_argmax_correct'] for x in points),
                score_argmax_severely_wrong_events=sum(x['score_argmax_severely_wrong'] for x in points),mean_score_argmax_iou=float(np.mean([x['score_argmax_iou'] for x in points])))
            values[key]=value
        summaries[name]=dict(events=len(rows),readouts=values)
    write(ROOT/'event_metrics.json',dict(spec_sha256=sha(ROOT/'spec.json'),events=metrics))
    result=dict(status='completed_fixed_state_cross_region_candidate_capacity',observed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__),spec_sha256=sha(ROOT/'spec.json'),receipts=receipts,summaries=summaries,
        event_metrics_sha256=sha(ROOT/'event_metrics.json'),public_track_calls=total_public,shadow_forward_calls=total_shadow,
        all_public_boxes_and_scores_exact=True,shadow_states_never_committed=True,
        scope='Post-hoc candidate capacity and fixed-state text-content sensitivity on reused Train development22. No rescue trajectories, deployment trigger, learned global verifier or new benchmark metrics.',
        M65_failed_gate_unchanged=True,M67_training_unchanged=True,public_evaluation_allowed=False,independent_model_review_pass=False)
    write(ROOT/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='summaries'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','check','idle','smoke','shard','analyze']);p.add_argument('--shard',type=int,choices=[0,1]);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='check':checked();print('SOURCE_AND_INPUTS_VERIFIED_NO_GPU_REPLAY')
    elif a.action=='idle':idle()
    elif a.action=='smoke':run(smoke=True)
    elif a.action=='shard':assert a.shard is not None;run(shard=a.shard)
    else:analyze()
