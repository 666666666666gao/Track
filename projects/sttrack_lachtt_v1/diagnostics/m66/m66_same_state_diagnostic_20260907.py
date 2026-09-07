"""Train-only failure diagnosis: exact public replay, noncommitting head probes."""
import argparse,hashlib,json,math,sys,time
from pathlib import Path
from datetime import datetime,timezone

BASE=Path('/root/autodl-tmp')
ROOT=BASE/'sttrack_m66_same_state_diagnostic_20260907'
PARENT=BASE/'sttrack_m65_category_null_support_20260907'
AUDITOR=BASE/'audit_m65_completed_20260907.py'
AUDITOR_SHA='ef6a3f5f9f7b8635497fc993fded0924e7f37f5b67d14c3fab46fc27b7e596db'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def write(path,x):Path(path).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def prepare():
    assert sha(PARENT/'recursive_result.json')=='0101fa8185855044dda7462df2f9606b6e58c88248e8a7f1cf87497fb055b150'
    assert sha(PARENT/'completed_evidence_audit.json')=='26a79a424ca8b0dec01ff3c12f8cda6653b4a156c12e5c3b297112f52cf69f88'
    assert not read(PARENT/'recursive_result.json')['primary_pass']
    windows={'glass03_indoor':[[600,675]],'cup13_indoor':[[970,1160]],
             'cup10_indoor':[[1750,1900],[2050,2196]],'mobilephone01_indoor':[[850,1020],[1300,1418]],
             'ball07_indoor':[[1380,1560]]}
    rs=read(PARENT/'recursive_spec.json');cases=[next(c for c in rs['cases'] if c['sequence']==n) for n in windows]
    train=read(PARENT/'training_spec.json')
    ROOT.mkdir()
    spec=dict(kind='post_result_Train_failure_mechanism_diagnostic',created_utc=datetime.now(timezone.utc).isoformat(),
        runner_sha256=sha(__file__),parent_result_sha256=sha(PARENT/'recursive_result.json'),
        parent_training_spec_sha256=sha(PARENT/'training_spec.json'),parent_recursive_spec_sha256=sha(PARENT/'recursive_spec.json'),
        integration_sha256=sha(PARENT/'integration.json'),bank_sha256=sha(PARENT/'text_development.pt'),
        base_sha256=train['native_checkpoint_sha256'],head_sha256={a:sha(PARENT/'training'/a/'final.pth') for a in ['control','null']},
        cases=cases,capture_windows=windows,also_capture_every_50_frames=True,
        diagnostic_heads=['actual_semantic','unadapted_head_same_current_features'],
        changes_to_public_tracking=False,new_text_content_controls=False,new_training=False,public_evaluation_allowed=False,
        GT_policy='Only frozen t0 box in run; all later GT opened in analyze after both exact replays seal.',
        selection_scope='Five already examined Train development failure/regression sequences; no deployment trigger or unbiased test.',
        candidate_protocol='Existing NMS top10; all 256 boxes decoded with native cal_bbox; GT used after sealing only.',
        state_policy='Head-only probe receives cloned pre-adapter fused tokens; no alternate crop, query, bbox or template commits.')
    write(ROOT/'spec.json',spec);print(json.dumps(dict(spec_sha256=sha(ROOT/'spec.json'),cases=len(cases),track_calls_per_arm=sum(c['frames']-1 for c in cases))))

def plans():
    s=read(ROOT/'spec.json');assert sha(__file__)==s['runner_sha256']
    for name,key in [('training_spec.json','parent_training_spec_sha256'),('recursive_spec.json','parent_recursive_spec_sha256'),
                     ('integration.json','integration_sha256'),('text_development.pt','bank_sha256')]:assert sha(PARENT/name)==s[key]
    integration=read(PARENT/'integration.json')
    for name,digest in integration['source_sha256'].items():assert sha(PARENT/'code'/name)==digest
    train=read(PARENT/'training_spec.json');assert sha(train['native_checkpoint'])==s['base_sha256']
    for arm,digest in s['head_sha256'].items():assert sha(PARENT/'training'/arm/'final.pth')==digest
    return s,train

def run(arm):
    s,t=plans()
    import copy
    import numpy as np
    import torch
    sys.path.insert(0,str(PARENT/'code'))
    from types import SimpleNamespace
    from lib.config.sttrack.config import cfg,update_config_from_file
    from lib.test.tracker.sttrack_semantic import STTrackSemantic
    from lib.test.tracker.sttrack_lachtt_observation import decode_nms_candidates,_map_local_box
    from lib.train.dataset.depth_utils import get_rgbd_frame
    torch.set_num_threads(1);torch.manual_seed(t['seed']);torch.cuda.manual_seed_all(t['seed'])
    update_config_from_file(str(PARENT/'code/experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    params=SimpleNamespace(cfg=cfg,checkpoint=t['native_checkpoint'],base_checkpoint_sha256=t['native_checkpoint_sha256'],
        template_factor=2.,template_size=128,search_factor=4.,search_size=256,save_all_boxes=False,debug=0)
    tracker=STTrackSemantic(params,str(PARENT/'training'/arm/'final.pth'))
    bank=torch.load(PARENT/'text_development.pt',map_location='cpu')
    outdir=ROOT/arm;outdir.mkdir()
    capture={'active':False};real_forward=tracker.network.forward
    def public_forward(*args,**kwargs):
        output=real_forward(*args,**kwargs)
        if capture['active']:
            capture['out']={k:output[0][k].detach().clone() for k in ['score_map','size_map','offset_map']}
            evidence=output[0]['semantic_features']['attribute_scores']
            mass=torch.cat((evidence,torch.zeros_like(evidence[...,:1])),dim=-1).softmax(-1)[...,-1]
            capture['null_mass']=mass.detach().cpu().flatten().tolist()
        return output
    tracker.network.forward=public_forward
    def fused_hook(module,args):
        if capture['active']:capture['fused']=args[2].detach().clone()
    hook=tracker.network.semantic_adapter.register_forward_pre_hook(fused_hook)
    identity=torch.eye(256,device='cuda').reshape(256,1,16,16)
    def decode(out,prior,image_shape):
        side=math.ceil(math.sqrt(prior[2]*prior[3])*4.);resize=256./side
        response=tracker.output_window*out['score_map']
        nms=decode_nms_candidates(response,out['size_map'],out['offset_map'],[prior],[resize],image_shape,256,10)
        local=tracker.network.box_head.cal_bbox(identity,out['size_map'].expand(256,-1,-1,-1),out['offset_map'].expand(256,-1,-1,-1))
        dense=[_map_local_box(x,prior,256,resize,image_shape[0],image_shape[1]) for x in (local*256/resize).cpu().tolist()]
        index=int(response.flatten().argmax());rawindex=int(out['score_map'].flatten().argmax())
        assert np.max(np.abs(np.asarray(dense[index])-np.asarray(nms[0]['bbox'])))<1e-4
        return dict(nms=nms,dense_boxes=dense,hann_scores=response.flatten().cpu().tolist(),
                    raw_scores=out['score_map'].flatten().cpu().tolist(),selected_index=index,raw_selected_index=rawindex)
    def equal_state(a,b):
        if torch.is_tensor(a):assert torch.equal(a,b)
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b)
            for x,y in zip(a,b):equal_state(x,y)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:equal_state(a[k],b[k])
        else:assert a==b
    started=time.time();receipts=[]
    for case in s['cases']:
        seq=case['sequence'];folder=Path(t['dataset_root'])/seq
        reference_path=PARENT/'recursive'/arm/(seq+'.json')
        original_receipt=read(PARENT/(arm+'_recursive_receipt.json'))
        assert sha(reference_path)==next(x['sha256'] for x in original_receipt['sequences'] if x['sequence']==seq)
        reference=read(reference_path)['rows']
        def frame(i):return get_rgbd_frame(str(folder/'color'/('%08d.jpg'%(i+1))),str(folder/'depth'/('%08d.png'%(i+1))),dtype='rgbcolormap',depth_clip=True)
        j=bank['sequences'].index(seq);capture['active']=False
        tracker.initialize(frame(0),dict(init_bbox=case['init_bbox'],text_tokens=bank['tokens'][j],text_mask=bank['mask'][j],empty_text=bank['empty']))
        records=[];geometry=[]
        for i in range(1,case['frames']):
            prior=list(tracker.state);image=frame(i)
            picked=i%50==0 or any(a<=i<b for a,b in s['capture_windows'][seq])
            capture['active']=picked
            prediction=tracker.track(image);capture['active']=False
            assert list(prediction['target_bbox'])==reference[i]['bbox'] and float(prediction['best_score'])==reference[i]['score'],(arm,seq,i)
            side=math.ceil(math.sqrt(prior[2]*prior[3])*4.)
            crop=[round(prior[0]+.5*prior[2]-.5*side),round(prior[1]+.5*prior[3]-.5*side),side,side]
            geometry.append(dict(frame=i,previous_bbox=prior,search_rectangle=crop,bbox=list(tracker.state),score=float(prediction['best_score']),
                                 template_write=i%50==0 and float(prediction['best_score'])>.75))
            if picked:
                state=[tracker.state,tracker.track_query_before,tracker.z_dict,tracker.box_mask_z,tracker.frame_id]
                frozen=copy.deepcopy(state)
                with torch.no_grad():
                    native=tracker.network.forward_head(capture['fused'].clone())
                    heads={'actual_semantic':decode(capture['out'],prior,image.shape),
                           'unadapted_head_same_current_features':decode(native,prior,image.shape)}
                equal_state(state,frozen)
                assert abs(heads['actual_semantic']['hann_scores'][heads['actual_semantic']['selected_index']]-reference[i]['score'])<1e-8
                assert np.max(np.abs(np.asarray(heads['actual_semantic']['dense_boxes'][heads['actual_semantic']['selected_index']])-np.asarray(reference[i]['bbox'])))<1e-4
                records.append(dict(frame=i,heads=heads,zero_slot_softmax_mass=capture['null_mass']))
        path=outdir/(seq+'.json')
        write(path,dict(sequence=seq,arm=arm,reference_sha256=sha(reference_path),all_public_boxes_and_scores_exact=True,
                       image_frames=case['frames'],geometry=geometry,probes=records))
        item=dict(sequence=seq,frames=case['frames'],probe_frames=len(records),sha256=sha(path),elapsed_seconds=time.time()-started)
        receipts.append(item);print(json.dumps(item),flush=True)
    hook.remove();plans()
    receipt=dict(status='complete_exact_replay_and_noncommitting_head_probes',arm=arm,spec_sha256=sha(ROOT/'spec.json'),
        head_sha256=s['head_sha256'][arm],sequences=receipts,subsequent_gt_opened=False,new_text_content_controls=False,
        new_optimizer_steps=0,all_public_boxes_and_scores_exact=True,probe_state_unchanged=True,elapsed_seconds=time.time()-started)
    write(ROOT/(arm+'_receipt.json'),receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='sequences'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','control','null']);args=p.parse_args()
    if args.action=='prepare':prepare()
    else:run(args.action)
