"""M81 full-t0 and matched-multistart development evaluation, sealed before GT."""
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import argparse,hashlib,json,sys,time
R=Path('/root/autodl-tmp/sttrack_m81_multistart_20260908')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def plans():
    f=read(R/'frozen.json');t=read(R/'training_spec.json');e=read(R/'evaluation_spec.json')
    assert sha(R/'training_spec.json')==f['training_spec_sha256'] and sha(R/'evaluation_spec.json')==f['evaluation_spec_sha256']
    assert sha(__file__)==e['evaluator_sha256'] and sha(R/'recursive_metric.py')==e['metric_sha256']
    assert sha(R/'integration.json')==t['integration_sha256']
    assert sha(R/'development_initializations.json')==t['development_initializations_sha256']
    for n,h in read(R/'integration.json')['source_sha256'].items():assert sha(R/'code'/n)==h
    results={}
    for a in ['category','empty']:
        assert (R/('training_'+a+'.exit')).read_text().strip()=='0'
        q=read(R/'training'/a/'result.json')
        assert q['status']=='one_full_causal_fit_pass_complete' and q['sequences']==130 and q['initializations']==426
        assert q['total_track_calls']==186694 and q['optimizer_steps']==5798
        assert q['training_spec_sha256']==sha(R/'training_spec.json') and q['initialization_manifest_sha256']==t['fit_initializations_sha256']
        assert q['base_parameters_and_buffers_unchanged'] and q['learned_parameters']==289154
        assert q['initial_adapter_state_sha256']=='56d03acff2972871788dd51f0eff40d0b6d888aad79fe74db4c44cdfb276ca42'
        assert q['base_state_before_sha256']=='c07022117c6efa8399b6df57e275ec81c6f3efce44eca5e4bc0282b52dd0dae0'
        assert sha(R/'training'/a/'final.pth')==q['final_checkpoint_sha256'];results[a]=q
        for b in [t['banks']['development'][a],e['multi_banks'][a],e['parent_heads'][a]]:assert sha(b['path'])==b['sha256']
    assert sha(e['swapped_bank']['path'])==e['swapped_bank']['sha256']
    return t,e,results
def family_spec(name,t,e,results):
    assert name in e['families']
    multi=name.startswith('multi_');native=name=='multi_native'
    arm='empty' if name.endswith('empty_trained') else 'category'
    if native:head=None
    elif '_M78_' in name:head=e['parent_heads'][arm]
    else:head=dict(path=str(R/'training'/arm/'final.pth'),sha256=results[arm]['final_checkpoint_sha256'])
    condition='empty' if name.endswith('empty_trained') or name.endswith('empty_content') else 'category'
    bank=e['multi_banks'][condition] if multi else t['banks']['development'][condition]
    if name=='t0_swapped':bank=e['swapped_bank']
    episodes=read(R/'development_initializations.json')['episodes'] if multi else [dict(id=x['sequence']+':0',sequence=x['sequence'],start_frame=0,end_frame_inclusive=x['frames']-1,init_bbox=x['init_bbox'],track_calls=x['frames']-1,groundtruth_sha256=x['gt_sha256']) for x in e['cases']]
    return dict(name=name,multi=multi,native=native,head=head,bank=bank,episodes=episodes)
def run(name):
    t,e,results=plans();f=family_spec(name,t,e,results)
    import torch
    sys.path.insert(0,str(R/'code'));sys.path.insert(0,str(R))
    from lib.config.sttrack.config import cfg,update_config_from_file
    from lib.test.tracker.sttrack_semantic import STTrackSemantic
    from lib.test.tracker.sttrack import STTrack
    from lib.train.dataset.depth_utils import get_rgbd_frame
    from multistart import initialize_episode
    torch.set_num_threads(1);torch.manual_seed(2027);torch.cuda.manual_seed_all(2027)
    update_config_from_file(str(R/'code/experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    params=SimpleNamespace(cfg=cfg,checkpoint=t['native_checkpoint'],base_checkpoint_sha256=t['native_checkpoint_sha256'],template_factor=2.,template_size=128,search_factor=4.,search_size=256,save_all_boxes=False,debug=0)
    tracker=STTrack(params) if f['native'] else STTrackSemantic(params,f['head']['path'])
    bank=torch.load(f['bank']['path'],map_location='cpu')
    output=R/'recursive'/name;output.mkdir(parents=True)
    receipts=[];started=time.time()
    for ep in f['episodes']:
        folder=Path(t['dataset_root'])/ep['sequence'];start=ep['start_frame'];end=ep['end_frame_inclusive']
        def frame(i):return get_rgbd_frame(str(folder/'color'/('%08d.jpg'%(i+1))),str(folder/'depth'/('%08d.png'%(i+1))),dtype='rgbcolormap',depth_clip=True)
        if f['native']:tracker.initialize(frame(start),dict(init_bbox=ep['init_bbox']))
        elif f['multi']:initialize_episode(tracker,bank,ep,frame(start))
        else:
            j=bank['sequences'].index(ep['sequence'])
            tracker.initialize(frame(start),dict(init_bbox=ep['init_bbox'],text_tokens=bank['tokens'][j],text_mask=bank['mask'][j],empty_text=bank['empty']))
        rows=[dict(frame=start,bbox=list(tracker.state),score=None)]
        for i in range(start+1,end+1):
            prediction=tracker.track(frame(i));rows.append(dict(frame=i,bbox=list(prediction['target_bbox']),score=float(prediction['best_score'])))
        filename=hashlib.sha256(ep['id'].encode()).hexdigest()+'.json'
        path=output/filename;write(path,dict(id=ep['id'],sequence=ep['sequence'],family=name,rows=rows))
        receipt=dict(id=ep['id'],sequence=ep['sequence'],path=filename,frames=len(rows),track_calls=len(rows)-1,sha256=sha(path),elapsed_seconds=time.time()-started)
        receipts.append(receipt);print(json.dumps(receipt),flush=True)
    plans()
    write(R/('receipt_'+name+'.json'),dict(status='complete',family=name,evaluation_spec_sha256=sha(R/'evaluation_spec.json'),head_sha256=f['head']['sha256'] if f['head'] else t['native_checkpoint_sha256'],bank_sha256=f['bank']['sha256'],episodes=receipts,total_track_calls=sum(x['track_calls'] for x in receipts),total_frames=sum(x['frames'] for x in receipts),metric_gt_opened=False,text_updated_online=False))
def analyze():
    t,e,results=plans()
    import numpy as np
    sys.path.insert(0,str(R));from recursive_metric import statistics
    predictions={};families={};receipts={}
    for name in e['families']:
        assert (R/('eval_'+name+'.exit')).read_text().strip()=='0'
        f=family_spec(name,t,e,results);families[name]=f
        receipt=read(R/('receipt_'+name+'.json'))
        assert receipt['status']=='complete' and receipt['family']==name and receipt['evaluation_spec_sha256']==sha(R/'evaluation_spec.json')
        assert receipt['head_sha256']==(f['head']['sha256'] if f['head'] else t['native_checkpoint_sha256'])
        assert receipt['bank_sha256']==f['bank']['sha256'] and receipt['total_track_calls']==33108
        assert len(receipt['episodes'])==len(f['episodes']) and receipt['total_frames']==33108+len(f['episodes'])
        predictions[name]={};receipts[name]=sha(R/('receipt_'+name+'.json'))
        for row,ep in zip(receipt['episodes'],f['episodes']):
            assert row['id']==ep['id']
            path=R/'recursive'/name/row['path'];assert sha(path)==row['sha256'];d=read(path)
            assert d['id']==ep['id'] and d['family']==name
            assert [x['frame'] for x in d['rows']]==list(range(ep['start_frame'],ep['end_frame_inclusive']+1))
            assert d['rows'][0]['bbox']==ep['init_bbox']
            boxes=np.asarray([x['bbox'] for x in d['rows']]);assert np.isfinite(boxes).all() and (boxes[:,2:]>0).all()
            predictions[name][ep['id']]=boxes
    # Nine complete families have been checked; only now read metric GT.
    per={};aggregates={};episode_metrics={}
    gt_cache={}
    for case in e['cases']:
        path=Path(t['dataset_root'])/case['sequence']/'groundtruth.txt';assert sha(path)==case['gt_sha256']
        gt_cache[case['sequence']]=np.loadtxt(path,delimiter=',').reshape(-1,4)
        assert len(gt_cache[case['sequence']])==case['frames']
    keys=['valid_frames','iou_sum','low_iou_frames','failure_episodes']
    for name,f in families.items():
        per[name]={};episode_metrics[name]={}
        for ep in f['episodes']:
            gt=gt_cache[ep['sequence']][ep['start_frame']:ep['end_frame_inclusive']+1]
            stat=statistics(predictions[name][ep['id']],gt);episode_metrics[name][ep['id']]=stat
            seq=per[name].setdefault(ep['sequence'],{k:0 for k in keys})
            for k in keys:seq[k]+=stat[k]
        for seq in per[name].values():seq['mean_iou']=seq['iou_sum']/seq['valid_frames']
        a={k:sum(x[k] for x in per[name].values()) for k in keys};assert a['valid_frames']==28897
        a.update(mean_iou=a['iou_sum']/a['valid_frames'],macro_sequence_mean_iou=float(np.mean([x['mean_iou'] for x in per[name].values()])))
        aggregates[name]=a
    assert sha(e['native_result_path'])==e['native_result_sha256'] and sha(e['parent_result_path'])==e['parent_result_sha256']
    native=read(e['native_result_path']);parent=read(e['parent_result_path'])
    per['t0_native']=native['per_sequence']['native'];aggregates['t0_native']=native['aggregates']['native']
    gates={};broken={}
    for prefix in ['t0','multi']:
        a=aggregates[prefix+'_category']
        for ref,margin in [(prefix+'_native',.002),(prefix+'_empty_trained',.001)]:
            b=aggregates[ref];key=prefix+'_vs_'+ref
            gates[key+'_pooled']=a['mean_iou']>=b['mean_iou']+margin
            gates[key+'_macro']=a['macro_sequence_mean_iou']>=b['macro_sequence_mean_iou']
            gates[key+'_low']=a['low_iou_frames']<=b['low_iou_frames']
            gates[key+'_H10']=a['failure_episodes']<=b['failure_episodes']
            broken[ref]=[n for n in per[ref] if per[ref][n]['failure_episodes']==0 and per[prefix+'_category'][n]['failure_episodes']>0]
            gates[key+'_protect']=not broken[ref]
    content={};a=aggregates['t0_category']
    for ref in ['t0_empty_content','t0_swapped']:
        b=aggregates[ref]
        content[ref+'_pooled']=a['mean_iou']>b['mean_iou'];content[ref+'_macro']=a['macro_sequence_mean_iou']>b['macro_sequence_mean_iou']
        content[ref+'_low']=a['low_iou_frames']<=b['low_iou_frames'];content[ref+'_H10']=a['failure_episodes']<=b['failure_episodes']
    a=aggregates['multi_category'];b=aggregates['multi_M78_category']
    coverage=dict(pooled=a['mean_iou']>=b['mean_iou']+.001,macro=a['macro_sequence_mean_iou']>=b['macro_sequence_mean_iou'],low=a['low_iou_frames']<=b['low_iou_frames'],H10=a['failure_episodes']<=b['failure_episodes'])
    output=dict(status='complete_M81_t0_and_multistart_development',observed_utc=datetime.now(timezone.utc).isoformat(),aggregates=aggregates,per_sequence=per,episode_metrics=episode_metrics,reference_M78_t0=parent['aggregates'],primary_gates=gates,content_gates=content,initialization_coverage_gates=coverage,broken_success_sequences=broken,primary_pass=all(gates.values()),content_pass=all(content.values()),coverage_pass=all(coverage.values()),all_checks_pass=all(list(gates.values())+list(content.values())+list(coverage.values())),receipt_sha256=receipts,training_spec_sha256=sha(R/'training_spec.json'),evaluation_spec_sha256=sha(R/'evaluation_spec.json'),public_evaluation_allowed=False,independent_model_review_pass=False,scope='Only reused DepthTrack Train development22. Multistart counts reset H10 at common boundaries and are not VOT metrics. t0 controls and multistart controls are distinct protocols.')
    write(R/'result.json',output)
    print(json.dumps({k:v for k,v in output.items() if k not in ['per_sequence','episode_metrics']},indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();a=p.add_mutually_exclusive_group(required=True);a.add_argument('--family');a.add_argument('--analyze',action='store_true');args=p.parse_args()
    analyze() if args.analyze else run(args.family)
