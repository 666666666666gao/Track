from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
import hashlib,importlib.util,json,sys,time
import numpy as np
import torch
ROOT=Path('/root/autodl-tmp/sttrack_m67_supervised_semantic_support_20260907')
PARENT=Path('/root/autodl-tmp/sttrack_m65_category_null_support_20260907')
sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT))
from causal_training import CausalTrainingTracker,supervision
from train_causal import sha,tensor_state_sha
from lib.config.sttrack.config import cfg,update_config_from_file
from lib.test.tracker.sttrack import STTrack
from lib.train.dataset.depth_utils import get_rgbd_frame

torch.set_num_threads(1);torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
spec=json.loads((ROOT/'prepared_training_spec.json').read_text());inv=json.loads((ROOT/'data_inventory.json').read_text())
assert sha(ROOT/'causal_training.py')==spec['causal_script_sha256'] and sha(ROOT/'support_loss.py')==spec['support_loss_sha256']
for n,h in json.loads((ROOT/'integration.json').read_text())['source_sha256'].items():assert sha(ROOT/'code'/n)==h
oldpath=PARENT/'causal_training.py';assert sha(oldpath)=='672b7575437f9d5000377311b949bb46f7908f0dce2f7cfe5891390b8f1bfb95'
ms=importlib.util.spec_from_file_location('m65_causal_reference',str(oldpath));old=importlib.util.module_from_spec(ms);ms.loader.exec_module(old)
update_config_from_file(str(ROOT/'code/experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
params=SimpleNamespace(cfg=cfg,checkpoint=spec['native_checkpoint'],base_checkpoint_sha256=spec['native_checkpoint_sha256'],
    template_factor=2.,template_size=128,search_factor=4.,search_size=256,save_all_boxes=False,debug=0)
native=STTrack(params)
trackers={a:CausalTrainingTracker(params,str(ROOT/'native_parity'/(a+'_zero.pth'))) for a in ['control','support']}
trackers['reference']=old.CausalTrainingTracker(params,str(PARENT/'native_parity/null_zero.pth'))
trackers['control']=old.CausalTrainingTracker(params,str(PARENT/'native_parity/null_zero.pth'))
initial={a:tensor_state_sha(t.network,adapter_only=True) for a,t in trackers.items()};assert len(set(initial.values()))==1
before={a:tensor_state_sha(t.network) for a,t in trackers.items()};assert len(set(before.values()))==1
row=next(r for r in inv['sequences_detail'] if r['sequence']=='chair01_indoor');assert row['split']=='fit'
folder=Path(inv['dataset_root'])/row['sequence'];bank=torch.load(ROOT/'text_fit.pt',map_location='cpu');j=bank['sequences'].index(row['sequence'])
info=dict(init_bbox=row['first_box'],text_tokens=bank['tokens'][j],text_mask=bank['mask'][j],empty_text=bank['empty'])
def frame(i):return get_rgbd_frame(str(folder/'color'/('%08d.jpg'%(i+1))),str(folder/'depth'/('%08d.png'%(i+1))),dtype='rgbcolormap',depth_clip=True)
for t in [native]+list(trackers.values()):t.initialize(frame(0),dict(info))
writes={a:[] for a in trackers};started=time.time()
with torch.no_grad():
    for i in range(1,2):
        image=frame(i);p=native.track(image)
        for a,t in trackers.items():
            _,state=t.step(image)
            assert p['target_bbox']==state['bbox'] and float(p['best_score'])==state['best_score'],(a,i)
            assert all(torch.equal(x,y) for x,y in zip(t.track_query_before,native.track_query_before))
            assert len(t.z_dict)==len(native.z_dict) and all(torch.equal(x,y) for x,y in zip(t.z_dict,native.z_dict))
            if state['template_write']:writes[a].append(i)
assert all(x==[] for x in writes.values())
del native
assert sha(folder/'groundtruth.txt')==row['groundtruth_sha256'];gt=np.loadtxt(folder/'groundtruth.txt',delimiter=',').reshape(-1,4)
results={};trajectories={};losses_by_arm={};final={}
for a in ['reference','control']:
    t=trackers[a];t.initialize(frame(0),dict(info));optim=torch.optim.AdamW(t.network.semantic_adapter.parameters(),lr=1e-4,weight_decay=1e-4)
    optim.zero_grad(set_to_none=True);pending=valid=steps=0;losses=[];states=[];labels={};last_grads={};support_stats=[]
    for i in range(1,97):
        out,state=t.step(frame(i))
        if a in ['reference','control']:loss,d=old.supervision(t.network,out,gt[i],state['previous_bbox'],state['resize_factor'],256)
        else:loss,d=supervision(t.network,out,gt[i],state['previous_bbox'],state['resize_factor'],256,support_weight=spec['support_loss_weights'][a])
        pending+=1;labels[d['label']]=labels.get(d['label'],0)+1;states.append(state)
        if loss is not None:
            (loss/32).backward();valid+=1;losses.append(float(loss.detach()))
            if 'support_loss' in d:support_stats.append({k:v for k,v in d.items() if k.startswith(('support_','positive_null','negative_null'))})
        if pending==32:
            assert valid>0
            for n,p in t.network.semantic_adapter.named_parameters():
                if p.grad is not None:
                    p.grad.mul_(32/valid);assert torch.isfinite(p.grad).all();last_grads[n]=float(p.grad.abs().sum())
            norm=torch.nn.utils.clip_grad_norm_(t.network.semantic_adapter.parameters(),1.);assert torch.isfinite(norm)
            optim.step();steps+=1;optim.zero_grad(set_to_none=True);pending=valid=0
        del out,loss
    assert steps==3 and all(last_grads[k+'.0.weight']>0 for k in ['rgb','depth','text'])
    assert tensor_state_sha(t.network)==before[a]
    assert all(p.grad is None for n,p in t.network.named_parameters() if not n.startswith('semantic_adapter.'))
    trajectories[a]=states;losses_by_arm[a]=losses;final[a]=tensor_state_sha(t.network,adapter_only=True)
    results[a]=dict(frames=96,optimizer_steps=3,labels=labels,mean_loss=float(np.mean(losses)),last_gradient_l1=last_grads,
                   support_first=support_stats[0] if support_stats else None,support_last=support_stats[-1] if support_stats else None,
                   base_parameters_and_buffers_unchanged=True)

x=trajectories['reference'];y=trajectories['control'];diff=[]
for i,(u,v) in enumerate(zip(x,y),1):
 if u!=v:diff.append(dict(frame=i,reference=u,control=v))
r=dict(status='two_original_trainers_numerical_repeat_diagnostic',first_differences=diff[:4],different_frames=len(diff),
 max_bbox_difference=max(abs(a-b) for u,v in zip(x,y) for a,b in zip(u['bbox'],v['bbox'])),
 max_score_difference=max(abs(u['best_score']-v['best_score']) for u,v in zip(x,y)),
 max_loss_difference=max(abs(u-v) for u,v in zip(losses_by_arm['reference'],losses_by_arm['control'])),
 initial_states=initial,final_states=final,loss_first_differences=[(i,u,v) for i,(u,v) in enumerate(zip(losses_by_arm['reference'],losses_by_arm['control']),1) if u!=v][:5])
r['maximum_parameter_difference']=max(float((u-v).abs().max()) for u,v in zip(trackers['reference'].network.semantic_adapter.state_dict().values(),trackers['control'].network.semantic_adapter.state_dict().values()))
(ROOT/'original_repeat_diagnostic.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
