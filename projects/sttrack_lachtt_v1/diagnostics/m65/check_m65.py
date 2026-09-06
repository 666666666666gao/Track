"""Real Train parity and gradient smoke; all smoke weights are discarded."""
from datetime import datetime,timezone
import hashlib,importlib.util,json
from pathlib import Path
import sys,time
from types import SimpleNamespace
import numpy as np
import torch

ROOT=Path('/root/autodl-tmp/sttrack_m65_category_null_support_20260907')
PARENT=Path('/root/autodl-tmp/sttrack_m58_semantic_spatial_v2_20260906')
sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT))
from causal_training import CausalTrainingTracker,supervision
from train_causal import sha,tensor_state_sha
from lib.config.sttrack.config import cfg,update_config_from_file
from lib.test.tracker.sttrack import STTrack
from lib.test.tracker.sttrack_semantic import STTrackSemantic
from lib.models.sttrack.semantic_spatial_adapter import SemanticSpatialAdapter
from lib.train.dataset.depth_utils import get_rgbd_frame

torch.set_num_threads(1);torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
i=json.loads((ROOT/'integration.json').read_text());prep=json.loads((ROOT/'preparation.json').read_text())
for p,h in i['source_sha256'].items():assert sha(ROOT/'code'/p)==h
inv=json.loads((ROOT/'data_inventory.json').read_text())
update_config_from_file(str(ROOT/'code/experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
params=SimpleNamespace(cfg=cfg,checkpoint=inv['native_checkpoint'],base_checkpoint_sha256=inv['native_checkpoint_sha256'],
    template_factor=2.,template_size=128,search_factor=4.,search_size=256,save_all_boxes=False,debug=0)
assert sha(params.checkpoint)==params.base_checkpoint_sha256
row=next(r for r in inv['sequences_detail'] if r['sequence']=='chair01_indoor');assert row['split']=='fit'
bank=torch.load(ROOT/'text_fit.pt',map_location='cpu');index=bank['sequences'].index(row['sequence'])
info=dict(init_bbox=row['first_box'],text_tokens=bank['tokens'][index],text_mask=bank['mask'][index],empty_text=bank['empty'])
folder=Path(inv['dataset_root'])/row['sequence']
def frame(n):return get_rgbd_frame(str(folder/'color'/('%08d.jpg'%(n+1))),str(folder/'depth'/('%08d.png'%(n+1))),dtype='rgbcolormap',depth_clip=True)
native=STTrack(params);causal={};deployed={}
for arm in ['control','null']:
    p=ROOT/'native_parity'/(arm+'_zero.pth');assert sha(p)==prep['initial_checkpoint_sha256'][arm]
    causal[arm]=CausalTrainingTracker(params,str(p));deployed[arm]=STTrackSemantic(params,str(p))
    assert sum(p.numel() for p in causal[arm].network.parameters() if p.requires_grad)==289154
before={a:tensor_state_sha(t.network) for a,t in causal.items()}
initial={a:tensor_state_sha(t.network,adapter_only=True) for a,t in causal.items()}
assert initial['control']==initial['null'] and before['control']==before['null']
for t in [native]+list(causal.values())+list(deployed.values()):t.initialize(frame(0),dict(info))
captured=[]
hook=causal['control'].network.semantic_adapter.register_forward_pre_hook(lambda module,args:captured.append(tuple(x.detach().clone() for x in args)))
started=time.time();writes={a:[] for a in causal}
with torch.no_grad():
    for n in range(1,102):
        image=frame(n);expected=native.track(image)
        for arm in causal:
            _,actual=causal[arm].step(image);p=deployed[arm].track(image)
            assert expected['target_bbox']==actual['bbox']==p['target_bbox'],(arm,n,'box')
            assert float(expected['best_score'])==actual['best_score']==float(p['best_score']),(arm,n,'score')
            for t in [causal[arm],deployed[arm]]:
                assert len(t.track_query_before)==len(native.track_query_before)
                assert all(torch.equal(x,y) for x,y in zip(t.track_query_before,native.track_query_before))
                assert len(t.z_dict)==len(native.z_dict) and all(torch.equal(x,y) for x,y in zip(t.z_dict,native.z_dict))
            if actual['template_write']:writes[arm].append(n)
        if n!=101:captured.clear()
hook.remove();assert writes['control']==writes['null'] and 100 in writes['control']
parity_seconds=time.time()-started
# The Control also reproduces the nonzero M58 adapter on the same real inputs.
module_spec=importlib.util.spec_from_file_location('m58_original_adapter',str(PARENT/'code/lib/models/sttrack/semantic_spatial_adapter.py'))
module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
checkpoint=PARENT/'training/text/final.pth'
assert sha(checkpoint)=='0f2303ae1f9c12b4a25e6e6531b1fe39fa79578938aa9bc8c517e10bec06b629'
learned=torch.load(checkpoint,map_location='cpu')['model']
original=module.SemanticSpatialAdapter().cuda().eval();control=SemanticSpatialAdapter(null_support=False).cuda().eval();null=SemanticSpatialAdapter(null_support=True).cuda().eval()
for a in [original,control,null]:a.load_state_dict(learned,strict=True)
with torch.no_grad():
    x,diag=original(*captured[0]);y,_=control(*captured[0]);z,nd=null(*captured[0])
    assert torch.equal(x,y) and torch.isfinite(z).all() and not torch.equal(x,z)
    scores=nd['attribute_scores'];mass=torch.cat((scores,torch.zeros_like(scores[...,:1])),dim=-1).softmax(-1)[...,-1]
    mechanism=dict(nonzero_control_matches_parent_exactly=True,null_changes_nonzero_real_features=True,
      null_mass_min=float(mass.min()),null_mass_mean=float(mass.mean()),null_mass_max=float(mass.max()),
      null_vs_control_max_abs=float((z-y).abs().max()),parameter_counts=[sum(p.numel() for p in a.parameters()) for a in [original,control,null]])
    assert mechanism['parameter_counts']==[289154]*3 and 0<mechanism['null_mass_min']<=mechanism['null_mass_max']<1
del original,control,null,learned,captured,x,y,z,diag,nd,scores,mass
gt=np.loadtxt(folder/'groundtruth.txt',delimiter=',').reshape(-1,4);assert sha(folder/'groundtruth.txt')==row['groundtruth_sha256']
results={}
for arm,t in causal.items():
    t.initialize(frame(0),dict(info));optimizer=torch.optim.AdamW(t.network.semantic_adapter.parameters(),lr=1e-4,weight_decay=1e-4)
    losses=[];gradients={};labels={};optimizer.zero_grad(set_to_none=True);train_start=time.time()
    for n in range(1,97):
        out,state=t.step(frame(n))
        loss,d=supervision(t.network,out,gt[n],state['previous_bbox'],state['resize_factor'],256)
        labels[d['label']]=labels.get(d['label'],0)+1
        if loss is not None:(loss/32).backward();losses.append(float(loss.detach()))
        if n%32==0:
            for name,p in t.network.semantic_adapter.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(),(arm,name)
                    gradients[name]=float(p.grad.abs().sum())
            torch.nn.utils.clip_grad_norm_(t.network.semantic_adapter.parameters(),1.)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
    assert all(gradients[k+'.0.weight']>0 for k in ['rgb','depth','text'])
    assert tensor_state_sha(t.network)==before[arm]
    assert all(p.grad is None for k,p in t.network.named_parameters() if not k.startswith('semantic_adapter.'))
    assert not t.network.training
    results[arm]=dict(training_frames=96,actual_dataset_optimizer_steps=3,seconds=time.time()-train_start,
        labels=labels,mean_loss=sum(losses)/len(losses),last_gradient_l1=gradients,base_parameters_and_buffers_unchanged=True)
for p,h in i['source_sha256'].items():assert sha(ROOT/'code'/p)==h
r=dict(status='paired_fit_smoke_complete',observed_utc=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
    preparation_sha256=sha(ROOT/'preparation.json'),integration_sha256=sha(ROOT/'integration.json'),sequence=row['sequence'],split='fit',
    native_zero_residual_parity_frames_each=101,causal_and_deployed_boxes_scores_queries_templates_exact=True,parity_template_writes=writes,
    parity_seconds=parity_seconds,initial_adapter_state_sha256=initial,base_state_sha256=before,mechanism=mechanism,arms=results,
    formal_optimizer_steps=0,smoke_weights_saved=False,source_unchanged=True,measured_tracking_improvement=False,
    elapsed_seconds=time.time()-started,independent_model_review_pass=False)
(ROOT/'causal_smoke_result.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps(r,indent=2),flush=True)
