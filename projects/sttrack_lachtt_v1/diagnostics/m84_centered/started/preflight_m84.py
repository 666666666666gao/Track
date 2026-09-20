"""Independent-native prefix parity and a discarded 96-frame fit smoke."""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
import hashlib,json,sys,time
import numpy as np
import torch

R=Path(__file__).parent;P=Path('/root/autodl-tmp/sttrack_m82_native_preservation_20260909')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
assert not (R/'training').exists() and not (R/'preflight_result.json').exists()
spec=read(R/'training_spec.json');assert sha(Path(__file__))==spec['preflight_source_sha256']
for n,h in read(R/'integration.json')['source_sha256'].items():assert sha(R/'code'/n)==h
sys.path.insert(0,str(R/'code'));sys.path.insert(0,str(R))
from causal_training import CausalTrainingTracker
from native_preservation import supervision
from train_causal import tensor_state_sha
from lib.test.tracker.sttrack import STTrack
from lib.config.sttrack.config import cfg,update_config_from_file
from lib.train.dataset.depth_utils import get_rgbd_frame

torch.set_num_threads(1);torch.manual_seed(2027);torch.cuda.manual_seed_all(2027)
update_config_from_file(str(R/'code/experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
params=SimpleNamespace(cfg=cfg,checkpoint=spec['native_checkpoint'],base_checkpoint_sha256=spec['native_checkpoint_sha256'],
    template_factor=2.,template_size=128,search_factor=4.,search_size=256,save_all_boxes=False,debug=0)
native=STTrack(params)
tracker=CausalTrainingTracker(params,str(R/'native_parity/category_zero.pth'))
assert sum(p.numel() for p in tracker.network.parameters() if p.requires_grad)==289154
base_before=tensor_state_sha(tracker.network)
initial={k:v.detach().clone() for k,v in tracker.network.semantic_adapter.state_dict().items()}
bank=torch.load(spec['banks']['fit']['category']['path'],map_location='cpu')
empty_bank=torch.load(spec['banks']['fit']['empty']['path'],map_location='cpu')
row=next(r for r in spec['sequence_order'] if r['sequence']=='chair01_indoor')
folder=Path(spec['dataset_root'])/row['sequence'];idx=bank['sequences'].index(row['sequence']);eidx=empty_bank['sequences'].index(row['sequence'])
assert torch.equal(bank['mask'][idx],empty_bank['mask'][eidx])
info=dict(init_bbox=row['first_box'],text_tokens=bank['tokens'][idx],text_mask=bank['mask'][idx],empty_text=bank['empty'])
empty_info=dict(info,text_tokens=empty_bank['tokens'][eidx])
def frame(i):
    return get_rgbd_frame(str(folder/'color'/('%08d.jpg'%(i+1))),str(folder/'depth'/('%08d.png'%(i+1))),dtype='rgbcolormap',depth_clip=True)

started=time.time();parity=[]
for label,setup in [('zero_category',info),('nonzero_M82_empty',empty_info)]:
    if label=='nonzero_M82_empty':
        weights=torch.load(P/'training/category/final.pth',map_location='cpu')['model']
        weights['empty_text']=bank['empty'].float()
        tracker.network.semantic_adapter.load_state_dict(weights,strict=True)
    writes=[]
    with torch.no_grad():
        native.initialize(frame(0),dict(init_bbox=row['first_box']))
        tracker.initialize(frame(0),setup)
        for i in range(1,102):
            image=frame(i);baseline=native.track(image);out,state=tracker.step(image)
            assert baseline['target_bbox']==state['bbox'] and float(baseline['best_score'])==state['best_score'],(label,i)
            assert out['native_bbox']==state['bbox'] and torch.equal(out['score_map'],out['native_score_map'])
            assert len(tracker.track_query_before)==len(native.track_query_before)
            assert all(torch.equal(a,b) for a,b in zip(tracker.track_query_before,native.track_query_before))
            assert len(tracker.z_dict)==len(native.z_dict) and all(torch.equal(a,b) for a,b in zip(tracker.z_dict,native.z_dict))
            if state['template_write']:writes.append(i)
    parity.append(dict(condition=label,frames=101,exact_bbox_score_query_template=True,template_writes=writes))
    assert tensor_state_sha(tracker.network)==base_before
del native
tracker.network.semantic_adapter.load_state_dict(initial,strict=True)
tracker.initialize(frame(0),info)
assert sha(folder/'groundtruth.txt')==row['groundtruth_sha256']
gt=np.loadtxt(folder/'groundtruth.txt',delimiter=',').reshape(-1,4)
optimizer=torch.optim.AdamW(tracker.network.semantic_adapter.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay'])
optimizer.zero_grad(set_to_none=True);steps=valid=0;losses=[];gradient_records=[];capture={}
tracker.network.semantic_adapter.register_forward_pre_hook(lambda m,args:capture.update(inputs=args))
for i in range(1,97):
    out,state=tracker.step(frame(i))
    loss,diagnostic=supervision(tracker.network,out,gt[i],state['previous_bbox'],state['resize_factor'],256,
        output_window=tracker.output_window,preservation_weight=spec['preservation_weight'])
    if loss is not None:
        assert bool(torch.isfinite(loss));(loss/32).backward();valid+=1;losses.append(float(loss.detach()))
    if i%32==0:
        assert valid>0
        for parameter in tracker.network.semantic_adapter.parameters():
            if parameter.grad is not None:parameter.grad.mul_(32/valid)
        gradients={n:float(p.grad.abs().max()) for n,p in tracker.network.semantic_adapter.named_parameters() if p.grad is not None}
        assert gradients['delta.2.weight']>0 and gradients['delta.2.bias']==0
        norm=torch.nn.utils.clip_grad_norm_(tracker.network.semantic_adapter.parameters(),spec['gradient_clip'])
        assert bool(torch.isfinite(norm))
        gradient_records.append(dict(step=steps+1,preclip_norm=float(norm),maximum_gradient=gradients))
        optimizer.step();steps+=1;optimizer.zero_grad(set_to_none=True);valid=0
    del out,loss
assert steps==3 and all(gradient_records[-1]['maximum_gradient'][n+'.0.weight']>0 for n in ['rgb','depth','text'])
assert tensor_state_sha(tracker.network)==base_before
assert all(p.grad is None for n,p in tracker.network.named_parameters() if not n.startswith('semantic_adapter.'))
assert not torch.equal(tracker.network.semantic_adapter.delta[-1].weight,initial['delta.2.weight'])
with torch.no_grad():
    rgb,depth,fused,initial_roi,text,mask=capture['inputs']
    empty=empty_bank['tokens'][eidx].float().cuda().unsqueeze(0)
    feature,_=tracker.network.semantic_adapter(rgb,depth,fused,initial_roi,empty,mask)
    assert torch.equal(feature,fused)
for n,h in read(R/'integration.json')['source_sha256'].items():assert sha(R/'code'/n)==h
report=dict(status='M84_native_parity_and_causal_smoke_pass',observed_utc=datetime.now(timezone.utc).isoformat(),
    source_sha256=sha(Path(__file__)),training_spec_sha256=sha(R/'training_spec.json'),seed=2027,sequence=row['sequence'],
    parity=parity,smoke_frames=96,supervised_frames=len(losses),smoke_optimizer_steps=steps,
    mean_smoke_loss=float(np.mean(losses)),gradients=gradient_records,base_state_sha256=base_before,
    base_parameters_and_buffers_unchanged=True,trained_smoke_empty_feature_exact_native=True,
    smoke_weights_saved=False,formal_optimizer_steps=0,elapsed_seconds=time.time()-started,
    scope='Correctness and trainability only; not development performance or final training results.')
(R/'preflight_result.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
