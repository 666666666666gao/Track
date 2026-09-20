"""101-frame CUDA interface probe on fixed M82 states, without subsequent GT."""
from pathlib import Path
import hashlib,json,sys,time
import torch

R=Path(__file__).parent
P=Path('/root/autodl-tmp/sttrack_m82_native_preservation_20260909')
S=Path('/root/autodl-tmp/sttrack_m83_same_state_20260920')
sys.path.insert(0,str(S))
import m83_same_state as shared
from centered_semantic_adapter import CenteredSemanticSpatialAdapter

read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
spec=read(P/'training_spec.json');plan=read(S/'spec.json');case=plan['cases'][0]
assert sha(S/'m83_same_state.py')==plan['source_sha256']
assert sha(P/'training_spec.json')==plan['training_spec_sha256']
assert sha(P/'training/category/final.pth')==plan['head_sha256']
torch.set_num_threads(1);torch.manual_seed(2027);torch.cuda.manual_seed_all(2027)
shared.update_config_from_file(str(P/'code/experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
params=shared.SimpleNamespace(cfg=shared.cfg,checkpoint=spec['native_checkpoint'],
    base_checkpoint_sha256=spec['native_checkpoint_sha256'],template_factor=2.,template_size=128,
    search_factor=4.,search_size=256,save_all_boxes=False,debug=0)
tracker=shared.STTrackSemantic(params,str(P/'training/category/final.pth'))
banks={}
for arm in ['category','empty']:
    item=spec['banks']['development'][arm];assert sha(Path(item['path']))==item['sha256']
    banks[arm]=torch.load(item['path'],map_location='cpu')
bank=banks['category'];idx=bank['sequences'].index(case['sequence'])
emptybank=banks['empty'];eidx=emptybank['sequences'].index(case['sequence'])
assert torch.equal(bank['mask'][idx],emptybank['mask'][eidx])
centered=CenteredSemanticSpatialAdapter(bank['empty'].float(),null_support=True)
weights=dict(tracker.network.semantic_adapter.state_dict());weights['empty_text']=bank['empty'].float()
centered.load_state_dict(weights,strict=True);centered.cuda().eval().requires_grad_(False)
empty_tokens=emptybank['tokens'][eidx].float().cuda().unsqueeze(0)
folder=Path(spec['dataset_root'])/case['sequence']
def frame(i):
    return shared.get_rgbd_frame(str(folder/'color'/('%08d.jpg'%(i+1))),
        str(folder/'depth'/('%08d.png'%(i+1))),dtype='rgbcolormap',depth_clip=True)
capture={}
tracker.network.semantic_adapter.register_forward_pre_hook(lambda module,inputs:capture.update(inputs=inputs))
sealed=P/'recursive/category'/(case['sequence']+'.json');assert sha(sealed)==case['sealed_category_sha256']
reference=read(sealed)['rows']
peak_changes=0;delta_rms=[];started=time.time()
with torch.no_grad():
    tracker.initialize(frame(0),dict(init_bbox=case['init_bbox'],text_tokens=bank['tokens'][idx],
        text_mask=bank['mask'][idx],empty_text=bank['empty']))
    for i in range(1,102):
        actual=tracker.track(frame(i))
        assert actual['target_bbox']==reference[i]['bbox'] and float(actual['best_score'])==reference[i]['score']
        rgb,depth,fused,initial,text,mask=capture['inputs']
        state=list(tracker.state);queries=[q.clone() for q in tracker.track_query_before];templates=list(tracker.z_dict)
        empty_feature,_=centered(rgb,depth,fused,initial,empty_tokens,mask)
        assert torch.equal(empty_feature,fused)
        native=tracker.network.forward_head(fused);empty_head=tracker.network.forward_head(empty_feature)
        assert all(torch.equal(native[k],empty_head[k]) for k in ['score_map','size_map','offset_map'])
        feature,_=centered(rgb,depth,fused,initial,text,mask)
        head=tracker.network.forward_head(feature)
        assert all(bool(torch.isfinite(head[k]).all()) for k in ['score_map','size_map','offset_map'])
        peak_changes+=int((head['score_map']*tracker.output_window).argmax()!=(native['score_map']*tracker.output_window).argmax())
        delta_rms.append(float((feature-fused).square().mean().sqrt()))
        assert tracker.state==state and len(tracker.z_dict)==len(templates)
        assert all(a is b for a,b in zip(tracker.z_dict,templates))
        assert all(torch.equal(a,b) for a,b in zip(tracker.track_query_before,queries))
result=dict(status='real_cuda_interface_pass',sequence=case['sequence'],positions=101,seed=2027,
    original_Category_saved_bbox_score_exact=True,nonzero_trained_weights_Empty_feature_and_head_exact_native=True,
    no_counterfactual_state_commit=True,centred_vs_native_hann_peak_changes=peak_changes,
    centred_feature_difference_rms_min=min(delta_rms),centred_feature_difference_rms_max=max(delta_rms),
    elapsed_seconds=time.time()-started,subsequent_gt_loaded=False,optimizer_steps=0,
    scope='Interface probe only, first fixed preflight sequence. No GT IoU, long-term native recovery or performance claim.',
    source_sha256={p.name:sha(p) for p in [Path(__file__),R/'centered_semantic_adapter.py',S/'m83_same_state.py']})
target=R/'centered_real_interface_result.json';assert not target.exists()
target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
