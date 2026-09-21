"""CPU functional checks for local-reference attention; no tracking metrics."""
from pathlib import Path
import importlib.util
import json, sys, torch

R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'code'))
from lib.models.sttrack.semantic_spatial_adapter import SemanticSpatialAdapter
from lib.models.sttrack.centered_semantic_adapter import CenteredSemanticSpatialAdapter

torch.set_num_threads(1)
torch.manual_seed(2027)
empty=torch.randn(768)
m=CenteredSemanticSpatialAdapter(empty,null_support=True)
assert sum(p.numel() for p in m.parameters())==289154
b,n,k,j=2,7,5,16
rgb,depth,fused=[torch.randn(b,n,768) for _ in range(3)]
initial=torch.randn(b,2,j,768)
text=torch.randn(b,k,768)
mask=torch.tensor([[1,1,0,0,1],[1,0,0,0,0]],dtype=torch.bool)
out,_=m(rgb,depth,fused,initial,text,mask)
assert torch.equal(out,fused)
torch.nn.init.normal_(m.delta[-1].weight,std=.01)
r=m.rgb(rgb);z=m.rgb(initial[:,0]);q=m.text(text)+m.slots[None]
bound=m.local_bound(r,z,q,m.rgb_bound)
manual=torch.empty_like(bound)
for batch in range(b):
    for p in range(n):
        for slot in range(k):
            logits=torch.stack([torch.dot(q[batch,slot]+r[batch,p],z[batch,t])*m.scale for t in range(j)])
            value=(logits.softmax(dim=0)[:,None]*z[batch]).sum(dim=0)
            manual[batch,p,slot]=m.rgb_bound(q[batch,slot]+value)
error=float((bound-manual).abs().max())
assert error<1e-5,error
assert not torch.allclose(bound[:,0],bound[:,1])
out,aux=m(rgb,depth,fused,initial,text,mask)
assert out.shape==fused.shape and aux['attribute_scores'].shape==(b,n,k)
assert torch.isneginf(aux['attribute_scores'].masked_select(~mask[:,None].expand(b,n,k))).all()
modified=text.clone();modified[~mask]=torch.randn_like(modified[~mask])*30
masked,_=m(rgb,depth,fused,initial,modified,mask)
assert torch.equal(masked,out)
permutation=torch.randperm(j)
permuted,_=m(rgb,depth,fused,initial[:,:,permutation],text,mask)
assert torch.allclose(permuted,out,atol=2e-6,rtol=2e-6)
permutation=torch.randperm(n)
permuted,_=m(rgb[:,permutation],depth[:,permutation],fused[:,permutation],initial,text,mask)
permutation_error=float((permuted-out[:,permutation]).abs().max())
print('search_permutation_max_error',permutation_error,flush=True)
assert torch.allclose(permuted,out[:,permutation],atol=2e-6,rtol=2e-6)
empty_input=empty[None,None].expand_as(text)
identity,_=m(rgb,depth,fused,initial,empty_input,mask)
assert torch.equal(identity,fused)
identity.sum().backward()
empty_gradient=max(float(p.grad.abs().max()) for p in m.parameters() if p.grad is not None)
print('empty_max_gradient',empty_gradient,flush=True)
original_path=Path('/root/autodl-tmp/sttrack_m84_centered_20260920/code/lib/models/sttrack/semantic_spatial_adapter.py')
spec=importlib.util.spec_from_file_location('m84_reference',str(original_path))
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
old=original.SemanticSpatialAdapter(null_support=True)
old.load_state_dict({name:value for name,value in m.state_dict().items() if name!='empty_text'},strict=True)
a,_=old(rgb,depth,torch.zeros_like(fused),initial,empty_input,mask)
c,_=old(rgb,depth,torch.zeros_like(fused),initial,empty_input,mask)
(fused+a-c).sum().backward()
original_empty_gradient=max(float(p.grad.abs().max()) for p in old.parameters() if p.grad is not None)
print('M84_empty_max_gradient',original_empty_gradient,flush=True)
assert empty_gradient<1e-6,empty_gradient
m.zero_grad(set_to_none=True)
out,_=m(rgb,depth,fused,initial,text,mask)
out.square().mean().backward()
gradients={name:float(p.grad.abs().max()) for name,p in m.named_parameters() if p.grad is not None}
assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
assert gradients['delta.2.bias']==0
assert all(gradients[name+'.0.weight']>0 for name in ['rgb','depth','text'])
report=dict(status='PASS',seed=2027,parameter_count=289154,explicit_loop_max_error=error,search_permutation_max_error=permutation_error,empty_max_gradient=empty_gradient,original_M84_empty_max_gradient=original_empty_gradient,
    batch_and_mask=True,reference_permutation=True,search_permutation=True,current_search_affects_binding=True,
    gradients=gradients,training_steps=0,tracker_calls=0,scope='Synthetic CPU functional checks only, not semantic or tracking performance.')
(R/'model_check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
