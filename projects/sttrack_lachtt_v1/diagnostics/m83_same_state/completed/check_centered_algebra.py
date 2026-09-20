"""Synthetic numerical interface checks; no data/GT or tracking-performance claim."""
from pathlib import Path
import hashlib,importlib.util,json,sys,types
import torch

R=Path(__file__).parent
base=Path('/root/autodl-tmp/sttrack_m82_native_preservation_20260909/code/lib/models/sttrack/semantic_spatial_adapter.py')
name='lib.models.sttrack.semantic_spatial_adapter'
spec=importlib.util.spec_from_file_location(name,base)
module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
from centered_semantic_adapter import CenteredSemanticSpatialAdapter

torch.set_num_threads(2);torch.manual_seed(2027)
empty=torch.randn(768)
m=CenteredSemanticSpatialAdapter(empty,null_support=True).eval()
rgb,depth,fused=[torch.randn(1,256,768) for _ in range(3)]
initial=torch.randn(1,2,16,768);text=torch.randn(1,5,768)
mask=torch.tensor([[True,True,True,False,False]])
empty_input=empty.reshape(1,1,768).expand_as(text)
assert sum(p.numel() for p in m.parameters())==289154
out,_=m(rgb,depth,fused,initial,text,mask)
assert torch.equal(out,fused)
out.square().mean().backward()
first={n:float(p.grad.abs().max()) for n,p in m.named_parameters() if p.grad is not None}
assert first['delta.2.weight']>0 and first['delta.2.bias']==0
assert all(v==0 for n,v in first.items() if n!='delta.2.weight')
with torch.no_grad():
    m.delta[-1].weight.normal_(std=.02);m.delta[-1].bias.normal_(std=.02)
m.zero_grad(set_to_none=True)
out,_=m(rgb,depth,fused,initial,empty_input,mask)
assert torch.equal(out,fused)
out.square().mean().backward()
empty_grad=max(float(p.grad.abs().max()) for p in m.parameters() if p.grad is not None)
assert empty_grad==0
m.zero_grad(set_to_none=True)
out,_=m(rgb,depth,fused,initial,text,mask)
assert not torch.equal(out,fused)
out.square().mean().backward()
nonempty={n:float(p.grad.abs().max()) for n,p in m.named_parameters() if p.grad is not None}
assert nonempty['delta.2.bias']==0 and nonempty['text.0.weight']>0 and nonempty['rgb.0.weight']>0
result=dict(status='synthetic_centered_algebra_pass',seed=2027,stored_parameters=289154,
    final_bias_cancelled_parameters=768,zero_initialization_exact_native=True,
    nonzero_weights_empty_exact_native=True,empty_task_gradient_max=empty_grad,
    first_step_gradient_max=first,nonzero_weights_nonempty_gradient_max=nonempty,
    scope='Synthetic CPU tensors only. No dataset samples, GT, training updates, recursive parity or performance evidence.',
    source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [base,R/'centered_semantic_adapter.py',Path(__file__)]})
(R/'centered_algebra_result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
