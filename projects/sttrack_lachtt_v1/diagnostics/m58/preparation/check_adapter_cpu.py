from pathlib import Path
import hashlib
import json
import sys

import torch

root = Path('/root/autodl-tmp/sttrack_m58_semantic_spatial_v1_20260906')
sys.path.insert(0, str(root))
from semantic_spatial_adapter import SemanticSpatialAdapter

torch.set_num_threads(2)
torch.manual_seed(2026)
model = SemanticSpatialAdapter()
rgb, depth, fused = [torch.randn(2, 256, 768) for _ in range(3)]
initial = torch.randn(2, 2, 16, 768)
text = torch.randn(2, 5, 768)
mask = torch.tensor([[True, False, False, False, False], [True, True, True, True, True]])
output, auxiliary = model(rgb, depth, fused, initial, text, mask)
assert torch.equal(output, fused)
assert torch.isfinite(auxiliary['attribute_scores'][mask[:, None, :].expand(2, 256, 5)]).all()
assert torch.allclose(auxiliary['modality_weights'].sum(-1), torch.ones(2, 5))
target = fused + .05 * torch.randn_like(fused)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
for _ in range(2):
    optimizer.zero_grad(set_to_none=True)
    output, _ = model(rgb, depth, fused, initial, text, mask)
    loss = (output - target).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
gradient = {name: float(module[0].weight.grad.abs().sum()) for name, module in [('rgb', model.rgb), ('depth', model.depth), ('text', model.text)]}
assert all(value > 0 for value in gradient.values())
changed, _ = model(rgb, depth, fused, initial, text, mask)
assert not torch.equal(changed, fused) and torch.isfinite(changed).all()
result = dict(status='synthetic_cpu_wiring_checks_complete', parameters=sum(p.numel() for p in model.parameters()),
    zero_initialized_output_exact=True, gradient_l1_after_second_synthetic_step=gradient,
    synthetic_optimizer_steps=2, actual_dataset_optimizer_steps=0,
    native_tracker_trajectory_equivalence_tested=False, public_evaluation=False,
    source_sha256=hashlib.sha256((root / 'semantic_spatial_adapter.py').read_bytes()).hexdigest(),
    checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    torch_version=torch.__version__)
(root / 'cpu_wiring_check.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
