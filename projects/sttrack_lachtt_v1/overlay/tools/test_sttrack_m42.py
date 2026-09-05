"""Small contracts for native ROI geometry and candidate permutation behavior."""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.test.tracker.sttrack_local_spatial_observation import sample_roi_tokens
from lib.models.sttrack.lachtt_local_spatial_association import LocalSpatialAssociation, select_candidate


def main():
    torch.set_num_threads(1); torch.manual_seed(2026)
    tokens = torch.arange(16, dtype=torch.float32).reshape(1,16,1)
    roi = sample_roi_tokens(tokens, [[0,0,4,4]], [0,0,4,4], 1., 4)
    assert torch.equal(roi, tokens), (roi, tokens)
    candidates = torch.randn(2,10,2,16,768)
    references = torch.randn(2,3,2,16,768)
    scalars = torch.randn(2,10,8); scalars[:,:,0] = -torch.arange(10)[None].float()
    spatial = LocalSpatialAssociation(True); pooled = LocalSpatialAssociation(False)
    pooled.load_state_dict(spatial.state_dict())
    with torch.no_grad():
        assert torch.equal(select_candidate(spatial(candidates,references,scalars)), torch.zeros(2,dtype=torch.long))
        permutation = torch.tensor([4,2,9,0,1,8,3,7,5,6])
        original = spatial.relation_features(candidates,references,scalars)
        permuted = spatial.relation_features(candidates[:,permutation],references,scalars[:,permutation])
        assert torch.allclose(original[:,permutation],permuted,atol=1e-6)
        reversed_roi = candidates.flip(-2)
        assert not torch.allclose(original, spatial.relation_features(reversed_roi,references,scalars))
        assert torch.allclose(pooled.relation_features(candidates,references,scalars),
                              pooled.relation_features(reversed_roi,references,scalars),atol=1e-6)
    optimizer = torch.optim.AdamW(spatial.parameters(), lr=.001)
    loss = F.cross_entropy(spatial(candidates,references,scalars),torch.tensor([1,10]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in spatial.parameters())
    optimizer.step()
    assert torch.isfinite(spatial(candidates,references,scalars)).all()
    print('PASS: patch-center ROI alignment, default preservation, candidate permutation, matched local-information control, finite training step')


if __name__ == '__main__':
    main()
