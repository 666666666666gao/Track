"""Matched local-versus-pooled candidate association for DepthTrack training."""
import torch
from torch import nn
import torch.nn.functional as F


class LocalSpatialAssociation(nn.Module):
    def __init__(self, spatial=True):
        super().__init__()
        self.spatial = spatial
        self.projection = nn.Linear(768, 16, bias=False)
        self.pair = nn.Sequential(nn.Linear(16 * 64, 32), nn.GELU(), nn.Linear(32, 8), nn.GELU())
        self.candidate = nn.Sequential(nn.Linear(3 * 2 * 8 + 8, 32), nn.GELU(), nn.Linear(32, 1))
        self.none = nn.Sequential(nn.Linear(2 * (3 * 2 * 8 + 8), 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.candidate[-1].weight)
        nn.init.zeros_(self.candidate[-1].bias)
        nn.init.zeros_(self.none[-1].weight)
        nn.init.constant_(self.none[-1].bias, -4.)

    def relation_features(self, candidates, references, scalars):
        # B,K,2,16,768 and B,3,2,16,768. Every candidate uses its own ROI.
        if not self.spatial:
            candidates = candidates.mean(-2, keepdim=True).expand_as(candidates)
            references = references.mean(-2, keepdim=True).expand_as(references)
        candidates = self.projection(F.normalize(candidates, dim=-1))
        references = self.projection(F.normalize(references, dim=-1))
        c = candidates[:, :, None].expand(-1, -1, 3, -1, -1, -1)
        r = references[:, None].expand(-1, candidates.shape[1], -1, -1, -1, -1)
        relation = torch.cat([c, r, c - r, c * r], dim=-1).flatten(-2)
        pair = self.pair(relation).flatten(-3)
        return torch.cat([pair, scalars], dim=-1)

    def forward(self, candidates, references, scalars):
        features = self.relation_features(candidates, references, scalars)
        residual = self.candidate(features).squeeze(-1)
        logits = residual + scalars[..., 0]  # log Hann response; zero residual preserves default top1.
        none = self.none(torch.cat([features.mean(1), features.max(1).values], dim=-1))
        return torch.cat([logits, none], dim=1)


def select_candidate(logits):
    """NONE leaves the default candidate unchanged; it never invents a box."""
    selected = logits.argmax(-1)
    return torch.where(selected == logits.shape[-1] - 1, torch.zeros_like(selected), selected)
