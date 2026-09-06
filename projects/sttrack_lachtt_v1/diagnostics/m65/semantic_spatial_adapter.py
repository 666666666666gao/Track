"""Experimental initial-instance phrase interaction for dense RGB-D search tokens."""
import math

import torch
from torch import nn


class SemanticSpatialAdapter(nn.Module):
    def __init__(self, channels=768, hidden=64, slots=5, null_support=False):
        super().__init__()
        self.rgb = nn.Sequential(nn.Linear(channels, hidden), nn.LayerNorm(hidden))
        self.depth = nn.Sequential(nn.Linear(channels, hidden), nn.LayerNorm(hidden))
        self.text = nn.Sequential(nn.Linear(channels, hidden), nn.LayerNorm(hidden))
        self.slots = nn.Parameter(torch.zeros(slots, hidden))
        self.rgb_bound = nn.LayerNorm(hidden)
        self.depth_bound = nn.LayerNorm(hidden)
        self.modality = nn.Linear(3 * hidden, 2)
        self.delta = nn.Sequential(nn.Linear(5 * hidden, 128), nn.GELU(), nn.Linear(128, channels))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
        self.scale = 1.0 / math.sqrt(hidden)
        assert isinstance(null_support, bool)
        self.null_support = null_support

    def forward(self, rgb, depth, fused, initial, text, mask):
        """initial: B,2,16,C; text/mask keep the same slots in lexical controls."""
        assert rgb.shape == depth.shape == fused.shape
        assert initial.shape[0] == rgb.shape[0] and initial.shape[1] == 2
        assert text.shape[:2] == mask.shape and text.shape[1] == self.slots.shape[0]
        assert mask.dtype == torch.bool and bool(mask.any(dim=1).all())
        r, d = self.rgb(rgb), self.depth(depth)
        reference_r, reference_d = self.rgb(initial[:, 0]), self.depth(initial[:, 1])
        query = self.text(text) + self.slots.unsqueeze(0)
        correspondence_r = (query @ reference_r.transpose(-1, -2) * self.scale).softmax(dim=-1)
        correspondence_d = (query @ reference_d.transpose(-1, -2) * self.scale).softmax(dim=-1)
        bound_r = self.rgb_bound(query + correspondence_r @ reference_r)
        bound_d = self.depth_bound(query + correspondence_d @ reference_d)
        modality = self.modality(torch.cat((query, bound_r, bound_d), dim=-1)).softmax(dim=-1)
        evidence_r = r @ bound_r.transpose(-1, -2) * self.scale
        evidence_d = d @ bound_d.transpose(-1, -2) * self.scale
        evidence = evidence_r * modality[:, None, :, 0] + evidence_d * modality[:, None, :, 1]
        evidence = evidence.masked_fill(~mask[:, None, :], float('-inf'))
        grounded = modality[:, :, 0:1] * bound_r + modality[:, :, 1:2] * bound_d
        if self.null_support:
            # Fixed zero-logit / zero-value alternative; no new learned parameter.
            scores = torch.cat((evidence, torch.zeros_like(evidence[..., :1])), dim=-1)
            context = scores.softmax(dim=-1)[..., :-1] @ grounded
        else:
            context = evidence.softmax(dim=-1) @ grounded
        local = torch.cat((r, d, context, r * context, d * context), dim=-1)
        enhanced = fused + self.delta(local)
        return enhanced, {'attribute_scores': evidence, 'modality_weights': modality}
