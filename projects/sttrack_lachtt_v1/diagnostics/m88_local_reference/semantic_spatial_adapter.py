"""Search and phrase jointly retrieve local initialization evidence (M88)."""
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

    def local_bound(self, search, reference, query, normalizer):
        phrase_logits = query @ reference.transpose(-1, -2) * self.scale
        search_logits = search @ reference.transpose(-1, -2) * self.scale
        weights = (phrase_logits[:, None] + search_logits[:, :, None]).softmax(dim=-1)
        return normalizer(query[:, None] + weights @ reference[:, None])

    def forward(self, rgb, depth, fused, initial, text, mask):
        assert rgb.shape == depth.shape == fused.shape
        assert initial.shape[0] == rgb.shape[0] and initial.shape[1] == 2
        assert text.shape[:2] == mask.shape and text.shape[1] == self.slots.shape[0]
        assert mask.dtype == torch.bool and bool(mask.any(dim=1).all())
        r, d = self.rgb(rgb), self.depth(depth)
        reference_r, reference_d = self.rgb(initial[:, 0]), self.depth(initial[:, 1])
        query = self.text(text) + self.slots.unsqueeze(0)
        # Keep the M84 initialization-only modality preference unchanged.
        cr = (query @ reference_r.transpose(-1, -2) * self.scale).softmax(dim=-1)
        cd = (query @ reference_d.transpose(-1, -2) * self.scale).softmax(dim=-1)
        static_r = self.rgb_bound(query + cr @ reference_r)
        static_d = self.depth_bound(query + cd @ reference_d)
        modality = self.modality(torch.cat((query, static_r, static_d), dim=-1)).softmax(dim=-1)
        bound_r = self.local_bound(r, reference_r, query, self.rgb_bound)
        bound_d = self.local_bound(d, reference_d, query, self.depth_bound)
        evidence_r = (r[:, :, None] * bound_r).sum(dim=-1) * self.scale
        evidence_d = (d[:, :, None] * bound_d).sum(dim=-1) * self.scale
        evidence = evidence_r * modality[:, None, :, 0] + evidence_d * modality[:, None, :, 1]
        evidence = evidence.masked_fill(~mask[:, None, :], float('-inf'))
        grounded = modality[:, None, :, 0:1] * bound_r + modality[:, None, :, 1:2] * bound_d
        if self.null_support:
            scores = torch.cat((evidence, torch.zeros_like(evidence[..., :1])), dim=-1)
            weights = scores.softmax(dim=-1)[..., :-1]
        else:
            weights = evidence.softmax(dim=-1)
        context = (weights[..., None] * grounded).sum(dim=2)
        local = torch.cat((r, d, context, r * context, d * context), dim=-1)
        return fused + self.delta(local), {'attribute_scores': evidence, 'modality_weights': modality}
