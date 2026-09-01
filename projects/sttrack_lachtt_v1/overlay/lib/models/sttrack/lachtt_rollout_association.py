"""Language-anchored dense RGB-D association for protected rollout training."""

import math

import torch
from torch import nn
import torch.nn.functional as F


class LanguageAnchoredDenseAssociation(nn.Module):
    """Compare every search token with immutable RGB/depth/text anchors.

    This head does not own tracker state.  A caller must keep the official
    protected state and any tentative rollout state separate.
    """

    def __init__(self, input_dim=768, hidden_dim=64):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.search_rgb = nn.Linear(input_dim, hidden_dim)
        self.search_depth = nn.Linear(input_dim, hidden_dim)
        self.search_fused = nn.Linear(input_dim, hidden_dim)
        self.anchor_rgb = nn.Linear(input_dim, hidden_dim)
        self.anchor_depth = nn.Linear(input_dim, hidden_dim)
        self.anchor_text = nn.Linear(input_dim, hidden_dim)
        relation_dim = hidden_dim * 12 + 1
        self.encoder = nn.Sequential(
            nn.LayerNorm(relation_dim),
            nn.Linear(relation_dim, 128),
            nn.GELU(),
        )
        self.dense_head = nn.Linear(128, 1)
        self.hazard_head = nn.Linear(128, 1)
        nn.init.zeros_(self.dense_head.weight)
        nn.init.zeros_(self.dense_head.bias)
        nn.init.zeros_(self.hazard_head.weight)
        nn.init.zeros_(self.hazard_head.bias)

    @staticmethod
    def _validate(name, value, ndim, width=768):
        if value.ndim != ndim or value.shape[-1] != width:
            raise ValueError("%s shape drifted" % name)
        if not torch.isfinite(value.float()).all().item():
            raise ValueError("%s is non-finite" % name)

    def _attend(self, query, anchor):
        query_n = F.normalize(query, dim=-1)
        anchor_n = F.normalize(anchor, dim=-1)
        similarity = torch.matmul(query_n, anchor_n.transpose(-1, -2))
        weights = F.softmax(similarity / math.sqrt(self.hidden_dim), dim=-1)
        return torch.matmul(weights, anchor)

    @staticmethod
    def _relation(query, target):
        return torch.cat((query, target, torch.abs(query - target),
                          query * target), dim=-1)

    def _encode(self, search_rgb, search_depth, search_fused,
                anchor_rgb, anchor_depth, anchor_text, depth_validity):
        for name, value, ndim in (
                ("search_rgb", search_rgb, 3),
                ("search_depth", search_depth, 3),
                ("search_fused", search_fused, 3),
                ("anchor_rgb", anchor_rgb, 3),
                ("anchor_depth", anchor_depth, 3),
                ("anchor_text", anchor_text, 2)):
            self._validate(name, value, ndim, self.input_dim)
        if (search_rgb.shape != search_depth.shape or
                search_rgb.shape != search_fused.shape or
                anchor_rgb.shape != anchor_depth.shape or
                search_rgb.shape[0] != anchor_rgb.shape[0] or
                search_rgb.shape[0] != anchor_text.shape[0]):
            raise ValueError("association batch/token alignment drifted")
        batch, tokens, _ = search_rgb.shape
        side = int(round(math.sqrt(tokens)))
        if side * side != tokens:
            raise ValueError("search token count is not square")
        if list(depth_validity.shape) != [batch, tokens, 1]:
            raise ValueError("depth validity shape drifted")
        if (not torch.isfinite(depth_validity).all().item() or
                depth_validity.min().item() < 0.0 or
                depth_validity.max().item() > 1.0):
            raise ValueError("depth validity is invalid")

        rgb = self.search_rgb(search_rgb.float())
        depth = self.search_depth(search_depth.float())
        fused = self.search_fused(search_fused.float())
        initial_rgb = self.anchor_rgb(anchor_rgb.float())
        initial_depth = self.anchor_depth(anchor_depth.float())
        text = self.anchor_text(anchor_text.float())[:, None, :].expand(
            -1, tokens, -1)
        rgb_target = self._attend(rgb, initial_rgb)
        depth_target = self._attend(depth, initial_depth)
        features = torch.cat((
            self._relation(rgb, rgb_target),
            self._relation(depth, depth_target),
            self._relation(fused, text),
            depth_validity.float(),
        ), dim=-1)
        hidden = self.encoder(features)
        if not torch.isfinite(hidden).all().item():
            raise RuntimeError("association hidden state is non-finite")
        return hidden, side

    def forward_all(self, search_rgb, search_depth, search_fused,
                    anchor_rgb, anchor_depth, anchor_text, depth_validity):
        hidden, side = self._encode(
            search_rgb, search_depth, search_fused, anchor_rgb,
            anchor_depth, anchor_text, depth_validity)
        batch = hidden.shape[0]
        logits = self.dense_head(hidden).transpose(1, 2).reshape(
            batch, 1, side, side)
        hazard = self.hazard_head(hidden).transpose(1, 2).reshape(
            batch, 1, side, side)
        if (not torch.isfinite(logits).all().item() or
                not torch.isfinite(hazard).all().item()):
            raise RuntimeError("association output is non-finite")
        return logits, hazard, hidden

    def forward_with_hazard(self, search_rgb, search_depth, search_fused,
                            anchor_rgb, anchor_depth, anchor_text,
                            depth_validity):
        logits, hazard, _ = self.forward_all(
            search_rgb, search_depth, search_fused, anchor_rgb,
            anchor_depth, anchor_text, depth_validity)
        return logits, hazard

    def forward(self, search_rgb, search_depth, search_fused,
                anchor_rgb, anchor_depth, anchor_text, depth_validity):
        logits, _ = self.forward_with_hazard(
            search_rgb, search_depth, search_fused, anchor_rgb,
            anchor_depth, anchor_text, depth_validity)
        return logits

    @staticmethod
    def refine_score(protected_score, association_logits, weight=0.20,
                     enabled=True):
        if not enabled:
            return protected_score
        if protected_score.shape != association_logits.shape:
            raise ValueError("protected/association map shape mismatch")
        probability = protected_score.float().clamp(1e-4, 1.0 - 1e-4)
        refined = torch.sigmoid(torch.logit(probability) +
                                float(weight) * association_logits.float())
        return refined.to(dtype=protected_score.dtype)


__all__ = ["LanguageAnchoredDenseAssociation"]
