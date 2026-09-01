"""Strict-H10 two-stage router over frozen candidate-specific RGB-D features."""

import math

import torch
from torch import nn
import torch.nn.functional as F


FEATURE_KEYS = (
    "clip_image",
    "native_depth",
    "native_fused",
    "native_rgb",
    "query_depth",
    "query_rgb",
    "raw_depth",
    "scalars",
)
NATIVE_KEYS = (
    "native_rgb",
    "native_depth",
    "native_fused",
    "query_rgb",
    "query_depth",
)


class CachedStrictTwoStageRouter(nn.Module):
    """Predict event commitability, then rank six fixed candidates.

    No sequence, frame, source, or peak-rank metadata is accepted.  Candidate
    order has no positional embedding, so candidate outputs are permutation
    equivariant and the event output is permutation invariant in eval mode.
    """

    def __init__(self, projection_dim=32, hidden_dim=128,
                 attention_heads=4, set_layers=2,
                 set_feedforward_dim=256, dropout=0.0):
        super().__init__()
        self.projection_dim = int(projection_dim)
        self.hidden_dim = int(hidden_dim)
        self.native_projection = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, projection_dim),
            nn.GELU(),
        )
        self.clip_projection = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, projection_dim),
            nn.GELU(),
        )
        self.raw_depth_encoder = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, projection_dim),
            nn.GELU(),
        )
        self.scalar_encoder = nn.Sequential(
            nn.LayerNorm(15),
            nn.Linear(15, projection_dim),
            nn.GELU(),
        )
        candidate_input_dim = projection_dim * 12 + 2
        self.candidate_input_dim = int(candidate_input_dim)
        self.candidate_encoder = nn.Sequential(
            nn.LayerNorm(candidate_input_dim),
            nn.Linear(candidate_input_dim, hidden_dim),
            nn.GELU(),
        )
        self.temporal = nn.GRU(
            hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        block = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=set_feedforward_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(
            block, num_layers=set_layers, norm=nn.LayerNorm(hidden_dim))
        self.event_commit_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.candidate_rank_head = nn.Linear(hidden_dim, 1)
        self.candidate_benefit_head = nn.Linear(hidden_dim, 1)
        self.candidate_catastrophe_head = nn.Linear(hidden_dim, 1)
        self.candidate_gain_head = nn.Linear(hidden_dim, 1)

    @staticmethod
    def _validate_inputs(features, initial_image, identity_text,
                         candidate_valid):
        if set(features) != set(FEATURE_KEYS):
            raise ValueError("cached feature keys drifted")
        reference = features["clip_image"]
        if reference.ndim != 4 or reference.shape[-1] != 768:
            raise ValueError("cached clip feature shape drifted")
        batch, horizon, candidates, _ = reference.shape
        if (horizon != 5 or candidates != 6 or
                tuple(initial_image.shape) != (batch, 1, 768) or
                tuple(identity_text.shape) != (batch, 1, 768) or
                tuple(candidate_valid.shape) != (batch, candidates)):
            raise ValueError("cached router input contract drifted")
        expected = {
            "clip_image": (batch, horizon, candidates, 768),
            "native_depth": (batch, horizon, candidates, 768),
            "native_fused": (batch, horizon, candidates, 768),
            "native_rgb": (batch, horizon, candidates, 768),
            "query_depth": (batch, horizon, candidates, 768),
            "query_rgb": (batch, horizon, candidates, 768),
            "raw_depth": (batch, horizon, candidates, 2, 16, 16),
            "scalars": (batch, horizon, candidates, 15),
        }
        for name, shape in expected.items():
            value = features[name]
            if tuple(value.shape) != shape:
                raise ValueError("cached feature shape drifted: %s" % name)
            if not torch.isfinite(value.float()).all().item():
                raise ValueError("cached feature is non-finite: %s" % name)
        for name, value in (("initial_image", initial_image),
                            ("identity_text", identity_text)):
            if not torch.isfinite(value.float()).all().item():
                raise ValueError("anchor is non-finite: %s" % name)
        if candidate_valid.dtype != torch.bool:
            raise ValueError("candidate_valid must be boolean")
        if not candidate_valid.any(dim=1).all().item():
            raise ValueError("each event must contain a valid candidate")
        return batch, horizon, candidates

    def forward(self, features, initial_image, identity_text,
                candidate_valid):
        batch, horizon, candidates = self._validate_inputs(
            features, initial_image, identity_text, candidate_valid)
        native = [self.native_projection(features[name].float())
                  for name in NATIVE_KEYS]
        clip_candidate = self.clip_projection(features["clip_image"].float())
        initial = self.clip_projection(initial_image.float()).view(
            batch, 1, 1, self.projection_dim).expand(
                -1, horizon, candidates, -1)
        text = self.clip_projection(identity_text.float()).view(
            batch, 1, 1, self.projection_dim).expand(
                -1, horizon, candidates, -1)
        clip_relations = [
            clip_candidate,
            torch.abs(clip_candidate - initial),
            clip_candidate * initial,
            torch.abs(clip_candidate - text),
            clip_candidate * text,
            F.cosine_similarity(clip_candidate, initial, dim=-1).unsqueeze(-1),
            F.cosine_similarity(clip_candidate, text, dim=-1).unsqueeze(-1),
        ]
        raw_depth = self.raw_depth_encoder(
            features["raw_depth"].float().flatten(start_dim=-3))
        scalars = self.scalar_encoder(features["scalars"].float())
        candidate_input = torch.cat(
            [*native, *clip_relations, raw_depth, scalars], dim=-1)
        if candidate_input.shape[-1] != self.candidate_input_dim:
            raise RuntimeError("candidate input width drifted")
        encoded = self.candidate_encoder(candidate_input)
        temporal_input = encoded.permute(0, 2, 1, 3).reshape(
            batch * candidates, horizon, self.hidden_dim)
        _, temporal_state = self.temporal(temporal_input)
        candidate_tokens = self.temporal_norm(
            temporal_state[-1].reshape(batch, candidates, self.hidden_dim))
        set_tokens = self.set_encoder(
            candidate_tokens, src_key_padding_mask=~candidate_valid)

        valid = candidate_valid.unsqueeze(-1)
        mean_pool = ((set_tokens * valid).sum(dim=1) /
                     valid.sum(dim=1).clamp_min(1))
        max_pool = set_tokens.masked_fill(~valid, -float("inf")).max(dim=1).values
        event_commit = self.event_commit_head(
            torch.cat((mean_pool, max_pool), dim=-1)).squeeze(-1)
        outputs = {
            "event_commit_logit": event_commit,
            "candidate_rank_logits": self.candidate_rank_head(
                set_tokens).squeeze(-1),
            "candidate_benefit_logits": self.candidate_benefit_head(
                set_tokens).squeeze(-1),
            "candidate_catastrophe_logits": self.candidate_catastrophe_head(
                set_tokens).squeeze(-1),
            "candidate_h10_gain": self.candidate_gain_head(
                set_tokens).squeeze(-1),
        }
        for name, value in outputs.items():
            if not torch.isfinite(value.float()).all().item():
                raise RuntimeError("router output is non-finite: %s" % name)
        return outputs


def cached_strict_router_loss(outputs, event_target, gain_target,
                              beneficial_target, catastrophic_target,
                              label_available, candidate_valid,
                              pairwise_margin=0.1):
    if event_target.ndim != 1:
        raise ValueError("event target rank drifted")
    batch, candidates = candidate_valid.shape
    expected = (batch, candidates)
    for name, value in (("gain_target", gain_target),
                        ("beneficial_target", beneficial_target),
                        ("catastrophic_target", catastrophic_target),
                        ("label_available", label_available)):
        if tuple(value.shape) != expected:
            raise ValueError("%s shape drifted" % name)
    if tuple(event_target.shape) != (batch,):
        raise ValueError("event target shape drifted")
    available = label_available & candidate_valid
    denominator = available.float().sum().clamp_min(1.0)
    zero = outputs["candidate_rank_logits"].sum() * 0.0
    event_commit = F.binary_cross_entropy_with_logits(
        outputs["event_commit_logit"], event_target.float())
    benefit = (F.binary_cross_entropy_with_logits(
        outputs["candidate_benefit_logits"], beneficial_target.float(),
        reduction="none") * available.float()).sum() / denominator
    catastrophe = (F.binary_cross_entropy_with_logits(
        outputs["candidate_catastrophe_logits"],
        catastrophic_target.float(), reduction="none") *
        available.float()).sum() / denominator
    gain = (F.smooth_l1_loss(
        outputs["candidate_h10_gain"], gain_target.float(),
        reduction="none") * available.float()).sum() / denominator

    rank_terms = []
    pairwise_terms = []
    for index in range(batch):
        beneficial = torch.nonzero(
            available[index] & beneficial_target[index].bool(),
            as_tuple=False).flatten()
        if beneficial.numel() == 0:
            continue
        best = beneficial[gain_target[index, beneficial].argmax()]
        masked_rank = outputs["candidate_rank_logits"][index].masked_fill(
            ~available[index], -float("inf"))
        rank_terms.append(F.cross_entropy(
            masked_rank.unsqueeze(0), best.reshape(1)))
        negatives = torch.nonzero(
            available[index] & ~beneficial_target[index].bool(),
            as_tuple=False).flatten()
        for positive in beneficial:
            for negative in negatives:
                pairwise_terms.append(F.relu(
                    float(pairwise_margin) -
                    outputs["candidate_rank_logits"][index, positive] +
                    outputs["candidate_rank_logits"][index, negative]))
    rank = torch.stack(rank_terms).mean() if rank_terms else zero
    pairwise = (torch.stack(pairwise_terms).mean()
                if pairwise_terms else zero)
    total = (event_commit + rank + 0.5 * benefit + catastrophe +
             0.5 * gain + 0.5 * pairwise)
    values = {
        "total": total,
        "event_commit": event_commit,
        "conditional_rank": rank,
        "benefit": benefit,
        "catastrophe": catastrophe,
        "gain": gain,
        "pairwise": pairwise,
    }
    if any(not math.isfinite(float(value.detach())) for value in values.values()):
        raise RuntimeError("router loss is non-finite")
    return values


__all__ = ["CachedStrictTwoStageRouter", "cached_strict_router_loss"]
