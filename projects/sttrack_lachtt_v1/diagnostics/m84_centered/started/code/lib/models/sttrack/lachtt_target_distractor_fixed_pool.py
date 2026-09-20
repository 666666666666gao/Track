"""Non-recurrent fixed temporal pooling for bounded target--distractor relations."""

import torch
from torch import nn

from .lachtt_target_distractor_memory import RELATION_DIM


TEMPORAL_STATISTICS = (
    "mean", "max", "min", "last", "last_minus_first",
)
TEMPORAL_WIDTH = RELATION_DIM * len(TEMPORAL_STATISTICS)


def fixed_temporal_pool(relations):
    """Return parameter-free [batch, six candidates, 245] H5 statistics."""
    if relations.ndim != 4 or relations.shape[1:] != (5, 6, RELATION_DIM):
        raise ValueError("fixed temporal pool relation contract drifted")
    if not torch.isfinite(relations).all().item():
        raise ValueError("fixed temporal pool relation is non-finite")
    return torch.cat((
        relations.mean(dim=1),
        relations.max(dim=1).values,
        relations.min(dim=1).values,
        relations[:, -1],
        relations[:, -1] - relations[:, 0],
    ), dim=-1)


class TargetDistractorFixedPoolRouter(nn.Module):
    """Small permutation-equivariant router without recurrent modules."""

    def __init__(self, hidden_dim=64, residual_scale=0.1):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.residual_scale = float(residual_scale)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(TEMPORAL_WIDTH, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.set_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.event_commit_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.candidate_rank_head = nn.Linear(hidden_dim, 1)
        self.candidate_benefit_head = nn.Linear(hidden_dim, 1)
        self.candidate_catastrophe_head = nn.Linear(hidden_dim, 1)
        self.candidate_gain_head = nn.Linear(hidden_dim, 1)

    def forward(self, relations, candidate_valid):
        if (relations.ndim != 4 or
                relations.shape[1:] != (5, 6, RELATION_DIM) or
                tuple(candidate_valid.shape) != (relations.shape[0], 6) or
                candidate_valid.dtype != torch.bool):
            raise ValueError("fixed-pool router input contract drifted")
        if not candidate_valid.any(dim=1).all().item():
            raise ValueError("fixed-pool router event has no valid candidate")
        pooled = fixed_temporal_pool(relations.float())
        tokens = self.candidate_encoder(pooled)
        valid = candidate_valid.unsqueeze(-1)
        mean_context = ((tokens * valid).sum(dim=1) /
                        valid.sum(dim=1).clamp_min(1))
        max_context = tokens.masked_fill(
            ~valid, -float("inf")).max(dim=1).values
        context = torch.cat((
            tokens,
            mean_context.unsqueeze(1).expand(-1, 6, -1),
            max_context.unsqueeze(1).expand(-1, 6, -1),
        ), dim=-1)
        set_tokens = tokens + self.residual_scale * self.set_mlp(context)
        event_mean = ((set_tokens * valid).sum(dim=1) /
                      valid.sum(dim=1).clamp_min(1))
        event_max = set_tokens.masked_fill(
            ~valid, -float("inf")).max(dim=1).values
        outputs = {
            "event_commit_logit": self.event_commit_head(
                torch.cat((event_mean, event_max), dim=-1)).squeeze(-1),
            "candidate_rank_logits": self.candidate_rank_head(
                set_tokens).squeeze(-1),
            "candidate_benefit_logits": self.candidate_benefit_head(
                set_tokens).squeeze(-1),
            "candidate_catastrophe_logits": self.candidate_catastrophe_head(
                set_tokens).squeeze(-1),
            "candidate_h10_gain": self.candidate_gain_head(
                set_tokens).squeeze(-1),
        }
        if any(not torch.isfinite(value).all().item()
               for value in outputs.values()):
            raise RuntimeError("fixed-pool router output is non-finite")
        return outputs


__all__ = [
    "TEMPORAL_STATISTICS",
    "TEMPORAL_WIDTH",
    "fixed_temporal_pool",
    "TargetDistractorFixedPoolRouter",
]
