"""Parameter-budget-matched task-decoupled target--distractor router."""

import torch
from torch import nn

from .lachtt_target_distractor_fixed_pool import (
    TEMPORAL_WIDTH,
    fixed_temporal_pool,
)


TASK_NAMES = ("commit", "rank", "benefit", "catastrophe", "gain")


class CandidateTaskTower(nn.Module):
    """One task-local permutation-equivariant candidate tower."""

    def __init__(self, hidden_dim=24, residual_scale=0.1):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.encoder = nn.Sequential(
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

    def forward(self, pooled, candidate_valid):
        tokens = self.encoder(pooled)
        valid = candidate_valid.unsqueeze(-1)
        mean_context = ((tokens * valid).sum(dim=1) /
                        valid.sum(dim=1).clamp_min(1))
        max_context = tokens.masked_fill(
            ~valid, -float("inf")).max(dim=1).values
        context = torch.cat((
            tokens,
            mean_context.unsqueeze(1).expand_as(tokens),
            max_context.unsqueeze(1).expand_as(tokens),
        ), dim=-1)
        return tokens + self.residual_scale * self.set_mlp(context)


class TargetDistractorTaskTowersRouter(nn.Module):
    """Five task-local towers sharing only parameter-free relation pooling."""

    def __init__(self, hidden_dim=24, residual_scale=0.1):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.towers = nn.ModuleDict({
            name: CandidateTaskTower(hidden_dim, residual_scale)
            for name in TASK_NAMES
        })
        self.event_commit_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.candidate_rank_head = nn.Linear(hidden_dim, 1)
        self.candidate_benefit_head = nn.Linear(hidden_dim, 1)
        self.candidate_catastrophe_head = nn.Linear(hidden_dim, 1)
        self.candidate_gain_head = nn.Linear(hidden_dim, 1)

    def task_parameters(self, task_name):
        if task_name not in TASK_NAMES:
            raise KeyError(task_name)
        parameters = list(self.towers[task_name].parameters())
        head = {
            "commit": self.event_commit_head,
            "rank": self.candidate_rank_head,
            "benefit": self.candidate_benefit_head,
            "catastrophe": self.candidate_catastrophe_head,
            "gain": self.candidate_gain_head,
        }[task_name]
        return parameters + list(head.parameters())

    @staticmethod
    def _candidate_output(head, tokens):
        return head(tokens).squeeze(-1)

    def forward(self, relations, candidate_valid):
        if (relations.ndim != 4 or relations.shape[1:] != (5, 6, 49) or
                tuple(candidate_valid.shape) != (relations.shape[0], 6) or
                candidate_valid.dtype != torch.bool):
            raise ValueError("task-towers router input contract drifted")
        if not candidate_valid.any(dim=1).all().item():
            raise ValueError("task-towers router event has no valid candidate")
        pooled = fixed_temporal_pool(relations.float())
        task_tokens = {
            name: tower(pooled, candidate_valid)
            for name, tower in self.towers.items()
        }
        valid = candidate_valid.unsqueeze(-1)
        commit_tokens = task_tokens["commit"]
        event_mean = ((commit_tokens * valid).sum(dim=1) /
                      valid.sum(dim=1).clamp_min(1))
        event_max = commit_tokens.masked_fill(
            ~valid, -float("inf")).max(dim=1).values
        outputs = {
            "event_commit_logit": self.event_commit_head(
                torch.cat((event_mean, event_max), dim=-1)).squeeze(-1),
            "candidate_rank_logits": self._candidate_output(
                self.candidate_rank_head, task_tokens["rank"]),
            "candidate_benefit_logits": self._candidate_output(
                self.candidate_benefit_head, task_tokens["benefit"]),
            "candidate_catastrophe_logits": self._candidate_output(
                self.candidate_catastrophe_head,
                task_tokens["catastrophe"]),
            "candidate_h10_gain": self._candidate_output(
                self.candidate_gain_head, task_tokens["gain"]),
        }
        if any(not torch.isfinite(value).all().item()
               for value in outputs.values()):
            raise RuntimeError("task-towers router output is non-finite")
        return outputs


__all__ = [
    "TASK_NAMES",
    "CandidateTaskTower",
    "TargetDistractorTaskTowersRouter",
]
