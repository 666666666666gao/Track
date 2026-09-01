"""Independent utility and multi-horizon safety paths for LACHTT candidates."""

import math

import torch
from torch import nn

from .lachtt_learned_bounded_roi_association import (
    BLOCK_FAMILY_INDICES,
    EMBEDDING_WIDTH,
    RAW_DIFFERENCE_BLOCKS,
    SCALAR_RELATION_DIM,
)
from .lachtt_rich_roi_relation import (
    FAMILY_NAMES,
    PROJECTION_WIDTH,
    RICH_RELATION_DIM,
    TEMPORAL_WIDTH,
    fixed_temporal_pool_rich,
)


HORIZONS = (3, 5, 10)
TRAJECTORY_METRICS = (
    "branch_mean_iou",
    "public_mean_iou",
    "gain",
    "low_overlap_fraction",
    "trailing_low_run_fraction",
)


def _fixed_projection(seed, device, dtype):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    signs = torch.randint(
        0, 2, (EMBEDDING_WIDTH, PROJECTION_WIDTH),
        generator=generator, dtype=torch.int64)
    matrix = (signs.float() * 2.0 - 1.0) / math.sqrt(PROJECTION_WIDTH)
    return matrix.to(device=device, dtype=dtype)


def _new_projectors(base_seed):
    projectors = nn.ModuleList([
        nn.Linear(EMBEDDING_WIDTH, PROJECTION_WIDTH, bias=False)
        for _ in FAMILY_NAMES
    ])
    with torch.no_grad():
        for family_index, projector in enumerate(projectors):
            matrix = _fixed_projection(
                int(base_seed) + family_index,
                projector.weight.device, projector.weight.dtype)
            projector.weight.copy_(matrix.transpose(0, 1))
    return projectors


def _project_relations(projectors, differences, block_gates, scalar):
    batch = differences.shape[0]
    if tuple(differences.shape) != (
            batch, 5, 6, RAW_DIFFERENCE_BLOCKS, EMBEDDING_WIDTH):
        raise ValueError("independent router difference input drifted")
    if tuple(block_gates.shape) != (
            batch, 5, 6, RAW_DIFFERENCE_BLOCKS):
        raise ValueError("independent router block gate drifted")
    if tuple(scalar.shape) != (batch, 5, 6, SCALAR_RELATION_DIM):
        raise ValueError("independent router scalar input drifted")
    projected = []
    for block_index, family_index in enumerate(BLOCK_FAMILY_INDICES):
        block = torch.tanh(projectors[family_index](
            differences[:, :, :, block_index]))
        projected.append(
            block * block_gates[:, :, :, block_index].unsqueeze(-1))
    relation = torch.cat((*projected, scalar), dim=-1)
    if tuple(relation.shape) != (batch, 5, 6, RICH_RELATION_DIM):
        raise RuntimeError("independent router relation width drifted")
    if not torch.isfinite(relation).all().item():
        raise RuntimeError("independent router relation is non-finite")
    return relation


class _IndependentSetBackbone(nn.Module):
    def __init__(self, hidden_dim=37, residual_scale=0.1):
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

    def set_tokens(self, relations, candidate_valid):
        if (relations.ndim != 4 or relations.shape[1:] != (
                5, 6, RICH_RELATION_DIM) or
                tuple(candidate_valid.shape) != (relations.shape[0], 6) or
                candidate_valid.dtype != torch.bool):
            raise ValueError("independent set input contract drifted")
        if not candidate_valid.any(dim=1).all().item():
            raise ValueError("independent set event has no valid candidate")
        tokens = self.candidate_encoder(
            fixed_temporal_pool_rich(relations.float()))
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


class UtilityRouter(_IndependentSetBackbone):
    """Commit, ranking and benefit only; no safety outputs."""

    def __init__(self, hidden_dim=37, residual_scale=0.1):
        super().__init__(hidden_dim=hidden_dim, residual_scale=residual_scale)
        self.event_commit_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.candidate_rank_head = nn.Linear(hidden_dim, 1)
        self.candidate_benefit_head = nn.Linear(hidden_dim, 1)

    def forward(self, relations, candidate_valid):
        tokens = self.set_tokens(relations, candidate_valid)
        valid = candidate_valid.unsqueeze(-1)
        event_mean = ((tokens * valid).sum(dim=1) /
                      valid.sum(dim=1).clamp_min(1))
        event_max = tokens.masked_fill(
            ~valid, -float("inf")).max(dim=1).values
        outputs = {
            "event_commit_logit": self.event_commit_head(
                torch.cat((event_mean, event_max), dim=-1)).squeeze(-1),
            "candidate_rank_logits": self.candidate_rank_head(
                tokens).squeeze(-1),
            "candidate_benefit_logits": self.candidate_benefit_head(
                tokens).squeeze(-1),
        }
        if any(not torch.isfinite(value).all().item()
               for value in outputs.values()):
            raise RuntimeError("utility output is non-finite")
        return outputs


class SafetyTrajectoryCritic(_IndependentSetBackbone):
    """Independent catastrophe and 3/5/10-frame trajectory critic."""

    def __init__(self, hidden_dim=37, residual_scale=0.1):
        super().__init__(hidden_dim=hidden_dim, residual_scale=residual_scale)
        self.candidate_catastrophe_head = nn.Linear(hidden_dim, 1)
        self.candidate_trajectory_head = nn.Linear(
            hidden_dim, len(HORIZONS) * len(TRAJECTORY_METRICS))

    def forward(self, relations, candidate_valid):
        tokens = self.set_tokens(relations, candidate_valid)
        raw = self.candidate_trajectory_head(tokens).reshape(
            relations.shape[0], 6, len(HORIZONS),
            len(TRAJECTORY_METRICS))
        trajectory = torch.stack((
            torch.sigmoid(raw[..., 0]),
            torch.sigmoid(raw[..., 1]),
            torch.tanh(raw[..., 2]),
            torch.sigmoid(raw[..., 3]),
            torch.sigmoid(raw[..., 4]),
        ), dim=-1)
        outputs = {
            "candidate_catastrophe_logits":
                self.candidate_catastrophe_head(tokens).squeeze(-1),
            "candidate_trajectory": trajectory,
        }
        if any(not torch.isfinite(value).all().item()
               for value in outputs.values()):
            raise RuntimeError("safety output is non-finite")
        return outputs


class IndependentUtilitySafetyRouter(nn.Module):
    """Two fully separate learned relation and set-processing paths."""

    def __init__(self, hidden_dim=37, residual_scale=0.1,
                 base_projection_seed=20260901):
        super().__init__()
        self.utility_projectors = _new_projectors(base_projection_seed)
        self.safety_projectors = _new_projectors(base_projection_seed)
        self.utility_router = UtilityRouter(
            hidden_dim=hidden_dim, residual_scale=residual_scale)
        self.safety_critic = SafetyTrajectoryCritic(
            hidden_dim=hidden_dim, residual_scale=residual_scale)

    def utility_relations(self, differences, block_gates, scalar):
        return _project_relations(
            self.utility_projectors, differences, block_gates, scalar)

    def safety_relations(self, differences, block_gates, scalar):
        return _project_relations(
            self.safety_projectors, differences, block_gates, scalar)

    def forward(self, differences, block_gates, scalar, candidate_valid):
        utility = self.utility_router(
            self.utility_relations(differences, block_gates, scalar),
            candidate_valid)
        safety = self.safety_critic(
            self.safety_relations(differences, block_gates, scalar),
            candidate_valid)
        outputs = {**utility, **safety}
        if set(outputs) != {
                "event_commit_logit", "candidate_rank_logits",
                "candidate_benefit_logits",
                "candidate_catastrophe_logits", "candidate_trajectory"}:
            raise RuntimeError("independent output contract drifted")
        return outputs

    def utility_parameters(self):
        yield from self.utility_projectors.parameters()
        yield from self.utility_router.parameters()

    def safety_parameters(self):
        yield from self.safety_projectors.parameters()
        yield from self.safety_critic.parameters()


__all__ = [
    "HORIZONS",
    "TRAJECTORY_METRICS",
    "UtilityRouter",
    "SafetyTrajectoryCritic",
    "IndependentUtilitySafetyRouter",
]
