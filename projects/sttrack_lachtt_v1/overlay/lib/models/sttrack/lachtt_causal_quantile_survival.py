"""Causal quantile-survival router for protected RGB-D template transactions.

The utility and safety towers own disjoint parameters. Candidate role ids only
canonicalize the candidate axis; they never enter a learned operation.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

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
)


CANDIDATE_COUNT = 6
CACHED_HORIZON = 5
STEP_HIDDEN_DIM = 48
TOKEN_DIM = 32
CONTEXT_DIM = TOKEN_DIM * 3
SURVIVAL_HORIZONS = (3, 5, 10)
CANDIDATE_ROLE_NAMES = (
    "current_peak0",
    "current_peak1",
    "last_reliable_peak0",
    "last_reliable_peak1",
    "velocity_peak0",
    "velocity_peak1",
)
FORBIDDEN_MODULE_NAMES = (
    "GRU", "RNN", "LSTM", "LayerNorm", "Transformer",
    "MultiheadAttention",
)


def _fixed_projection(seed, device, dtype):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    signs = torch.randint(
        0, 2, (EMBEDDING_WIDTH, PROJECTION_WIDTH),
        generator=generator, dtype=torch.int64)
    matrix = (signs.float() * 2.0 - 1.0) / math.sqrt(PROJECTION_WIDTH)
    return matrix.to(device=device, dtype=dtype)


def _new_family_projectors(base_seed):
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
            batch, CACHED_HORIZON, CANDIDATE_COUNT,
            RAW_DIFFERENCE_BLOCKS, EMBEDDING_WIDTH):
        raise ValueError("causal router difference input drifted")
    if tuple(block_gates.shape) != (
            batch, CACHED_HORIZON, CANDIDATE_COUNT,
            RAW_DIFFERENCE_BLOCKS):
        raise ValueError("causal router gate input drifted")
    if tuple(scalar.shape) != (
            batch, CACHED_HORIZON, CANDIDATE_COUNT,
            SCALAR_RELATION_DIM):
        raise ValueError("causal router scalar input drifted")
    projected = []
    for block_index, family_index in enumerate(BLOCK_FAMILY_INDICES):
        block = torch.tanh(projectors[family_index](
            differences[:, :, :, block_index]))
        projected.append(
            block * block_gates[:, :, :, block_index].unsqueeze(-1))
    relations = torch.cat((*projected, scalar), dim=-1)
    if tuple(relations.shape) != (
            batch, CACHED_HORIZON, CANDIDATE_COUNT, RICH_RELATION_DIM):
        raise RuntimeError("causal router relation width drifted")
    if not torch.isfinite(relations).all().item():
        raise RuntimeError("causal router relation is non-finite")
    return relations


def _candidate_gather(value, indices, axis):
    if (indices.dtype != torch.int64 or value.shape[0] != indices.shape[0] or
            value.shape[axis] != CANDIDATE_COUNT):
        raise ValueError("causal candidate gather contract drifted")
    view = [indices.shape[0]] + [1] * (value.ndim - 1)
    view[axis] = CANDIDATE_COUNT
    expand = list(value.shape)
    expand[axis] = CANDIDATE_COUNT
    return torch.gather(
        value, axis, indices.reshape(view).expand(expand))


def canonical_to_input_indices(candidate_role_ids, device):
    if (candidate_role_ids.dtype != torch.int64 or
            candidate_role_ids.device != device or
            candidate_role_ids.ndim != 2 or
            candidate_role_ids.shape[1] != CANDIDATE_COUNT):
        raise ValueError("candidate role id contract drifted")
    canonical = torch.arange(
        CANDIDATE_COUNT, device=device, dtype=torch.int64)
    sorted_ids, canonical_to_input = torch.sort(candidate_role_ids, dim=1)
    if not torch.equal(
            sorted_ids,
            canonical.unsqueeze(0).expand(candidate_role_ids.shape[0], -1)):
        raise ValueError("candidate role ids must permute 0..5")
    return canonical_to_input


class _CausalSetTower(nn.Module):
    def __init__(self, projection_seed, residual_scale=0.1):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.family_projectors = _new_family_projectors(projection_seed)
        self.step_encoder = nn.Sequential(
            nn.Linear(RICH_RELATION_DIM, STEP_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(STEP_HIDDEN_DIM, TOKEN_DIM),
            nn.GELU(),
        )
        self.set_context_mixer = nn.Linear(CONTEXT_DIM, TOKEN_DIM)
        self.temporal_context_encoder = nn.Sequential(
            nn.Linear(CONTEXT_DIM, TOKEN_DIM),
            nn.GELU(),
        )

    def causal_tokens(self, differences, block_gates, scalar,
                      candidate_valid):
        batch = differences.shape[0]
        if (tuple(candidate_valid.shape) != (batch, CANDIDATE_COUNT) or
                candidate_valid.dtype != torch.bool or
                not candidate_valid.any(dim=1).all().item()):
            raise ValueError("causal candidate validity contract drifted")
        relations = _project_relations(
            self.family_projectors, differences, block_gates, scalar)
        step = self.step_encoder(relations.float())
        valid = candidate_valid[:, None, :, None]
        valid_float = valid.to(step.dtype)
        denominator = valid_float.sum(dim=2).clamp_min(1.0)
        mean_context = (step * valid_float).sum(dim=2) / denominator
        max_context = step.masked_fill(~valid, -float("inf")).max(
            dim=2).values
        set_context = torch.cat((
            step,
            mean_context[:, :, None].expand_as(step),
            max_context[:, :, None].expand_as(step),
        ), dim=-1)
        step = step + self.residual_scale * self.set_context_mixer(
            set_context)
        prefix_denominator = torch.arange(
            1, CACHED_HORIZON + 1, device=step.device,
            dtype=step.dtype).reshape(1, CACHED_HORIZON, 1, 1)
        prefix_mean = step.cumsum(dim=1) / prefix_denominator
        previous = torch.cat((torch.zeros_like(step[:, :1]), step[:, :-1]),
                             dim=1)
        current_minus_previous = step - previous
        prefix_minimum = torch.cummin(step, dim=1).values
        temporal = self.temporal_context_encoder(torch.cat((
            prefix_mean, current_minus_previous, prefix_minimum,
        ), dim=-1))
        if (tuple(temporal.shape) != (
                batch, CACHED_HORIZON, CANDIDATE_COUNT, TOKEN_DIM) or
                not torch.isfinite(temporal).all().item()):
            raise RuntimeError("causal temporal token contract drifted")
        return temporal


class UtilityQuantileTower(_CausalSetTower):
    def __init__(self, projection_seed=20260918, residual_scale=0.1):
        super().__init__(projection_seed, residual_scale=residual_scale)
        self.gain_q10_head = nn.Linear(TOKEN_DIM, 1)
        self.branch_mean_q10_head = nn.Linear(TOKEN_DIM, 1)

    def forward(self, differences, block_gates, scalar, candidate_valid):
        final = self.causal_tokens(
            differences, block_gates, scalar, candidate_valid)[:, -1]
        return {
            "gain_q10_lcb": torch.tanh(
                self.gain_q10_head(final).squeeze(-1)),
            "branch_mean_q10_lcb": torch.sigmoid(
                self.branch_mean_q10_head(final).squeeze(-1)),
        }


class SafetySurvivalTower(_CausalSetTower):
    def __init__(self, projection_seed=20261918, residual_scale=0.1):
        super().__init__(projection_seed, residual_scale=residual_scale)
        self.hazard_h3_head = nn.Linear(TOKEN_DIM, 1)
        self.hazard_h5_head = nn.Linear(TOKEN_DIM, 1)
        self.hazard_h10_head = nn.Linear(TOKEN_DIM, 1)
        self.catastrophe_head = nn.Linear(TOKEN_DIM, 1)

    def forward(self, differences, block_gates, scalar, candidate_valid):
        temporal = self.causal_tokens(
            differences, block_gates, scalar, candidate_valid)
        increments = torch.stack((
            F.softplus(self.hazard_h3_head(temporal[:, 2]).squeeze(-1)),
            F.softplus(self.hazard_h5_head(temporal[:, 4]).squeeze(-1)),
            F.softplus(self.hazard_h10_head(temporal[:, 4]).squeeze(-1)),
        ), dim=-1)
        survival = torch.exp(-torch.cumsum(increments, dim=-1))
        return {
            "hazard_increments": increments,
            "survival": survival,
            "risk_q90_ucb": 1.0 - survival,
            "catastrophe_probability": torch.sigmoid(
                self.catastrophe_head(temporal[:, -1]).squeeze(-1)),
        }


class CausalQuantileSurvivalRouter(nn.Module):
    """Disjoint utility/safety towers with deterministic dominance output."""

    def __init__(self, utility_projection_seed=20260918,
                 safety_projection_seed=20261918, residual_scale=0.1):
        super().__init__()
        self.utility_tower = UtilityQuantileTower(
            utility_projection_seed, residual_scale=residual_scale)
        self.safety_tower = SafetySurvivalTower(
            safety_projection_seed, residual_scale=residual_scale)

    def forward(self, differences, block_gates, scalar, candidate_valid,
                candidate_role_ids):
        if differences.ndim != 5:
            raise ValueError("causal router difference rank drifted")
        canonical_to_input = canonical_to_input_indices(
            candidate_role_ids, differences.device)
        canonical_differences = _candidate_gather(
            differences, canonical_to_input, axis=2)
        canonical_gates = _candidate_gather(
            block_gates, canonical_to_input, axis=2)
        canonical_scalar = _candidate_gather(
            scalar, canonical_to_input, axis=2)
        canonical_valid = _candidate_gather(
            candidate_valid, canonical_to_input, axis=1)
        canonical_outputs = {
            **self.utility_tower(
                canonical_differences, canonical_gates,
                canonical_scalar, canonical_valid),
            **self.safety_tower(
                canonical_differences, canonical_gates,
                canonical_scalar, canonical_valid),
        }
        invalid = ~canonical_valid
        canonical_outputs["gain_q10_lcb"] = canonical_outputs[
            "gain_q10_lcb"].masked_fill(invalid, -1.0)
        canonical_outputs["branch_mean_q10_lcb"] = canonical_outputs[
            "branch_mean_q10_lcb"].masked_fill(invalid, 0.0)
        canonical_outputs["hazard_increments"] = canonical_outputs[
            "hazard_increments"].masked_fill(invalid.unsqueeze(-1), 80.0)
        canonical_outputs["survival"] = torch.exp(-torch.cumsum(
            canonical_outputs["hazard_increments"], dim=-1))
        canonical_outputs["risk_q90_ucb"] = 1.0 - canonical_outputs[
            "survival"]
        canonical_outputs["catastrophe_probability"] = canonical_outputs[
            "catastrophe_probability"].masked_fill(invalid, 1.0)
        canonical_outputs["dominance_score"] = (
            canonical_outputs["gain_q10_lcb"] -
            2.0 * canonical_outputs["risk_q90_ucb"][:, :, -1] -
            0.5 * canonical_outputs["catastrophe_probability"])
        canonical_outputs["dominance_score"] = canonical_outputs[
            "dominance_score"].masked_fill(invalid, -1.0e9)
        outputs = {
            name: _candidate_gather(value, candidate_role_ids, axis=1)
            for name, value in canonical_outputs.items()
        }
        if any(not torch.isfinite(value).all().item()
               for value in outputs.values()):
            raise RuntimeError("causal quantile-survival output is non-finite")
        return outputs

    def utility_parameters(self):
        yield from self.utility_tower.parameters()

    def safety_parameters(self):
        yield from self.safety_tower.parameters()

    def parameter_partition(self):
        utility = {id(parameter) for parameter in self.utility_parameters()}
        safety = {id(parameter) for parameter in self.safety_parameters()}
        return utility, safety


def trainable_parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters()
               if parameter.requires_grad)


__all__ = [
    "CANDIDATE_COUNT",
    "CACHED_HORIZON",
    "RICH_RELATION_DIM",
    "STEP_HIDDEN_DIM",
    "TOKEN_DIM",
    "SURVIVAL_HORIZONS",
    "CANDIDATE_ROLE_NAMES",
    "FORBIDDEN_MODULE_NAMES",
    "UtilityQuantileTower",
    "SafetySurvivalTower",
    "CausalQuantileSurvivalRouter",
    "canonical_to_input_indices",
    "trainable_parameter_count",
]
