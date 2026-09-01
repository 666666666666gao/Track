"""Direct selective router over deduplicated causal RGB-D hypotheses.

The utility and safety towers remain parameter-disjoint. Exact duplicate
five-step trajectories are collapsed by the caller through ``candidate_valid``;
this module never observes ground truth or candidate multiplicity.
"""

import torch
from torch import nn

from .lachtt_causal_quantile_survival import (
    CANDIDATE_COUNT,
    _CausalSetTower,
    _candidate_gather,
    canonical_to_input_indices,
)


class DirectBenefitTower(_CausalSetTower):
    def __init__(self, projection_seed=20260923, residual_scale=0.1):
        super().__init__(projection_seed, residual_scale=residual_scale)
        self.benefit_head = nn.Linear(32, 1)

    def forward(self, differences, block_gates, scalar, candidate_valid):
        final = self.causal_tokens(
            differences, block_gates, scalar, candidate_valid)[:, -1]
        return self.benefit_head(final).squeeze(-1)


class DirectCatastropheTower(_CausalSetTower):
    def __init__(self, projection_seed=20261923, residual_scale=0.1):
        super().__init__(projection_seed, residual_scale=residual_scale)
        self.catastrophe_head = nn.Linear(32, 1)

    def forward(self, differences, block_gates, scalar, candidate_valid):
        final = self.causal_tokens(
            differences, block_gates, scalar, candidate_valid)[:, -1]
        return self.catastrophe_head(final).squeeze(-1)


class UniqueHypothesisSelectiveRouter(nn.Module):
    """Predict direct benefit and catastrophe probabilities per hypothesis."""

    def __init__(self, benefit_projection_seed=20260923,
                 catastrophe_projection_seed=20261923,
                 residual_scale=0.1, catastrophe_penalty=4.0):
        super().__init__()
        self.catastrophe_penalty = float(catastrophe_penalty)
        self.benefit_tower = DirectBenefitTower(
            benefit_projection_seed, residual_scale=residual_scale)
        self.catastrophe_tower = DirectCatastropheTower(
            catastrophe_projection_seed, residual_scale=residual_scale)

    def forward(self, differences, block_gates, scalar, candidate_valid,
                candidate_role_ids):
        if differences.ndim != 5:
            raise ValueError("unique-hypothesis difference rank drifted")
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

        benefit_logit = self.benefit_tower(
            canonical_differences, canonical_gates,
            canonical_scalar, canonical_valid)
        catastrophe_logit = self.catastrophe_tower(
            canonical_differences, canonical_gates,
            canonical_scalar, canonical_valid)
        invalid = ~canonical_valid
        benefit_logit = benefit_logit.masked_fill(invalid, -80.0)
        catastrophe_logit = catastrophe_logit.masked_fill(invalid, 80.0)
        benefit_probability = torch.sigmoid(benefit_logit)
        catastrophe_probability = torch.sigmoid(catastrophe_logit)
        dominance_score = (
            benefit_probability -
            self.catastrophe_penalty * catastrophe_probability)
        dominance_score = dominance_score.masked_fill(invalid, -1.0e9)

        canonical = {
            "benefit_logit": benefit_logit,
            "catastrophe_logit": catastrophe_logit,
            "benefit_probability": benefit_probability,
            "catastrophe_probability": catastrophe_probability,
            "dominance_score": dominance_score,
        }
        outputs = {
            name: _candidate_gather(value, candidate_role_ids, axis=1)
            for name, value in canonical.items()
        }
        if any(not torch.isfinite(value).all().item()
               for value in outputs.values()):
            raise RuntimeError("unique-hypothesis output is non-finite")
        return outputs

    def utility_parameters(self):
        yield from self.benefit_tower.parameters()

    def safety_parameters(self):
        yield from self.catastrophe_tower.parameters()

    def parameter_partition(self):
        utility = {id(parameter) for parameter in self.utility_parameters()}
        safety = {id(parameter) for parameter in self.safety_parameters()}
        return utility, safety


def trainable_parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters()
               if parameter.requires_grad)


__all__ = [
    "DirectBenefitTower",
    "DirectCatastropheTower",
    "UniqueHypothesisSelectiveRouter",
    "trainable_parameter_count",
]
