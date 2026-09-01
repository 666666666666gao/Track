"""Low-capacity protected-harm guard for sequence-nested calibration.

The utility tower uses the full candidate-own RGB-D/language relation.  The
safety tower is deliberately restricted to a linear summary of the frozen
49-D scalar relation and predicts signed candidate-versus-protected harm at
H3, H5 and H10.  Calibration is performed by the experiment runner, not by
this module.
"""

import torch
from torch import nn

from .lachtt_causal_quantile_survival import (
    CACHED_HORIZON,
    CANDIDATE_COUNT,
    _candidate_gather,
    canonical_to_input_indices,
)
from .lachtt_learned_bounded_roi_association import SCALAR_RELATION_DIM
from .lachtt_unique_hypothesis_selective_router import DirectBenefitTower


HARM_HORIZONS = (3, 5, 10)
HARM_SUMMARY_MULTIPLIER = 4
HARM_INPUT_DIM = SCALAR_RELATION_DIM * HARM_SUMMARY_MULTIPLIER


class LinearProtectedCounterfactualHarmTower(nn.Module):
    """Predict signed protected harm from a bounded temporal scalar summary."""

    def __init__(self):
        super().__init__()
        self.harm_head = nn.Linear(HARM_INPUT_DIM, len(HARM_HORIZONS))

    def forward(self, scalar, candidate_valid):
        batch = scalar.shape[0]
        if tuple(scalar.shape) != (
                batch, CACHED_HORIZON, CANDIDATE_COUNT,
                SCALAR_RELATION_DIM):
            raise ValueError("protected-harm scalar input drifted")
        if (tuple(candidate_valid.shape) != (batch, CANDIDATE_COUNT) or
                candidate_valid.dtype != torch.bool or
                not candidate_valid.any(dim=1).all().item()):
            raise ValueError("protected-harm candidate validity drifted")
        scalar = scalar.float()
        summary = torch.cat((
            scalar[:, -1],
            scalar.mean(dim=1),
            scalar.min(dim=1).values,
            scalar.max(dim=1).values,
        ), dim=-1)
        if tuple(summary.shape) != (
                batch, CANDIDATE_COUNT, HARM_INPUT_DIM):
            raise RuntimeError("protected-harm summary shape drifted")
        harm = torch.tanh(self.harm_head(summary))
        harm = harm.masked_fill(~candidate_valid.unsqueeze(-1), 1.0)
        if (tuple(harm.shape) != (
                batch, CANDIDATE_COUNT, len(HARM_HORIZONS)) or
                not torch.isfinite(harm).all().item()):
            raise RuntimeError("protected-harm output drifted")
        return harm


class NestedCalibratedCounterfactualHarmRouter(nn.Module):
    """Full-relation utility plus disjoint linear protected-harm prediction."""

    def __init__(self, benefit_projection_seed=20260923,
                 residual_scale=0.1):
        super().__init__()
        self.benefit_tower = DirectBenefitTower(
            benefit_projection_seed, residual_scale=residual_scale)
        self.harm_tower = LinearProtectedCounterfactualHarmTower()

    def forward(self, differences, block_gates, scalar, candidate_valid,
                candidate_role_ids):
        if differences.ndim != 5:
            raise ValueError("nested protected-harm difference rank drifted")
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
        benefit_logit = benefit_logit.masked_fill(~canonical_valid, -80.0)
        benefit_probability = torch.sigmoid(benefit_logit)
        predicted_harm = self.harm_tower(
            canonical_scalar, canonical_valid)
        canonical = {
            "benefit_logit": benefit_logit,
            "benefit_probability": benefit_probability,
            "predicted_harm": predicted_harm,
        }
        outputs = {
            name: _candidate_gather(value, candidate_role_ids, axis=1)
            for name, value in canonical.items()
        }
        if any(not torch.isfinite(value).all().item()
               for value in outputs.values()):
            raise RuntimeError("nested protected-harm output is non-finite")
        return outputs

    def utility_parameters(self):
        yield from self.benefit_tower.parameters()

    def safety_parameters(self):
        yield from self.harm_tower.parameters()

    def parameter_partition(self):
        utility = {id(parameter) for parameter in self.utility_parameters()}
        safety = {id(parameter) for parameter in self.safety_parameters()}
        return utility, safety


def trainable_parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters()
               if parameter.requires_grad)


__all__ = [
    "HARM_HORIZONS",
    "HARM_INPUT_DIM",
    "LinearProtectedCounterfactualHarmTower",
    "NestedCalibratedCounterfactualHarmRouter",
    "trainable_parameter_count",
]
