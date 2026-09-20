"""Utility-conditioned temporal prediction of strict-benefit components."""

import torch
from torch import nn

from lib.models.sttrack.lachtt_utility_conditioned_temporal_harm import (
    CACHED_HORIZON,
    IDENTITY_RELATION_DIM,
    SAFETY_INPUT_DIM,
    SCALAR_RELATION_DIM,
    TEMPORAL_HIDDEN_DIM,
    bounded_identity_relation,
)


class UtilityConditionedStrictBenefitTemporalHead(nn.Module):
    """Predict the three frozen components of a beneficial action."""

    def __init__(self):
        super().__init__()
        self.age_projection = nn.Linear(SAFETY_INPUT_DIM, TEMPORAL_HIDDEN_DIM)
        self.temporal = nn.GRU(
            TEMPORAL_HIDDEN_DIM, TEMPORAL_HIDDEN_DIM, batch_first=True)
        self.gain_head = nn.Linear(TEMPORAL_HIDDEN_DIM, 1)
        self.branch_mean_head = nn.Linear(TEMPORAL_HIDDEN_DIM, 1)
        self.early_hit_rate_head = nn.Linear(TEMPORAL_HIDDEN_DIM, 1)

    def forward(self, scalar, identity):
        batch = scalar.shape[0]
        if tuple(scalar.shape) != (
                batch, CACHED_HORIZON, SCALAR_RELATION_DIM):
            raise ValueError("utility-top scalar relation shape drifted")
        if tuple(identity.shape) != (
                batch, CACHED_HORIZON, IDENTITY_RELATION_DIM):
            raise ValueError("utility-top identity relation shape drifted")
        model_input = torch.cat((scalar.float(), identity.float()), dim=-1)
        if tuple(model_input.shape) != (
                batch, CACHED_HORIZON, SAFETY_INPUT_DIM):
            raise RuntimeError("strict-benefit temporal input shape drifted")
        age_hidden = torch.tanh(self.age_projection(model_input))
        temporal_output, _ = self.temporal(age_hidden)
        last = temporal_output[:, -1]
        gain = torch.tanh(self.gain_head(last)).squeeze(-1)
        branch_mean = torch.sigmoid(self.branch_mean_head(last)).squeeze(-1)
        early_hit_rate = torch.sigmoid(
            self.early_hit_rate_head(last)).squeeze(-1)
        for name, value in (
                ("predicted_gain_h10", gain),
                ("predicted_branch_mean_h10", branch_mean),
                ("predicted_early_hit_rate_h5", early_hit_rate)):
            if tuple(value.shape) != (batch,) or not torch.isfinite(value).all():
                raise RuntimeError(f"{name} output drifted")
        return {
            "predicted_gain_h10": gain,
            "predicted_branch_mean_h10": branch_mean,
            "predicted_early_hit_rate_h5": early_hit_rate,
            "age_hidden": age_hidden,
        }


def trainable_parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters()
               if parameter.requires_grad)


__all__ = [
    "UtilityConditionedStrictBenefitTemporalHead",
    "bounded_identity_relation",
    "trainable_parameter_count",
]
