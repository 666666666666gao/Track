"""Utility-conditioned candidate-only and paired-protected temporal harm."""

import torch
from torch import nn


CACHED_HORIZON = 5
CANDIDATE_COUNT = 6
MODALITY_COUNT = 3
NATIVE_DIM = 768
SCALAR_RELATION_DIM = 49
IDENTITY_RELATION_DIM = MODALITY_COUNT * 3
SAFETY_INPUT_DIM = SCALAR_RELATION_DIM + IDENTITY_RELATION_DIM
TEMPORAL_HIDDEN_DIM = 32
HARM_HORIZONS = (3, 5, 10)


def bounded_identity_relation(candidate, protected, mode):
    """Return the per-age CAND-TEMP or PAIR-TEMP identity relation."""
    batch = candidate.shape[0]
    if tuple(candidate.shape) != (
            batch, CACHED_HORIZON, CANDIDATE_COUNT,
            MODALITY_COUNT, NATIVE_DIM):
        raise ValueError("candidate native feature shape drifted")
    if tuple(protected.shape) != (
            batch, CACHED_HORIZON, MODALITY_COUNT, NATIVE_DIM):
        raise ValueError("protected native feature shape drifted")
    if mode == "candidate_temporal":
        reference = candidate[:, :1].expand_as(candidate)
    elif mode == "paired_temporal":
        reference = protected.unsqueeze(2).expand_as(candidate)
    else:
        raise ValueError("identity relation mode drifted")

    candidate = candidate.float()
    reference = reference.float()
    epsilon = 1.0e-6
    candidate_norm = torch.linalg.vector_norm(candidate, dim=-1)
    reference_norm = torch.linalg.vector_norm(reference, dim=-1)
    cosine = ((candidate * reference).sum(dim=-1) /
              (candidate_norm * reference_norm + epsilon)).clamp(-1.0, 1.0)
    normalized_l2 = (torch.linalg.vector_norm(
        candidate - reference, dim=-1) /
        (candidate_norm + reference_norm + epsilon)).clamp(0.0, 1.0)
    log_norm_ratio = (torch.log(
        (candidate_norm + epsilon) / (reference_norm + epsilon)
    ).clamp(-4.0, 4.0) / 4.0)
    relation = torch.stack(
        (cosine, normalized_l2, log_norm_ratio), dim=-1)
    if (tuple(relation.shape) != (
            batch, CACHED_HORIZON, CANDIDATE_COUNT,
            MODALITY_COUNT, 3) or
            not torch.isfinite(relation).all().item() or
            float(relation.min()) < -1.0 or float(relation.max()) > 1.0):
        raise RuntimeError("bounded identity relation contract drifted")
    return relation


class UtilityConditionedTemporalHarmHead(nn.Module):
    """Matched 5-age temporal signed protected-harm head."""

    def __init__(self):
        super().__init__()
        self.age_projection = nn.Linear(SAFETY_INPUT_DIM, TEMPORAL_HIDDEN_DIM)
        self.temporal = nn.GRU(
            TEMPORAL_HIDDEN_DIM, TEMPORAL_HIDDEN_DIM, batch_first=True)
        self.harm_head = nn.Linear(
            TEMPORAL_HIDDEN_DIM, len(HARM_HORIZONS))

    def forward(self, scalar, identity):
        batch = scalar.shape[0]
        if tuple(scalar.shape) != (
                batch, CACHED_HORIZON, SCALAR_RELATION_DIM):
            raise ValueError("utility-top scalar relation shape drifted")
        if tuple(identity.shape) != (
                batch, CACHED_HORIZON, IDENTITY_RELATION_DIM):
            raise ValueError("utility-top identity relation shape drifted")
        safety_input = torch.cat((scalar.float(), identity.float()), dim=-1)
        if tuple(safety_input.shape) != (
                batch, CACHED_HORIZON, SAFETY_INPUT_DIM):
            raise RuntimeError("temporal safety input shape drifted")
        age_hidden = torch.tanh(self.age_projection(safety_input))
        temporal_output, _ = self.temporal(age_hidden)
        harm = torch.tanh(self.harm_head(temporal_output[:, -1]))
        if (tuple(harm.shape) != (batch, len(HARM_HORIZONS)) or
                not torch.isfinite(harm).all().item()):
            raise RuntimeError("utility-conditioned harm output drifted")
        return {
            "predicted_harm": harm,
            "age_hidden": age_hidden,
        }


def trainable_parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters()
               if parameter.requires_grad)


__all__ = [
    "HARM_HORIZONS",
    "IDENTITY_RELATION_DIM",
    "SAFETY_INPUT_DIM",
    "TEMPORAL_HIDDEN_DIM",
    "UtilityConditionedTemporalHarmHead",
    "bounded_identity_relation",
    "trainable_parameter_count",
]
