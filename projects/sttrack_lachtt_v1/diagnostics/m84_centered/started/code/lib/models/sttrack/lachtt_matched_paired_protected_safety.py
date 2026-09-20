"""Matched candidate-only and candidate-versus-protected safety evidence."""

import torch
from torch import nn


CACHED_HORIZON = 5
CANDIDATE_COUNT = 6
MODALITY_COUNT = 3
NATIVE_DIM = 768
SCALAR_RELATION_DIM = 49
SCALAR_SUMMARY_DIM = SCALAR_RELATION_DIM * 4
IDENTITY_RELATION_DIM = MODALITY_COUNT * 3 * 4
SAFETY_INPUT_DIM = SCALAR_SUMMARY_DIM + IDENTITY_RELATION_DIM
HARM_HORIZONS = (3, 5, 10)


def _temporal_summary(value):
    return torch.cat((
        value[:, -1],
        value.mean(dim=1),
        value.min(dim=1).values,
        value.max(dim=1).values,
    ), dim=-1)


def bounded_identity_relation(candidate, protected, mode):
    """Return the per-age CAND or PAIR identity relation."""
    batch = candidate.shape[0]
    if tuple(candidate.shape) != (
            batch, CACHED_HORIZON, CANDIDATE_COUNT,
            MODALITY_COUNT, NATIVE_DIM):
        raise ValueError("candidate native feature shape drifted")
    if tuple(protected.shape) != (
            batch, CACHED_HORIZON, MODALITY_COUNT, NATIVE_DIM):
        raise ValueError("protected native feature shape drifted")
    if mode == "candidate":
        reference = candidate[:, :1].expand_as(candidate)
    elif mode == "paired":
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


def summarize_identity_relation(relation, candidate_valid):
    """Summarize an already duplicate-aggregated identity relation."""
    batch = relation.shape[0]
    if tuple(relation.shape) != (
            batch, CACHED_HORIZON, CANDIDATE_COUNT,
            MODALITY_COUNT, 3):
        raise ValueError("identity relation shape drifted")
    if (tuple(candidate_valid.shape) != (batch, CANDIDATE_COUNT) or
            candidate_valid.dtype != torch.bool or
            not candidate_valid.any(dim=1).all().item()):
        raise ValueError("candidate validity drifted")
    summary = _temporal_summary(relation).reshape(
        batch, CANDIDATE_COUNT, IDENTITY_RELATION_DIM)
    summary = summary.masked_fill(~candidate_valid.unsqueeze(-1), 0.0)
    if (tuple(summary.shape) != (
            batch, CANDIDATE_COUNT, IDENTITY_RELATION_DIM) or
            not torch.isfinite(summary).all().item() or
            float(summary.min()) < -1.0 or float(summary.max()) > 1.0):
        raise RuntimeError("bounded identity summary contract drifted")
    return summary


class MatchedProtectedHarmHead(nn.Module):
    """A parameter-matched 232-to-3 signed protected-harm head."""

    def __init__(self):
        super().__init__()
        self.harm_head = nn.Linear(SAFETY_INPUT_DIM, len(HARM_HORIZONS))

    def forward(self, scalar, identity_summary, candidate_valid):
        batch = scalar.shape[0]
        if tuple(scalar.shape) != (
                batch, CACHED_HORIZON, CANDIDATE_COUNT,
                SCALAR_RELATION_DIM):
            raise ValueError("scalar relation shape drifted")
        if tuple(identity_summary.shape) != (
                batch, CANDIDATE_COUNT, IDENTITY_RELATION_DIM):
            raise ValueError("identity relation shape drifted")
        scalar_summary = _temporal_summary(scalar.float())
        safety_input = torch.cat(
            (scalar_summary, identity_summary.float()), dim=-1)
        if tuple(safety_input.shape) != (
                batch, CANDIDATE_COUNT, SAFETY_INPUT_DIM):
            raise RuntimeError("matched safety input shape drifted")
        harm = torch.tanh(self.harm_head(safety_input))
        harm = harm.masked_fill(~candidate_valid.unsqueeze(-1), 1.0)
        if (tuple(harm.shape) != (
                batch, CANDIDATE_COUNT, len(HARM_HORIZONS)) or
                not torch.isfinite(harm).all().item()):
            raise RuntimeError("matched protected-harm output drifted")
        return {
            "predicted_harm": harm,
            "identity_summary": identity_summary,
        }


def trainable_parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters()
               if parameter.requires_grad)


__all__ = [
    "HARM_HORIZONS",
    "IDENTITY_RELATION_DIM",
    "MatchedProtectedHarmHead",
    "SAFETY_INPUT_DIM",
    "bounded_identity_relation",
    "summarize_identity_relation",
    "trainable_parameter_count",
]
