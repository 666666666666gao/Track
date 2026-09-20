"""Exact candidate-role canonicalization for independent LACHTT routers.

Candidate role identifiers are bookkeeping metadata only.  They are used to
restore the fixed candidate-generation order before the learned router and to
restore the caller's order afterwards.  They never enter a learned layer.
"""

import torch

from .lachtt_independent_utility_safety import (
    IndependentUtilitySafetyRouter,
)


CANDIDATE_ROLE_NAMES = (
    "current_peak0",
    "current_peak1",
    "last_reliable_peak0",
    "last_reliable_peak1",
    "velocity_peak0",
    "velocity_peak1",
)
CANDIDATE_ROLE_COUNT = len(CANDIDATE_ROLE_NAMES)


def _candidate_gather(value, indices, axis):
    """Gather a per-event candidate axis with a `[batch, candidate]` index."""
    if value.ndim <= axis or value.shape[0] != indices.shape[0]:
        raise ValueError("candidate gather input shape drifted")
    if value.shape[axis] != CANDIDATE_ROLE_COUNT:
        raise ValueError("candidate gather axis width drifted")
    view = [indices.shape[0]] + [1] * (value.ndim - 1)
    view[axis] = CANDIDATE_ROLE_COUNT
    expand = list(value.shape)
    expand[axis] = CANDIDATE_ROLE_COUNT
    gather_index = indices.reshape(view).expand(expand)
    return torch.gather(value, axis, gather_index)


def validate_candidate_role_ids(candidate_role_ids, batch_size, device):
    """Validate and return canonical-to-input indices for every event."""
    if not isinstance(candidate_role_ids, torch.Tensor):
        raise TypeError("candidate role ids must be a tensor")
    if candidate_role_ids.dtype != torch.int64:
        raise TypeError("candidate role ids must use torch.int64")
    if tuple(candidate_role_ids.shape) != (
            int(batch_size), CANDIDATE_ROLE_COUNT):
        raise ValueError("candidate role id shape drifted")
    if candidate_role_ids.device != device:
        raise ValueError("candidate role ids must share the input device")
    canonical = torch.arange(
        CANDIDATE_ROLE_COUNT, dtype=torch.int64, device=device)
    sorted_ids, canonical_to_input = torch.sort(
        candidate_role_ids, dim=1)
    if not torch.equal(
            sorted_ids, canonical.unsqueeze(0).expand(int(batch_size), -1)):
        raise ValueError("candidate role ids must be a permutation of 0..5")
    return canonical_to_input


class CanonicalRoleIndependentUtilitySafetyRouter(
        IndependentUtilitySafetyRouter):
    """M15 independent router with parameter-free role canonicalization."""

    def forward(self, differences, block_gates, scalar, candidate_valid,
                candidate_role_ids):
        if differences.ndim != 5:
            raise ValueError("canonical router difference rank drifted")
        batch_size = differences.shape[0]
        canonical_to_input = validate_candidate_role_ids(
            candidate_role_ids, batch_size, differences.device)
        canonical_differences = _candidate_gather(
            differences, canonical_to_input, axis=2)
        canonical_block_gates = _candidate_gather(
            block_gates, canonical_to_input, axis=2)
        canonical_scalar = _candidate_gather(
            scalar, canonical_to_input, axis=2)
        canonical_valid = _candidate_gather(
            candidate_valid, canonical_to_input, axis=1)

        canonical_outputs = super().forward(
            canonical_differences, canonical_block_gates,
            canonical_scalar, canonical_valid)
        outputs = {
            "event_commit_logit": canonical_outputs["event_commit_logit"],
        }
        for name in (
                "candidate_rank_logits", "candidate_benefit_logits",
                "candidate_catastrophe_logits", "candidate_trajectory"):
            outputs[name] = _candidate_gather(
                canonical_outputs[name], candidate_role_ids, axis=1)
        return outputs


__all__ = [
    "CANDIDATE_ROLE_COUNT",
    "CANDIDATE_ROLE_NAMES",
    "CanonicalRoleIndependentUtilitySafetyRouter",
    "validate_candidate_role_ids",
]
