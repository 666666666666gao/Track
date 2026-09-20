"""Bounded scalar target--distractor memory over frozen RGB-D trajectories."""

import math

import torch
from torch import nn
import torch.nn.functional as F


RELATION_DIM = 49


def _normalize(value, epsilon):
    return F.normalize(value.float(), p=2.0, dim=-1, eps=float(epsilon))


def _maximum_other(values):
    candidates = values.shape[-1]
    expanded = values.unsqueeze(-2).expand(*values.shape[:-1], candidates, -1)
    diagonal = torch.eye(candidates, dtype=torch.bool, device=values.device)
    diagonal = diagonal.reshape(*([1] * (expanded.ndim - 2)), candidates, candidates)
    return expanded.masked_fill(diagonal, -float("inf")).max(dim=-1).values


def _distractor_similarity(value):
    pairwise = torch.einsum("bhcd,bhkd->bhck", value, value)
    candidates = value.shape[2]
    diagonal = torch.eye(candidates, dtype=torch.bool, device=value.device)
    return pairwise.masked_fill(
        diagonal.reshape(1, 1, candidates, candidates),
        -float("inf")).max(dim=-1).values


def _fixed_ema_relations(value, anchor, anchor_score, alpha, epsilon):
    batch, horizon, candidates, width = value.shape
    memory = anchor.reshape(batch, 1, width).expand(-1, candidates, -1).clone()
    anchor_vector = anchor.reshape(batch, 1, width)
    rows = []
    distractors = _distractor_similarity(value)
    strongest_anchor_other = _maximum_other(anchor_score)
    for age in range(horizon):
        current = value[:, age]
        memory_cosine = (current * memory).sum(dim=-1)
        memory_anchor = (memory * anchor_vector).sum(dim=-1)
        gap = 0.5 * (anchor_score[:, age] - strongest_anchor_other[:, age])
        rows.append(torch.stack((
            anchor_score[:, age],
            memory_cosine,
            memory_anchor,
            distractors[:, age],
            gap,
        ), dim=-1))
        memory = _normalize(
            (1.0 - float(alpha)) * memory + float(alpha) * current,
            epsilon)
    return torch.stack(rows, dim=1)


def _unanchored_relations(value, alpha, epsilon):
    batch, horizon, candidates, _ = value.shape
    memory = value[:, 0].clone()
    distractors = _distractor_similarity(value)
    rows = []
    for age in range(horizon):
        current = value[:, age]
        rows.append(torch.stack((
            (current * memory).sum(dim=-1),
            distractors[:, age],
        ), dim=-1))
        memory = _normalize(
            (1.0 - float(alpha)) * memory + float(alpha) * current,
            epsilon)
    return torch.stack(rows, dim=1)


def _topk_anchor_score(value, bank, top_k, epsilon):
    value = _normalize(value, epsilon)
    bank = _normalize(bank, epsilon)
    similarity = torch.einsum("bhcd,btd->bhct", value, bank)
    return similarity.topk(int(top_k), dim=-1).values.mean(dim=-1)


def _validity_adjust(value, validity, floor):
    while validity.ndim < value.ndim:
        validity = validity.unsqueeze(-1)
    return validity * value + (1.0 - validity) * float(floor)


def _weighted_mean(value, validity, dimensions):
    numerator = (value * validity).sum(dim=dimensions)
    denominator = validity.sum(dim=dimensions).clamp_min(1.0)
    return numerator / denominator


@torch.no_grad()
def build_target_distractor_relations(
        features, initial_image, identity_text, native_rgb_bank,
        native_depth_bank, alpha=0.2, epsilon=1e-6, top_k=4,
        depth_missing_floor=-1.0):
    """Return detached [batch, H5, six candidates, 49] relations."""
    expected_keys = {
        "clip_image", "native_depth", "native_fused", "native_rgb",
        "query_depth", "query_rgb", "raw_depth", "scalars",
    }
    if set(features) != expected_keys:
        raise ValueError("cached feature keys drifted")
    batch, horizon, candidates, width = features["clip_image"].shape
    if (horizon, candidates, width) != (5, 6, 768):
        raise ValueError("cached trajectory shape drifted")
    if (tuple(initial_image.shape) != (batch, 1, 768) or
            tuple(identity_text.shape) != (batch, 1, 768) or
            tuple(native_rgb_bank.shape) != (batch, 64, 768) or
            tuple(native_depth_bank.shape) != (batch, 64, 768)):
        raise ValueError("target anchor shape drifted")
    if not all(torch.isfinite(value.float()).all().item()
               for value in (*features.values(), initial_image, identity_text,
                             native_rgb_bank, native_depth_bank)):
        raise ValueError("non-finite cached memory input")

    clip = _normalize(features["clip_image"], epsilon)
    rgb = _normalize(features["native_rgb"], epsilon)
    depth = _normalize(features["native_depth"], epsilon)
    fused = _normalize(features["native_fused"], epsilon)
    query_rgb = _normalize(features["query_rgb"], epsilon)
    query_depth = _normalize(features["query_depth"], epsilon)
    initial = _normalize(initial_image[:, 0], epsilon)
    text = _normalize(identity_text[:, 0], epsilon)
    rgb_anchor = _normalize(native_rgb_bank.mean(dim=1), epsilon)
    depth_anchor = _normalize(native_depth_bank.mean(dim=1), epsilon)

    clip_anchor_score = torch.einsum("bhcd,bd->bhc", clip, initial)
    clip_text_score = torch.einsum("bhcd,bd->bhc", clip, text)
    clip_block = _fixed_ema_relations(
        clip, initial, clip_anchor_score, alpha, epsilon)
    text_gap = 0.5 * (clip_text_score - _maximum_other(clip_text_score))
    clip_block = torch.cat((
        clip_block,
        clip_text_score.unsqueeze(-1),
        text_gap.unsqueeze(-1),
    ), dim=-1)

    rgb_anchor_score = _topk_anchor_score(
        rgb, native_rgb_bank, top_k, epsilon)
    rgb_block = _fixed_ema_relations(
        rgb, rgb_anchor, rgb_anchor_score, alpha, epsilon)

    raw_depth = features["raw_depth"].float()
    validity_map = raw_depth[:, :, :, 1].clamp(0.0, 1.0)
    validity = validity_map.mean(dim=(-1, -2))
    depth_anchor_score = _topk_anchor_score(
        depth, native_depth_bank, top_k, epsilon)
    depth_block = _fixed_ema_relations(
        depth, depth_anchor, depth_anchor_score, alpha, epsilon)
    depth_block = _validity_adjust(
        depth_block, validity, depth_missing_floor)
    depth_block = torch.cat((depth_block, validity.unsqueeze(-1)), dim=-1)

    fused_block = _unanchored_relations(fused, alpha, epsilon)
    query_rgb_block = _unanchored_relations(query_rgb, alpha, epsilon)
    query_depth_block = _validity_adjust(
        _unanchored_relations(query_depth, alpha, epsilon),
        validity, depth_missing_floor)

    depth_value = raw_depth[:, :, :, 0]
    depth_mean = _weighted_mean(depth_value, validity_map, (-1, -2))
    centered = depth_value - depth_mean.unsqueeze(-1).unsqueeze(-1)
    depth_variance = _weighted_mean(
        centered * centered, validity_map, (-1, -2))
    depth_std = torch.sqrt(depth_variance.clamp_min(0.0))
    depth_abs_mean = _weighted_mean(
        torch.abs(depth_value), validity_map, (-1, -2))
    center_value = depth_value[..., 4:12, 4:12]
    center_valid = validity_map[..., 4:12, 4:12]
    center_mean = _weighted_mean(
        center_value, center_valid, (-1, -2))
    ring_mask = torch.ones_like(validity_map)
    ring_mask[..., 4:12, 4:12] = 0.0
    ring_valid = validity_map * ring_mask
    ring_mean = _weighted_mean(depth_value, ring_valid, (-1, -2))
    center_ring = center_mean - ring_mean
    flat_depth = depth_value.flatten(start_dim=-2)
    flat_valid = validity_map.flatten(start_dim=-2)
    structural = _normalize(flat_depth * flat_valid, epsilon)
    previous = torch.cat((structural[:, :1], structural[:, :-1]), dim=1)
    structural_cosine = (structural * previous).sum(dim=-1)
    previous_validity = torch.cat((validity[:, :1], validity[:, :-1]), dim=1)
    structural_validity = torch.minimum(validity, previous_validity)
    structural_cosine = _validity_adjust(
        structural_cosine, structural_validity, depth_missing_floor)
    raw_depth_block = torch.stack((
        validity,
        torch.tanh(depth_mean),
        torch.tanh(depth_std),
        torch.tanh(depth_abs_mean),
        torch.tanh(center_ring),
    ), dim=-1)
    raw_depth_block = torch.cat((
        raw_depth_block, structural_cosine.unsqueeze(-1)), dim=-1)

    target_scores = torch.stack((
        clip_anchor_score,
        rgb_anchor_score,
        _validity_adjust(depth_anchor_score, validity,
                         depth_missing_floor),
    ), dim=-1)
    cross_modal = torch.stack((
        target_scores.min(dim=-1).values,
        0.5 * (target_scores.max(dim=-1).values -
               target_scores.min(dim=-1).values),
        0.5 * torch.abs(target_scores[..., 0] - target_scores[..., 1]),
        0.5 * torch.abs(target_scores[..., 1] - target_scores[..., 2]),
    ), dim=-1)
    scalar_block = torch.tanh(features["scalars"].float())
    relations = torch.cat((
        clip_block,
        rgb_block,
        depth_block,
        fused_block,
        query_rgb_block,
        query_depth_block,
        raw_depth_block,
        scalar_block,
        cross_modal,
    ), dim=-1).detach()
    if tuple(relations.shape) != (batch, 5, 6, RELATION_DIM):
        raise RuntimeError("target--distractor relation width drifted")
    if not torch.isfinite(relations).all().item():
        raise RuntimeError("target--distractor relations are non-finite")
    return relations


class TargetDistractorMemoryRouter(nn.Module):
    """Small scalar-only commit-then-rank router."""

    def __init__(self, hidden_dim=64, residual_scale=0.1):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.residual_scale = float(residual_scale)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(RELATION_DIM, hidden_dim),
            nn.GELU(),
        )
        self.temporal = nn.GRU(
            hidden_dim, hidden_dim, num_layers=1, batch_first=True)
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
        if (relations.ndim != 4 or relations.shape[1:] != (5, 6, RELATION_DIM) or
                tuple(candidate_valid.shape) != (relations.shape[0], 6) or
                candidate_valid.dtype != torch.bool):
            raise ValueError("memory router input contract drifted")
        if not torch.isfinite(relations).all().item():
            raise ValueError("memory router relation is non-finite")
        batch = relations.shape[0]
        encoded = self.candidate_encoder(relations.float())
        temporal_input = encoded.permute(0, 2, 1, 3).reshape(
            batch * 6, 5, self.hidden_dim)
        _, temporal_state = self.temporal(temporal_input)
        tokens = temporal_state[-1].reshape(batch, 6, self.hidden_dim)
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
            raise RuntimeError("memory router output is non-finite")
        return outputs


__all__ = [
    "RELATION_DIM",
    "build_target_distractor_relations",
    "TargetDistractorMemoryRouter",
]
