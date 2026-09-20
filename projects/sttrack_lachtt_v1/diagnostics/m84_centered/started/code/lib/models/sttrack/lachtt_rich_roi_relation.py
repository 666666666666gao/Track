"""Detached bounded rich RoI instance relations for target--distractor association."""

import math

import torch
from torch import nn
import torch.nn.functional as F

from .lachtt_target_distractor_memory import (
    build_target_distractor_relations,
)


RICH_RELATION_DIM = 177
PROJECTION_WIDTH = 8
DIFFERENCE_BLOCKS = 16
TEMPORAL_STATISTICS = (
    "mean", "max", "min", "last", "last_minus_first",
)
TEMPORAL_WIDTH = RICH_RELATION_DIM * len(TEMPORAL_STATISTICS)
FAMILY_NAMES = (
    "clip_image", "native_rgb", "native_depth", "native_fused",
    "query_rgb", "query_depth",
)


def _normalize(value, epsilon):
    return F.normalize(value.float(), p=2.0, dim=-1, eps=float(epsilon))


def _fixed_projection(seed, device, dtype):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    signs = torch.randint(
        0, 2, (768, PROJECTION_WIDTH), generator=generator,
        dtype=torch.int64)
    matrix = (signs.float() * 2.0 - 1.0) / math.sqrt(PROJECTION_WIDTH)
    return matrix.to(device=device, dtype=dtype)


def _dynamic_memory(value, anchor, alpha, epsilon):
    batch, horizon, candidates, width = value.shape
    if anchor is None:
        memory = value[:, 0].clone()
    else:
        memory = anchor.reshape(batch, 1, width).expand(
            -1, candidates, -1).clone()
    rows = []
    for age in range(horizon):
        rows.append(memory)
        memory = _normalize(
            (1.0 - float(alpha)) * memory +
            float(alpha) * value[:, age], epsilon)
    return torch.stack(rows, dim=1)


def _soft_distractor(value, scale):
    candidates = value.shape[2]
    pairwise = torch.einsum("bhcd,bhkd->bhck", value, value)
    diagonal = torch.eye(candidates, dtype=torch.bool, device=value.device)
    pairwise = pairwise.masked_fill(
        diagonal.reshape(1, 1, candidates, candidates), -float("inf"))
    weights = torch.softmax(float(scale) * pairwise, dim=-1)
    return torch.einsum("bhck,bhkd->bhcd", weights, value)


def _project_difference(left, right, projection):
    return torch.tanh(torch.matmul(left - right, projection))


def _gate_blocks(blocks, validity):
    gate = validity.unsqueeze(-1)
    return [block * gate for block in blocks]


def fixed_temporal_pool_rich(relations):
    if relations.ndim != 4 or relations.shape[1:] != (
            5, 6, RICH_RELATION_DIM):
        raise ValueError("rich temporal pool relation contract drifted")
    if not torch.isfinite(relations).all().item():
        raise ValueError("rich temporal pool relation is non-finite")
    return torch.cat((
        relations.mean(dim=1),
        relations.max(dim=1).values,
        relations.min(dim=1).values,
        relations[:, -1],
        relations[:, -1] - relations[:, 0],
    ), dim=-1)


@torch.no_grad()
def build_rich_roi_relations(
        features, initial_image, identity_text, native_rgb_bank,
        native_depth_bank, base_projection_seed=20260901, ema_alpha=0.2,
        soft_distractor_scale=4.0, epsilon=1e-6,
        native_anchor_top_k=4, depth_missing_floor=-1.0):
    """Return detached [batch, H5, six candidates, 177] relations."""
    expected_keys = {
        "clip_image", "native_depth", "native_fused", "native_rgb",
        "query_depth", "query_rgb", "raw_depth", "scalars",
    }
    if set(features) != expected_keys:
        raise ValueError("rich relation cached feature keys drifted")
    batch, horizon, candidates, width = features["clip_image"].shape
    if (horizon, candidates, width) != (5, 6, 768):
        raise ValueError("rich relation trajectory shape drifted")
    if (tuple(initial_image.shape) != (batch, 1, 768) or
            tuple(identity_text.shape) != (batch, 1, 768) or
            tuple(native_rgb_bank.shape) != (batch, 64, 768) or
            tuple(native_depth_bank.shape) != (batch, 64, 768)):
        raise ValueError("rich relation anchor shape drifted")

    values = {
        name: _normalize(features[name], epsilon)
        for name in FAMILY_NAMES
    }
    anchors = {
        "clip_initial": _normalize(initial_image[:, 0], epsilon),
        "clip_text": _normalize(identity_text[:, 0], epsilon),
        "native_rgb": _normalize(native_rgb_bank.mean(dim=1), epsilon),
        "native_depth": _normalize(native_depth_bank.mean(dim=1), epsilon),
    }
    projections = {
        name: _fixed_projection(
            int(base_projection_seed) + index,
            values[name].device, values[name].dtype)
        for index, name in enumerate(FAMILY_NAMES)
    }
    raw_depth = features["raw_depth"].float()
    validity = raw_depth[:, :, :, 1].clamp(0.0, 1.0).mean(dim=(-1, -2))

    rich_blocks = []
    clip = values["clip_image"]
    clip_memory = _dynamic_memory(
        clip, anchors["clip_initial"], ema_alpha, epsilon)
    clip_distractor = _soft_distractor(clip, soft_distractor_scale)
    clip_initial = anchors["clip_initial"].reshape(
        batch, 1, 1, 768).expand_as(clip)
    clip_text = anchors["clip_text"].reshape(
        batch, 1, 1, 768).expand_as(clip)
    rich_blocks.extend((
        _project_difference(clip, clip_initial, projections["clip_image"]),
        _project_difference(clip, clip_text, projections["clip_image"]),
        _project_difference(clip, clip_memory, projections["clip_image"]),
        _project_difference(
            clip, clip_distractor, projections["clip_image"]),
    ))

    rgb = values["native_rgb"]
    rgb_anchor = anchors["native_rgb"].reshape(
        batch, 1, 1, 768).expand_as(rgb)
    rgb_memory = _dynamic_memory(
        rgb, anchors["native_rgb"], ema_alpha, epsilon)
    rgb_distractor = _soft_distractor(rgb, soft_distractor_scale)
    rich_blocks.extend((
        _project_difference(rgb, rgb_anchor, projections["native_rgb"]),
        _project_difference(rgb, rgb_memory, projections["native_rgb"]),
        _project_difference(
            rgb, rgb_distractor, projections["native_rgb"]),
    ))

    depth = values["native_depth"]
    depth_anchor = anchors["native_depth"].reshape(
        batch, 1, 1, 768).expand_as(depth)
    depth_memory = _dynamic_memory(
        depth, anchors["native_depth"], ema_alpha, epsilon)
    depth_distractor = _soft_distractor(depth, soft_distractor_scale)
    rich_blocks.extend(_gate_blocks((
        _project_difference(
            depth, depth_anchor, projections["native_depth"]),
        _project_difference(
            depth, depth_memory, projections["native_depth"]),
        _project_difference(
            depth, depth_distractor, projections["native_depth"]),
    ), validity))

    for name in ("native_fused", "query_rgb", "query_depth"):
        value = values[name]
        memory = _dynamic_memory(value, None, ema_alpha, epsilon)
        distractor = _soft_distractor(value, soft_distractor_scale)
        blocks = (
            _project_difference(value, memory, projections[name]),
            _project_difference(value, distractor, projections[name]),
        )
        if name == "query_depth":
            blocks = tuple(_gate_blocks(blocks, validity))
        rich_blocks.extend(blocks)

    if len(rich_blocks) != DIFFERENCE_BLOCKS:
        raise RuntimeError("rich relation difference-block count drifted")
    projected = torch.cat(rich_blocks, dim=-1)
    scalar = build_target_distractor_relations(
        features, initial_image, identity_text, native_rgb_bank,
        native_depth_bank, alpha=ema_alpha, epsilon=epsilon,
        top_k=native_anchor_top_k,
        depth_missing_floor=depth_missing_floor)
    relations = torch.cat((projected, scalar), dim=-1).detach()
    if tuple(relations.shape) != (batch, 5, 6, RICH_RELATION_DIM):
        raise RuntimeError("rich relation width drifted")
    if not torch.isfinite(relations).all().item():
        raise RuntimeError("rich relation is non-finite")
    return relations


class RichRoIRelationRouter(nn.Module):
    """Shared bounded-relation router with no recurrent or norm modules."""

    def __init__(self, hidden_dim=64, residual_scale=0.1):
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
        if (relations.ndim != 4 or relations.shape[1:] != (
                5, 6, RICH_RELATION_DIM) or
                tuple(candidate_valid.shape) != (relations.shape[0], 6) or
                candidate_valid.dtype != torch.bool):
            raise ValueError("rich relation router input contract drifted")
        if not candidate_valid.any(dim=1).all().item():
            raise ValueError("rich relation event has no valid candidate")
        pooled = fixed_temporal_pool_rich(relations.float())
        tokens = self.candidate_encoder(pooled)
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
            raise RuntimeError("rich relation router output is non-finite")
        return outputs


__all__ = [
    "RICH_RELATION_DIM",
    "PROJECTION_WIDTH",
    "DIFFERENCE_BLOCKS",
    "TEMPORAL_STATISTICS",
    "TEMPORAL_WIDTH",
    "FAMILY_NAMES",
    "fixed_temporal_pool_rich",
    "build_rich_roi_relations",
    "RichRoIRelationRouter",
]
