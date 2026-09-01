"""Learned bounded RoI association without trainable cosine normalization."""

import math

import torch
from torch import nn
import torch.nn.functional as F

from .lachtt_rich_roi_relation import (
    FAMILY_NAMES,
    PROJECTION_WIDTH,
    RICH_RELATION_DIM,
    RichRoIRelationRouter,
)
from .lachtt_target_distractor_memory import (
    build_target_distractor_relations,
)


EMBEDDING_WIDTH = 768
RAW_DIFFERENCE_BLOCKS = 16
SCALAR_RELATION_DIM = 49
FAMILY_BLOCK_COUNTS = (4, 3, 3, 2, 2, 2)
BLOCK_FAMILY_INDICES = (
    0, 0, 0, 0,
    1, 1, 1,
    2, 2, 2,
    3, 3,
    4, 4,
    5, 5,
)


def _normalize(value, epsilon):
    return F.normalize(value.float(), p=2.0, dim=-1, eps=float(epsilon))


def _fixed_projection(seed, device, dtype):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    signs = torch.randint(
        0, 2, (EMBEDDING_WIDTH, PROJECTION_WIDTH),
        generator=generator, dtype=torch.int64)
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
    weights = torch.softmax(float(scale) * pairwise, dim=-1).detach()
    distractor = torch.einsum("bhck,bhkd->bhcd", weights, value)
    return distractor.detach(), weights


@torch.no_grad()
def build_detached_roi_differences(
        features, initial_image, identity_text, native_rgb_bank,
        native_depth_bank, ema_alpha=0.2, soft_distractor_scale=4.0,
        epsilon=1e-6, native_anchor_top_k=4,
        depth_missing_floor=-1.0):
    """Build detached raw differences, depth gates and the frozen 49-D block."""
    expected_keys = {
        "clip_image", "native_depth", "native_fused", "native_rgb",
        "query_depth", "query_rgb", "raw_depth", "scalars",
    }
    if set(features) != expected_keys:
        raise ValueError("learned association cached feature keys drifted")
    batch, horizon, candidates, width = features["clip_image"].shape
    if (horizon, candidates, width) != (5, 6, EMBEDDING_WIDTH):
        raise ValueError("learned association trajectory shape drifted")
    if (tuple(initial_image.shape) != (batch, 1, EMBEDDING_WIDTH) or
            tuple(identity_text.shape) != (batch, 1, EMBEDDING_WIDTH) or
            tuple(native_rgb_bank.shape) != (batch, 64, EMBEDDING_WIDTH) or
            tuple(native_depth_bank.shape) !=
            (batch, 64, EMBEDDING_WIDTH)):
        raise ValueError("learned association anchor shape drifted")

    values = {
        name: _normalize(features[name].detach(), epsilon)
        for name in FAMILY_NAMES
    }
    anchors = {
        "clip_initial": _normalize(initial_image[:, 0].detach(), epsilon),
        "clip_text": _normalize(identity_text[:, 0].detach(), epsilon),
        "native_rgb": _normalize(
            native_rgb_bank.detach().mean(dim=1), epsilon),
        "native_depth": _normalize(
            native_depth_bank.detach().mean(dim=1), epsilon),
    }
    raw_depth = features["raw_depth"].detach().float()
    depth_validity = raw_depth[:, :, :, 1].clamp(0.0, 1.0).mean(
        dim=(-1, -2)).detach()

    blocks = []
    block_gates = []

    def add(left, right, gate=None):
        blocks.append((left - right).detach())
        if gate is None:
            block_gates.append(torch.ones(
                left.shape[:-1], device=left.device, dtype=left.dtype))
        else:
            block_gates.append(gate.to(
                device=left.device, dtype=left.dtype).detach())

    clip = values["clip_image"]
    clip_anchor = anchors["clip_initial"].reshape(
        batch, 1, 1, EMBEDDING_WIDTH).expand_as(clip)
    clip_text = anchors["clip_text"].reshape(
        batch, 1, 1, EMBEDDING_WIDTH).expand_as(clip)
    clip_memory = _dynamic_memory(
        clip, anchors["clip_initial"], ema_alpha, epsilon)
    clip_distractor, _ = _soft_distractor(
        clip, soft_distractor_scale)
    add(clip, clip_anchor)
    add(clip, clip_text)
    add(clip, clip_memory)
    add(clip, clip_distractor)

    rgb = values["native_rgb"]
    rgb_anchor = anchors["native_rgb"].reshape(
        batch, 1, 1, EMBEDDING_WIDTH).expand_as(rgb)
    rgb_memory = _dynamic_memory(
        rgb, anchors["native_rgb"], ema_alpha, epsilon)
    rgb_distractor, _ = _soft_distractor(rgb, soft_distractor_scale)
    add(rgb, rgb_anchor)
    add(rgb, rgb_memory)
    add(rgb, rgb_distractor)

    depth = values["native_depth"]
    depth_anchor = anchors["native_depth"].reshape(
        batch, 1, 1, EMBEDDING_WIDTH).expand_as(depth)
    depth_memory = _dynamic_memory(
        depth, anchors["native_depth"], ema_alpha, epsilon)
    depth_distractor, _ = _soft_distractor(
        depth, soft_distractor_scale)
    add(depth, depth_anchor, depth_validity)
    add(depth, depth_memory, depth_validity)
    add(depth, depth_distractor, depth_validity)

    for name in ("native_fused", "query_rgb", "query_depth"):
        value = values[name]
        memory = _dynamic_memory(value, None, ema_alpha, epsilon)
        distractor, _ = _soft_distractor(
            value, soft_distractor_scale)
        gate = depth_validity if name == "query_depth" else None
        add(value, memory, gate)
        add(value, distractor, gate)

    if (len(blocks) != RAW_DIFFERENCE_BLOCKS or
            len(block_gates) != RAW_DIFFERENCE_BLOCKS or
            sum(FAMILY_BLOCK_COUNTS) != RAW_DIFFERENCE_BLOCKS or
            len(BLOCK_FAMILY_INDICES) != RAW_DIFFERENCE_BLOCKS):
        raise RuntimeError("learned association block contract drifted")
    differences = torch.stack(blocks, dim=3).detach()
    gates = torch.stack(block_gates, dim=3).detach()
    scalar = build_target_distractor_relations(
        features, initial_image, identity_text, native_rgb_bank,
        native_depth_bank, alpha=ema_alpha, epsilon=epsilon,
        top_k=native_anchor_top_k,
        depth_missing_floor=depth_missing_floor).detach()
    if tuple(differences.shape) != (
            batch, 5, 6, RAW_DIFFERENCE_BLOCKS, EMBEDDING_WIDTH):
        raise RuntimeError("learned association difference shape drifted")
    if tuple(gates.shape) != (batch, 5, 6, RAW_DIFFERENCE_BLOCKS):
        raise RuntimeError("learned association gate shape drifted")
    if tuple(scalar.shape) != (batch, 5, 6, SCALAR_RELATION_DIM):
        raise RuntimeError("learned association scalar shape drifted")
    if (not torch.isfinite(differences).all().item() or
            not torch.isfinite(gates).all().item() or
            not torch.isfinite(scalar).all().item()):
        raise RuntimeError("learned association detached input is non-finite")
    return differences, gates, scalar


class LearnedBoundedRoIAssociationRouter(nn.Module):
    """Family projectors followed by the audited non-recurrent set router."""

    def __init__(self, hidden_dim=37, residual_scale=0.1,
                 base_projection_seed=20260901):
        super().__init__()
        self.base_projection_seed = int(base_projection_seed)
        self.projectors = nn.ModuleList([
            nn.Linear(EMBEDDING_WIDTH, PROJECTION_WIDTH, bias=False)
            for _ in FAMILY_NAMES
        ])
        with torch.no_grad():
            for family_index, projector in enumerate(self.projectors):
                matrix = _fixed_projection(
                    self.base_projection_seed + family_index,
                    projector.weight.device, projector.weight.dtype)
                projector.weight.copy_(matrix.transpose(0, 1))
        self.router = RichRoIRelationRouter(
            hidden_dim=int(hidden_dim), residual_scale=float(residual_scale))

    def project_relations(self, differences, block_gates, scalar):
        batch = differences.shape[0]
        if tuple(differences.shape) != (
                batch, 5, 6, RAW_DIFFERENCE_BLOCKS, EMBEDDING_WIDTH):
            raise ValueError("learned association difference input drifted")
        if tuple(block_gates.shape) != (
                batch, 5, 6, RAW_DIFFERENCE_BLOCKS):
            raise ValueError("learned association block gate drifted")
        if tuple(scalar.shape) != (batch, 5, 6, SCALAR_RELATION_DIM):
            raise ValueError("learned association scalar input drifted")
        projected = []
        for block_index, family_index in enumerate(BLOCK_FAMILY_INDICES):
            block = torch.tanh(self.projectors[family_index](
                differences[:, :, :, block_index]))
            projected.append(
                block * block_gates[:, :, :, block_index].unsqueeze(-1))
        relation = torch.cat((*projected, scalar), dim=-1)
        if tuple(relation.shape) != (batch, 5, 6, RICH_RELATION_DIM):
            raise RuntimeError("learned association relation width drifted")
        if not torch.isfinite(relation).all().item():
            raise RuntimeError("learned association relation is non-finite")
        return relation

    def forward(self, differences, block_gates, scalar, candidate_valid):
        relation = self.project_relations(
            differences, block_gates, scalar)
        return self.router(relation, candidate_valid)


__all__ = [
    "EMBEDDING_WIDTH",
    "RAW_DIFFERENCE_BLOCKS",
    "SCALAR_RELATION_DIM",
    "FAMILY_BLOCK_COUNTS",
    "BLOCK_FAMILY_INDICES",
    "build_detached_roi_differences",
    "LearnedBoundedRoIAssociationRouter",
]
