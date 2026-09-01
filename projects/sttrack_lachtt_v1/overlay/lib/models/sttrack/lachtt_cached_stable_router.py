"""Gradient-stable strict-H10 two-stage router for frozen RGB-D candidates."""

import torch
from torch import nn
import torch.nn.functional as F

from .lachtt_cached_strict_router import (
    FEATURE_KEYS,
    NATIVE_KEYS,
    cached_strict_router_loss,
)


class CachedStableTwoStageRouter(nn.Module):
    """A shallow permutation-equivariant DeepSets router.

    The model keeps candidate-specific RGB-D/language evidence and the
    commit-then-rank decomposition while avoiding cascaded normalization layers.
    It accepts no sequence, frame, source, or rank metadata.
    """

    def __init__(self, projection_dim=32, hidden_dim=128,
                 l2_normalization_eps=1e-4, residual_scale=0.1):
        super().__init__()
        self.projection_dim = int(projection_dim)
        self.hidden_dim = int(hidden_dim)
        self.l2_normalization_eps = float(l2_normalization_eps)
        self.residual_scale = float(residual_scale)
        self.native_projection = nn.Sequential(
            nn.Linear(768, projection_dim),
            nn.GELU(),
        )
        self.clip_projection = nn.Sequential(
            nn.Linear(768, projection_dim),
            nn.GELU(),
        )
        self.raw_depth_projection = nn.Sequential(
            nn.Linear(512, projection_dim),
            nn.GELU(),
        )
        self.scalar_projection = nn.Sequential(
            nn.Linear(15, projection_dim),
            nn.GELU(),
        )
        self.candidate_input_dim = projection_dim * 12 + 2
        self.candidate_encoder = nn.Sequential(
            nn.Linear(self.candidate_input_dim, hidden_dim),
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

    @staticmethod
    def _validate_inputs(features, initial_image, identity_text,
                         candidate_valid):
        if set(features) != set(FEATURE_KEYS):
            raise ValueError("cached feature keys drifted")
        reference = features["clip_image"]
        if reference.ndim != 4 or reference.shape[-1] != 768:
            raise ValueError("cached clip feature shape drifted")
        batch, horizon, candidates, _ = reference.shape
        expected = {
            "clip_image": (batch, 5, 6, 768),
            "native_depth": (batch, 5, 6, 768),
            "native_fused": (batch, 5, 6, 768),
            "native_rgb": (batch, 5, 6, 768),
            "query_depth": (batch, 5, 6, 768),
            "query_rgb": (batch, 5, 6, 768),
            "raw_depth": (batch, 5, 6, 2, 16, 16),
            "scalars": (batch, 5, 6, 15),
        }
        if horizon != 5 or candidates != 6:
            raise ValueError("cached horizon/candidate contract drifted")
        for name, shape in expected.items():
            value = features[name]
            if tuple(value.shape) != shape:
                raise ValueError("cached feature shape drifted: %s" % name)
            if not torch.isfinite(value.float()).all().item():
                raise ValueError("cached feature is non-finite: %s" % name)
        if (tuple(initial_image.shape) != (batch, 1, 768) or
                tuple(identity_text.shape) != (batch, 1, 768) or
                tuple(candidate_valid.shape) != (batch, 6)):
            raise ValueError("anchor or candidate-valid shape drifted")
        if not torch.isfinite(initial_image.float()).all().item() or \
                not torch.isfinite(identity_text.float()).all().item():
            raise ValueError("anchor tensor is non-finite")
        if candidate_valid.dtype != torch.bool or \
                not candidate_valid.any(dim=1).all().item():
            raise ValueError("candidate-valid contract drifted")
        return batch, horizon, candidates

    def _normalized_projection(self, projection, value):
        normalized = F.normalize(
            value.float(), p=2.0, dim=-1, eps=self.l2_normalization_eps)
        return projection(normalized)

    def forward(self, features, initial_image, identity_text,
                candidate_valid):
        batch, horizon, candidates = self._validate_inputs(
            features, initial_image, identity_text, candidate_valid)
        native = [self._normalized_projection(
            self.native_projection, features[name]) for name in NATIVE_KEYS]
        clip_candidate = self._normalized_projection(
            self.clip_projection, features["clip_image"])
        initial = self._normalized_projection(
            self.clip_projection, initial_image).view(
                batch, 1, 1, self.projection_dim).expand(
                    -1, horizon, candidates, -1)
        text = self._normalized_projection(
            self.clip_projection, identity_text).view(
                batch, 1, 1, self.projection_dim).expand(
                    -1, horizon, candidates, -1)
        clip_relations = [
            clip_candidate,
            torch.abs(clip_candidate - initial),
            clip_candidate * initial,
            torch.abs(clip_candidate - text),
            clip_candidate * text,
            F.cosine_similarity(
                clip_candidate, initial, dim=-1,
                eps=self.l2_normalization_eps).unsqueeze(-1),
            F.cosine_similarity(
                clip_candidate, text, dim=-1,
                eps=self.l2_normalization_eps).unsqueeze(-1),
        ]
        raw_depth = self.raw_depth_projection(
            features["raw_depth"].float().flatten(start_dim=-3))
        scalars = self.scalar_projection(torch.tanh(features["scalars"].float()))
        candidate_input = torch.cat(
            [*native, *clip_relations, raw_depth, scalars], dim=-1)
        if candidate_input.shape[-1] != self.candidate_input_dim:
            raise RuntimeError("candidate input width drifted")
        encoded = self.candidate_encoder(candidate_input)
        temporal_input = encoded.permute(0, 2, 1, 3).reshape(
            batch * candidates, horizon, self.hidden_dim)
        _, temporal_state = self.temporal(temporal_input)
        tokens = temporal_state[-1].reshape(
            batch, candidates, self.hidden_dim)
        tokens = F.normalize(
            tokens, p=2.0, dim=-1, eps=self.l2_normalization_eps)

        valid = candidate_valid.unsqueeze(-1)
        mean_context = ((tokens * valid).sum(dim=1) /
                        valid.sum(dim=1).clamp_min(1))
        max_context = tokens.masked_fill(
            ~valid, -float("inf")).max(dim=1).values
        context = torch.cat((
            tokens,
            mean_context.unsqueeze(1).expand(-1, candidates, -1),
            max_context.unsqueeze(1).expand(-1, candidates, -1),
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
        for name, value in outputs.items():
            if not torch.isfinite(value.float()).all().item():
                raise RuntimeError("stable router output is non-finite: %s" % name)
        return outputs


__all__ = [
    "CachedStableTwoStageRouter",
    "cached_strict_router_loss",
    "FEATURE_KEYS",
]
