"""Candidate only: shared empty-reference subtraction before feature addition."""
import torch
from lib.models.sttrack.semantic_spatial_adapter import SemanticSpatialAdapter


class CenteredSemanticSpatialAdapter(SemanticSpatialAdapter):
    def __init__(self, empty_text, **kwargs):
        super().__init__(**kwargs)
        assert empty_text.shape == (self.text[0].in_features,)
        self.register_buffer('empty_text', empty_text.detach().clone())

    def forward(self, rgb, depth, fused, initial, text, mask):
        # The base adapter uses fused only in its last additive residual operation.
        zero = torch.zeros_like(fused)
        delta, auxiliary = super().forward(rgb, depth, zero, initial, text, mask)
        empty = self.empty_text.reshape(1, 1, -1).expand_as(text)
        empty_delta, _ = super().forward(rgb, depth, zero, initial, empty, mask)
        # Both branches retain gradients; subtraction precedes addition to fused.
        residual = delta - empty_delta
        return fused + residual, auxiliary
