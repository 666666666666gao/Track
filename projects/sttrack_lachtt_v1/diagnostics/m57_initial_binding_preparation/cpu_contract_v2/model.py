"""Untrained preparation for initialization-grounded phrase interaction."""
import torch
from torch import nn

from lib.models.sttrack.lachtt_attribute_candidate_set import (
    AttributeCandidateSetAssociation,
    AttributeLocalAlignment,
)


def content_tokens(tokens, empty_embedding, use_text):
    """Keep the caller's slot count and mask in the visual-query control."""
    if use_text:
        return tokens
    return empty_embedding.to(tokens).reshape(1, 1, -1).expand_as(tokens)


class InitialInstanceLocalAlignment(AttributeLocalAlignment):
    def __init__(self, reference_mode):
        super().__init__()
        assert reference_mode in ['candidate', 'initial']
        self.reference_mode = reference_mode
        self.slot_embedding = nn.Parameter(torch.empty(5, 32))
        nn.init.normal_(self.slot_embedding, std=.02)

    def queries(self, text_tokens):
        return self.text(text_tokens) + self.slot_embedding[None]

    def initial_phrases(self, initial_cells, text_tokens):
        """Only initialization cells and fixed text determine this reference."""
        text = self.queries(text_tokens)
        initial = initial_cells.flatten(1, 2)
        observed, _ = self.read_visual(text, initial, initial, need_weights=False)
        return text + observed

    def forward(self, cells, text_tokens, text_mask):
        # All arms supply the same t0-only ROI at object 20 (first reference).
        batch, objects = cells.shape[:2]
        visual = cells.flatten(2, 3).flatten(0, 1)
        if self.reference_mode == 'initial':
            conditioned = self.initial_phrases(cells[:, 20], text_tokens)
            conditioned = conditioned[:, None].expand(-1, objects, -1, -1).flatten(0, 1)
        else:
            text = self.queries(text_tokens)
            text = text[:, None].expand(-1, objects, -1, -1).flatten(0, 1)
            observed, _ = self.read_visual(text, visual, visual, need_weights=False)
            conditioned = text + observed
        padding = ~text_mask[:, None].expand(-1, objects, -1).flatten(0, 1)
        guided, _ = self.guide_visual(visual, conditioned, conditioned,
                                      key_padding_mask=padding, need_weights=False)
        return cells + self.output(guided).reshape_as(cells)


class InitialInstanceCandidateSetAssociation(AttributeCandidateSetAssociation):
    def __init__(self, reference_mode):
        super().__init__()
        self.attribute_alignment = InitialInstanceLocalAlignment(reference_mode)
