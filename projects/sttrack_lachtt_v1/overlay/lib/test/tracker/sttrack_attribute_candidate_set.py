"""M56 text-conditioned head on the existing causal candidate-set runtime."""
import torch
from torch import nn

from lib.models.sttrack.lachtt_attribute_candidate_set import AttributeCandidateSetAssociation, condition_text
from lib.test.tracker.sttrack import STTrack
from lib.test.tracker.sttrack_candidate_set import STTrackCandidateSet


class BoundAttributeAssociation(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head
        self.register_buffer('tokens', torch.zeros(1, 5, 768, device='cuda'))
        self.register_buffer('mask', torch.zeros(1, 5, dtype=torch.bool, device='cuda'))

    def set_text(self, tokens, mask):
        assert tokens.shape == self.tokens.shape and mask.shape == self.mask.shape
        assert mask.any(1).all() and torch.isfinite(tokens).all()
        self.tokens.copy_(tokens)
        self.mask.copy_(mask)

    def forward(self, current, previous, references, geometry, scores, previous_choice):
        return self.head(current, previous, references, geometry, scores,
                         previous_choice, self.tokens, self.mask)


class STTrackAttributeCandidateSet(STTrackCandidateSet):
    def __init__(self, params, association_checkpoint):
        STTrack.__init__(self, params)
        assert not params.save_all_boxes and params.debug == 0
        checkpoint = torch.load(association_checkpoint, map_location='cpu')
        self.variant = checkpoint['variant']
        assert self.variant in ['attributes', 'pooled', 'empty']
        head = AttributeCandidateSetAssociation().cuda().eval()
        head.load_state_dict(checkpoint['model'], strict=True)
        self.association = BoundAttributeAssociation(head).eval()

    def initialize(self, image, info):
        super().initialize(image, info)
        tokens, mask = condition_text(info['text_tokens'][None].cuda().float(),
            info['text_mask'][None].cuda(), info['empty_text'].cuda().float(), self.variant)
        self.association.set_text(tokens, mask)
