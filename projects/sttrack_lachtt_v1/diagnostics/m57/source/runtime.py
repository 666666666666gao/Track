"""M57 same-slot language comparison with a causal t0 instance reference."""
import torch

from lib.models.sttrack.lachtt_initial_instance_alignment import (
    InitialInstanceCandidateSetAssociation,
    content_tokens,
)
from lib.test.tracker.sttrack import STTrack
from lib.test.tracker.sttrack_attribute_candidate_set import BoundAttributeAssociation
from lib.test.tracker.sttrack_candidate_set import STTrackCandidateSet
from lib.test.tracker.sttrack_initial_instance_observation import extract_initial_instance_roi


class STTrackInitialInstanceCandidateSet(STTrackCandidateSet):
    def __init__(self, params, association_checkpoint):
        STTrack.__init__(self, params)
        assert not params.save_all_boxes and params.debug == 0
        checkpoint = torch.load(association_checkpoint, map_location='cpu')
        self.reference_mode = checkpoint['reference_mode']
        self.use_text = checkpoint['use_text']
        self.variant = self.reference_mode + ('_text' if self.use_text else '_visual')
        assert checkpoint['variant'] == self.variant
        head = InitialInstanceCandidateSetAssociation(self.reference_mode).cuda().eval()
        head.load_state_dict(checkpoint['model'], strict=True)
        self.association = BoundAttributeAssociation(head).eval()

    def initialize(self, image, info):
        super().initialize(image, info)
        self.reference_bank.initial = extract_initial_instance_roi(self, image, info['init_bbox'])
        assert self.reference_bank.dynamic is None and self.reference_bank.encoded_dynamic is None
        assert self.reference_bank.previous is None and self.previous_set is None
        tokens = content_tokens(info['text_tokens'][None].cuda().float(),
                                info['empty_text'].cuda().float(), self.use_text)
        self.association.set_text(tokens, info['text_mask'][None].cuda())
