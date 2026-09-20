"""A fixed admission rule around the frozen M45 candidate-set tracker."""
import torch
from torch import nn
import torch.nn.functional as F

from lib.models.sttrack.lachtt_candidate_set import select_candidate
from lib.test.tracker.sttrack_candidate_set import STTrackCandidateSet


def native_continuity(current, previous, previous_choice, proposal):
    batch = torch.arange(len(current), device=current.device)
    reference = previous[batch, previous_choice].flatten(-2)
    similarity = F.cosine_similarity(current.flatten(-2), reference[:, None], dim=-1)
    delta = similarity[batch, proposal] - similarity[:, 0]
    return (delta >= 0).all(-1), delta


class NativeContinuityAdmission(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head
        self.reset_record()

    def reset_record(self):
        self.record = dict(association_proposal=0, association_vetoed=False,
            continuity_delta=None, previous_choice_input=0)

    def forward(self, current, previous, references, geometry, scores, previous_choice):
        logits, affinity = self.head(current, previous, references, geometry, scores, previous_choice)
        proposal = select_candidate(logits)
        accepted, delta = native_continuity(current, previous, previous_choice, proposal)
        vetoed = (proposal != 0) & ~accepted
        self.record = dict(association_proposal=int(proposal[0]), association_vetoed=bool(vetoed[0]),
            continuity_delta=delta[0].tolist(), previous_choice_input=int(previous_choice[0]))
        # A veto becomes the existing native-default action before state updates.
        gated = logits.clone()
        gated[vetoed] = -torch.inf
        gated[vetoed, 0] = 0
        return gated, affinity


class STTrackCandidateContinuity(STTrackCandidateSet):
    def __init__(self, params, association_checkpoint):
        super().__init__(params, association_checkpoint)
        self.association = NativeContinuityAdmission(self.association).eval()

    def initialize(self, image, info):
        super().initialize(image, info)
        self.association.reset_record()

    def track(self, image, info=None):
        output = super().track(image, info)
        output.update(self.association.record)
        return output
