"""Express candidate motion and scale in the preceding selected box's units."""
import torch
from torch import nn

from lib.models.sttrack.lachtt_candidate_set import CandidateSetAssociation


def relative_geometry(geometry, previous_choice):
    index = torch.arange(len(geometry), device=geometry.device)
    reference = geometry[index, 10 + previous_choice][:, None]
    center_offset = (geometry[..., :2] - reference[..., :2]) / reference[..., 2:]
    log_size_ratio = (geometry[..., 2:] / reference[..., 2:]).log()
    return torch.cat([center_offset, log_size_ratio], dim=-1)


class RelativeCandidateSetAssociation(CandidateSetAssociation):
    def forward(self, current, previous, references, geometry, scores, previous_choice):
        return super().forward(current, previous, references,
                               relative_geometry(geometry, previous_choice), scores, previous_choice)


class RelativeGeometryInference(nn.Module):
    """Use the same transform with a head loaded by the frozen tracker class."""
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, current, previous, references, geometry, scores, previous_choice):
        return self.head(current, previous, references,
                         relative_geometry(geometry, previous_choice), scores, previous_choice)
