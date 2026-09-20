"""M51 keeps the frozen candidate-set state path and changes only geometry."""
from lib.models.sttrack.lachtt_relative_geometry import RelativeGeometryInference
from lib.test.tracker.sttrack_candidate_set import STTrackCandidateSet


class STTrackRelativeCandidateSet(STTrackCandidateSet):
    def __init__(self, params, association_checkpoint):
        super().__init__(params, association_checkpoint)
        self.association = RelativeGeometryInference(self.association).eval()
