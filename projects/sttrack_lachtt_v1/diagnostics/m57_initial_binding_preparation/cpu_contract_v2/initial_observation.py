"""Extract a t0-only reference without committing the auxiliary query output."""
import torch

from lib.train.data.processing_utils import sample_target
from lib.test.tracker.sttrack_local_spatial_observation import template_roi


def extract_initial_instance_roi(tracker, image, init_bbox):
    """Call immediately after native initialize, using that same RGB-D frame."""
    assert tracker.frame_id == 0 and tracker.track_query_before is None
    assert list(tracker.state) == list(init_bbox)
    patch, _, _ = sample_target(image, init_bbox, tracker.params.search_factor,
        output_sz=tracker.params.search_size)
    search = tracker.preprocessor.process(patch)
    with torch.no_grad():
        output = tracker.network.forward(template=tracker.z_dict, search=[search],
            ce_template_mask=tracker.box_mask_z, track_query_before=None,
            keep_rate=tracker.keep_rate, return_candidate_features=True)[0]
        reference = template_roi(output['candidate_features'], 0, init_bbox).detach().clone()
    return reference
