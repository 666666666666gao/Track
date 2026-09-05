"""Observe all candidates in one native forward, without ground truth."""
import torch
from lib.test.tracker.sttrack_lachtt_observation import decode_nms_candidates
from lib.test.tracker.sttrack_local_spatial_observation import search_rois


def observe_candidate_set(output,window,prior,resize,image_shape):
    candidates=decode_nms_candidates(window*output['score_map'],output['size_map'],output['offset_map'],
                                     [prior],[resize],image_shape,256,10,3)
    boxes=torch.tensor([c['bbox'] for c in candidates],dtype=torch.float32)
    size=torch.tensor([image_shape[1],image_shape[0]],dtype=torch.float32)
    geometry=torch.cat([(boxes[:,:2]+boxes[:,2:]/2)/size,boxes[:,2:]/size],dim=1)
    scores=torch.tensor([c['score'] for c in candidates],dtype=torch.float32).clamp_min(1e-6).log()
    rois=search_rois(output['candidate_features'],candidates,prior,resize)
    return dict(rois=rois,geometry=geometry,scores=scores,boxes=boxes,candidates=candidates)
