"""STTrack with a trained local association head and its own recursive state."""
import torch
from lib.test.tracker.sttrack import STTrack
from lib.test.tracker.sttrack_lachtt_observation import decode_nms_candidates
from lib.test.tracker.sttrack_local_spatial_observation import search_rois, candidate_scalars, NativeReferenceBank
from lib.models.sttrack.lachtt_local_spatial_association import LocalSpatialAssociation, select_candidate
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box


class STTrackLocalSpatial(STTrack):
    def __init__(self, params, association_checkpoint, parity_smoke=False):
        super().__init__(params)
        assert not params.save_all_boxes and params.debug == 0
        if parity_smoke:
            self.association = LocalSpatialAssociation(spatial=True).cuda().eval()
        else:
            checkpoint = torch.load(association_checkpoint, map_location='cpu')
            self.association = LocalSpatialAssociation(spatial=checkpoint['variant'] == 'spatial').cuda().eval()
            self.association.load_state_dict(checkpoint['model'], strict=True)

    def initialize(self, image, info):
        super().initialize(image, info)
        self.reference_bank = NativeReferenceBank(info['init_bbox'])

    def track(self, image, info=None):
        height, width = image.shape[:2]
        self.frame_id += 1
        prior = list(self.state)
        dynamic_before = self.z_dict[1]
        patch, resize, _ = sample_target(image, self.state, self.params.search_factor,
                                        output_sz=self.params.search_size)
        search = self.preprocessor.process(patch)
        with torch.no_grad():
            output = self.network.forward(template=self.z_dict, search=[search],
                ce_template_mask=self.box_mask_z, track_query_before=self.track_query_before,
                keep_rate=self.keep_rate, return_candidate_features=True)[0]
            self.track_query_before = output['track_query_before']
            features = output['candidate_features']
            response = self.output_window * output['score_map']
            candidates = decode_nms_candidates(response, output['size_map'], output['offset_map'],
                [prior], [resize], image.shape, self.params.search_size, 10, 3)
            references = self.reference_bank.before_decision(features, dynamic_before)
            rois = search_rois(features, candidates, prior, resize)
            scalars = candidate_scalars(candidates, prior).cuda()
            # Match the frozen feature cache's FP16->FP32 conversion exactly.
            logits = self.association(rois[None].half().float(), references[None].half().float(), scalars[None])
            chosen = int(select_candidate(logits)[0])
            if chosen == 0:
                boxes = self.network.box_head.cal_bbox(response, output['size_map'], output['offset_map'])
                confidence = response.flatten(1).max(1, keepdim=True).values
            else:
                row, col = candidates[chosen]['grid_row'], candidates[chosen]['grid_column']
                offsets, sizes = output['offset_map'], output['size_map']
                boxes = torch.stack([(col + offsets[0,0,row,col]) / self.feat_sz,
                    (row + offsets[0,1,row,col]) / self.feat_sz,
                    sizes[0,0,row,col], sizes[0,1,row,col]]).reshape(1,4)
                confidence = response[0,0,row,col].reshape(1,1)
            box = (boxes.view(-1,4).mean(0) * self.params.search_size / resize).tolist()
            self.state = clip_box(self.map_box_back(box, resize), height, width, margin=10)
            if self.frame_id % self.update_intervals == 0 and confidence > self.update_threshold:
                template_patch, _, _ = sample_target(image, self.state, self.params.template_factor,
                                                      output_sz=self.params.template_size)
                self.z_patch_arr = template_patch
                template = self.preprocessor.process(template_patch)
                self.z_dict.append(template)
                if len(self.z_dict) > self.num_template:
                    self.z_dict.pop(1)
            self.reference_bank.after_decision(features, prior, resize, self.state, self.z_dict[1])
        return dict(target_bbox=self.state, best_score=confidence.cpu().numpy()[0][0],
                    association_candidate=chosen, association_none=int(logits.argmax(1)[0]) == 10)
