"""Read initial/current template combinations and commit one complete prediction."""
import torch
from lib.test.tracker.sttrack import STTrack
from lib.test.tracker.sttrack_lachtt_observation import clone_query_state
from lib.test.tracker.sttrack_local_spatial_observation import search_rois, template_roi
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box


READER_FIELDS = ['rois', 'references', 'maps', 'geometry', 'scores']


class STTrackTemplateReader(STTrack):
    def __init__(self, params, reader):
        super().__init__(params)
        assert self.num_template == 2 and not params.save_all_boxes and params.debug == 0
        self.reader = reader.cuda().eval()

    def initialize(self, image, info):
        super().initialize(image, info)
        self.initial_bbox = list(info['init_bbox'])
        self.initial_roi = None
        self.previous_roi = None

    def read_alternate(self):
        # Native initialization repeats the same tensor in both template slots.
        return self.z_dict[0] is not self.z_dict[1]

    def choose_view(self, observation):
        values = [observation[k][None].cuda().float() for k in READER_FIELDS]
        return int(self.reader(*values).argmax(1)[0])

    def track(self, image, info=None):
        self.frame_id += 1
        prior = list(self.state)
        patch, resize, _ = sample_target(image, prior, self.params.search_factor, output_sz=self.params.search_size)
        search = self.preprocessor.process(patch)
        with torch.no_grad():
            def forward(templates):
                return self.network.forward(template=templates, search=[search], ce_template_mask=self.box_mask_z,
                    track_query_before=clone_query_state(self.track_query_before), keep_rate=self.keep_rate,
                    return_candidate_features=True)[0]

            current = forward(self.z_dict)
            outputs = [current, forward([self.z_dict[0], self.z_dict[0]]) if self.read_alternate() else current]
            boxes, scores, maps, rois = [], [], [], []
            for out in outputs:
                response = self.output_window * out['score_map']
                box = self.network.box_head.cal_bbox(response, out['size_map'], out['offset_map'])
                box = (box.view(-1, 4).mean(0) * self.params.search_size / resize).tolist()
                bbox = clip_box(self.map_box_back(box, resize), image.shape[0], image.shape[1], margin=10)
                boxes.append(bbox)
                scores.append(response.max())
                maps.append(torch.cat([out['score_map'], response], dim=1)[0])
                rois.append(search_rois(out['candidate_features'], [{'bbox': bbox}], prior, resize)[0])
            if self.initial_roi is None:
                self.initial_roi = template_roi(current['candidate_features'], 0, self.initial_bbox).detach().clone()
                self.previous_roi = self.initial_roi
            bbox_tensor = torch.tensor(boxes, dtype=torch.float32, device='cuda')
            image_size = torch.tensor([image.shape[1], image.shape[0]], device='cuda')
            geometry = torch.cat([(bbox_tensor[:, :2] + bbox_tensor[:, 2:] / 2) / image_size,
                                  bbox_tensor[:, 2:] / image_size], dim=1)
            observation = dict(rois=torch.stack(rois).half(),
                references=torch.stack([self.initial_roi, self.previous_roi]).half(),
                maps=torch.stack(maps), geometry=geometry, scores=torch.stack(scores))
            chosen = self.choose_view(observation)
            assert chosen in (0, 1)
            self.state = boxes[chosen]
            self.track_query_before = outputs[chosen]['track_query_before']
            self.previous_roi = rois[chosen].detach().clone()
            confidence = scores[chosen]
            if self.frame_id % self.update_intervals == 0 and confidence > self.update_threshold:
                patch, _, _ = sample_target(image, self.state, self.params.template_factor, output_sz=self.params.template_size)
                self.z_patch_arr = patch
                self.z_dict.append(self.preprocessor.process(patch))
                self.z_dict.pop(1)
            self.last_observation = observation
            self.last_boxes = boxes
        return dict(target_bbox=self.state, best_score=float(confidence), template_read=chosen)
