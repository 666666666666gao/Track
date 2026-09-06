"""Dense adapter learning on the tracker's own causal crop/query/template states."""
import torch
import torch.nn.functional as F

from lib.test.tracker.sttrack_semantic import STTrackSemantic
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, clip_box, giou_loss
from lib.utils.focal_loss import FocalLoss
from lib.utils.heapmap_utils import CenterNetHeatMap


class CausalTrainingTracker(STTrackSemantic):
    def __init__(self, params, checkpoint_path):
        super().__init__(params, checkpoint_path)
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)
        self.network.semantic_adapter.requires_grad_(True)
        self.network.eval()

    def step(self, image):
        """No GT argument: commit only Hann-decoded predictions and default updates."""
        previous = list(self.state)
        self.frame_id += 1
        patch, resize, _ = sample_target(image, previous, self.params.search_factor,
                                         output_sz=self.params.search_size)
        search = self.preprocessor.process(patch)
        with torch.no_grad():
            native = self.network.forward(template=self.z_dict, search=[search],
                ce_template_mask=self.box_mask_z, track_query_before=self.track_query_before,
                keep_rate=self.keep_rate, return_candidate_features=True)[0]
        features = native['candidate_features']
        context = self.semantic_context
        enhanced, _ = self.network.semantic_adapter(features['search_rgb_tokens'],
            features['search_depth_tokens'], features['search_fused_tokens'],
            context['initial'], context['text'], context['mask'])
        out = self.network.forward_head(enhanced)
        with torch.no_grad():
            self.track_query_before = native['track_query_before']
            response = self.output_window * out['score_map']
            boxes = self.network.box_head.cal_bbox(response, out['size_map'], out['offset_map']).view(-1, 4)
            prediction = (boxes.mean(dim=0) * self.params.search_size / resize).tolist()
            h, w, _ = image.shape
            self.state = clip_box(self.map_box_back(prediction, resize), h, w, margin=10)
            score = response.flatten(1).max(dim=1, keepdim=True)[0]
            wrote = False
            if self.num_template > 1 and self.frame_id % self.update_intervals == 0 and score > self.update_threshold:
                z, _, _ = sample_target(image, self.state, self.params.template_factor,
                                         output_sz=self.params.template_size)
                self.z_patch_arr = z
                self.z_dict.append(self.preprocessor.process(z))
                if len(self.z_dict) > self.num_template:
                    self.z_dict.pop(1)
                wrote = True
        return out, dict(previous_bbox=previous, resize_factor=resize, bbox=list(self.state),
                         best_score=float(score.item()), template_write=wrote)


def supervision(network, out, groundtruth, previous, resize, search_size):
    """Training labels only, after prediction/state commitment; invalid GT is masked."""
    gt = torch.as_tensor(groundtruth, dtype=torch.float32, device=out['score_map'].device)
    if not bool(torch.isfinite(gt).all()) or not bool((gt[2:] > 0).all()):
        return None, dict(label='invalid')
    side = search_size / resize
    # Inverse of the actual native map_box_back convention, including its unrounded centre.
    origin = gt.new_tensor([previous[0] + .5 * previous[2] - .5 * side,
                            previous[1] + .5 * previous[3] - .5 * side])
    target = torch.cat(((gt[:2] - origin) / side, gt[2:] / side))
    centre = target[:2] + .5 * target[2:]
    inside = bool(((centre >= 0) & (centre < 1)).all())
    heatmap = torch.zeros_like(out['score_map'])
    if inside:
        fmap = torch.zeros(1, 16, 16)
        wh = target[None, 2:].detach().cpu() * 16
        cell = (centre.detach().cpu()[None] * 16).floor()
        CenterNetHeatMap.generate_score_map(fmap, torch.tensor([0]), wh, cell, .7)
        heatmap.copy_(fmap[:, None].to(heatmap.device))
        assert float(heatmap.max()) == 1.
    focal = FocalLoss()(out['score_map'], heatmap)
    regression = focal * 0.
    if inside:
        # GT chooses a supervised cell only; it never chooses the public prediction.
        box = network.box_head.cal_bbox(heatmap, out['size_map'], out['offset_map'])
        xyxy = box_cxcywh_to_xyxy(box)
        expected = box_xywh_to_xyxy(target[None])
        regression = 2. * giou_loss(xyxy, expected)[0] + 5. * F.l1_loss(xyxy, expected)
    loss = focal + regression
    assert bool(torch.isfinite(loss)), (target, focal, regression)
    return loss, dict(label='centre_inside' if inside else 'centre_outside',
                      focal=float(focal.detach()), regression=float(regression.detach()))
