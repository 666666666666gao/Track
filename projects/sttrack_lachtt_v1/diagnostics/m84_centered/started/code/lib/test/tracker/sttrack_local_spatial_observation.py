"""Native spatial observations; no GT access or public state mutation."""
import math
import torch
import torch.nn.functional as F
from lib.test.tracker.sttrack_lachtt_observation import _search_origin, bbox_iou


def sample_roi_tokens(tokens, boxes, prior, resize_factor, image_size, samples=4):
    """Sample ROI cell centers on patch-center aligned feature maps."""
    side = math.isqrt(tokens.shape[1])
    assert side * side == tokens.shape[1] and tokens.shape[0] == 1
    left, top, _ = _search_origin(prior, image_size, resize_factor)
    unit = (torch.arange(samples, device=tokens.device, dtype=tokens.dtype) + .5) / samples
    grids = []
    for x, y, w, h in boxes:
        xs = ((x + unit * w - left) * resize_factor / image_size) * 2 - 1
        ys = ((y + unit * h - top) * resize_factor / image_size) * 2 - 1
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        grids.append(torch.stack([xx, yy], dim=-1))
    maps = tokens.transpose(1, 2).reshape(1, tokens.shape[-1], side, side).expand(len(boxes), -1, -1, -1)
    roi = F.grid_sample(maps, torch.stack(grids), mode='bilinear', padding_mode='zeros', align_corners=False)
    return roi.flatten(-2).transpose(1, 2)


def search_rois(features, candidates, prior, resize_factor):
    boxes = [c['bbox'] for c in candidates]
    return torch.stack([sample_roi_tokens(features['search_' + m + '_tokens'], boxes, prior, resize_factor, 256)
                        for m in ['rgb', 'depth']], dim=1)


def template_roi(features, index, template_bbox):
    side = math.ceil(math.sqrt(template_bbox[2] * template_bbox[3]) * 2.)
    return torch.stack([sample_roi_tokens(features['template_' + m + '_tokens'][:, index * 64:(index + 1) * 64],
                       [template_bbox], template_bbox, 128 / side, 128)[0] for m in ['rgb', 'depth']], dim=0)


def candidate_scalars(candidates, previous_bbox):
    x, y, w, h = previous_bbox
    scale = math.sqrt(w * h)
    result = []
    for c in candidates:
        a, b, cw, ch = c['bbox']
        result.append([math.log(max(c['score'], 1e-6)),
            (a + cw / 2 - x - w / 2) / scale, (b + ch / 2 - y - h / 2) / scale,
            math.log(cw / w), math.log(ch / h), bbox_iou(c['bbox'], previous_bbox), c['margin'], c['entropy']])
    return torch.tensor(result, dtype=torch.float32)


class NativeReferenceBank:
    """Freeze template features on first use; previous ROI follows predictions."""
    def __init__(self, init_bbox):
        self.initial = None
        self.dynamic = None
        self.previous = None
        self.initial_bbox = list(init_bbox)
        self.dynamic_bbox = list(init_bbox)
        self.encoded_dynamic = None

    def before_decision(self, features, dynamic_tensor):
        if self.initial is None:
            self.initial = template_roi(features, 0, self.initial_bbox).detach().clone()
        if dynamic_tensor is not self.encoded_dynamic:
            self.dynamic = template_roi(features, 1, self.dynamic_bbox).detach().clone()
            self.encoded_dynamic = dynamic_tensor
        previous = self.initial if self.previous is None else self.previous
        return torch.stack([self.initial, self.dynamic, previous])

    def after_decision(self, features, prior, resize_factor, bbox, dynamic_tensor):
        self.previous = search_rois(features, [{'bbox': bbox}], prior, resize_factor)[0].detach().clone()
        if dynamic_tensor is not self.encoded_dynamic:
            self.dynamic_bbox = list(bbox)
