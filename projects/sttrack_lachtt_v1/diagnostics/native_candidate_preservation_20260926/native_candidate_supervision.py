"""Add candidate response preservation after the frozen M82 training objective."""
import torch

from candidate_margin import candidate_margin_loss
from lib.train.data.processing_utils import transform_image_to_crop


def augment_loss(loss, diagnostic, out, groundtruth, previous, resize, window, weight, probe_gradient=False):
    diagnostic.update(candidate_eligible=False, candidate_competitors=0,
                      candidate_loss=None, candidate_weight=weight, candidate_score_gradient_norm=None)
    if weight == 0 or loss is None or not diagnostic['native_eligible']:
        return loss, diagnostic
    assert not out['native_score_map'].requires_grad
    with torch.no_grad():
        gt = out['score_map'].new_tensor(groundtruth)
        target = transform_image_to_crop(gt, gt.new_tensor(previous), resize, gt.new_tensor([256, 256]), normalize=True)
        axis = torch.arange(16, device=gt.device, dtype=gt.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        offset = out['offset_map'].detach()[0]
        centers = torch.stack((xx + offset[0], yy + offset[1]), -1).reshape(-1, 2) / 16
        size = out['size_map'].detach()[0].permute(1, 2, 0).reshape(-1, 2)
        lo, hi = centers - size / 2, centers + size / 2
        intersection = (torch.minimum(hi, target[:2] + target[2:]) - torch.maximum(lo, target[:2])).clamp(min=0).prod(-1)
        quality = intersection / (size.prod(-1) + target[2:].prod() - intersection)
    auxiliary, values = candidate_margin_loss(out['score_map'], out['native_score_map'], quality, window, True)
    diagnostic.update(candidate_eligible=values['eligible'], candidate_competitors=values['competitors'],
                      candidate_loss=float(auxiliary.detach()))
    if probe_gradient and diagnostic['candidate_loss'] > 0:
        gradient = torch.autograd.grad(auxiliary, out['score_map'], retain_graph=True)[0]
        diagnostic['candidate_score_gradient_norm'] = float(gradient.norm())
        assert torch.isfinite(gradient).all() and gradient.norm() > 0
    combined = loss + weight * auxiliary
    assert torch.isfinite(combined)
    return combined, diagnostic
