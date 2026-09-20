"""Raw-response competition control; public inference still uses native Hann.

Training-only GT labels are read after the causal tracker has committed its state.
The native focal and box losses, network, decoder and template rule are retained.
"""
import torch
from causal_training import base_supervision
from lib.train.data.processing_utils import transform_image_to_crop


def competition_loss(out, target, output_window):
    """Rounded GT center versus up to nine decoded, severely wrong competitors."""
    score = out['score_map'].reshape(16, 16)
    assert score.shape == output_window.reshape(16, 16).shape
    center = target[:2] + .5 * target[2:]
    cell = (center * 16).round().clamp(0, 15).long()
    positive = cell[1] * 16 + cell[0]
    with torch.no_grad():
        axis = torch.arange(16, device=score.device, dtype=score.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        offset = out['offset_map'].detach()[0]
        centers = torch.stack((xx + offset[0], yy + offset[1]), -1).reshape(-1, 2) / 16
        size = out['size_map'].detach()[0].permute(1, 2, 0).reshape(-1, 2)
        lo, hi = centers - size / 2, centers + size / 2
        overlap = (torch.minimum(hi, target[:2]+target[2:]) - torch.maximum(lo, target[:2])).clamp(min=0)
        inter = overlap.prod(-1)
        iou = inter / (size.prod(-1) + target[2:].prod() - inter)
        # Exclude target-overlapping hypotheses; different cells can decode the same instance.
        outside = ((centers < target[:2]) | (centers > target[:2]+target[2:])).any(-1)
        negative = outside & (iou <= .1)
        negative[positive] = False
    weighted = score.flatten()
    # Center Head already clamps scores to [1e-4,1-1e-4]; raw competition does not use the Hann values.
    assert bool((weighted > 0).all())
    log_weighted = weighted.log()
    competitors = log_weighted[negative]
    hard = competitors.topk(min(9, competitors.numel())).values
    values = torch.cat((log_weighted[positive:positive+1], hard))
    loss = torch.logsumexp(values, dim=0) - log_weighted[positive]
    return loss, dict(competition_loss=float(loss.detach()), competition_negatives=hard.numel(),
        competition_positive_index=int(positive), competition_raw_top1_index=int(weighted.detach().argmax()),
        competition_positive_probability=float(values.softmax(0)[0].detach()))


def supervision(network, out, groundtruth, previous, resize, search_size, output_window):
    loss, diagnostic = base_supervision(network, out, groundtruth, previous, resize, search_size)
    if loss is None or diagnostic['label'] != 'centre_inside':
        # Existing invalid/empty-crop supervision has no positive target for this ranking task.
        diagnostic.update(competition_loss=None, competition_negatives=0)
        return loss, diagnostic
    gt = torch.as_tensor(groundtruth, dtype=torch.float32, device=out['score_map'].device)
    target = transform_image_to_crop(gt, gt.new_tensor(previous), resize,
        gt.new_tensor([search_size, search_size]), normalize=True)
    rank, values = competition_loss(out, target, output_window)
    diagnostic.update(values)
    combined = loss + rank
    assert bool(torch.isfinite(combined))
    return combined, diagnostic
