"""Training-only GT-supported same-state native spatial preservation."""
import torch
from window_competition import supervision as tracking_supervision


def spatial_kl(student_score, teacher_score):
    # Native Center Head already clamps sigmoid outputs to [1e-4, 1-1e-4].
    # Normalize raw spatial evidence, sum 256 positions, average batch only.
    student = student_score.flatten(1)
    teacher = teacher_score.detach().flatten(1)
    student = student / student.sum(dim=1, keepdim=True)
    teacher = teacher / teacher.sum(dim=1, keepdim=True)
    return (teacher * (teacher.log() - student.log())).sum(dim=1).mean()


def bbox_iou_xywh(box, gt):
    box = gt.new_tensor(box)
    intersection = (torch.minimum(box[:2]+box[2:], gt[:2]+gt[2:])-
                    torch.maximum(box[:2], gt[:2])).clamp(min=0).prod()
    return float(intersection / (box[2:].prod()+gt[2:].prod()-intersection))


def supervision(network, out, groundtruth, previous, resize, search_size,
                output_window, preservation_weight=1.):
    loss, diagnostic = tracking_supervision(network, out, groundtruth, previous,
                                            resize, search_size, output_window=output_window)
    diagnostic.update(native_eligible=False, native_iou=None, preservation_kl=None,
                      preservation_weight=preservation_weight, preservation_weighted=0.)
    if diagnostic['label'] != 'centre_inside':
        return loss, diagnostic
    gt = torch.as_tensor(groundtruth, dtype=torch.float32, device=out['score_map'].device)
    iou = bbox_iou_xywh(out['native_bbox'], gt)
    diagnostic['native_iou'] = iou
    if iou < .5:
        return loss, diagnostic
    kl = spatial_kl(out['score_map'], out['native_score_map'])
    diagnostic.update(native_eligible=True, preservation_kl=float(kl.detach()),
                      preservation_weighted=float(kl.detach())*preservation_weight)
    # Explicit disabled-loss control keeps the exact original tensor and graph.
    if preservation_weight == 0.:
        return loss, diagnostic
    combined = loss + preservation_weight * kl
    assert bool(torch.isfinite(combined))
    return combined, diagnostic
