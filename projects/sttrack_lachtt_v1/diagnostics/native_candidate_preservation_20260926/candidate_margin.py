"""Training-only preservation of a GT-supported native candidate's relative response."""
import torch


def candidate_margin_loss(student_score, teacher_score, candidate_iou, window, teacher_reliable):
    """IoUs label localization quality, not physical-instance identity.

    The caller establishes native GT IoU >= .5. The same position must also
    have student geometry IoU >= .5; negatives have student geometry IoU <= .1.
    Teacher scores and geometry labels never receive gradients.
    """
    zero = student_score.sum() * 0.
    diagnostic = dict(eligible=False, competitors=0)
    if not teacher_reliable:
        return zero, diagnostic
    student = student_score.flatten() * window.flatten()
    teacher = teacher_score.detach().flatten() * window.flatten()
    quality = candidate_iou.detach().flatten()
    positive = teacher.argmax()
    if quality[positive] < .5:
        return zero, diagnostic
    negative = torch.where(quality <= .1)[0]
    if negative.numel() == 0:
        return zero, diagnostic
    negative = negative[student.detach()[negative].topk(min(9, negative.numel())).indices]
    # The positive Hann response is strictly positive for the clamped Center
    # Head. Zero-window competitors are harmless and require no log epsilon.
    student_gap = (student[positive] - student[negative]) / (student[positive] + student[negative])
    teacher_gap = (teacher[positive] - teacher[negative]) / (teacher[positive] + teacher[negative])
    loss = torch.relu(teacher_gap - student_gap).mean()
    diagnostic.update(eligible=True, competitors=negative.numel(), positive_index=int(positive),
                      loss=float(loss.detach()))
    return loss, diagnostic
