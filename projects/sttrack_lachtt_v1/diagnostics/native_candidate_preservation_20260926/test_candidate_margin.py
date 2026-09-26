"""Synthetic localized-choice regression checks; not tracking-performance evidence."""
import torch

from candidate_margin import candidate_margin_loss


teacher = torch.tensor([.8, .2, .1, .1], requires_grad=True)
student = torch.tensor([.2, .8, .1, .1], requires_grad=True)
quality = torch.tensor([.9, .0, .8, .0])
window = torch.ones(4)
loss, diagnostic = candidate_margin_loss(student, teacher, quality, window, True)
assert diagnostic['eligible'] and diagnostic['positive_index'] == 0
assert diagnostic['competitors'] == 2 and loss > 0
loss.backward()
assert teacher.grad is None
assert student.grad[0] < 0 and student.grad[1] > 0 and student.grad[2] == 0
improved = student.detach() - .1 * student.grad
better, _ = candidate_margin_loss(improved, teacher, quality, window, True)
assert better < loss
same, _ = candidate_margin_loss(teacher, teacher, quality, window, True)
assert same == 0
for reliable, geometry in [(False, quality), (True, torch.tensor([.2, .0, .8, .0])), (True, torch.ones(4))]:
    skipped, diag = candidate_margin_loss(student, teacher, geometry, window, reliable)
    assert skipped == 0 and not diag['eligible']
edge, _ = candidate_margin_loss(student, teacher, quality, torch.tensor([1., 1., 1., 0.]), True)
assert torch.isfinite(edge)
print('candidate margin: corrected ranking reduces loss; no teacher gradient; good alternative and ineligible geometry protected; zero-window finite PASS')
