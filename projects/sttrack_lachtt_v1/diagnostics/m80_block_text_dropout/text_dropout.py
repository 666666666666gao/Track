"""Training-only content selection; keep slots, padding, initial RoI and mask."""
import torch

def make_contexts(tracker, empty):
    category = tracker.semantic_context['text']
    mask = tracker.semantic_context['mask'].unsqueeze(-1)
    blank = torch.where(mask, empty.to(category).reshape(1, 1, -1), category)
    return category, blank
