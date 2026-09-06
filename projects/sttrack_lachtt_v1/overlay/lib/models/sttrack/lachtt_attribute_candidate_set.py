"""M56 hypothesis: contextual text tokens interact with local RGB-D cells."""
import torch
from torch import nn
import torch.nn.functional as F

from lib.models.sttrack.lachtt_candidate_set import CandidateSetAssociation


def condition_text(tokens, mask, empty, variant):
    """Matched phrase-set, phrase-mean, and valid empty-string inputs."""
    assert variant in ['attributes', 'pooled', 'empty']
    if variant == 'attributes':
        return tokens, mask
    result = torch.zeros_like(tokens)
    valid = torch.zeros_like(mask)
    valid[:, 0] = True
    if variant == 'pooled':
        result[:, 0] = (tokens * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
    else:
        result[:, 0] = empty
    return result, valid


class AttributeLocalAlignment(nn.Module):
    def __init__(self):
        super().__init__()
        self.text = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 32), nn.GELU())
        self.read_visual = nn.MultiheadAttention(32, 4, dropout=0., batch_first=True)
        self.guide_visual = nn.MultiheadAttention(32, 4, dropout=0., batch_first=True)
        self.output = nn.Linear(32, 32)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, cells, text_tokens, text_mask):
        # Cells: B,objects,2 modalities,16 spatial cells,32 channels.
        # Empty-text controls supply a valid frozen CLIP empty-string embedding.
        batch, objects = cells.shape[:2]
        visual = cells.flatten(2, 3).flatten(0, 1)
        text = self.text(text_tokens)
        text = text[:, None].expand(-1, objects, -1, -1).flatten(0, 1)
        padding = ~text_mask[:, None].expand(-1, objects, -1).flatten(0, 1)
        observed, _ = self.read_visual(text, visual, visual, need_weights=False)
        conditioned = text + observed
        guided, _ = self.guide_visual(visual, conditioned, conditioned,
                                      key_padding_mask=padding, need_weights=False)
        return cells + self.output(guided).reshape_as(cells)


class AttributeCandidateSetAssociation(CandidateSetAssociation):
    def __init__(self):
        super().__init__(use_geometry=True)
        self.attribute_alignment = AttributeLocalAlignment()

    def forward(self, current, previous, references, geometry, scores,
                previous_choice, text_tokens, text_mask):
        batch = current.shape[0]
        cells = self.cell(torch.cat([current, previous, references], dim=1))
        cells = self.attribute_alignment(cells, text_tokens, text_mask)
        x = self.object(cells.flatten(-3))
        x = torch.cat([x[:, :20] + self.position(geometry) + self.response(scores[..., None]),
                       x[:, 20:]], dim=1)
        roles = torch.tensor([0] * 10 + [1] * 10 + [2, 3], device=current.device)
        x = x + self.role(roles)[None]
        marker = F.one_hot(previous_choice, 10).to(x.dtype)[..., None] * self.previous_target
        x = torch.cat([x[:, :10], x[:, 10:20] + marker, x[:, 20:]], dim=1)
        x = self.context(x)
        logits = torch.cat([self.identity(x[:, :10]).squeeze(-1) + scores[:, :10],
                            self.none(x[:, 20])], dim=1)
        current_match = F.normalize(self.match(x[:, :10]), dim=-1)
        previous_match = F.normalize(self.match(x[:, 10:20]), dim=-1)
        affinity = torch.matmul(current_match, previous_match.transpose(1, 2)) / .1
        affinity = torch.cat([affinity, self.unmatched.expand(batch, 10, 1)], dim=2)
        affinity = torch.cat([affinity, self.unmatched.expand(batch, 1, 11)], dim=1)
        return logits, affinity
