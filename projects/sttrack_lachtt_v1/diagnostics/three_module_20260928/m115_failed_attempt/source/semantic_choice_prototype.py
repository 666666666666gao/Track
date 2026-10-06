"""Direct semantic candidate scoring on an unchanged M101 visual parent."""
import torch
from torch import nn
from instance_ab_prototype import InstanceCandidatePrototype


TEXT_PREFIXES = ('evidence.text.', 'evidence.phrase_slots',
                 'evidence.phrase_read.', 'evidence.text_read.', 'evidence.evidence.')


class SemanticChoicePrototype(nn.Module):
    def __init__(self):
        super().__init__()
        self.parent = InstanceCandidatePrototype()
        self.semantic_choice = nn.Sequential(nn.Linear(131, 64), nn.GELU(),
                                             nn.Linear(64, 1, bias=False))
        nn.init.zeros_(self.semantic_choice[-1].weight)

    def initialize_parent(self, state):
        self.parent.load_state_dict(state, strict=True)
        nn.init.zeros_(self.parent.semantic.weight)
        nn.init.zeros_(self.parent.phrase.weight)
        self.parent.requires_grad_(False)
        for name, parameter in self.parent.named_parameters():
            if name.startswith(TEXT_PREFIXES):
                parameter.requires_grad_(True)

    def forward(self, data):
        out = self.parent(data)
        visual = out['candidate_features']
        phrase = out['phrase_delta'].sum(-2) / data['text_mask'].sum(-1)[:, None, None]
        semantic = torch.cat((out['semantic_delta'], phrase), dim=-1)
        full = self.semantic_choice(torch.cat((visual, semantic), dim=-1)).squeeze(-1)
        null = self.semantic_choice(torch.cat((visual, torch.zeros_like(semantic)), dim=-1)).squeeze(-1)
        raw_delta = full - null
        delta = raw_delta - raw_delta.mean(-1, keepdim=True)
        score = out['visual_selection_logits'] + delta
        selected = score.argmax(-1)
        row = torch.arange(len(selected), device=selected.device)
        out.update(selection_logits=score, selected_index=selected,
                   selected_box=data['boxes'][row, selected], selected_score=score[row, selected],
                   selected_quality=out['quality_logits'][row, selected],
                   selected_feature=visual[row, selected], semantic_score_delta=delta,
                   raw_semantic_score_delta=raw_delta)
        return out
