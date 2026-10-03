"""Isolated A+B wiring prototype; no tracker state or trained semantics."""

import torch
from torch import nn


RING = [0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15]


def roi_positions():
    axis = (torch.arange(4, dtype=torch.float32) + .5) / 4 * 2 - 1
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    return torch.stack((x, y), dim=-1).reshape(16, 2)


def masked_mean(tokens, valid):
    weight = valid.to(tokens.dtype)[..., None]
    return (tokens * weight).sum(-2) / weight.sum(-2).clamp_min(1)


class InstanceEvidence(nn.Module):
    def __init__(self, channels=768, hidden=64, slots=5):
        super().__init__()
        self.projection = nn.ModuleList([
            nn.Sequential(nn.Linear(channels, hidden), nn.LayerNorm(hidden)) for _ in range(2)])
        self.position = nn.Linear(2, hidden, bias=False)
        self.roles = nn.Parameter(torch.zeros(4, hidden))
        self.modalities = nn.Parameter(torch.zeros(2, hidden))
        self.phrase_slots = nn.Parameter(torch.zeros(slots, hidden))
        self.modality = nn.Linear(2 * hidden + 1, 2)
        self.text = nn.Sequential(nn.Linear(channels, hidden), nn.LayerNorm(hidden))
        self.visual_read = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.phrase_read = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.text_read = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.evidence = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))
        self.register_buffer('roi_xy', roi_positions())
        self.register_buffer('ring_xy', roi_positions()[RING] * 2)

    def encode(self, region, valid, depth_valid, role, ring=False):
        projected = torch.stack([self.projection[m](region[:, :, m]) for m in range(2)], dim=2)
        pooled = torch.stack([masked_mean(projected[:, :, m], valid) for m in range(2)], dim=2)
        weights = self.modality(torch.cat((pooled[:, :, 0], pooled[:, :, 1],
                                           depth_valid[..., None]), dim=-1)).softmax(-1)
        xy = self.ring_xy if ring else self.roi_xy
        encoded = (projected * weights[..., None, None] + self.position(xy)[None, None, None]
                   + self.modalities[None, None, :, None] + self.roles[role])
        return encoded.flatten(2, 3), valid[:, :, None].expand(-1, -1, 2, -1).flatten(2, 3), weights

    def phrase_branch(self, text, text_mask, memory, memory_valid, current, current_valid):
        batch, count = current.shape[:2]
        query = self.text(text.contiguous()) + self.phrase_slots
        query = query[:, None].expand(-1, count, -1, -1).reshape(batch * count, text.shape[1], -1)
        bound = query + self.phrase_read(query, memory, memory,
                                         key_padding_mask=~memory_valid, need_weights=False)[0]
        mask = text_mask[:, None].expand(-1, count, -1).reshape(batch * count, -1)
        conditioned = self.text_read(current.flatten(0, 1), bound, bound,
                                     key_padding_mask=~mask, need_weights=False)[0]
        pooled = masked_mean(conditioned, current_valid.flatten(0, 1)).reshape(batch, count, -1)
        evidence = self.evidence(torch.cat((query, bound), dim=-1))
        evidence = (evidence * mask[..., None]).reshape(batch, count, text.shape[1], 3)
        return pooled, evidence

    def forward(self, data):
        count = data['candidate_rois'].shape[1]
        initial = data['initial_rois'][:, None].expand(-1, count, -1, -1, -1)
        initial_context = data['initial_context'][:, None].expand(-1, count, -1, -1, -1)
        initial_mask = data['initial_mask'][:, None].expand(-1, count, -1)
        initial_context_mask = data['initial_context_mask'][:, None].expand(-1, count, -1)
        initial_depth = data['initial_depth_valid'][:, None].expand(-1, count)
        origin, origin_valid, _ = self.encode(initial, initial_mask, initial_depth, 0)
        surround, surround_valid, _ = self.encode(initial_context, initial_context_mask, initial_depth, 1, True)
        current, current_valid, weights = self.encode(data['candidate_rois'], data['candidate_mask'],
                                                       data['depth_valid'], 2)
        context, context_valid, _ = self.encode(data['contexts'], data['context_mask'],
                                                data['depth_valid'], 3, True)
        reference = torch.cat((origin, surround, context), dim=2).flatten(0, 1)
        reference_valid = torch.cat((origin_valid, surround_valid, context_valid), dim=2).flatten(0, 1)
        assert bool(reference_valid.any(-1).all())
        visual = current.flatten(0, 1) + self.visual_read(current.flatten(0, 1), reference, reference,
                    key_padding_mask=~reference_valid, need_weights=False)[0]
        visual = masked_mean(visual, current_valid.flatten(0, 1)).reshape(*current.shape[:2], -1)
        memory = torch.cat((origin, surround, current, context), dim=2).flatten(0, 1)
        memory_valid = torch.cat((origin_valid, surround_valid, current_valid, context_valid), dim=2).flatten(0, 1)
        empty = data['empty_text'][None, None].expand_as(data['text'])
        empty = empty * data['text_mask'][..., None]
        full, evidence = self.phrase_branch(data['text'], data['text_mask'], memory, memory_valid,
                                             current, current_valid)
        null, null_evidence = self.phrase_branch(empty, data['text_mask'], memory, memory_valid,
                                                 current, current_valid)
        return dict(visual=visual, semantic_delta=full - null,
                    phrase_delta=evidence - null_evidence, phrase_logits=evidence,
                    modality_weights=weights)


class InstanceCandidatePrototype(nn.Module):
    def __init__(self, channels=768, hidden=64, slots=5):
        super().__init__()
        self.evidence = InstanceEvidence(channels, hidden, slots)
        self.geometry = nn.Linear(13, hidden)
        self.candidate_read = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.semantic = nn.Linear(hidden, hidden, bias=False)
        self.phrase = nn.Linear(3, hidden, bias=False)
        self.selection = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.quality = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.observation = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.selection[-1].weight)
        nn.init.zeros_(self.selection[-1].bias)

    def forward(self, data):
        evidence = self.evidence(data)
        visual = evidence['visual'] + self.geometry(data['geometry'])
        visual = visual + self.candidate_read(visual, visual, visual, need_weights=False)[0]
        phrase = evidence['phrase_delta'].sum(-2) / data['text_mask'].sum(-1)[:, None, None]
        instance = visual + self.semantic(evidence['semantic_delta']) + self.phrase(phrase)
        score = data['base_scores'] + self.selection(instance).squeeze(-1)
        visual_score = data['base_scores'] + self.selection(visual).squeeze(-1)
        quality = self.quality(visual).squeeze(-1)
        selected = score.argmax(-1)
        row = torch.arange(len(selected), device=selected.device)
        return dict(selection_logits=score, visual_selection_logits=visual_score,
                    quality_logits=quality, observation_logits=self.observation(visual.mean(1)).squeeze(-1),
                    selected_index=selected, selected_box=data['boxes'][row, selected],
                    selected_score=score[row, selected], selected_quality=quality[row, selected],
                    selected_feature=instance[row, selected], candidate_features=instance,
                    phrase_delta=evidence['phrase_delta'], semantic_delta=evidence['semantic_delta'],
                    modality_weights=evidence['modality_weights'])
