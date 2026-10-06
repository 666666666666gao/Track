"""M118 attributed regional evidence, followed by quality-aware residual selection.

The frozen dense tokens are from the measured M117 CLIP Surgery mechanism.
This module does not claim current phrase visibility or physical-instance truth.
"""
import torch
from torch import nn
from torch.nn import functional as F
from instance_ab_prototype import roi_positions, RING


def sample_regions(grid, valid, boxes, origins, xy, ring=False):
    if ring:
        boxes = torch.cat((boxes[..., :2] - .5 * boxes[..., 2:], 2. * boxes[..., 2:]), dim=-1)
    points = boxes[:, :, None, :2] + xy[None, None] * boxes[:, :, None, 2:]
    coordinates = 2. * (points - origins[:, None, None, :2]) / origins[:, None, None, 2:] - 1.
    batch, count, samples, _ = coordinates.shape
    coordinates = coordinates.reshape(batch, count, samples, 2)
    image = grid.reshape(batch, 16, 16, 768).permute(0, 3, 1, 2)
    values = F.grid_sample(image, coordinates, align_corners=False, padding_mode='zeros')
    weights = F.grid_sample(valid.reshape(batch, 1, 16, 16), coordinates,
                            align_corners=False, padding_mode='zeros')
    return values.permute(0, 2, 3, 1), weights[:, 0]


def weighted_mean(values, weights):
    return (values * weights[..., None]).sum(-2) / weights.sum(-1, keepdim=True).clamp_min(1.)


class DenseRegionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.region_projection = nn.Sequential(nn.Linear(768, 64), nn.LayerNorm(64))
        self.query_projection = nn.Sequential(nn.Linear(768, 64), nn.LayerNorm(64))
        self.position = nn.Linear(2, 64, bias=False)
        self.roles = nn.Parameter(torch.zeros(4, 64))
        self.query_slots = nn.Parameter(torch.zeros(5, 64))
        self.query_read = nn.MultiheadAttention(64, 4, dropout=0., batch_first=True)
        self.visual_read = nn.MultiheadAttention(64, 4, dropout=0., batch_first=True)
        self.decision = nn.Sequential(nn.Linear(277, 64), nn.GELU(), nn.Linear(64, 1, bias=False))
        nn.init.zeros_(self.decision[-1].weight)
        self.register_buffer('roi_xy', roi_positions())
        self.register_buffer('ring_xy', roi_positions()[RING] * 2.)

    def observe(self, data):
        boxes = data['boxes']
        current, valid = sample_regions(data['dense_grid'], data['dense_valid'], boxes,
                                        data['dense_origin'], (self.roi_xy + 1.) * .5)
        context, context_valid = sample_regions(data['dense_grid'], data['dense_valid'], boxes,
                                                 data['dense_origin'], (self.roi_xy[RING] + 1.) * .5, True)
        initial, initial_valid = sample_regions(data['initial_grid'], data['initial_valid'],
                                                 data['initial_box'][:, None], data['initial_origin'],
                                                 (self.roi_xy + 1.) * .5)
        initial_context, initial_context_valid = sample_regions(data['initial_grid'], data['initial_valid'],
                                                 data['initial_box'][:, None], data['initial_origin'],
                                                 (self.roi_xy[RING] + 1.) * .5, True)
        current = F.normalize(current, dim=-1)
        context = F.normalize(context, dim=-1)
        initial = F.normalize(initial, dim=-1)
        initial_context = F.normalize(initial_context, dim=-1)
        count = boxes.shape[1]
        initial = initial.expand(-1, count, -1, -1)
        initial_context = initial_context.expand(-1, count, -1, -1)
        initial_valid = initial_valid.expand(-1, count, -1)
        initial_context_valid = initial_context_valid.expand(-1, count, -1)
        parts = [initial, initial_context, current, context]
        masks = [initial_valid, initial_context_valid, valid, context_valid]
        positions = [self.roi_xy, self.ring_xy, self.roi_xy, self.ring_xy]
        encoded = [self.region_projection(value) + self.position(xy)[None, None] + self.roles[i]
                   for i, (value, xy) in enumerate(zip(parts, positions))]
        memory = torch.cat(encoded, dim=-2).flatten(0, 1)
        memory_valid = torch.cat(masks, dim=-1).flatten(0, 1) > 0
        assert bool(memory_valid.any(-1).all())
        identity = (F.normalize(weighted_mean(current, valid), dim=-1) *
                    F.normalize(weighted_mean(initial, initial_valid), dim=-1)).sum(-1, keepdim=True)
        return dict(current=current, valid=valid, projected=encoded[2], memory=memory,
                    memory_valid=memory_valid, identity=identity,
                    initial_query=F.normalize(weighted_mean(initial, initial_valid)[:, 0], dim=-1))

    def branch(self, data, observed, query):
        batch, count = data['boxes'].shape[:2]
        query = F.normalize(query, dim=-1)
        mask = data['query_mask'][:, None].expand(-1, count, -1).flatten(0, 1)
        projected = self.query_projection(query) + self.query_slots
        projected = projected[:, None].expand(-1, count, -1, -1).flatten(0, 1)
        bound = projected + self.query_read(projected, observed['memory'], observed['memory'],
                    key_padding_mask=~observed['memory_valid'], need_weights=False)[0]
        current = observed['projected'].flatten(0, 1)
        conditioned = current + self.visual_read(current, bound, bound,
                    key_padding_mask=~mask, need_weights=False)[0]
        conditioned = weighted_mean(conditioned, observed['valid'].flatten(0, 1)).reshape(batch, count, 64)
        bound_mean = weighted_mean(bound, mask.float()).reshape(batch, count, 64)
        cosine = observed['current'] @ query[:, None].transpose(-2, -1)
        cosine = weighted_mean(cosine, observed['valid']) * data['query_mask'][:, None]
        features = torch.cat((data['parent_features'], weighted_mean(observed['projected'], observed['valid']),
                              conditioned, bound_mean, cosine, observed['identity'], data['geometry'],
                              data['parent_scores'][..., None], data['parent_quality'][..., None]), dim=-1)
        assert features.shape[-1] == 277
        return self.decision(features).squeeze(-1), features

    def forward(self, data):
        observed = self.observe(data)
        empty = data['empty_query'][None, None].expand(len(data['boxes']), 5, -1)
        query = observed['initial_query'][:, None].expand(-1, 5, -1) if data['condition'] == 'visual_query' else data['query']
        full, features = self.branch(data, observed, query)
        null, _ = self.branch(data, observed, empty)
        delta = full - null
        delta = delta - delta.mean(-1, keepdim=True)
        scores = data['parent_scores'] + delta
        selected = scores.argmax(-1)
        row = torch.arange(len(selected), device=selected.device)
        return dict(selection_logits=scores, visual_selection_logits=data['parent_scores'],
                    quality_logits=data['parent_quality'], semantic_score_delta=delta,
                    selected_index=selected, selected_box=data['boxes'][row, selected],
                    selected_score=scores[row, selected], selected_quality=data['parent_quality'][row, selected],
                    selected_feature=features[row, selected], candidate_features=features)
