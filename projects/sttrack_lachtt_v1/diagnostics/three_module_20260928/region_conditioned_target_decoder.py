"""Dense box-region evidence with visual geometry and first-frame/phrase reads.

ROI, language and visual controls share every trainable parameter. This predicts
target-location support and IoU quality, not physical-identity or visibility truth.
"""
import torch
from torch import nn
from torch.nn import functional as F

from dense_target_decoder import DenseTargetDecoder, clip_boxes, odds_residual
from dense_region_decoder import sample_regions, weighted_mean
from instance_ab_prototype import RING


def sample_projected_regions(grid, valid, boxes, origins, xy):
    """Read the shared 16x16 grid in the inherited global-pixel crop coordinates."""
    batch, _, channels = grid.shape
    points = boxes[:, :, None, :2] + xy[None, None] * boxes[:, :, None, 2:]
    coordinates = 2. * (points - origins[:, None, None, :2]) / origins[:, None, None, 2:] - 1.
    image = grid.reshape(batch, 16, 16, channels).permute(0, 3, 1, 2)
    values = F.grid_sample(image, coordinates, align_corners=False, padding_mode='zeros')
    weights = F.grid_sample(valid.reshape(batch, 1, 16, 16), coordinates,
                            align_corners=False, padding_mode='zeros')
    return values.permute(0, 2, 3, 1), weights[:, 0]


class RegionConditionedTargetDecoder(DenseTargetDecoder):
    NEW_PREFIXES = ('region_encoder.', 'region_support.', 'region_box_delta.', 'region_quality.')

    def __init__(self, arm):
        assert arm in ['roi', 'language', 'visual']
        super().__init__()
        self.arm = arm
        self.region_encoder = nn.Sequential(nn.Linear(512, 64), nn.GELU(), nn.LayerNorm(64))
        self.region_support = nn.Linear(64, 1)
        self.region_box_delta = nn.Linear(64, 4)
        self.region_quality = nn.Linear(64, 1)
        for head in [self.region_support, self.region_box_delta, self.region_quality]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def load_common_warm(self, common):
        state = self.state_dict()
        inherited = {name for name in state if not name.startswith(self.NEW_PREFIXES)}
        assert set(common) == inherited
        state.update(common)
        self.load_state_dict(state, strict=True)

    def observe(self, data):
        # Keep the inherited observation math; additionally retain its four projected grids.
        references = []
        current = []
        initial_means = []
        initial_weights = None
        initial_clip = None
        for kind in ['rgb', 'depth', 'fused', 'clip']:
            grid = data['initial_' + kind]
            valid = data['initial_clip_valid' if kind == 'clip' else 'initial_native_valid']
            roi, weight = sample_regions(grid, valid, data['initial_box'][:, None],
                data['initial_origin'], (self.roi_xy + 1.) * .5)
            roi = F.normalize(roi[:, 0], dim=-1)
            weight = weight[:, 0]
            projection = self.semantic_projection if kind == 'clip' else self.native[kind]
            projected = projection(roi)
            references.append(projected + self.position(self.roi_xy)[None])
            initial_means.append(weighted_mean(projected, weight))
            current.append(projection(F.normalize(data[kind], dim=-1)))
            if kind == 'clip':
                initial_clip = weighted_mean(roi, weight)
            if initial_weights is None:
                initial_weights = weight
            else:
                assert bool((weight.sum(-1) > 0).all())
        reference = torch.cat(references, dim=1)
        _, clip_weight = sample_regions(data['initial_clip'], data['initial_clip_valid'],
            data['initial_box'][:, None], data['initial_origin'], (self.roi_xy + 1.) * .5)
        mask = torch.cat([initial_weights] * 3 + [clip_weight[:, 0]], dim=-1) > 0
        assert bool(mask.any(-1).all())
        search = self.search_fusion(torch.cat(current, dim=-1)) + self.position(self.search_xy)[None]
        context = self.reference_read(search, reference, reference, key_padding_mask=~mask,
                                      need_weights=False)[0]
        visual = self.visual_fusion(torch.cat((search, context), dim=-1))
        return dict(reference=reference, reference_mask=mask, visual=visual, fused=current[2],
                    initial_query=initial_clip, projected_grids=current, initial_means=initial_means)

    def read_regions(self, data, observed, boxes):
        values = []
        weights = []
        inside = []
        outside = []
        expanded = torch.cat((boxes[..., :2] - .5 * boxes[..., 2:], 2. * boxes[..., 2:]), dim=-1)
        for index, grid in enumerate(observed['projected_grids']):
            valid = data['clip_valid' if index == 3 else 'native_valid']
            roi, weight = sample_projected_regions(grid, valid, boxes, data['origin'],
                                                    (self.roi_xy + 1.) * .5)
            ring, ring_weight = sample_projected_regions(grid, valid, expanded, data['origin'],
                                                        (self.roi_xy[RING] + 1.) * .5)
            values.append(roi)
            weights.append(weight)
            inside.append(weighted_mean(roi, weight))
            outside.append(weighted_mean(ring, ring_weight))
        return dict(values=values, weights=weights, outside=torch.cat(outside, dim=-1),
                    ordinary=torch.cat(inside + outside, dim=-1))

    def conditioned_read(self, regions, observed, phrase_scores, phrase_valid, query_mask):
        pooled = []
        for values, valid, initial in zip(regions['values'], regions['weights'], observed['initial_means']):
            reference_similarity = (F.normalize(values, dim=-1) *
                                    F.normalize(initial, dim=-1)[:, None, None]).sum(-1)
            # Masked exponential weights define a zero read when a box has no crop samples.
            # Padding validity is not depth-measurement reliability or attribute visibility.
            weights = (reference_similarity[..., None] + phrase_scores).exp() * valid[..., None] * phrase_valid[..., None]
            by_phrase = (values[..., None, :] * weights[..., None]).sum(2)
            by_phrase = by_phrase / weights.sum(2).clamp_min(1.)[..., None]
            pooled.append(weighted_mean(by_phrase, query_mask[:, None].float()))
        return torch.cat(pooled + [regions['outside']], dim=-1)

    def forward(self, data):
        if self.arm == 'visual':
            # No phrase content or phrase-count mask enters the visual-control arm.
            data = dict(data, query_mask=torch.ones_like(data['query_mask']))
        observed = self.observe(data)
        batch = len(data['boxes'])
        visual_query = self.arm == 'visual' or data['condition'] == 'visual_query'
        query = observed['initial_query'][:, None].expand(-1, 5, -1) if visual_query else data['query']
        empty = data['empty_query'][None, None].expand(batch, 5, -1)
        full, semantic_features, cosine = self.branch(data, observed, query)
        null, _, _ = self.branch(data, observed, empty)

        geometry_input = torch.cat((observed['visual'], observed['fused']), dim=-1)
        local = self.geometry(geometry_input.transpose(1, 2).reshape(batch, 128, 16, 16))
        delta = self.box_delta(local).flatten(2).transpose(1, 2)
        base = data['boxes']
        scale = base[..., 2:]
        inherited_raw = torch.cat((base[..., :2] + delta[..., :2] * scale -
            .5 * scale * torch.expm1(delta[..., 2:]), scale * torch.exp(delta[..., 2:])), dim=-1)
        inherited_boxes = clip_boxes(inherited_raw, data['image_shape'])
        regions = self.read_regions(data, observed, inherited_boxes)
        encoded = self.region_encoder(regions['ordinary'])
        region_visual = self.region_support(encoded).squeeze(-1)
        region_geometry = self.region_box_delta(encoded)
        region_quality = self.region_quality(encoded).squeeze(-1)

        if self.arm == 'roi':
            region_semantic = torch.zeros_like(region_visual)
        else:
            phrases = torch.cat((F.normalize(query, dim=-1), F.normalize(empty, dim=-1)), dim=1)
            similarity_grid = F.normalize(data['clip'], dim=-1) @ phrases.transpose(-2, -1)
            scores, phrase_valid = sample_projected_regions(similarity_grid, data['clip_valid'],
                inherited_boxes, data['origin'], (self.roi_xy + 1.) * .5)
            full_region = self.conditioned_read(regions, observed, scores[..., :5], phrase_valid, data['query_mask'])
            null_region = self.conditioned_read(regions, observed, scores[..., 5:], phrase_valid, data['query_mask'])
            region_semantic = (self.region_support(self.region_encoder(full_region)) -
                               self.region_support(self.region_encoder(null_region))).squeeze(-1)

        scale = inherited_raw[..., 2:]
        raw_boxes = torch.cat((inherited_raw[..., :2] + region_geometry[..., :2] * scale -
            .5 * scale * torch.expm1(region_geometry[..., 2:]),
            scale * torch.exp(region_geometry[..., 2:])), dim=-1)
        boxes = clip_boxes(raw_boxes, data['image_shape'])
        quality_logits = self.quality(local).flatten(1) + region_quality
        observation = self.observation(torch.cat((observed['visual'].mean(1),
            observed['reference'].mean(1)), dim=-1)).squeeze(-1)
        native = data['native_score']
        assert bool(((native > 0) & (native < 1)).all())
        base_logits = torch.log(native) - torch.log1p(-native)
        visual_delta = self.visual_support(observed['visual']).squeeze(-1) + region_visual
        semantic_delta = full - null + region_semantic
        visual_logits = base_logits + visual_delta
        support_logits = visual_logits + semantic_delta
        probability = odds_residual(native, visual_delta + semantic_delta)
        visual_probability = odds_residual(native, visual_delta)
        response = data['native_response'] * (probability / native) * quality_logits.sigmoid()
        visual_response = data['native_response'] * (visual_probability / native) * quality_logits.sigmoid()
        selected = response.argmax(-1)
        row = torch.arange(batch, device=selected.device)
        features = torch.cat((observed['visual'], semantic_features), dim=-1)
        return dict(support_logits=support_logits, visual_support_logits=visual_logits,
            semantic_delta=semantic_delta, response=response, visual_response=visual_response,
            boxes=boxes, raw_boxes=raw_boxes, quality_logits=quality_logits,
            observation_logits=observation, phrase_cosine=cosine, selected_index=selected,
            selected_box=boxes[row, selected], selected_score=response[row, selected],
            selected_quality=quality_logits.sigmoid()[row, selected], selected_feature=features[row, selected],
            spatial_features=features, region_visual_delta=region_visual,
            region_semantic_delta=region_semantic, region_geometry_delta=region_geometry,
            region_quality_delta=region_quality,
            region_padding_fraction=torch.stack([w.mean(-1) for w in regions['weights']], dim=-1).mean(-1))
