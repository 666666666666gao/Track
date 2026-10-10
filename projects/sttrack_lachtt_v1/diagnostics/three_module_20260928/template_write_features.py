"""Fixed 519-value current/past input shared by both prospective C arms."""
import torch
from torch.nn import functional as F
from dense_region_decoder import sample_regions, weighted_mean


INPUT_DIM = 519
CONTRACT = 'M122-C-current128-past128-initial4x64-quality3-motion4-v1'


@torch.no_grad()
def initial_reference(actor):
    assert not actor.decoder.training
    blocks = []
    for kind in ['rgb', 'depth', 'fused', 'clip']:
        grid = actor.initial['initial_' + kind]
        valid = actor.initial['initial_clip_valid' if kind == 'clip' else 'initial_native_valid']
        values, weights = sample_regions(grid, valid, actor.initial['initial_box'][:, None],
            actor.initial['initial_origin'], (actor.decoder.roi_xy + 1.) * .5)
        projection = actor.decoder.semantic_projection if kind == 'clip' else actor.decoder.native[kind]
        values = projection(F.normalize(values[:, 0], dim=-1)) + actor.decoder.position(actor.decoder.roi_xy)[None]
        blocks.append(weighted_mean(values, weights[:, 0]))
    reference = torch.cat(blocks, dim=-1)
    assert reference.shape == (1, 256) and bool(torch.isfinite(reference).all())
    return reference.detach()


@torch.no_grad()
def write_features(current, previous, reference, record):
    assert current.shape == previous.shape == (1, 128) and reference.shape == (1, 256)
    before = current.new_tensor(record['previous_bbox'])
    now = current.new_tensor(record['bbox'])
    before_center = before[:2] + .5 * before[2:]
    now_center = now[:2] + .5 * now[2:]
    motion = torch.cat(((now_center - before_center) / before[2:], (now[2:] / before[2:]).log()))[None]
    quality = current.new_tensor([[record['selected_quality'], record['observation_intersection_probability'],
                                   record['native_same_position_response']]])
    result = torch.cat((current, previous, reference, quality, motion), dim=-1).float()
    assert result.shape == (1, INPUT_DIM) and bool(torch.isfinite(result).all())
    return result.detach()
