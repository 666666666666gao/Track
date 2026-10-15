"""Synthetic CPU interface checks, with no tracking, images, or encoder models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from dense_target_decoder import DenseTargetDecoder
from dense_region_decoder import sample_regions
from instance_ab_prototype import roi_positions, RING
from region_conditioned_target_decoder import RegionConditionedTargetDecoder, sample_projected_regions
from template_write_features import initial_reference, write_features
from train_m122_full_causal import precision_objective


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_data():
    batch = 2
    axis = (torch.arange(16) + .5) * 16
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    origin = torch.tensor([[100., 80., 256.], [50., 60., 256.]])
    centers = torch.stack((x, y), dim=-1).reshape(1, 256, 2) + origin[:, None, :2]
    size = torch.tensor([40., 50.]).expand(batch, 256, 2)
    boxes = torch.cat((centers - .5 * size, size), dim=-1).clone()
    boxes[:, 0] = torch.tensor([600., 450., 20., 20.])
    probability = (.5 * torch.randn(batch, 256)).sigmoid()
    probability[:, 136] = .95
    window = torch.hann_window(16, periodic=False)
    window = (window[:, None] * window[None]).reshape(1, 256)
    data = {kind: torch.randn(batch, 256, 768) for kind in ['rgb', 'depth', 'fused', 'clip']}
    data.update({'initial_' + kind: torch.randn(batch, 256, 768)
                 for kind in ['rgb', 'depth', 'fused', 'clip']})
    valid = torch.ones(batch, 256)
    valid[:, :16] = 0
    data.update(native_valid=valid, clip_valid=valid.clone(),
        initial_native_valid=torch.ones(batch, 256), initial_clip_valid=torch.ones(batch, 256),
        boxes=boxes, native_score=probability, native_response=probability * window,
        origin=origin, initial_origin=torch.tensor([[150., 120., 160.], [80., 90., 160.]]),
        initial_box=torch.tensor([[190., 160., 40., 70.], [120., 130., 40., 70.]]),
        image_shape=torch.tensor([[512., 640.], [512., 640.]]),
        query=torch.randn(batch, 5, 768),
        query_mask=torch.tensor([[True, True, True, False, False], [True, True, False, True, False]]),
        empty_query=torch.randn(768), condition='human_text')
    target = torch.tensor([[216., 191., 40., 50.], [166., 171., 40., 50.]])
    return data, target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--warm-final', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert sha(args.warm_final) == 'cc71e0b4f42c2e489325627d0a16dcdfc4fa6bcb941908c30bc20c8c3c5828c0'
    assert not args.output.exists()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.random.default_generator.manual_seed(2027)
    warm = torch.load(args.warm_final, map_location='cpu')
    data, target = make_data()
    single = {key: value[:1] if torch.is_tensor(value) and key != 'empty_query' else value for key, value in data.items()}
    checks = []
    forwards = []

    def run(model, inputs):
        forwards.append(dict(model=type(model).__name__, arm=getattr(model, 'arm', 'inherited')))
        result = model(inputs)
        assert all(bool(torch.isfinite(value).all()) for value in result.values())
        assert result['boxes'].shape == (2, 256, 4)
        assert result['spatial_features'].shape == (2, 256, 128)
        row = torch.arange(2)
        index = result['selected_index']
        assert torch.equal(result['selected_feature'], result['spatial_features'][row, index])
        assert torch.equal(result['selected_quality'], result['quality_logits'].sigmoid()[row, index])
        return result

    # Independently check the inherited coordinate interface and a known affine grid.
    positions = (roi_positions() + 1.) * .5
    legacy, old_weight = sample_regions(data['fused'], data['native_valid'], data['boxes'], data['origin'], positions)
    sampled, new_weight = sample_projected_regions(data['fused'], data['native_valid'], data['boxes'], data['origin'], positions)
    assert torch.equal(legacy, sampled) and torch.equal(old_weight, new_weight)
    assert torch.equal(new_weight[:, 0], torch.zeros_like(new_weight[:, 0]))
    checks.append('global-pixel sampler exactly matches inherited768 interface, including outside-crop candidate')
    axis = (torch.arange(16) + .5) / 16
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    field = torch.stack((xx, yy), dim=-1).reshape(1, 256, 2)
    affine, _ = sample_projected_regions(field, torch.ones(1, 256),
        torch.tensor([[[100., 200., 256., 256.]]]), torch.tensor([[100., 200., 256.]]), positions)
    torch.testing.assert_close(affine[0, 0], positions, atol=1e-7, rtol=0.)
    expanded = torch.cat((data['boxes'][..., :2] - .5 * data['boxes'][..., 2:], 2. * data['boxes'][..., 2:]), dim=-1)
    ring, ring_weight = sample_projected_regions(data['fused'], data['native_valid'], expanded,
        data['origin'], positions[RING])
    old_ring, old_ring_weight = sample_regions(data['fused'], data['native_valid'], data['boxes'],
        data['origin'], positions[RING], ring=True)
    assert torch.equal(ring, old_ring) and torch.equal(ring_weight, old_ring_weight)
    checks.append('known affine64-compatible coordinates and inherited outer ring')

    base = DenseTargetDecoder().cpu().eval()
    base.load_state_dict(warm['A_B'], strict=True)
    models = {}
    new_parameters = []
    for arm in ['roi', 'language', 'visual']:
        torch.random.default_generator.manual_seed(2027)
        model = RegionConditionedTargetDecoder(arm).cpu().eval()
        model.load_common_warm(warm['A_B'])
        assert sum(p.numel() for p in model.parameters()) == 413517
        new = {key: value.clone() for key, value in model.state_dict().items()
               if key.startswith(model.NEW_PREFIXES)}
        new_parameters.append(new)
        for name, value in warm['A_B'].items():assert torch.equal(model.state_dict()[name], value)
        inputs = data
        if arm == 'visual':inputs = dict(data, condition='visual_query', query_mask=torch.ones_like(data['query_mask']))
        with torch.no_grad():
            expected = run(base, inputs)
            actual = run(model, data)
        for name in expected:assert torch.equal(expected[name], actual[name]), (arm, name)
        assert not bool(actual['region_semantic_delta'].any())
        assert not bool(actual['region_geometry_delta'].any())
        old_reference = initial_reference(SimpleNamespace(decoder=base, initial=single))
        new_reference = initial_reference(SimpleNamespace(decoder=model, initial=single))
        assert torch.equal(old_reference, new_reference)
        assert old_reference.shape == (1, 256)
        models[arm] = model
    for candidate in new_parameters[1:]:
        assert all(torch.equal(candidate[name], new_parameters[0][name]) for name in candidate)
    checks.append('all three arms inherit exact warm tensors, equal33350 new parameters and zero-tail full-output parity')
    checks.append('unchanged256-value C initial reference for every arm')

    # Exercise the 519-feature C input with its actual one-instance contract.
    reference = initial_reference(SimpleNamespace(decoder=models['language'], initial=single))
    current = torch.randn(1, 128)
    record = dict(previous_bbox=[120., 100., 40., 60.], bbox=[122., 101., 42., 58.],
        selected_quality=.6, observation_intersection_probability=.8, native_same_position_response=.7)
    assert write_features(current, current.clone(), reference, record).shape == (1, 519)
    checks.append('unchanged actual one-instance519 C input interface')

    for model in models.values():
        with torch.no_grad():
            for head in [model.region_support, model.region_box_delta, model.region_quality]:
                nn.init.normal_(head.weight, std=.01)
    changed = dict(data, query=-data['query'], query_mask=torch.ones_like(data['query_mask']))
    with torch.no_grad():
        language = run(models['language'], data)
        conflict = run(models['language'], changed)
        for name in ['boxes', 'raw_boxes', 'quality_logits', 'visual_response']:
            assert torch.equal(language[name], conflict[name]), name
        assert not torch.equal(language['region_semantic_delta'], conflict['region_semantic_delta'])
        visual = run(models['visual'], data)
        visual_changed = run(models['visual'], changed)
        for name in visual:assert torch.equal(visual[name], visual_changed[name]), name
        plain = run(models['roi'], data)
        assert not bool(plain['region_semantic_delta'].any())
        empty = dict(data, query=data['empty_query'][None, None].expand(2, 5, -1), condition='empty')
        no_words = run(models['language'], empty)
        assert not bool(no_words['semantic_delta'].any())
        assert torch.equal(no_words['response'], no_words['visual_response'])
    checks.append('nonzero region heads: language changes support while all256 geometry/quality stay query-invariant')
    checks.append('visual control ignores supplied phrase content and phrase-count mask')
    checks.append('plain ROI adds no regional language increment; Empty full-null difference exactly zero')

    model = models['language'].train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=.01)
    optimizer.zero_grad(set_to_none=True)
    output = run(model, data)
    loss, _, _ = precision_objective(output, data, target, 1)
    assert bool(torch.isfinite(loss))
    loss.backward()
    gradients = {prefix: sum(float(p.grad.norm()) for name, p in model.named_parameters()
        if name.startswith(prefix) and p.grad is not None) for prefix in model.NEW_PREFIXES}
    assert all(value > 0 for value in gradients.values())
    assert all(bool(p.grad.isfinite().all()) for p in model.parameters() if p.grad is not None)
    optimizer.step()
    args.output.mkdir()
    state_path = args.output / 'synthetic_unit_state_only.pt'
    torch.save(model.state_dict(), state_path)
    restored = RegionConditionedTargetDecoder('language').cpu().eval()
    restored.load_state_dict(torch.load(state_path, map_location='cpu'), strict=True)
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items())
    with torch.no_grad():
        before = run(model.eval(), data)
        after = run(restored, data)
    assert all(torch.equal(before[name], after[name]) for name in before)
    checks.append('inherited precision loss gives finite nonzero gradients to every new parameter group')
    checks.append('strict complete-state save/reload preserves every synthetic output')
    result = dict(status='complete_synthetic_CPU_region_decoder_interface_checks', seed=2027,
        evaluation_type='synthetic_interface_only', checks=checks, check_count=len(checks),
        inherited_AB_parameters=380167, additional_region_parameters=33350, AB_parameters=413517,
        unchanged_C_parameters=70721, combined_trainable_parameters=484238,
        new_group_gradient_norms=gradients, decoder_forward_calls=len(forwards), forward_log=forwards,
        real_tracking_calls=0, dataset_images_opened=0, STTrack_CLIP_models_instantiated=0,
        GPU_queries=0, CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'], formal_training_result=False,
        synthetic_state_used_for_any_formal_initialization=False, warm_sha256=sha(args.warm_final),
        source_sha256=sha(Path(__file__).with_name('region_conditioned_target_decoder.py')),
        test_source_sha256=sha(__file__), synthetic_state_sha256=sha(state_path))
    (args.output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
