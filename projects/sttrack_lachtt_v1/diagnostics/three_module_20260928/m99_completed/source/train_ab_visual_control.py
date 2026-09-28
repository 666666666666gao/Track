"""M97 same-architecture Empty visual control; IoU quality, never identity truth."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from analyze_train_states import overlaps, sha
from collect_ab_interface_panel import geometry_features
from instance_ab_prototype import InstanceCandidatePrototype
from train_fixed_visual_selector import load_initial_search


def read(path):
    return json.loads(Path(path).read_text())


def load_inputs(cache, contexts, origins, smoke):
    prep = read(cache/'preparation.json')
    assert prep['training_labels_sha256'] == sha(cache/'training_labels.json')
    labels = read(cache/'training_labels.json')
    initial = load_initial_search(origins, cache)
    if smoke:
        receipt = read(contexts/'smoke_shard0.json')
        assert receipt['smoke'] and receipt['status'] == 'complete_context_collection_only'
        items = receipt['sequences']
        directory = contexts/'smoke_features'
        assert len(items) == 1 and items[0]['sequence'] == 'cube04_indoor'
    else:
        assert (contexts/'queue.exit').read_text().strip() == '0'
        audit = read(contexts/'input_audit.json')
        assert audit['status'] == 'complete_context_input_audit_only'
        assert (audit['sequences'], audit['events'], audit['frames']) == (152, 3502, 219194)
        items = [r for shard in (0, 1) for r in read(contexts/f'shard{shard}.json')['sequences']]
        directory = contexts/'features'
    fields = {split: defaultdict(list) for split in ('fit', 'development')}
    for row in items:
        sequence = row['sequence']
        source_path = cache/'features'/f'{sequence}.pt'
        context_path = directory/f'{sequence}.pt'
        assert sha(source_path) == row['original_feature_sha256']
        assert sha(context_path) == row['feature_sha256']
        original = torch.load(source_path, map_location='cpu')
        data = torch.load(context_path, map_location='cpu')
        assert not original['labels_loaded'] and not data['GT_loaded'] and not data['text_loaded']
        assert not data['auxiliary_query_committed'] and not data['prototype_state_committed']
        assert data['split'] == original['split'] == row['split']
        assert data['original_feature_sha256'] == row['original_feature_sha256']
        indices = {frame:index for index,frame in enumerate(original['event_frames'])}
        for index, frame in enumerate(data['event_frames']):
            key = f'{sequence}@{frame}'
            label = labels[key]
            assert label['split'] == data['split']
            if label['current'] is None:
                continue
            at = indices[frame]
            split = fields[data['split']]
            split['key'].append(key)
            split['strata'].append(label['strata'])
            split['candidate_rois'].append(original['candidate_rois'][at])
            split['initial_rois'].append(initial[sequence])
            for name in ('initial_context', 'initial_mask', 'initial_context_mask', 'initial_depth_valid'):
                split[name].append(data[name])
            for name in ('contexts', 'candidate_mask', 'context_mask', 'depth_valid'):
                split[name].append(data[name][index])
            split['geometry'].append(geometry_features(original, at))
            split['base_scores'].append(original['scores'][at])
            split['boxes'].append(original['boxes'][at])
            split['iou'].append(overlaps(original['boxes'][at], torch.tensor(label['current'], dtype=torch.float32)))
    panel = {split:{key:values if key in ('key', 'strata') else torch.stack(values)
                    for key,values in group.items()} for split,group in fields.items() if group}
    if smoke:
        assert panel['fit']['key'] == ['cube04_indoor@10', 'cube04_indoor@12', 'cube04_indoor@14']
        assert 'development' not in panel
    else:
        assert len(panel['fit']['key']) == 2544 and len(panel['development']['key']) == 495
    return panel


def batch(panel, indices, empty, device):
    data = {key:value[indices].to(device) for key,value in panel.items()
            if torch.is_tensor(value) and key != 'iou'}
    data = {key:value if value.dtype == torch.bool else value.float() for key,value in data.items()}
    count = len(indices)
    data['empty_text'] = empty
    data['text'] = empty[None, None].expand(count, 5, -1)
    data['text_mask'] = torch.ones(count, 5, dtype=torch.bool, device=device)
    return data


def freeze_unused(model):
    modules = (model.evidence.text, model.evidence.phrase_read, model.evidence.text_read,
               model.evidence.evidence, model.semantic, model.phrase, model.observation)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    model.evidence.phrase_slots.requires_grad_(False)
    return {name:parameter.detach().cpu().clone() for name,parameter in model.named_parameters()
            if not parameter.requires_grad}


def check_frozen(model, frozen):
    for name, parameter in model.named_parameters():
        if name in frozen:
            assert torch.equal(parameter.detach().cpu(), frozen[name]), name


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups['all'].append(row)
        for tag in row['strata']:
            groups[tag].append(row)
    return {tag:dict(valid_gt=len(values),
                    native_iou50=sum(r['native_iou'] >= .5 for r in values),
                    selected_iou50=sum(r['selected_iou'] >= .5 for r in values),
                    oracle_iou50=sum(r['oracle_iou'] >= .5 for r in values),
                    rescues=sum(r['native_iou'] < .5 and r['selected_iou'] >= .5 for r in values),
                    breaks=sum(r['native_iou'] >= .5 and r['selected_iou'] < .5 for r in values),
                    native_mean_iou=sum(r['native_iou'] for r in values)/len(values),
                    selected_mean_iou=sum(r['selected_iou'] for r in values)/len(values))
            for tag,values in groups.items()}


def evaluate(model, panel, empty, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(panel['key']), 64):
            indices = torch.arange(start, min(start+64, len(panel['key'])))
            output = model(batch(panel, indices, empty, device))
            assert torch.equal(output['selection_logits'], output['visual_selection_logits'])
            selected = output['selected_index'].cpu()
            for offset,index in enumerate(indices.tolist()):
                iou = panel['iou'][index]
                rows.append(dict(key=panel['key'][index], strata=panel['strata'][index],
                                 native_iou=float(iou[0]), selected_iou=float(iou[selected[offset]]),
                                 oracle_iou=float(iou.max()), selected=int(selected[offset])))
    return summarize(rows), rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--contexts', type=Path, required=True)
    parser.add_argument('--origins', type=Path, required=True)
    parser.add_argument('--empty-bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('cpu-smoke', 'gpu-sanity', 'train'), required=True)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    random.seed(2027); np.random.seed(2027); torch.manual_seed(2027); torch.cuda.manual_seed_all(2027)
    assert (args.mode == 'cpu-smoke') == (args.device == 'cpu')
    panel = load_inputs(args.cache, args.contexts, args.origins, args.mode == 'cpu-smoke')
    device = torch.device(args.device)
    bank = torch.load(args.empty_bank, map_location='cpu')
    empty = bank['empty'].to(device).float()
    assert empty.shape == (768,) and bool(torch.isfinite(empty).all())
    model = InstanceCandidatePrototype().to(device)
    frozen = freeze_unused(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    fit = panel['fit']
    initial_indices = torch.arange(min(64, len(fit['key'])))
    with torch.no_grad():
        initial = model(batch(fit, initial_indices, empty, device))
    assert torch.equal(initial['selection_logits'], fit['base_scores'][initial_indices].to(device))
    assert bool((initial['selected_index'] == 0).all())
    started = time.time()
    history = []
    if args.mode != 'train':
        indices_list = [initial_indices, initial_indices]
        epochs = 1
    else:
        epochs = 12
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        if args.mode == 'train':
            indices_list = torch.randperm(len(fit['key']), generator=torch.Generator().manual_seed(2027+epoch)).split(64)
        losses = []
        for indices in indices_list:
            inputs = batch(fit, indices, empty, device)
            target = fit['iou'][indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(inputs)
            assert torch.equal(output['selection_logits'], output['visual_selection_logits'])
            loss = F.binary_cross_entropy_with_logits(output['visual_selection_logits'], target) + F.binary_cross_entropy_with_logits(output['quality_logits'], target)
            assert bool(torch.isfinite(loss))
            loss.backward()
            if args.mode != 'train' and len(losses) == 1:
                gradient = {name:sum(float(p.grad.norm()) for p in module.parameters() if p.grad is not None)
                            for name,module in [('visual_read', model.evidence.visual_read), ('candidate_relation', model.candidate_read)]}
                assert all(value > 0 for value in gradient.values())
                assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
            optimizer.step()
            losses.append(float(loss.detach()))
        row = dict(epoch=epoch+1, optimizer_steps=len(losses), mean_loss=sum(losses)/len(losses), seconds=time.time()-epoch_start)
        history.append(row)
        print(json.dumps(row), flush=True)
    check_frozen(model, frozen)
    result = dict(status='complete_visual_control' if args.mode == 'train' else 'complete_visual_control_sanity_only',
                  mode=args.mode, seed=2027, batch_size=64, learning_rate=3e-4, epochs=epochs,
                  optimizer_steps=sum(r['optimizer_steps'] for r in history), elapsed_seconds=time.time()-started,
                  total_parameters=sum(p.numel() for p in model.parameters()),
                  optimized_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  frozen_semantic_observation_parameters_unchanged=True, history=history,
                  category_attribute_identity_labels_used=False, fixed_five_slot_empty_input=True,
                  no_public_evaluation=True, no_recursive_action=True,
                  active_capacity_matched_to_semantic_variant=False, empty_bank_sha256=sha(args.empty_bank))
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'train':
        result['fit'], _ = evaluate(model, fit, empty, device)
        result['development'], result['development_rows'] = evaluate(model, panel['development'], empty, device)
        dev = result['development']
        result['fixed_state_capacity_checks'] = dict(
            correct_exceeds_native=dev['all']['selected_iou50'] > dev['all']['native_iou50'],
            mean_iou_exceeds_native=dev['all']['selected_mean_iou'] > dev['all']['native_mean_iou'],
            healthy_breaks_zero=dev['healthy']['breaks'] == 0,
            transition_correct_at_least_native=dev['transition']['selected_iou50'] >= dev['transition']['native_iou50'])
        weights = args.output/'final.pt'
        assert not weights.exists()
        torch.save(model.state_dict(), weights)
        result['final_weights_sha256'] = sha(weights)
    else:
        result['gradient_norms'] = gradient
        result['sanity_examples'] = len(initial_indices)
        result['development_evaluated'] = False
        result['checkpoint_saved'] = False
    if device.type == 'cuda':
        result['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
        result['gpu_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
    path = args.output/'result.json'
    assert not path.exists()
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('history', 'development_rows')}), flush=True)


if __name__ == '__main__':
    main()
