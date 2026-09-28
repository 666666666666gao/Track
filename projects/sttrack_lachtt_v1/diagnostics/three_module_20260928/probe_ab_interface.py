"""Actual-input A+B invariants and two-step localization-only gradient smoke."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from analyze_train_states import overlaps, sha
from instance_ab_prototype import InstanceCandidatePrototype


CANDIDATE_FIELDS = ('candidate_rois', 'contexts', 'candidate_mask', 'context_mask',
                    'depth_valid', 'geometry', 'base_scores', 'boxes')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--text-bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(2027)
    torch.cuda.manual_seed_all(2027)
    shards, records = [], []
    for shard in (0, 1):
        receipt = json.loads((args.panel / f'shard{shard}.json').read_text())
        path = args.panel / f'shard{shard}.pt'
        assert receipt['status'] == 'complete_train_only' and not receipt['smoke']
        assert receipt['panel_sha256'] == sha(path)
        data = torch.load(path, map_location='cpu')
        assert not data['GT_loaded'] and not data['text_loaded']
        assert data['records'] == receipt['records']
        shards.append(data)
        records.extend(data['records'])
    assert len(records) == 8 and all(r['split'] == 'fit' for r in records)
    tensor_keys = [key for key, value in shards[0].items() if torch.is_tensor(value)]
    device = torch.device(args.device)
    data = {key: torch.cat([shard[key] for shard in shards]).to(device)
            for key in tensor_keys}
    data = {key: value if value.dtype == torch.bool else value.float() for key, value in data.items()}
    bank = torch.load(args.text_bank, map_location='cpu')
    indices = [bank['sequences'].index(r['sequence']) for r in records]
    data['text'] = bank['tokens'][indices].to(device).float()
    data['text_mask'] = bank['mask'][indices].to(device).bool()
    data['empty_text'] = bank['empty'].to(device).float()
    active_attrs = data['text'][:, 1:][data['text_mask'][:, 1:]]
    assert torch.equal(active_attrs, data['empty_text'][None].expand_as(active_attrs))
    assert bool(data['text_mask'][:, 0].all())
    model = InstanceCandidatePrototype().to(device)
    with torch.no_grad():
        initial = model(data)
    assert torch.equal(initial['selection_logits'], data['base_scores'])
    assert bool((initial['selected_index'] == 0).all())
    assert initial['phrase_delta'].shape == (8, 10, 5, 3)

    labels = json.loads((args.cache / 'training_labels.json').read_text())
    preparation = json.loads((args.cache / 'preparation.json').read_text())
    assert preparation['training_labels_sha256'] == sha(args.cache / 'training_labels.json')
    valid, targets = [], []
    for index, row in enumerate(records):
        label = labels[f"{row['sequence']}@{row['frame']}"]
        assert label['split'] == 'fit'
        if label['current'] is not None:
            valid.append(index)
            targets.append(overlaps(data['boxes'][index], torch.tensor(label['current'], device=device)))
    assert len(valid) > 0
    target = torch.stack(targets)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_history = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = (F.binary_cross_entropy_with_logits(output['selection_logits'][valid], target)
                + F.binary_cross_entropy_with_logits(output['quality_logits'][valid], target))
        assert bool(torch.isfinite(loss))
        loss.backward()
        if step == 1:
            gradient = {}
            for name, module in [('visual_read', model.evidence.visual_read),
                                 ('text_projection', model.evidence.text),
                                 ('visual_to_phrase', model.evidence.phrase_read),
                                 ('phrase_to_visual', model.evidence.text_read),
                                 ('phrase_evidence', model.evidence.evidence),
                                 ('candidate_relation', model.candidate_read)]:
                norm = sum(float(p.grad.norm()) for p in module.parameters() if p.grad is not None)
                assert norm > 0 and all(bool(torch.isfinite(p.grad).all()) for p in module.parameters() if p.grad is not None)
                gradient[name] = norm
        optimizer.step()
        loss_history.append(float(loss.detach()))
    model.eval()
    empty = dict(data)
    empty['text'] = data['empty_text'][None, None].expand_as(data['text']) * data['text_mask'][..., None]
    with torch.no_grad():
        output = model(data)
        blank = model(empty)
        assert torch.equal(blank['selection_logits'], blank['visual_selection_logits'])
        assert bool((blank['semantic_delta'] == 0).all()) and bool((blank['phrase_delta'] == 0).all())
        assert torch.equal(output['quality_logits'], blank['quality_logits'])
        permutation = torch.tensor([3, 9, 1, 5, 0, 8, 4, 2, 7, 6], device=device)
        reordered = dict(data)
        for key in CANDIDATE_FIELDS:
            reordered[key] = data[key][:, permutation]
        permuted = model(reordered)
        score_error = float((permuted['selection_logits'] - output['selection_logits'][:, permutation]).abs().max())
        quality_error = float((permuted['quality_logits'] - output['quality_logits'][:, permutation]).abs().max())
        assert torch.allclose(permuted['selection_logits'], output['selection_logits'][:, permutation], atol=1e-5, rtol=1e-5)
        assert torch.allclose(permuted['quality_logits'], output['quality_logits'][:, permutation], atol=1e-5, rtol=1e-5)
        row = torch.arange(8, device=device)
        selected = output['selected_index']
        assert torch.equal(output['selected_box'], data['boxes'][row, selected])
        assert torch.equal(output['selected_score'], output['selection_logits'][row, selected])
        assert torch.equal(output['selected_quality'], output['quality_logits'][row, selected])
        assert torch.equal(output['selected_feature'], output['candidate_features'][row, selected])
        assert all(bool(torch.isfinite(value).all()) for value in output.values())
    quality_only = model(data)
    text_gradient = torch.autograd.grad(quality_only['quality_logits'].sum(),
                                        list(model.evidence.text.parameters()), allow_unused=True)
    assert all(value is None for value in text_gradient)
    report = dict(status='complete_interface_sanity_only', cases=8, seed=2027,
                  trainable_parameters=sum(p.numel() for p in model.parameters()),
                  native_zero_residual_choice=True, empty_semantic_increment_exact_zero=True,
                  empty_selection_exact_visual=True, quality_independent_of_text=True,
                  selection_permutation_max_error=score_error, quality_permutation_max_error=quality_error,
                  selected_fields_share_one_index=True, finite_outputs=True,
                  gradient_norms_after_zero_head_warmup=gradient,
                  location_iou_optimizer_steps=2, valid_location_targets=len(valid),
                  diagnostic_loss_history=loss_history, semantic_labels_used=False,
                  physical_identity_labels_used=False, observation_head_trained=False,
                  text_input='Unverified old category; active attributes exactly empty; original five-slot mask retained',
                  text_bank_sha256=sha(args.text_bank), no_public_evaluation=True,
                  no_recursive_action=True, official_checkpoint_saved=False,
                  collection_receipt_sha256={str(shard): sha(args.panel / f'shard{shard}.json') for shard in (0, 1)})
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'interface_sanity.json'
    assert not path.exists()
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
