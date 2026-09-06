"""Frozen-weight development diagnostics and text counterfactuals, after fitting."""
import argparse
import json
from pathlib import Path
import sys

import torch

from train import binding, sha, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    spec = binding(root)
    assert sha(__file__) == spec['static_analyzer_sha256']
    assert not (root / 'static_result.json').exists()
    training = json.loads((root / 'training_result.json').read_text())
    assert training['status'] == 'complete_train' and training['equal_initialization_order_budget_parameters']
    assert training['spec_sha256'] == sha(root / 'spec.json')
    for variant, row in training['variants'].items():
        assert sha(root / 'training' / (variant + '_final.pth')) == row['checkpoint_sha256']
        assert sha(root / 'training' / (variant + '_result.json')) == row['result_sha256']
    # All three final weights are sealed before the development targets are parsed.
    sys.path.insert(0, spec['repository'])
    from lib.models.sttrack.lachtt_attribute_candidate_set import AttributeCandidateSetAssociation, condition_text
    from lib.models.sttrack.lachtt_candidate_set import select_candidate
    from tools.train_sttrack_m42 import overlaps
    torch.set_num_threads(4)
    parent = Path(spec['source_root'])
    parent_spec = json.loads((parent / 'spec.json').read_text())
    assert sha(parent / 'training_labels.json') == parent_spec['labels_sha256']
    labels = json.loads((parent / 'training_labels.json').read_text())
    manifest = json.loads((root / 'text_manifest.json').read_text())
    controls = json.loads((root / 'text_controls.json').read_text())
    names = {row['sequence'] for row in manifest if row['split'] == 'development'}
    assert len(names) == 22
    bank = torch.load(root / 'text_bank.pt', map_location='cpu')
    receipts = sum([json.loads((parent / ('shard%d_receipt.json' % s)).read_text())['sequences'] for s in [0, 1]], [])
    collected = {key: [] for key in ['current', 'previous', 'references', 'geometry', 'scores']}
    keys, sequences, ious, valid_gt = [], [], [], []
    for receipt in sorted(receipts, key=lambda x: x['sequence']):
        name = receipt['sequence']
        if name not in names:
            continue
        path = parent / 'features' / (name + '.pt')
        assert sha(path) == receipt['feature_sha256']
        data = torch.load(path, map_location='cpu')
        assert data['spec_sha256'] == spec['parent_spec_sha256'] and data['fold'] == 5
        for key in collected:
            collected[key].append(data[key])
        for i, row in enumerate(data['records']):
            label = labels[row['key']]
            assert label['sequence'] == name and label['fold'] == 5 and row['previous_choice'] == 0
            if label['current'] is None:
                value = torch.zeros(10)
            else:
                gt = torch.tensor(label['current'])
                value = overlaps(data['current_boxes'][i], gt)
                value[0] = overlaps(data['public_bbox'][i], gt)
            ious.append(value)
            valid_gt.append(label['current'] is not None)
            keys.append(row['key'])
            sequences.append(name)
    assert len(keys) == len(set(keys)) == 590
    tensors = {key: torch.cat(values) for key, values in collected.items()}
    del collected, data, labels
    ious = torch.stack(ious)
    reports = {}
    modes = ['attributes', 'pooled', 'empty', 'attributes_empty', 'attributes_shuffled', 'attributes_conflict']
    for mode in modes:
        checkpoint_variant = mode if mode in spec['variants'] else 'attributes'
        checkpoint = root / 'training' / (checkpoint_variant + '_final.pth')
        model = AttributeCandidateSetAssociation().eval()
        saved = torch.load(checkpoint, map_location='cpu')
        assert saved['m56_spec_sha256'] == sha(root / 'spec.json')
        model.load_state_dict(saved['model'], strict=True)
        text_variant = 'empty' if mode == 'attributes_empty' else checkpoint_variant
        text, mask = condition_text(bank['tokens'], bank['mask'], bank['empty'], text_variant)
        selected_indices, donor_indices = [], []
        for i, name in enumerate(sequences):
            donor = name
            if mode == 'attributes_shuffled':
                donor = controls['rows'][name]['shuffled_donor']
            elif mode == 'attributes_conflict':
                donor = controls['rows'][name]['same_category_conflict_donor']
                if donor is None:
                    continue
            selected_indices.append(i)
            donor_indices.append(bank['sequences'].index(donor))
        indices = torch.tensor(selected_indices)
        donors = torch.tensor(donor_indices)
        logits_all = []
        with torch.no_grad():
            for batch in torch.arange(len(indices)).split(32):
                index = indices[batch]
                inputs = [tensors[key][index].float() for key in ['current', 'previous', 'references', 'geometry', 'scores']]
                inputs += [torch.zeros(len(index), dtype=torch.long), text[donors[batch]], mask[donors[batch]]]
                logits, _ = model(*inputs)
                logits_all.append(logits)
        logits = torch.cat(logits_all)
        selected = select_candidate(logits)
        rows = [dict(key=keys[i], sequence=sequences[i], candidate=int(k), none=int(l.argmax()) == 10,
            default_iou=float(ious[i, 0]), selected_iou=float(ious[i, k]), oracle_iou=float(ious[i].max()),
            valid_gt=valid_gt[i]) for i, k, l in zip(indices.tolist(), selected.tolist(), logits)]
        reports[mode] = dict(rows=rows, checkpoint_sha256=sha(checkpoint))
    def summary(rows):
        count = len(rows)
        values = torch.tensor([x['selected_iou'] for x in rows])
        default = torch.tensor([x['default_iou'] for x in rows])
        return dict(events=count, sequences=len({x['sequence'] for x in rows}),
            valid_gt_events=sum(x['valid_gt'] for x in rows), mean_iou=float(values.mean()), default_mean_iou=float(default.mean()),
            correct=int((values >= .5).sum()), default_correct=int((default >= .5).sum()),
            rescues=int(((default <= .1) & (values >= .5)).sum()), breaks=int(((default >= .5) & (values <= .1)).sum()),
            nondefault=sum(x['candidate'] != 0 for x in rows), none=sum(x['none'] for x in rows))
    for mode, report in reports.items():
        report['aggregate'] = summary(report['rows'])
        report['per_sequence'] = {name: summary([row for row in report['rows'] if row['sequence'] == name])
            for name in sorted({row['sequence'] for row in report['rows']})}
    real = {row['key']: row for row in reports['attributes']['rows']}
    comparisons = {}
    for mode in ['attributes_empty', 'attributes_shuffled', 'attributes_conflict']:
        counterfactual = reports[mode]['rows']
        paired_real = [real[row['key']] for row in counterfactual]
        comparisons[mode] = dict(real_on_same_coverage=summary(paired_real), counterfactual=summary(counterfactual),
            choice_changes=sum(a['candidate'] != b['candidate'] for a, b in zip(paired_real, counterfactual)),
            real_correct_counterfactual_severe=sum(a['selected_iou'] >= .5 and b['selected_iou'] <= .1 for a, b in zip(paired_real, counterfactual)),
            real_severe_counterfactual_correct=sum(a['selected_iou'] <= .1 and b['selected_iou'] >= .5 for a, b in zip(paired_real, counterfactual)))
        if mode in ['attributes_shuffled', 'attributes_conflict']:
            field = 'shuffled_phrase_count_changed' if mode == 'attributes_shuffled' else 'conflict_phrase_count_changed'
            subsets = {'same_phrase_count': [row for row in counterfactual if not controls['rows'][row['sequence']][field]]}
            if mode == 'attributes_shuffled':
                subsets['same_category'] = [row for row in counterfactual if not controls['rows'][row['sequence']]['shuffled_category_changed']]
                subsets['same_category_and_phrase_count'] = [row for row in subsets['same_phrase_count']
                    if not controls['rows'][row['sequence']]['shuffled_category_changed']]
            comparisons[mode]['predefined_subsets'] = {name: dict(events=len(rows),
                real=summary([real[row['key']] for row in rows]) if rows else None,
                counterfactual=summary(rows) if rows else None) for name, rows in subsets.items()}
    result = dict(status='complete_static_diagnostic', analyzer_sha256=sha(__file__), spec_sha256=sha(root / 'spec.json'),
        training_result_sha256=sha(root / 'training_result.json'), text_controls_sha256=sha(root / 'text_controls.json'),
        scope='Repeated development native-state snapshots; unavailable GT assigned zero IoU/NONE as in M45 cache evaluation',
        all_final_weights_frozen_before_development_labels=True, variants=reports, text_counterfactuals=comparisons,
        promotion_decision=False, recursive_required=True,
        claim='Static choice sensitivity and same-budget diagnostics only; no full-trajectory or VOT language gain')
    write(root / 'static_result.json', result)
    print(json.dumps({mode: report['aggregate'] for mode, report in reports.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
