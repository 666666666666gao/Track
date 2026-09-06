"""Frozen M57 native-state diagnostics with fixed-mask content interventions."""
import json
from pathlib import Path
import sys

import torch

from run_recursive import bound
from train import sha, write


root = Path('/root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906')
spec, recursive, training, cases = bound(root)
frozen = json.loads((root / 'static_spec.json').read_text())
assert sha(__file__) == frozen['analyzer_sha256']
assert sha(root / 'recursive_spec.json') == frozen['recursive_spec_sha256']
assert not (root / 'static_result.json').exists()
for path, digest in frozen['input_sha256'].items():
    assert sha(path) == digest, path
sys.path.insert(0, spec['repository'])
from lib.models.sttrack.lachtt_initial_instance_alignment import InitialInstanceCandidateSetAssociation, content_tokens
from lib.models.sttrack.lachtt_candidate_set import select_candidate
from tools.train_sttrack_m42 import overlaps

torch.set_num_threads(4)
parent = Path(spec['source_root'])
labels = json.loads((parent / 'training_labels.json').read_text())
bank = torch.load(root / 'text_development.pt', map_location='cpu')
initial_bank = torch.load(root / 'initial_references/initial_development.pt', map_location='cpu')
initial = dict(zip(initial_bank['sequences'], initial_bank['references']))
names = {case['sequence'] for case in cases}
assert bank['split'] == initial_bank['split'] == 'development'
assert set(bank['sequences']) == set(initial) == names
collected = {key: [] for key in ['current', 'previous', 'references', 'geometry', 'scores']}
keys, sequences, ious, valid_gt = [], [], [], []
for receipt in frozen['development_features']:
    name = receipt['sequence']
    path = parent / 'features' / (name + '.pt')
    assert name in names and sha(path) == receipt['sha256']
    data = torch.load(path, map_location='cpu')
    assert data['spec_sha256'] == spec['parent_spec_sha256'] and data['fold'] == 5
    assert data['references'].dtype == initial[name].dtype == torch.float16
    data['references'][:, 0] = initial[name]
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
tensors = {key: torch.cat(value) for key, value in collected.items()}
del collected, data, labels
ious = torch.stack(ious)
reports = {}
for mode in frozen['modes']:
    variant = mode['variant']
    checkpoint = root / 'training' / (variant + '_final.pth')
    saved = torch.load(checkpoint, map_location='cpu')
    assert sha(checkpoint) == training['variants'][variant]['checkpoint_sha256']
    assert saved['m57_spec_sha256'] == sha(root / 'spec.json') and saved['variant'] == variant
    assert saved['epochs'] == 20 and saved['optimizer_steps'] == 960
    assert saved['reference_mode'] == variant.split('_')[0]
    model = InitialInstanceCandidateSetAssociation(saved['reference_mode']).eval()
    model.load_state_dict(saved['model'], strict=True)
    use_text = saved['use_text'] and mode['content'] != 'empty'
    tokens = content_tokens(bank['tokens'], bank['empty'], use_text)
    selected_indices, own_indices, donor_indices = [], [], []
    for i, name in enumerate(sequences):
        own_index = bank['sequences'].index(name)
        if mode['content'] == 'same_mask_shuffled':
            if name not in frozen['same_mask_donors']:
                continue
            donor = frozen['same_mask_donors'][name]
        else:
            donor = name
        donor_index = bank['sequences'].index(donor)
        assert torch.equal(bank['mask'][own_index], bank['mask'][donor_index])
        selected_indices.append(i)
        own_indices.append(own_index)
        donor_indices.append(donor_index)
    indices = torch.tensor(selected_indices)
    own = torch.tensor(own_indices)
    donors = torch.tensor(donor_indices)
    logits_all = []
    with torch.no_grad():
        for batch in torch.arange(len(indices)).split(32):
            index = indices[batch]
            inputs = [tensors[key][index].float() for key in ['current', 'previous', 'references', 'geometry', 'scores']]
            inputs += [torch.zeros(len(index), dtype=torch.long), tokens[donors[batch]], bank['mask'][own[batch]]]
            logits, _ = model(*inputs)
            logits_all.append(logits)
    logits = torch.cat(logits_all)
    selected = select_candidate(logits)
    rows = [dict(key=keys[i], sequence=sequences[i], candidate=int(k), none=int(value.argmax()) == 10,
        default_iou=float(ious[i, 0]), selected_iou=float(ious[i, k]), oracle_iou=float(ious[i].max()),
        valid_gt=valid_gt[i]) for i, k, value in zip(indices.tolist(), selected.tolist(), logits)]
    reports[mode['name']] = dict(rows=rows, checkpoint_sha256=sha(checkpoint), mode=mode)
    del model


def summary(rows):
    values = torch.tensor([row['selected_iou'] for row in rows])
    default = torch.tensor([row['default_iou'] for row in rows])
    return dict(events=len(rows), sequences=len({row['sequence'] for row in rows}),
        valid_gt_events=sum(row['valid_gt'] for row in rows), mean_iou=float(values.mean()),
        default_mean_iou=float(default.mean()), correct=int((values >= .5).sum()), default_correct=int((default >= .5).sum()),
        rescues=int(((default <= .1) & (values >= .5)).sum()), breaks=int(((default >= .5) & (values <= .1)).sum()),
        nondefault=sum(row['candidate'] != 0 for row in rows), none=sum(row['none'] for row in rows))


for report in reports.values():
    report['aggregate'] = summary(report['rows'])
    report['per_sequence'] = {name: summary([row for row in report['rows'] if row['sequence'] == name])
        for name in sorted({row['sequence'] for row in report['rows']})}
comparisons = {}
for variant in ['candidate_text', 'initial_text']:
    real = {row['key']: row for row in reports[variant]['rows']}
    for content in ['empty', 'same_mask_shuffled']:
        name = variant + '_' + content
        cf = reports[name]['rows']
        paired = [real[row['key']] for row in cf]
        comparisons[name] = dict(real_on_same_coverage=summary(paired), counterfactual=summary(cf),
            choice_changes=sum(a['candidate'] != b['candidate'] for a, b in zip(paired, cf)),
            real_correct_counterfactual_severe=sum(a['selected_iou'] >= .5 and b['selected_iou'] <= .1 for a, b in zip(paired, cf)),
            real_severe_counterfactual_correct=sum(a['selected_iou'] <= .1 and b['selected_iou'] >= .5 for a, b in zip(paired, cf)))
result = dict(status='complete_static_diagnostic', spec_sha256=sha(root / 'spec.json'),
    static_spec_sha256=sha(root / 'static_spec.json'), training_result_sha256=sha(root / 'training_result.json'),
    all_final_weights_frozen_before_development_labels=True, variants=reports, text_counterfactuals=comparisons,
    original_slot_masks_unchanged=True, shuffle_unpaired_sequences=frozen['shuffle_unpaired_sequences'],
    scope='590 reused native-state development snapshots; invalid GT assigned zero IoU as in the earlier cached diagnostic',
    promotion_decision=False, recursive_required=True,
    claim='Same-slot local content sensitivity only; not a full-trajectory, attribute-truth, VOT or three-dataset gain')
write(root / 'static_result.json', result)
print(json.dumps({name: report['aggregate'] for name, report in reports.items()}, indent=2), flush=True)
print(json.dumps(comparisons, indent=2), flush=True)
