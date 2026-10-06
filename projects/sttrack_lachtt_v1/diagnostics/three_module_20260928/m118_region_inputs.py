"""Load measured Train-only M117 observations; cache the frozen Parent once."""
import json
import torch
from analyze_train_states import sha
from instance_ab_prototype import InstanceCandidatePrototype
from train_ab_visual_control import load_inputs
from train_m110_no_weak_rank import inputs as parent_inputs


def load_panel(args, bank, device):
    panel = load_inputs(args.cache, args.contexts, args.origins, False)
    prior = json.loads(args.parent_result.read_text())
    assert sha(args.parent) == prior['final_weights_sha256'] == '1d187d78c1dd76ebf59e41fd31af16fa69cecc37b4cb6c74fbdfe3ed5c237293'
    parent = InstanceCandidatePrototype().to(device)
    parent.load_state_dict(torch.load(args.parent, map_location='cpu'), strict=True)
    parent.requires_grad_(False).eval()
    frozen = {k: v.cpu().clone() for k, v in parent.state_dict().items()}
    for split, part in panel.items():
        fields = {name: [] for name in ['parent_scores', 'parent_quality', 'parent_features']}
        with torch.no_grad():
            for start in range(0, len(part['key']), 64):
                ids = torch.arange(start, min(start + 64, len(part['key'])))
                out = parent(parent_inputs(part, ids, bank, 'empty', device))
                assert torch.equal(out['selection_logits'], out['visual_selection_logits'])
                for name, key in [('parent_scores', 'selection_logits'), ('parent_quality', 'quality_logits'),
                                  ('parent_features', 'candidate_features')]:
                    fields[name].append(out[key].cpu())
        part.update({k: torch.cat(v) for k, v in fields.items()})
    assert panel['development']['parent_scores'].argmax(-1).tolist() == [r['selected'] for r in prior['development_rows']]
    assert all(torch.equal(v.cpu(), frozen[k]) for k, v in parent.state_dict().items())
    del parent, frozen
    # The old high-volume RGB-D tensors served the actual frozen Parent forward.
    for part in panel.values():
        for key in list(part):
            if key not in ['key', 'strata', 'iou', 'geometry', 'boxes', 'parent_scores', 'parent_quality', 'parent_features']:
                del part[key]
    assert (args.dense / 'driver.exit').read_text().strip() == '0'
    receipt = json.loads((args.dense / 'result.json').read_text())
    assert receipt['status'] == 'complete_M117_region_pair' and receipt['optimizer_steps'] == 0
    assert sum(r['events'] for r in receipt['full']) == 3502
    preparation = json.loads((args.cache / 'preparation.json').read_text())
    assert preparation['inference_inputs_sha256'] == sha(args.cache / 'inference_inputs.json')
    plans = {r['sequence']: r for r in json.loads((args.cache / 'inference_inputs.json').read_text())}
    by_key = {k: (split, i) for split, part in panel.items() for i, k in enumerate(part['key'])}
    values = {split: [None] * len(part['key']) for split, part in panel.items()}
    initial = []
    for shard in [0, 1]:
        for row in receipt['full'][shard]['sequences']:
            name = row['sequence'];path = args.dense / ('full_shard' + str(shard)) / 'features' / (name + '.pt')
            assert sha(path) == row['feature_sha256']
            data = torch.load(path, map_location='cpu')
            assert data['human_confirmed'] and not data['GT_loaded'] and data['model_updates'] == 0
            assert data['split'] == plans[name]['split'] and data['event_frames'] == plans[name]['event_frames']
            init_index = len(initial)
            initial.append(dict(grid=data['initial_search_tokens'], valid=data['initial_observed_fraction'],
                                origin=torch.tensor(data['crop_origins'][0], dtype=torch.float32),
                                box=torch.tensor(plans[name]['init_bbox'], dtype=torch.float32)))
            for j, frame in enumerate(data['event_frames']):
                key = name + '@' + str(frame)
                if key not in by_key:
                    continue  # Existing 463 GT-invalid Train states are not optimization labels.
                split, i = by_key[key]
                assert split == data['split']
                values[split][i] = dict(dense_grid=data['search_tokens'][j], dense_valid=data['observed_fraction'][j],
                    dense_origin=torch.tensor(data['crop_origins'][j + 1], dtype=torch.float32), initial_index=init_index)
    assert len(initial) == 152
    initial = {k: torch.stack([r[k] for r in initial]) for k in ['grid', 'valid', 'origin', 'box']}
    for split, rows in values.items():
        assert all(r is not None for r in rows)
        for key in ['dense_grid', 'dense_valid', 'dense_origin']:
            panel[split][key] = torch.stack([r[key] for r in rows])
        panel[split]['initial_index'] = torch.tensor([r['initial_index'] for r in rows])
    return panel, initial


def inputs(panel, indices, initial, bank, condition, device):
    data = {k: v[indices].to(device).float() for k, v in panel.items() if torch.is_tensor(v) and k != 'iou'}
    initial_ids = panel['initial_index'][indices]
    for key in ['grid', 'valid', 'origin', 'box']:
        data['initial_' + key] = initial[key][initial_ids].to(device).float()
    bank_ids = torch.tensor([bank['sequences'].index(panel['key'][i].rsplit('@', 1)[0]) for i in indices.tolist()])
    mask = bank['mask'][bank_ids].to(device)
    assert bool(mask[:, 0].all())
    query = bank['tokens'][bank_ids].to(device).float()
    if condition in ['empty', 'generic']:
        query = bank['empty' if condition == 'empty' else 'generic'].to(device).float()[None, None].expand_as(query)
    data.update(query=query, query_mask=mask, empty_query=bank['empty'].to(device).float(), condition=condition)
    return data
