"""Audit sealed policy-state data and train the two fixed M52 comparison arms."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch


FIELDS = ['current', 'previous', 'references', 'geometry', 'scores']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def check_sources(root):
    plan = json.loads((root/'spec.json').read_text())
    parent = Path(plan['source_root'])
    spec = json.loads((parent/'spec.json').read_text())
    repo = Path(spec['repository'])
    sys.path.insert(0, str(repo))
    from tools.train_sttrack_m44 import check_binding
    check_binding(parent, spec)
    binding = json.loads((root/'training_binding.json').read_text())
    assert binding['spec_sha256'] == sha(root/'spec.json')
    assert sha(parent/'spec.json') == plan['source_spec_sha256']
    for name, digest in {**plan['source_sha256'], **binding['source_sha256']}.items():
        assert sha(repo/name) == digest, name
    for name, digest in binding['run_files_sha256'].items():
        assert sha(root/name) == digest, name
    assert sha(Path(plan['policy_root'])/'geometry_result.json') == plan['policy_training_result_sha256']
    assert sha(plan['policy_checkpoint']) == plan['policy_checkpoint_sha256']
    assert sha(Path(plan['m51_root'])/'recursive_result.json') == plan['m51_result_sha256']
    return plan, parent, spec


def load_data(root, plan, parent, spec):
    """Verify every sealed input before reading the original supervision labels."""
    assert (root/'collection.exit').read_text().strip() == '0'
    receipt = json.loads((root/'collection_receipt.json').read_text())
    assert receipt['status'] == 'complete' and receipt['spec_sha256'] == sha(root/'spec.json')
    assert receipt['events'] == 1511 and receipt['frames'] == 93362 and not receipt['labels_opened']
    assert receipt['policy_checkpoint_sha256'] == plan['policy_checkpoint_sha256']
    assert sha(parent/'inference_inputs.json') == spec['inference_inputs_sha256']
    cases = {c['sequence']: c for c in json.loads((parent/'inference_inputs.json').read_text())}
    old_receipts = []
    for shard in [0, 1]:
        old = json.loads((parent/f'shard{shard}_receipt.json').read_text())
        assert old['status'] == 'complete' and old['spec_sha256'] == sha(parent/'spec.json')
        old_receipts.extend(old['sequences'])
    assert len(old_receipts) == 85 and {r['sequence'] for r in old_receipts} == set(cases)
    assert len(receipt['sequences']) == 63 and sorted(r['sequence'] for r in receipt['sequences']) == plan['sequences']
    packets = []
    state_rows = []
    for kind, directory, receipts in [('native', parent, old_receipts), ('policy', root, receipt['sequences'])]:
        for item in sorted(receipts, key=lambda r: r['sequence']):
            name = item['sequence']
            case = cases[name]
            path = directory/'features'/(name+'.pt')
            assert sha(path) == item['feature_sha256'], name
            data = torch.load(path, map_location='cpu')
            assert data['spec_sha256'] == sha(directory/'spec.json')
            assert data['sequence'] == name and data['fold'] == case['fold'] and data['split'] == case['split']
            records = data['records']
            assert [r['frame'] for r in records] == sorted(case['event_frames'])
            assert [r['key'] for r in records] == [f'{name}@{r["frame"]}' for r in records]
            assert all(r['previous_frame'] == r['frame']-1 for r in records)
            assert all(torch.isfinite(data[key]).all() for key in FIELDS)
            if kind == 'native':
                assert all(r['previous_choice'] == 0 for r in records)
            else:
                assert case['split'] == 'fit' and data['policy_checkpoint_sha256'] == plan['policy_checkpoint_sha256']
                trace_path = root/'traces'/(name+'.json')
                assert sha(trace_path) == item['trace_sha256'], name
                trace = json.loads(trace_path.read_text())['rows']
                assert [r['frame'] for r in trace] == list(range(max(case['event_frames'])+1))
                assert trace[0]['bbox'] == case['init_bbox']
                boxes = torch.tensor([r['bbox'] for r in trace])
                assert torch.isfinite(boxes).all() and (boxes[:, 2:] > 0).all()
                for i, row in enumerate(records):
                    frame = row['frame']
                    assert row['previous_choice'] == trace[frame-1]['choice']
                    assert row['current_choice'] == trace[frame]['choice']
                    assert 0 <= row['previous_choice'] < 10 and 0 <= row['current_choice'] < 10
                    assert torch.equal(data['selected_bbox'][i], boxes[frame])
                    assert torch.equal(data['previous_selected_bbox'][i], boxes[frame-1])
                nonzero = sum(r['previous_choice'] != 0 for r in records)
                assert nonzero == item['nonzero_previous_choice_events']
                assert sum(r['choice'] != 0 for r in trace) == item['changes']
                state_rows.append(dict(sequence=name, events=len(records), nonzero_previous_choice_events=nonzero,
                                       trace_changes=item['changes'], feature_sha256=item['feature_sha256']))
            packets.append((kind, data))

    # Labels cannot influence collection: all 148 feature files and 63 new
    # trajectories have been checked against completed receipts above.
    assert sha(parent/'training_labels.json') == spec['labels_sha256']
    labels = json.loads((parent/'training_labels.json').read_text())
    from tools.train_sttrack_m42 import overlaps
    arrays = {key: [] for key in FIELDS}
    keys, kinds, folds, choices, current_ious, previous_ious = [], [], [], [], [], []
    new_selected = []
    old_default = []
    for kind, data in packets:
        for key in FIELDS:
            arrays[key].append(data[key])
        for i, record in enumerate(data['records']):
            label = labels[record['key']]
            assert label['fold'] == data['fold'] and label['sequence'] == data['sequence']
            pair = []
            for side, boxes_key, default_key in [('current', 'current_boxes', 'public_bbox'),
                                                  ('previous', 'previous_boxes', 'previous_public_bbox')]:
                if label[side] is None:
                    values = torch.zeros(10)
                else:
                    gt = torch.tensor(label[side])
                    values = overlaps(data[boxes_key][i], gt)
                    values[0] = overlaps(data[default_key][i], gt)
                pair.append(values)
            keys.append(record['key']); kinds.append(kind); folds.append(data['fold'])
            choices.append(record['previous_choice']); current_ious.append(pair[0]); previous_ious.append(pair[1])
            if kind == 'policy':
                new_selected.append(data['selected_bbox'][i])
            elif data['split'] == 'fit':
                old_default.append(data['public_bbox'][i])
    tensors = {key: torch.cat(value) for key, value in arrays.items()}
    del arrays, packets, data
    current_ious, previous_ious = torch.stack(current_ious), torch.stack(previous_ious)
    def targets(ious):
        result = ious.argmax(1)
        result[ious.max(1).values < .5] = 10
        result[ious[:, 0] >= .5] = 0
        return result
    current_target, previous_target = targets(current_ious), targets(previous_ious)
    original_fit = torch.tensor([i for i, (k, f) in enumerate(zip(kinds, folds)) if k == 'native' and f in spec['fit_folds']])
    development = torch.tensor([i for i, (k, f) in enumerate(zip(kinds, folds)) if k == 'native' and f == spec['development_fold']])
    policy_fit = torch.tensor([i for i, k in enumerate(kinds) if k == 'policy'])
    assert len(original_fit) == len(policy_fit) == 1511 and len(development) == 590 and len(keys) == 3612
    assert [keys[i] for i in original_fit] == [keys[i] for i in policy_fit]
    assert {keys[i].split('@')[0] for i in original_fit}.isdisjoint({keys[i].split('@')[0] for i in development})
    previous_choice = torch.tensor(choices, dtype=torch.long)
    old_result = json.loads((Path(plan['policy_root'])/'geometry_result.json').read_text())
    old_labels = {row['key']: (row['target'], row['previous_target'])
                  for split in ['fit', 'development'] for row in old_result[split]['rows']}
    assert len(old_labels) == 2101
    for i in torch.cat([original_fit, development]):
        assert (int(current_target[i]), int(previous_target[i])) == old_labels[keys[i]], keys[i]
    audit = dict(status='PASS', spec_sha256=sha(root/'spec.json'), collection_receipt_sha256=sha(root/'collection_receipt.json'),
                 training_binding_sha256=sha(root/'training_binding.json'), labels_sha256=spec['labels_sha256'],
                 native_feature_files=85, policy_feature_files=63, new_trace_files=63,
                 labels_read_after_sealed_inputs_verified=True, original_targets_match_all_2101_m45_records=True,
                 physical_fit_events=1511, paired_fit_views=3022, fit_sequences=63, development_sequences=22,
                 new_nonzero_previous_choice_events=int((previous_choice[policy_fit] != 0).sum()),
                 changed_event_predictions=int((torch.stack(new_selected) != torch.stack(old_default)).any(1).sum()),
                 current_labels_changed=int((current_target[original_fit] != current_target[policy_fit]).sum()),
                 previous_labels_changed=int((previous_target[original_fit] != previous_target[policy_fit]).sum()),
                 native_current_correct_candidates=int((current_ious[original_fit].max(1).values >= .5).sum()),
                 policy_current_correct_candidates=int((current_ious[policy_fit].max(1).values >= .5).sum()),
                 paired_current_gt_unavailable=sum(labels[keys[i]]['current'] is None for i in original_fit),
                 paired_previous_gt_unavailable=sum(labels[keys[i]]['previous'] is None for i in original_fit),
                 state_rows=state_rows, scope='Fitting-only state comparison; candidate counts are capacities, not recursive rescues.')
    return tensors, keys, previous_choice, current_ious, current_target, previous_target, original_fit, policy_fit, development, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--audit', action='store_true')
    mode.add_argument('--arm', choices=['control', 'mixed'])
    args = parser.parse_args()
    root = args.root
    torch.set_num_threads(1)
    plan, parent, spec = check_sources(root)
    tensors, keys, previous_choice, ious, target, previous_target, original_fit, policy_fit, dev, audit = load_data(root, plan, parent, spec)
    check_sources(root)
    if args.audit:
        assert args.arm is None
        (root/'data_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
        print(json.dumps({k: v for k, v in audit.items() if k != 'state_rows'}, indent=2))
        return
    assert args.arm in ['control', 'mixed']
    assert audit == json.loads((root/'data_audit.json').read_text())
    folder = root/args.arm
    folder.mkdir()
    from lib.models.sttrack.lachtt_candidate_set import CandidateSetAssociation, select_candidate, supervised_loss
    opt = plan['pair_training']
    seed = spec['optimization']['seed']
    assert seed == 2026
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)
    model = CandidateSetAssociation(True).cuda()
    initial = {name: tensor_sha(value) for name, value in model.state_dict().items()}
    paired_reference = json.loads((Path(plan['policy_root'])/'geometry_result.json').read_text())
    assert initial == paired_reference['initial_state_sha256']
    assert sum(p.numel() for p in model.parameters()) == 448739
    train = torch.cat([original_fit, original_fit if args.arm == 'control' else policy_fit])
    assert len(train) == 3022 and all(keys[i].split('@')[0] in plan['sequences'] for i in train)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt['lr'], weight_decay=opt['weight_decay'])
    order = torch.Generator().manual_seed(seed)
    order_digest = hashlib.sha256()
    losses, steps = [], 0
    started = time.time()
    def inputs(index):
        return [tensors[key][index].cuda().float() for key in FIELDS]+[previous_choice[index].cuda()]
    for epoch in range(opt['epochs']):
        model.train()
        positions = torch.randperm(len(train), generator=order)
        order_digest.update(positions.numpy().tobytes())
        totals = [0., 0., 0.]
        for index in train[positions].split(opt['batch_size']):
            logits, affinity = model(*inputs(index))
            values = supervised_loss(logits, affinity, target[index].cuda(), previous_target[index].cuda())
            assert torch.isfinite(values[0])
            optimizer.zero_grad(set_to_none=True); values[0].backward()
            assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), opt['grad_clip']))
            optimizer.step(); steps += 1
            totals = [a+float(v.detach())*len(index) for a, v in zip(totals, values)]
        row = dict(epoch=epoch+1, loss=totals[0]/len(train), identity_loss=totals[1]/len(train), matching_loss=totals[2]/len(train))
        losses.append(row)
        print(json.dumps(dict(arm=args.arm, optimizer_steps=steps, elapsed_seconds=time.time()-started, **row)), flush=True)
    assert steps == opt['optimizer_steps'] == 1900
    checkpoint = folder/'geometry_final.pth'
    torch.save(dict(model=model.state_dict(), variant='geometry', arm=args.arm, optimizer_steps=steps, epochs=opt['epochs'],
                    base_checkpoint_sha256=spec['checkpoint_sha256'], m52_spec_sha256=sha(root/'spec.json'),
                    training_binding_sha256=sha(root/'training_binding.json'), data_audit_sha256=sha(root/'data_audit.json'),
                    target_rule='default_if_iou_at_least_half'), checkpoint)

    def evaluate(net, index):
        net.eval(); pieces = []
        with torch.no_grad():
            for batch in index.split(opt['batch_size']):
                logits, _ = net(*inputs(batch)); pieces.append(logits.cpu())
        logits = torch.cat(pieces)
        selected = select_candidate(logits)
        values = ious[index].gather(1, selected[:, None]).flatten()
        default = ious[index, 0]
        result = dict(events=len(index), mean_iou=float(values.mean()), default_mean_iou=float(default.mean()),
                      correct=int((values >= .5).sum()), default_correct=int((default >= .5).sum()),
                      changes=int((selected != 0).sum()), none=int((logits.argmax(1) == 10).sum()),
                      rescues=int(((default <= .1) & (values >= .5)).sum()), breaks=int(((default >= .5) & (values <= .1)).sum()),
                      rows=[dict(key=keys[i], chosen=int(k), action_none=bool(n == 10), default_iou=float(a), selected_iou=float(b),
                                 target=int(target[i]), previous_target=int(previous_target[i]), previous_choice=int(previous_choice[i]))
                            for i, k, n, a, b in zip(index, selected, logits.argmax(1), default, values)])
        return result, logits
    native_stats, _ = evaluate(model, original_fit)
    policy_stats, _ = evaluate(model, policy_fit)
    dev_stats, dev_logits = evaluate(model, dev)
    loaded = CandidateSetAssociation(True).cuda()
    loaded.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'], strict=True)
    _, restored = evaluate(loaded, dev)
    assert torch.equal(restored, dev_logits)
    changed = [name for name, value in model.state_dict().items() if tensor_sha(value) != initial[name]]
    assert changed
    check_sources(root)
    result = dict(status='complete', arm=args.arm, parameters=448739, optimizer_steps=steps, epochs=opt['epochs'],
                  initial_state_sha256=initial, logical_sample_order_sha256=order_digest.hexdigest(), losses=losses,
                  seed=seed, changed_tensors=changed,
                  checkpoint_sha256=sha(checkpoint), checkpoint_bytes=checkpoint.stat().st_size, reload_logits_exact=True,
                  spec_sha256=sha(root/'spec.json'), training_binding_sha256=sha(root/'training_binding.json'),
                  data_audit_sha256=sha(root/'data_audit.json'), trainer_sha256=sha(Path(__file__)),
                  native_fit=native_stats, policy_fit=policy_stats, development=dev_stats, elapsed_seconds=time.time()-started,
                  scope='Same physical fitting events, two views. Static diagnostics do not replace full paired recursion.')
    (folder/'training_result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status='complete', arm=args.arm, development={k: v for k, v in dev_stats.items() if k != 'rows'})), flush=True)


if __name__ == '__main__': main()
