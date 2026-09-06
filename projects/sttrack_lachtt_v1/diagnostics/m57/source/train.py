"""Same-slot four-arm M57 fitting on the fixed native base; no development evaluation."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def binding(root):
    spec = json.loads((root / 'spec.json').read_text())
    assert sha(__file__) == spec['trainer_sha256']
    for name, digest in spec['source_sha256'].items():
        assert sha(Path(spec['repository']) / name) == digest, name
    for path, digest in spec['input_sha256'].items():
        assert sha(path) == digest, path
    return spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    spec = binding(root)
    spec_sha = sha(root / 'spec.json')
    assert not (root / 'training').exists()
    (root / 'training').mkdir()
    sys.path.insert(0, spec['repository'])
    from lib.models.sttrack.lachtt_initial_instance_alignment import InitialInstanceCandidateSetAssociation, content_tokens
    from lib.models.sttrack.lachtt_candidate_set import supervised_loss
    from tools.train_sttrack_m42 import overlaps
    opt = spec['optimization']
    assert opt['device'] == 'cuda' and os.environ['CUDA_VISIBLE_DEVICES'] == '0'
    device = torch.device('cuda:0')
    torch.set_num_threads(opt['cpu_threads'])
    labels = json.loads((root / 'fit_labels.json').read_text())
    bank = torch.load(root / 'text_fit.pt', map_location='cpu')
    initial_bank = torch.load(root / 'initial_references/initial_fit.pt', map_location='cpu')
    fit_names = set(bank['sequences'])
    assert bank['split'] == initial_bank['split'] == 'fit'
    assert len(fit_names) == 63 and {x['sequence'] for x in labels.values()} == set(initial_bank['sequences']) == fit_names
    assert initial_bank['checkpoint_sha256'] == spec['base_checkpoint_sha256']
    initial_references = dict(zip(initial_bank['sequences'], initial_bank['references']))
    parent = Path(spec['source_root'])
    collected = {key: [] for key in ['current', 'previous', 'references', 'geometry', 'scores']}
    keys, sequence_ids, current_ious, previous_ious = [], [], [], []
    for receipt in spec['fit_features']:
        name = receipt['sequence']
        assert name in fit_names
        path = parent / 'features' / (name + '.pt')
        assert sha(path) == receipt['sha256']
        data = torch.load(path, map_location='cpu')
        assert data['sequence'] == name and data['fold'] in [2, 3, 4]
        assert data['spec_sha256'] == spec['parent_spec_sha256']
        assert data['references'].dtype == initial_references[name].dtype == torch.float16
        data['references'][:, 0] = initial_references[name]
        for key in collected:
            collected[key].append(data[key])
        for index, row in enumerate(data['records']):
            label = labels[row['key']]
            assert label['sequence'] == name and label['fold'] == data['fold']
            assert row['previous_choice'] == 0 and row['previous_frame'] == row['frame'] - 1
            for side, boxkey, publickey, destination in [
                ('current', 'current_boxes', 'public_bbox', current_ious),
                ('previous', 'previous_boxes', 'previous_public_bbox', previous_ious)]:
                if label[side] is None:
                    value = torch.zeros(10)
                else:
                    gt = torch.tensor(label[side])
                    value = overlaps(data[boxkey][index], gt)
                    value[0] = overlaps(data[publickey][index], gt)
                destination.append(value)
            keys.append(row['key'])
            sequence_ids.append(bank['sequences'].index(name))
    assert len(keys) == len(set(keys)) == 1511 and set(keys) == set(labels)
    tensors = {key: torch.cat(value) for key, value in collected.items()}
    del collected, data, labels
    sequence_ids = torch.tensor(sequence_ids)
    targets, changed_counts = [], []
    for values in [torch.stack(current_ious), torch.stack(previous_ious)]:
        target = values.argmax(1)
        target[values.max(1).values < .5] = 10
        before = target.clone()
        target[values[:, 0] >= .5] = 0
        changed_counts.append(int((target != before).sum()))
        targets.append(target)
    assert changed_counts[0] == 121
    results = {}
    for variant in spec['variants']:
        torch.manual_seed(opt['seed'])
        np.random.seed(opt['seed'])
        random.seed(opt['seed'])
        reference_mode = variant.split('_')[0]
        use_text = variant.endswith('_text')
        model = InitialInstanceCandidateSetAssociation(reference_mode).to(device)
        assert sum(p.numel() for p in model.parameters()) == spec['parameters']
        initial = {name: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
                   for name, value in model.state_dict().items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt['lr'], weight_decay=opt['weight_decay'])
        order = torch.Generator().manual_seed(opt['seed'])
        digest = hashlib.sha256()
        text = content_tokens(bank['tokens'], bank['empty'], use_text).to(device)
        text_mask = bank['mask'].to(device)
        losses, steps = [], 0
        started = time.time()
        model.train()
        for epoch in range(opt['epochs']):
            shuffled = torch.randperm(len(keys), generator=order)
            digest.update(shuffled.numpy().tobytes())
            totals = np.zeros(3)
            for index in shuffled.split(opt['batch_size']):
                inputs = [tensors[key][index].float().to(device) for key in ['current', 'previous', 'references', 'geometry', 'scores']]
                indices = sequence_ids[index].to(device)
                inputs += [torch.zeros(len(index), dtype=torch.long, device=device), text[indices], text_mask[indices]]
                logits, affinity = model(*inputs)
                loss, main, pair = supervised_loss(logits, affinity, targets[0][index].to(device), targets[1][index].to(device))
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), opt['grad_clip'])
                assert torch.isfinite(norm)
                optimizer.step()
                steps += 1
                totals += np.asarray([float(x.detach()) for x in [loss, main, pair]]) * len(index)
            row = dict(epoch=epoch + 1, loss=totals[0] / len(keys), identity_loss=totals[1] / len(keys),
                       matching_loss=totals[2] / len(keys), optimizer_steps=steps, elapsed_seconds=time.time() - started)
            losses.append(row)
            print(json.dumps(dict(variant=variant, **row)), flush=True)
        assert steps == 960
        checkpoint = root / 'training' / (variant + '_final.pth')
        torch.save(dict(model=model.state_dict(), variant=variant, reference_mode=reference_mode, use_text=use_text, m57_spec_sha256=spec_sha,
            base_checkpoint_sha256=spec['base_checkpoint_sha256'], text_bank_sha256=sha(root / 'text_fit.pt'),
            initial_fit_sha256=sha(root / 'initial_references/initial_fit.pt'),
            epochs=opt['epochs'], optimizer_steps=steps, target_rule='default_if_iou_at_least_half'), checkpoint)
        result = dict(status='complete_train', variant=variant, reference_mode=reference_mode, use_text=use_text, spec_sha256=spec_sha, parameters=spec['parameters'],
            optimizer_steps=steps, epochs=opt['epochs'], fit_sequences=63, fit_events=len(keys),
            sample_order_sha256=digest.hexdigest(), ordered_event_keys_sha256=hashlib.sha256('\n'.join(keys).encode()).hexdigest(),
            initial_state_sha256=initial, checkpoint_sha256=sha(checkpoint), checkpoint_bytes=checkpoint.stat().st_size,
            target_counts=dict(Counter(targets[0].tolist())), changed_current_labels=changed_counts[0],
            changed_previous_labels=changed_counts[1], losses=losses, elapsed_seconds=time.time() - started,
            changed_tensors=[name for name, value in model.state_dict().items()
                if hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest() != initial[name]],
            development_features_loaded=False, development_targets_loaded=False, development_text_bank_loaded=False,
            initial_reference_protocol='All arms replace reference0 with the same causal t0 float16 reference',
            optimization_device=str(device), gpu_name=torch.cuda.get_device_name(), torch_version=torch.__version__, cuda_version=torch.version.cuda,
            source_state='native STTrack cached trajectory; previous_choice always zero',
            scope='Train fitting only; no tracking-performance or language-gain claim')
        write(root / 'training' / (variant + '_result.json'), result)
        results[variant] = result
        del model, optimizer
    for field in ['sample_order_sha256', 'ordered_event_keys_sha256', 'initial_state_sha256', 'optimizer_steps', 'parameters']:
        assert all(results[variant][field] == results[spec['variants'][0]][field] for variant in spec['variants']), field
    binding(root)
    assert sha(root / 'spec.json') == spec_sha
    write(root / 'training_result.json', dict(status='complete_train', spec_sha256=spec_sha,
        equal_initialization_order_budget_parameters=True, variants={name: dict(
            result_sha256=sha(root / 'training' / (name + '_result.json')), checkpoint_sha256=value['checkpoint_sha256'])
            for name, value in results.items()}, development_evaluated=False, recursive_evaluated=False))
    print(json.dumps(dict(status='complete_train', next='Freeze weights, evaluate text controls and full recursive development')), flush=True)


if __name__ == '__main__':
    main()
