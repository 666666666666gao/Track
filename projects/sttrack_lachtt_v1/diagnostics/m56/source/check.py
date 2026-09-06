"""Real fit-cache shape, zero-residual parity, mask and gradient contract."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    assert not (root / 'contract.json').exists()
    parent = Path('/root/autodl-tmp/sttrack_m44_candidate_set_v1_20260905')
    repo = Path('/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1')
    sys.path.insert(0, str(repo))
    from lib.models.sttrack.lachtt_candidate_set import CandidateSetAssociation, supervised_loss
    from lib.models.sttrack.lachtt_attribute_candidate_set import AttributeCandidateSetAssociation, condition_text
    from tools.train_sttrack_m42 import overlaps
    torch.set_num_threads(4)
    bank = torch.load(root / 'text_bank.pt', map_location='cpu')
    cache_path = parent / 'features/chair01_indoor.pt'
    receipts = sum([json.loads((parent / ('shard%d_receipt.json' % s)).read_text())['sequences'] for s in [0, 1]], [])
    receipt = next(x for x in receipts if x['sequence'] == 'chair01_indoor')
    assert sha(cache_path) == receipt['feature_sha256']
    cache = torch.load(cache_path, map_location='cpu')
    assert cache['fold'] in [2, 3, 4]
    indices = torch.arange(32) % len(cache['records'])
    args_model = [cache[key][indices].float() for key in ['current', 'previous', 'references', 'geometry', 'scores']]
    args_model.append(torch.zeros(32, dtype=torch.long))
    row = bank['sequences'].index('chair01_indoor')
    tokens = bank['tokens'][row:row + 1].expand(32, -1, -1)
    mask = bank['mask'][row:row + 1].expand(32, -1)
    torch.manual_seed(2026)
    native = CandidateSetAssociation().eval()
    torch.manual_seed(2026)
    model = AttributeCandidateSetAssociation().eval()
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in native.state_dict().items())
    with torch.no_grad():
        reference = native(*args_model)
        out = model(*args_model, tokens, mask)
    assert all(torch.equal(a, b) for a, b in zip(reference, out))
    labels = json.loads((root / 'fit_labels.json').read_text())
    targets = []
    for side, key, public in [('current', 'current_boxes', 'public_bbox'), ('previous', 'previous_boxes', 'previous_public_bbox')]:
        values = []
        for index in indices.tolist():
            gt = labels[cache['records'][index]['key']][side]
            if gt is None:
                value = torch.zeros(10)
            else:
                gt = torch.tensor(gt)
                value = overlaps(cache[key][index], gt)
                value[0] = overlaps(cache[public][index], gt)
            target = 0 if value[0] >= .5 else (int(value.argmax()) if value.max() >= .5 else 10)
            values.append(target)
        targets.append(torch.tensor(values))
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    durations = []
    for _ in range(3):
        started = time.time()
        logits, affinity = model(*args_model, tokens, mask)
        loss, _, _ = supervised_loss(logits, affinity, *targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        durations.append(time.time() - started)
    changed = [name for name, value in model.state_dict().items() if not torch.equal(value, initial[name])]
    assert any(name.startswith('attribute_alignment.output') for name in changed)
    assert any(name.startswith('attribute_alignment.text') for name in changed)
    perturbed = tokens.clone()
    perturbed[~mask] = 1000.
    permutation = torch.tensor([4, 2, 0, 3, 1])
    with torch.no_grad():
        actual = model(*args_model, tokens, mask)
        padded = model(*args_model, perturbed, mask)
        reordered = model(*args_model, tokens[:, permutation], mask[:, permutation])
    assert all(torch.equal(a, b) for a, b in zip(actual, padded))
    maximum_permutation_error = max(float((a - b).abs().max()) for a, b in zip(actual, reordered))
    assert maximum_permutation_error < 1e-5
    for variant in ['attributes', 'pooled', 'empty']:
        t, m = condition_text(tokens, mask, bank['empty'], variant)
        with torch.no_grad():
            output = model(*args_model, t, m)
        assert all(torch.isfinite(value).all() for value in output)
    result = dict(status='pass', checker_sha256=sha(__file__),
        model_sha256=sha(repo / 'lib/models/sttrack/lachtt_attribute_candidate_set.py'),
        text_bank_sha256=sha(root / 'text_bank.pt'), cache_sha256=sha(cache_path),
        sequence='chair01_indoor', split='fit', real_cache_records=len(cache['records']),
        contract_batch=32, cyclic_repetition_for_contract_only=True, discarded_optimizer_steps=3,
        trained_checkpoint_saved=False, parent_parameter_count=sum(p.numel() for p in native.parameters()),
        parameter_count=sum(p.numel() for p in model.parameters()),
        added_parameter_count=sum(p.numel() for p in model.attribute_alignment.parameters()),
        shared_initial_weights_exact=True, zero_residual_logits_and_affinity_exact=True,
        padding_invariance_exact=True, phrase_permutation_max_abs_error=maximum_permutation_error,
        finite_backward=True, changed_attribute_tensors=[n for n in changed if n.startswith('attribute_alignment.')],
        cpu_threads=4, torch_version=torch.__version__, batch_step_seconds=durations,
        estimated_three_arm_train_seconds=sum(durations) / len(durations) * 960 * 3,
        scope='Engineering contract on existing fit cache, not performance or a retained trained weight')
    (root / 'contract.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
