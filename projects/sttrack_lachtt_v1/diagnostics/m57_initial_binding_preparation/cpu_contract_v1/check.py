"""CPU engineering contract only; does not extract a new t0 reference or retain weights."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_digest(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    assert not (root / 'contract.json').exists()
    repository = Path('/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1')
    source = Path('/root/autodl-tmp/sttrack_m44_candidate_set_v1_20260905')
    previous = Path('/root/autodl-tmp/sttrack_m56_attribute_alignment_v1_20260906')
    sys.path.insert(0, str(repository))
    from lib.models.sttrack.lachtt_candidate_set import CandidateSetAssociation, supervised_loss
    from tools.train_sttrack_m42 import overlaps
    module_spec = importlib.util.spec_from_file_location('m57_preparation', root / 'model.py')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    torch.set_num_threads(4)
    assert sha(previous / 'text_bank.pt') == '93cd72856177f338d14ae05959459e36bf3082ddda340e9b332e5a33089ae967'
    assert sha(previous / 'fit_labels.json') == '0913d2121efc2e5a5261eebc58d932fa763d039623c11cbe72e14c9796742080'
    cache_path = source / 'features/chair01_indoor.pt'
    receipts = sum([json.loads((source / ('shard%d_receipt.json' % s)).read_text())['sequences'] for s in [0, 1]], [])
    assert sha(cache_path) == next(r['feature_sha256'] for r in receipts if r['sequence'] == 'chair01_indoor')
    cache = torch.load(cache_path, map_location='cpu')
    bank = torch.load(previous / 'text_bank.pt', map_location='cpu')
    labels = json.loads((previous / 'fit_labels.json').read_text())
    assert cache['fold'] in [2, 3, 4]
    indices = torch.arange(32) % len(cache['records'])
    inputs = [cache[k][indices].float() for k in ['current', 'previous', 'references', 'geometry', 'scores']]
    inputs.append(torch.zeros(32, dtype=torch.long))
    bank_row = bank['sequences'].index('chair01_indoor')
    original = bank['tokens'][bank_row:bank_row + 1].expand(32, -1, -1)
    mask = bank['mask'][bank_row:bank_row + 1].expand(32, -1)
    targets = []
    for side, key, public in [('current', 'current_boxes', 'public_bbox'), ('previous', 'previous_boxes', 'previous_public_bbox')]:
        values = []
        for index in indices.tolist():
            gt = labels[cache['records'][index]['key']][side]
            value = torch.zeros(10) if gt is None else overlaps(cache[key][index], torch.tensor(gt))
            if gt is not None:
                value[0] = overlaps(cache[public][index], torch.tensor(gt))
            values.append(0 if value[0] >= .5 else (int(value.argmax()) if value.max() >= .5 else 10))
        targets.append(torch.tensor(values))
    torch.manual_seed(2026)
    parent = CandidateSetAssociation().eval()
    results = []
    for reference_mode in ['candidate', 'initial']:
        for use_text in [True, False]:
            torch.manual_seed(2026)
            model = module.InitialInstanceCandidateSetAssociation(reference_mode).eval()
            assert all(torch.equal(value, model.state_dict()[name]) for name, value in parent.state_dict().items())
            initial_sha = tensor_digest(model.state_dict())
            tokens = module.content_tokens(original, bank['empty'], use_text)
            assert tokens.shape == original.shape and mask.shape == (32, 5)
            with torch.no_grad():
                expected = parent(*inputs)
                actual = model(*inputs, tokens, mask)
            assert all(torch.equal(a, b) for a, b in zip(expected, actual))
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
            gradients = []
            for step in range(3):
                output = model(*inputs, tokens, mask)
                loss, _, _ = supervised_loss(*output, *targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                assert torch.isfinite(loss)
                group_norms = {}
                for group in ['text', 'read_visual', 'guide_visual', 'output', 'slot_embedding']:
                    prefix = 'attribute_alignment.' + group
                    values = [p.grad for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
                    assert values and all(torch.isfinite(v).all() for v in values)
                    group_norms[group] = float(torch.sqrt(sum(v.square().sum() for v in values)))
                gradients.append(group_norms)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            assert all(any(row[group] > 0 for row in gradients[1:]) for group in gradients[0])
            with torch.no_grad():
                actual = model(*inputs, tokens, mask)
                padded = tokens.clone()
                padded[~mask] = 1000.
                changed_padding = model(*inputs, padded, mask)
                assert all(torch.equal(a, b) for a, b in zip(actual, changed_padding))
                cells = model.cell(torch.cat(inputs[:3], dim=1))
                changed = cells.clone()
                changed[:, :20] += .5
                changed[:, 21] += .5
                changed_initial = cells.clone()
                changed_initial[:, 20] += .5
                observed = []

                def record_conditioning(_, arguments):
                    observed.append(arguments[1].detach().clone().reshape(32, 22, 5, 32))

                hook = model.attribute_alignment.guide_visual.register_forward_pre_hook(record_conditioning)
                model.attribute_alignment(cells, tokens, mask)
                model.attribute_alignment(changed, tokens, mask)
                model.attribute_alignment(changed_initial, tokens, mask)
                hook.remove()
                stable_under_mutable_change = torch.equal(observed[0], observed[1])
                candidate_conditioning_responds_to_initial = not torch.equal(observed[0][:, :20], observed[2][:, :20])
                assert stable_under_mutable_change == (reference_mode == 'initial')
                assert candidate_conditioning_responds_to_initial == (reference_mode == 'initial')
            results.append(dict(reference_mode=reference_mode, text='real' if use_text else 'visual_queries',
                parameters=sum(p.numel() for p in model.parameters()), initial_state_sha256=initial_sha,
                zero_residual_parent_parity_exact=True, padding_invariance_exact=True,
                actual_conditioning_independent_of_mutable_cells=stable_under_mutable_change,
                actual_candidate_conditioning_responds_to_initial_cells=candidate_conditioning_responds_to_initial,
                data_gradient_norms_before_optimizer=gradients, discarded_optimizer_steps=3))
    assert len({r['initial_state_sha256'] for r in results}) == 1
    assert {r['parameters'] for r in results} == {484547}
    assert not torch.cuda.is_initialized()
    result = dict(status='pass', checker_sha256=sha(__file__), model_sha256=sha(root / 'model.py'),
        extractor_sha256=sha(root / 'initial_observation.py'), cache_sha256=sha(cache_path),
        text_bank_sha256=sha(previous / 'text_bank.pt'), fit_labels_sha256=sha(previous / 'fit_labels.json'),
        source_sequence='chair01_indoor', split='fit', real_cache_records=len(cache['records']), batch=32,
        cyclic_repetition_for_contract_only=True, arms=results, cpu_threads=4, torch_version=torch.__version__,
        initial_reference_used='Historical t1 reference from existing cache; only tensor/gradient contract, not a new t0 observation',
        t0_extractor_executed=False, gpu_execution=False, trained_checkpoint_saved=False,
        final_base_selected=False, formal_feature_collection_started=False, development_evaluation=False,
        scope='Untrained interface preparation; no evidence of semantic accuracy or tracking improvement')
    (root / 'contract.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
