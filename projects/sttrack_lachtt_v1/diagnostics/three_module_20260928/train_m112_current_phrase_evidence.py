"""M112 current-candidate phrase supervision, with the completed zero arm reused."""
import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from analyze_train_states import sha
from instance_ab_prototype import InstanceCandidatePrototype
from train_ab_visual_control import load_inputs, native_candidate_preservation
from train_m110_no_weak_rank import TRAIN_PREFIXES, inputs, snapshot, evaluate


PARENT_SHA = '1d187d78c1dd76ebf59e41fd31af16fa69cecc37b4cb6c74fbdfe3ed5c237293'
BANK_SHA = '8ff9887ceae2a26f36867824a3a04941fb076fc15b0e93575538f5374633de10'
CONTROL_SHA = '36f3cdf300522eeb68252f538d04c4b8e02eff6a69a45c09102c3bebf82da4cc'
MANIFEST_SHA = '1527c6b86d1035f76ed85f07c2a08cc98726d47cb8c9b6db7509bcc7c3eca217'
CLASSES = ['supported', 'conflicting', 'unknown']


def phrase_panel(fit, bank, labels, manifest):
    assert manifest['classes'] == CLASSES and not manifest['human_confirmed']
    assert manifest['optimized_split'] == 'fit' and not manifest['physical_instance_labels']
    assert manifest['input_sha256']['weak_labels'] == bank['labels_sha256']
    locations = {key: i for i, key in enumerate(fit['key'])}
    initial = {r['sequence']: r for r in labels['initial']}
    rows = manifest['rows']
    assert len(rows) == 202 and Counter(r['label'] for r in rows) == Counter({0: 98, 1: 15, 2: 89})
    seen = set()
    for r in rows:
        token = (r['audit_id'], r['candidate'], r['slot'])
        assert token not in seen
        seen.add(token)
        assert r['key'] == r['sequence'] + '@' + str(r['frame']) and r['key'] in locations
        at = bank['sequences'].index(r['sequence'])
        assert r['split'] == initial[r['sequence']]['split'] == bank['splits'][at] == 'fit'
        assert 0 <= r['candidate_index'] < fit['candidate_rois'].shape[1]
        assert 0 <= r['slot'] < 5 and bool(bank['mask'][at, r['slot']])
        assert r['query'] == initial[r['sequence']]['phrases'][r['slot']]
        assert r['label_name'] == CLASSES[r['label']] and r['evidence_note'] and r['reviewer']
    assert len({r['key'] for r in rows}) == 24
    pairs = defaultdict(dict)
    for i, r in enumerate(rows):
        pairs[(r['audit_id'], r['slot'])][r['candidate']] = i
    assert len(pairs) == 101
    for pair in pairs.values():
        assert set(pair) == {'A', 'B'}
        a, b = (rows[pair[k]] for k in ('A', 'B'))
        assert a['key'] == b['key'] and a['query'] == b['query']
        assert a['candidate_index'] != b['candidate_index']
    return dict(rows=rows, cache_ids=torch.tensor([locations[r['key']] for r in rows]),
                candidates=torch.tensor([r['candidate_index'] for r in rows]),
                slots=torch.tensor([r['slot'] for r in rows]),
                labels=torch.tensor([r['label'] for r in rows]), pairs=pairs)


def phrase_logits(model, fit, review, ids, bank, device, initial_surrogate=False):
    data = inputs(fit, review['cache_ids'][ids], bank, 'weak_text', device)
    if initial_surrogate:
        count = data['candidate_rois'].shape[1]
        # Read-only input intervention, not a deployment policy or identity label.
        for current, initial in [('candidate_rois', 'initial_rois'), ('contexts', 'initial_context'),
                                 ('candidate_mask', 'initial_mask'), ('context_mask', 'initial_context_mask'),
                                 ('depth_valid', 'initial_depth_valid')]:
            value = data[initial][:, None]
            data[current] = value.expand(-1, count, *value.shape[2:])
    output = model.evidence(data)['phrase_logits']
    row = torch.arange(len(ids), device=device)
    return output[row, review['candidates'][ids].to(device), review['slots'][ids].to(device)]


def phrase_readout(model, fit, review, bank, device, output, prefix):
    model.eval()
    actual, surrogate = [], []
    with torch.no_grad():
        for start in range(0, len(review['rows']), 64):
            ids = torch.arange(start, min(start + 64, len(review['rows'])))
            actual.append(phrase_logits(model, fit, review, ids, bank, device).cpu())
            surrogate.append(phrase_logits(model, fit, review, ids, bank, device, True).cpu())
    logits, replaced = torch.cat(actual), torch.cat(surrogate)
    assert bool(torch.isfinite(logits).all()) and bool(torch.isfinite(replaced).all())
    target = review['labels']; pred = logits.argmax(-1)
    confusion = torch.zeros(3, 3, dtype=torch.long)
    for label, chosen in zip(target.tolist(), pred.tolist()):
        confusion[label, chosen] += 1
    margins = []
    probability = logits.softmax(-1)[:, 0]
    for pair in review['pairs'].values():
        a, b = pair['A'], pair['B']
        if {int(target[a]), int(target[b])} == {0, 1}:
            support, conflict = (a, b) if target[a] == 0 else (b, a)
            margins.append(float(probability[support] - probability[conflict]))
    assert len(margins) == 10
    records = [dict(audit_id=r['audit_id'], candidate=r['candidate'], slot=r['slot'],
                    target=r['label'], prediction=int(pred[i]), logits=logits[i].tolist(),
                    initial_surrogate_logits=replaced[i].tolist()) for i, r in enumerate(review['rows'])]
    (output / (prefix + '_phrase_rows.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in records))
    return dict(fit_examples=len(target), correct=int((pred == target).sum()),
                cross_entropy=float(F.cross_entropy(logits, target)),
                confusion_target_by_prediction=confusion.tolist(),
                recalls=(confusion.diag() / confusion.sum(-1)).tolist(),
                constant_majority_correct=int(confusion.sum(-1).max()),
                supported_conflicting_pairs=len(margins), paired_support_margins=margins,
                paired_support_margin_positive=sum(x > 0 for x in margins),
                initial_surrogate_max_logit_change=float((logits - replaced).abs().max()),
                initial_surrogate_prediction_changes=int((pred != replaced.argmax(-1)).sum()),
                initial_surrogate_cross_entropy=float(F.cross_entropy(replaced, target)),
                human_confirmed=False, semantic_development_examples=0,
                physical_instance_truth=False, transfer_established=False)


def selection_readout(model, panel, bank, device, parent_indices, output):
    fit_summary, fit_rows = evaluate(model, panel['fit'], bank, 'weak_text', device, parent_indices['fit'])
    (output / 'fit_events.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in fit_rows))
    summaries, rows = {}, {}
    for mode in ('empty', 'generic', 'weak_text'):
        summaries[mode], rows[mode] = evaluate(model, panel['development'], bank, mode, device,
                                             parent_indices['development'])
        (output / (mode + '_development.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in rows[mode]))
    empty, weak = rows['empty'], rows['weak_text']
    assert [r['key'] for r in empty] == [r['key'] for r in weak]
    delta = dict(events=len(weak), changed_selections=sum(x['selected'] != y['selected'] for x, y in zip(empty, weak)),
                 iou_improved=sum(y['selected_iou'] > x['selected_iou'] for x, y in zip(empty, weak)),
                 iou_worsened=sum(y['selected_iou'] < x['selected_iou'] for x, y in zip(empty, weak)),
                 delta_mean_iou=sum(y['selected_iou'] - x['selected_iou'] for x, y in zip(empty, weak)) / len(weak),
                 threshold_rescues=sum(x['selected_iou'] < .5 <= y['selected_iou'] for x, y in zip(empty, weak)),
                 threshold_breaks=sum(y['selected_iou'] < .5 <= x['selected_iou'] for x, y in zip(empty, weak)))
    delta['content_vs_own_empty'] = {}
    for mode in ('generic', 'weak_text'):
        assert [r['key'] for r in empty] == [r['key'] for r in rows[mode]]
        groups = {}
        for tag in ('all', 'healthy', 'transition'):
            paired = [(x, y) for x, y in zip(empty, rows[mode]) if tag == 'all' or tag in x['strata']]
            assert paired, tag
            groups[tag] = dict(events=len(paired), empty_iou50=sum(x['selected_iou'] >= .5 for x, _ in paired),
                               condition_iou50=sum(y['selected_iou'] >= .5 for _, y in paired),
                               delta_mean_iou=sum(y['selected_iou'] - x['selected_iou'] for x, y in paired) / len(paired),
                               rescues=sum(x['selected_iou'] < .5 <= y['selected_iou'] for x, y in paired),
                               harms=sum(y['selected_iou'] < .5 <= x['selected_iou'] for x, y in paired))
        delta['content_vs_own_empty'][mode] = groups
    return fit_summary, summaries, delta


def main():
    p = argparse.ArgumentParser()
    for name in ('cache', 'contexts', 'origins', 'bank', 'labels', 'parent', 'parent-result',
                 'manifest', 'control', 'control-result', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--mode', choices=('sanity', 'train', 'reference'), required=True)
    a = p.parse_args(); assert not a.output.exists()
    torch.set_num_threads(1); random.seed(2027); np.random.seed(2027)
    torch.manual_seed(2027); torch.cuda.manual_seed_all(2027)
    device = torch.device('cuda'); started = time.time()
    panel = load_inputs(a.cache, a.contexts, a.origins, False)
    bank = torch.load(a.bank, map_location='cpu'); labels = json.loads(a.labels.read_text())
    manifest = json.loads(a.manifest.read_text())
    assert sha(a.bank) == BANK_SHA and sha(a.manifest) == MANIFEST_SHA
    assert bank['labels_sha256'] == sha(a.labels) and not bank['human_confirmed']
    old = json.loads(a.parent_result.read_text()); control = json.loads(a.control_result.read_text())
    assert sha(a.parent) == old['final_weights_sha256'] == PARENT_SHA
    assert sha(a.control) == control['final_sha256'] == CONTROL_SHA
    assert control['optimizer_steps'] == 480 and control['evidence_weight'] == 0
    assert control['parent_sha256'] == PARENT_SHA and control['bank_sha256'] == BANK_SHA
    review = phrase_panel(panel['fit'], bank, labels, manifest)
    model = InstanceCandidatePrototype().to(device)
    model.load_state_dict(torch.load(a.parent, map_location='cpu'), strict=True)
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        if name.startswith(TRAIN_PREFIXES):
            param.requires_grad_(True)
    torch.nn.init.zeros_(model.semantic.weight); torch.nn.init.zeros_(model.phrase.weight)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 95683
    frozen = {k: v.detach().cpu().clone() for k, v in model.named_parameters() if not v.requires_grad}
    buffers = {k: v.detach().cpu().clone() for k, v in model.named_buffers()}
    initial = {s: snapshot(model, panel[s], bank, device) for s in ('fit', 'development')}
    parent_indices = {s: x[0].argmax(-1) for s, x in initial.items()}
    assert parent_indices['development'].tolist() == [r['selected'] for r in old['development_rows']]
    a.output.mkdir()
    result = dict(mode=a.mode, seed=2027, source_sha256=sha(__file__),
                  prototype_sha256=sha(Path(__file__).with_name('instance_ab_prototype.py')),
                  parent_sha256=PARENT_SHA, bank_sha256=BANK_SHA, labels_sha256=sha(a.labels),
                  manifest_sha256=sha(a.manifest), reused_control_sha256=CONTROL_SHA,
                  human_confirmed=False, classes=CLASSES, semantic_fit_examples=202,
                  semantic_development_examples=0, semantic_transfer_established=False,
                  no_recursive_or_public_evaluation=True, no_automatic_promotion=True)
    if a.mode == 'reference':
        model.load_state_dict(torch.load(a.control, map_location='cpu'), strict=True)
        model.requires_grad_(False)
        result.update(status='complete_M112_readonly_reference', optimizer_steps=0,
                      checkpoint_saved=False, optimized_parameters=0)
    else:
        if a.mode == 'train':
            result['initial_phrase_readout'] = phrase_readout(model, panel['fit'], review, bank, device,
                                                            a.output, 'initial')
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
        history, gradients, draws = [], [], Counter(); step = 0
        for epoch in range(12 if a.mode == 'train' else 1):
            batches = (torch.randperm(2544, generator=torch.Generator().manual_seed(2027 + epoch)).split(64)
                       if a.mode == 'train' else [torch.arange(64)] * 3)
            terms = []; model.train()
            for ids in batches:
                optimizer.zero_grad(set_to_none=True)
                data = inputs(panel['fit'], ids, bank, 'weak_text', device)
                out = model(data); target = panel['fit']['iou'][ids].to(device)
                localization = F.binary_cross_entropy_with_logits(out['selection_logits'], target)
                preservation, _ = native_candidate_preservation(out['selection_logits'], data['base_scores'], target)
                phrase_ids = torch.randint(202, (32,), generator=torch.Generator().manual_seed(32027 + step))
                phrase_target = review['labels'][phrase_ids].to(device)
                draws.update(phrase_target.cpu().tolist())
                ce = F.cross_entropy(phrase_logits(model, panel['fit'], review, phrase_ids, bank, device), phrase_target)
                loss = localization + preservation + ce
                assert bool(torch.isfinite(loss)) and bool(torch.isfinite(ce)); loss.backward()
                assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
                assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
                if a.mode == 'sanity':
                    gradients.append({n: sum(float(p.grad.norm()) for k, p in model.named_parameters()
                                             if k.startswith(n + '.') and p.grad is not None)
                                      for n in ('evidence.text', 'evidence.evidence', 'semantic', 'phrase')})
                optimizer.step(); step += 1
                terms.append([float(x.detach()) for x in (loss, localization, preservation, ce)])
            history.append(dict(epoch=epoch + 1, optimizer_steps=len(terms),
                                mean_losses=np.mean(terms, axis=0).tolist(), seconds=time.time() - started))
            print(json.dumps(history[-1]), flush=True)
        result.update(status='complete_M112_current_phrase_training' if a.mode == 'train' else 'complete_M112_gpu_sanity',
                      optimizer_steps=step, epochs=len(history), batch_size=64, phrase_batch_size=32,
                      learning_rate=3e-4, evidence_weight=1, trainable_prefixes=TRAIN_PREFIXES,
                      optimized_parameters=95683, history=history, phrase_example_draw_counts=dict(draws),
                      weak_identity_rank_optimized=False, loss_components=['localization', 'preservation', 'current_phrase_ce'])
        if a.mode == 'sanity':
            assert gradients[0]['evidence.evidence'] > 0
            assert gradients[-1]['evidence.text'] > 0 and gradients[-1]['semantic'] > 0
            result.update(gradient_norms=gradients, checkpoint_saved=False, development_metrics_reported=False)
    for name, value in model.named_parameters():
        if name in frozen:
            assert torch.equal(value.detach().cpu(), frozen[name]), name
    for name, value in model.named_buffers():
        assert torch.equal(value.cpu(), buffers[name]), name
    for split in initial:
        final = snapshot(model, panel[split], bank, device)
        assert all(torch.equal(x, y) for x, y in zip(final, initial[split])), split
    result.update(empty_all3039_scores_quality_exact=True, frozen_parameters_buffers_exact=True)
    if a.mode != 'sanity':
        result['final_phrase_readout'] = phrase_readout(model, panel['fit'], review, bank, device, a.output, 'final')
        result['fit'], result['content_conditions'], result['weak_vs_own_empty'] = selection_readout(
            model, panel, bank, device, parent_indices, a.output)
        if a.mode == 'reference':
            assert result['fit'] == control['fit']
            assert result['content_conditions'] == control['content_conditions']
            for name in ('fit_events.jsonl', 'empty_development.jsonl', 'generic_development.jsonl', 'weak_text_development.jsonl'):
                assert (a.output / name).read_bytes() == (a.control_result.parent / name).read_bytes(), name
            result['all_stored_control_selection_rows_reproduced'] = True
        else:
            torch.save(model.state_dict(), a.output / 'final.pt')
            reloaded = torch.load(a.output / 'final.pt', map_location='cpu')
            assert all(torch.equal(v.cpu(), reloaded[k]) for k, v in model.state_dict().items())
            result.update(final_state_roundtrip_exact=True, final_sha256=sha(a.output / 'final.pt'), checkpoint_saved=True)
    result.update(elapsed_seconds=time.time() - started, gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    (a.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('history', 'fit', 'content_conditions')}), flush=True)


if __name__ == '__main__':
    main()
