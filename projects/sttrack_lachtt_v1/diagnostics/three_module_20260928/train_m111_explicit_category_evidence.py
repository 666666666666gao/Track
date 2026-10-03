"""M111 matched 0/1 first-frame category evidence supervision; frozen visual B."""
import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from analyze_train_states import sha
from instance_ab_prototype import InstanceCandidatePrototype
from train_ab_visual_control import load_inputs, native_candidate_preservation
from train_m110_no_weak_rank import TRAIN_PREFIXES, inputs, snapshot, evaluate


INITIAL_FIELDS = ('initial_rois', 'initial_context', 'initial_mask',
                  'initial_context_mask', 'initial_depth_valid')


def initial_panel(panel, manifest, category_bank, split):
    locations = {}
    for i, key in enumerate(panel['key']):
        locations.setdefault(key.rsplit('@', 1)[0], []).append(i)
    rows = [r for r in manifest['rows'] if r['split'] == split]
    ids = []
    for r in rows:
        at = locations[r['sequence']]
        ids.append(at[0])
        for name in INITIAL_FIELDS:
            value = panel[name][at]
            assert torch.equal(value, value[:1].expand_as(value)), (r['sequence'], name)
    ids = torch.tensor(ids)
    bank_ids = torch.tensor([category_bank['sequences'].index(r['sequence']) for r in rows])
    return dict(rows=rows, labels=torch.tensor([r['label'] for r in rows]),
                tokens=category_bank['tokens'][bank_ids],
                **{name: panel[name][ids] for name in INITIAL_FIELDS})


def category_inputs(panel, ids, empty, device):
    data = {name: panel[name][ids].to(device) for name in INITIAL_FIELDS}
    data = {name: value if value.dtype == torch.bool else value.float()
            for name, value in data.items()}
    # Both observations are the immutable initialization region. No event ROI,
    # candidate GT, review-video frame, or corrected phrase is read here.
    for target, source in [('candidate_rois', 'initial_rois'), ('contexts', 'initial_context'),
                           ('candidate_mask', 'initial_mask'), ('context_mask', 'initial_context_mask'),
                           ('depth_valid', 'initial_depth_valid')]:
        data[target] = data[source][:, None]
    data['text'] = torch.zeros(len(ids), 5, 768, device=device)
    data['text'][:, 0] = panel['tokens'][ids].to(device)
    data['text_mask'] = torch.zeros(len(ids), 5, dtype=torch.bool, device=device)
    data['text_mask'][:, 0] = True
    data['empty_text'] = empty.to(device)
    return data


def category_readout(model, panel, empty, device):
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(panel['rows']), 32):
            ids = torch.arange(start, min(start + 32, len(panel['rows'])))
            out = model.evidence(category_inputs(panel, ids, empty, device))
            logits.append(out['phrase_logits'][:, 0, 0].cpu())
    logits = torch.cat(logits)
    target = panel['labels']
    prediction = logits.argmax(-1)
    confusion = torch.zeros(3, 3, dtype=torch.long)
    for label, pred in zip(target.tolist(), prediction.tolist()):
        confusion[label, pred] += 1
    query_labels = {}
    for r in panel['rows']:
        query_labels.setdefault(r['query'], set()).add(r['label'])
    mixed_ids = torch.tensor([i for i, r in enumerate(panel['rows'])
                              if len(query_labels[r['query']]) > 1])
    return dict(examples=len(target), correct=int((prediction == target).sum()),
                cross_entropy=float(F.cross_entropy(logits, target)),
                confusion_target_by_prediction=confusion.tolist(),
                recalls=(confusion.diag() / confusion.sum(-1)).tolist(),
                constant_majority_correct=int(confusion.sum(-1).max()),
                mixed_label_same_query_examples=len(mixed_ids),
                mixed_label_same_query_correct=int((prediction[mixed_ids] == target[mixed_ids]).sum()),
                supervision='model weak original-category status; immutable first-frame regions only')


def main():
    p = argparse.ArgumentParser()
    for k in ['cache', 'contexts', 'origins', 'bank', 'labels', 'parent', 'parent-result',
              'manifest', 'category-bank', 'output']:
        p.add_argument('--' + k, type=Path, required=True)
    p.add_argument('--evidence-weight', type=int, choices=(0, 1), required=True)
    p.add_argument('--mode', choices=('sanity', 'train'), required=True)
    a = p.parse_args()
    assert not a.output.exists()
    torch.set_num_threads(1)
    random.seed(2027); np.random.seed(2027)
    torch.manual_seed(2027); torch.cuda.manual_seed_all(2027)
    device = torch.device('cuda')
    panel = load_inputs(a.cache, a.contexts, a.origins, False)
    bank = torch.load(a.bank, map_location='cpu')
    manifest = json.loads(a.manifest.read_text())
    category_bank = torch.load(a.category_bank, map_location='cpu')
    assert bank['labels_sha256'] == sha(a.labels) == manifest['labels_sha256']
    assert not bank['human_confirmed'] and not manifest['human_confirmed'] and not category_bank['human_confirmed']
    assert sha(a.bank) == '8ff9887ceae2a26f36867824a3a04941fb076fc15b0e93575538f5374633de10'
    assert category_bank['manifest_sha256'] == sha(a.manifest)
    assert category_bank['encoder_sha256'] == bank['encoder_sha256']
    assert category_bank['sequences'] == [r['sequence'] for r in manifest['rows']]
    assert manifest['classes'] == ['supported', 'conflicting', 'uncertain']
    old = json.loads(a.parent_result.read_text())
    assert sha(a.parent) == old['final_weights_sha256'] == '1d187d78c1dd76ebf59e41fd31af16fa69cecc37b4cb6c74fbdfe3ed5c237293'
    category = {s: initial_panel(panel[s], manifest, category_bank, s) for s in ['fit', 'development']}
    assert Counter(category['fit']['labels'].tolist()) == Counter({0: 80, 1: 17, 2: 1})
    assert Counter(category['development']['labels'].tolist()) == Counter({0: 15, 1: 3, 2: 1})
    model = InstanceCandidatePrototype().to(device)
    model.load_state_dict(torch.load(a.parent, map_location='cpu'), strict=True)
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        if name.startswith(TRAIN_PREFIXES):
            param.requires_grad_(True)
    torch.nn.init.zeros_(model.semantic.weight); torch.nn.init.zeros_(model.phrase.weight)
    frozen = {k: v.detach().cpu().clone() for k, v in model.named_parameters() if not v.requires_grad}
    buffers = {k: v.detach().cpu().clone() for k, v in model.named_buffers()}
    initial = {s: snapshot(model, panel[s], bank, device) for s in ['fit', 'development']}
    parent_indices = {s: x[0].argmax(-1) for s, x in initial.items()}
    assert parent_indices['development'].tolist() == [r['selected'] for r in old['development_rows']]
    initial_category = ({s: category_readout(model, category[s], bank['empty'], device) for s in category}
                        if a.mode == 'train' else {})
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    started = time.time(); history = []; gradients = []; category_draw_counts = Counter()
    epochs = 12 if a.mode == 'train' else 1
    for epoch in range(epochs):
        batches = (torch.randperm(len(panel['fit']['key']), generator=torch.Generator().manual_seed(2027 + epoch)).split(64)
                   if a.mode == 'train' else [torch.arange(64)] * 3)
        terms = []; model.train()
        for ids in batches:
            optimizer.zero_grad(set_to_none=True)
            data = inputs(panel['fit'], ids, bank, 'weak_text', device)
            out = model(data); target = panel['fit']['iou'][ids].to(device)
            localization = F.binary_cross_entropy_with_logits(out['selection_logits'], target)
            preservation, _ = native_candidate_preservation(out['selection_logits'], data['base_scores'], target)
            step = sum(r['optimizer_steps'] for r in history) + len(terms)
            category_ids = torch.randint(len(category['fit']['rows']), (32,),
                                         generator=torch.Generator().manual_seed(32027 + step))
            category_target = category['fit']['labels'][category_ids].to(device)
            category_draw_counts.update(category_target.cpu().tolist())
            evidence = model.evidence(category_inputs(category['fit'], category_ids, bank['empty'], device))
            ce = F.cross_entropy(evidence['phrase_logits'][:, 0, 0], category_target)
            base_loss = localization + preservation
            loss = base_loss if a.evidence_weight == 0 else base_loss + ce
            assert bool(torch.isfinite(loss)) and bool(torch.isfinite(ce))
            loss.backward()
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
            if a.mode == 'sanity':
                norms = {n: sum(float(p.grad.norm()) for k, p in model.named_parameters()
                               if k.startswith(n + '.') and p.grad is not None)
                         for n in ['evidence.text', 'evidence.evidence', 'semantic', 'phrase']}
                gradients.append(norms)
            optimizer.step()
            terms.append([float(x.detach()) for x in [loss, localization, preservation, ce]])
        history.append(dict(epoch=epoch + 1, optimizer_steps=len(terms),
                            mean_losses=np.mean(terms, axis=0).tolist(), seconds=time.time() - started))
        print(json.dumps(history[-1]), flush=True)
    for k, v in model.named_parameters():
        if k in frozen:
            assert torch.equal(v.detach().cpu(), frozen[k]), k
    for k, v in model.named_buffers():
        assert torch.equal(v.cpu(), buffers[k]), k
    for split in initial:
        final = snapshot(model, panel[split], bank, device)
        assert all(torch.equal(x, y) for x, y in zip(final, initial[split])), split
    result = dict(status='complete_M111_explicit_category_arm' if a.mode == 'train' else 'complete_M111_gpu_sanity',
                  evidence_weight=a.evidence_weight, mode=a.mode, seed=2027, epochs=epochs,
                  batch_size=64, category_batch_size=32, learning_rate=3e-4,
                  optimizer_steps=sum(r['optimizer_steps'] for r in history),
                  category_example_draw_counts=dict(category_draw_counts),
                  trainable_prefixes=TRAIN_PREFIXES,
                  optimized_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  source_sha256=sha(__file__), prototype_sha256=sha(Path(__file__).with_name('instance_ab_prototype.py')),
                  parent_sha256=sha(a.parent), bank_sha256=sha(a.bank), labels_sha256=sha(a.labels),
                  manifest_sha256=sha(a.manifest), category_bank_sha256=sha(a.category_bank),
                  human_confirmed=False, category_classes=manifest['classes'], history=history,
                  empty_all3039_scores_quality_exact=True, frozen_parameters_buffers_exact=True,
                  original_category_labels_not_inherited_by_corrected_phrase=True,
                  no_recursive_or_public_evaluation=True, no_automatic_promotion=True,
                  initial_category_readout=initial_category, elapsed_seconds=time.time() - started)
    a.output.mkdir()
    if a.mode == 'sanity':
        assert gradients[-1]['evidence.text'] > 0 and gradients[-1]['semantic'] > 0
        if a.evidence_weight == 1:
            assert gradients[0]['evidence.evidence'] > 0
        result.update(gradient_norms=gradients, checkpoint_saved=False, development_metrics_reported=False)
    else:
        result['final_category_readout'] = {s: category_readout(model, category[s], bank['empty'], device) for s in category}
        result['fit'], rows = evaluate(model, panel['fit'], bank, 'weak_text', device, parent_indices['fit'])
        (a.output / 'fit_events.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        result['content_conditions'] = {}; content_rows = {}
        for mode in ['empty', 'generic', 'weak_text']:
            summary, rows = evaluate(model, panel['development'], bank, mode, device, parent_indices['development'])
            result['content_conditions'][mode] = summary; content_rows[mode] = rows
            (a.output / (mode + '_development.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in rows))
        empty_rows, weak_rows = content_rows['empty'], content_rows['weak_text']
        result['weak_vs_own_empty'] = dict(events=495,
            changed_selections=sum(x['selected'] != y['selected'] for x, y in zip(empty_rows, weak_rows)),
            iou_improved=sum(y['selected_iou'] > x['selected_iou'] for x, y in zip(empty_rows, weak_rows)),
            iou_worsened=sum(y['selected_iou'] < x['selected_iou'] for x, y in zip(empty_rows, weak_rows)),
            delta_mean_iou=sum(y['selected_iou'] - x['selected_iou'] for x, y in zip(empty_rows, weak_rows)) / 495,
            threshold_rescues=sum(x['selected_iou'] < .5 <= y['selected_iou'] for x, y in zip(empty_rows, weak_rows)),
            threshold_breaks=sum(y['selected_iou'] < .5 <= x['selected_iou'] for x, y in zip(empty_rows, weak_rows)))
        dev = result['content_conditions']['weak_text']
        result['fixed_state_checks'] = dict(correct_exceeds_native=dev['all']['selected_iou50'] > dev['all']['native_iou50'],
            mean_iou_exceeds_native=dev['all']['selected_mean_iou'] > dev['all']['native_mean_iou'],
            healthy_breaks_zero=dev['healthy']['breaks'] == 0,
            transition_correct_at_least_native=dev['transition']['selected_iou50'] >= dev['transition']['native_iou50'])
        torch.save(model.state_dict(), a.output / 'final.pt')
        reloaded = torch.load(a.output / 'final.pt', map_location='cpu')
        assert all(torch.equal(v.cpu(), reloaded[k]) for k, v in model.state_dict().items())
        result.update(final_state_roundtrip_exact=True, final_sha256=sha(a.output / 'final.pt'))
    result['gpu_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    (a.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['history', 'content_conditions', 'fit']}), flush=True)


if __name__ == '__main__':
    main()
