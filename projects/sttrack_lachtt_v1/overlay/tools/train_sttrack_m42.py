"""Fixed, matched local/pooled fitting on Train folds 2-4; evaluate fold 5 once."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def overlaps(boxes, gt):
    corner = boxes[..., :2]
    end = corner + boxes[..., 2:]
    intersection = (torch.minimum(end, gt[:2] + gt[2:]) - torch.maximum(corner, gt[:2])).clamp_min(0).prod(-1)
    return intersection / (boxes[..., 2:].prod(-1) + gt[2:].prod() - intersection)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    args = p.parse_args(); root = args.root
    spec = json.loads((root / 'spec.json').read_text())
    assert sha(root / 'training_labels.json') == spec['labels_sha256']
    repo = Path(spec['repository']); sys.path.insert(0, str(repo))
    for name, digest in spec['source_sha256'].items():
        assert sha(repo / name) == digest, name
    from lib.models.sttrack.lachtt_local_spatial_association import LocalSpatialAssociation, select_candidate
    labels = json.loads((root / 'training_labels.json').read_text())
    sequence_receipts = []
    for shard in [0, 1]:
        receipt = json.loads((root / f'shard{shard}_receipt.json').read_text())
        assert receipt['status'] == 'complete' and receipt['spec_sha256'] == sha(root / 'spec.json')
        sequence_receipts.extend(receipt['sequences'])
    assert len(sequence_receipts) == 85 and len({r['sequence'] for r in sequence_receipts}) == 85
    all_tensors = {k: [] for k in ['candidates', 'references', 'scalars']}
    keys, folds, ious = [], [], []
    for receipt in sorted(sequence_receipts, key=lambda r: r['sequence']):
        path = root / 'features' / (receipt['sequence'] + '.pt')
        assert sha(path) == receipt['feature_sha256']
        data = torch.load(path, map_location='cpu')
        assert data['spec_sha256'] == sha(root / 'spec.json')
        for name in all_tensors:
            all_tensors[name].append(data[name])
        for i, record in enumerate(data['records']):
            key = record['key']; label = labels[key]
            assert label['fold'] == data['fold'] and label['sequence'] == data['sequence']
            if label['visible']:
                gt = torch.tensor(label['gt_bbox'])
                values = overlaps(data['bboxes'][i], gt)
                # Selecting index 0 preserves the exact default output, not its rounded re-decoding.
                values[0] = overlaps(data['public_bbox'][i], gt)
            else:
                values = torch.zeros(10)
            keys.append(key); folds.append(data['fold']); ious.append(values)
    assert len(keys) == spec['event_count'] and set(keys) == set(labels)
    tensors = {k: torch.cat(v) for k, v in all_tensors.items()}
    ious = torch.stack(ious)
    targets = ious.argmax(1)
    targets[ious.max(1).values < .5] = 10
    fit = torch.tensor([i for i, fold in enumerate(folds) if fold in spec['fit_folds']])
    heldout = torch.tensor([i for i, fold in enumerate(folds) if fold == 5])
    assert len(fit) + len(heldout) == len(keys)
    assert set(keys[i].split('@')[0] for i in fit).isdisjoint(keys[i].split('@')[0] for i in heldout)
    torch.set_num_threads(1)
    opt = spec['optimization']
    results = {}
    start = time.time()

    def batch_inputs(index):
        return [tensors[k][index].cuda().float() for k in ['candidates', 'references', 'scalars']]

    def evaluate(model, index):
        model.eval()
        predictions = []
        with torch.no_grad():
            for b in index.split(opt['batch_size']):
                predictions.append(model(*batch_inputs(b)).cpu())
        logits = torch.cat(predictions)
        selected = select_candidate(logits)
        values = ious[index].gather(1, selected[:, None]).flatten()
        baseline = ious[index, 0]
        sequence_stats = {}
        for seq in sorted({keys[i].split('@')[0] for i in index}):
            mask = torch.tensor([keys[i].split('@')[0] == seq for i in index])
            sequence_stats[seq] = dict(events=int(mask.sum()), default_mean_iou=float(baseline[mask].mean()),
                selected_mean_iou=float(values[mask].mean()), gain=float((values-baseline)[mask].mean()))
        stats = dict(events=len(index), mean_iou=float(values.mean()), default_mean_iou=float(baseline.mean()),
            correct=int((values >= .5).sum()), default_correct=int((baseline >= .5).sum()),
            changes=int((selected != 0).sum()), none=int((logits.argmax(1) == 10).sum()),
            rescues=int(((baseline <= .1) & (values >= .5)).sum()),
            breaks=int(((baseline >= .5) & (values <= .1)).sum()),
            positive_sequences=sum(v['gain'] > 0 for v in sequence_stats.values()),
            sequence_stats=sequence_stats,
            rows=[dict(key=keys[i], chosen=int(chosen), action_none=bool(pred == 10),
                       default_iou=float(d), selected_iou=float(v), oracle_iou=float(ious[i].max()))
                  for i, chosen, pred, d, v in zip(index, selected, logits.argmax(1), baseline, values)])
        return stats, logits

    for variant in spec['variants']:
        torch.manual_seed(opt['seed']); torch.cuda.manual_seed_all(opt['seed'])
        random.seed(opt['seed']); np.random.seed(opt['seed'])
        model = LocalSpatialAssociation(spatial=variant == 'spatial').cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt['lr'], weight_decay=opt['weight_decay'])
        order = torch.Generator().manual_seed(opt['seed'])
        losses = []
        for epoch in range(opt['epochs']):
            model.train(); epoch_loss = 0.
            shuffled = fit[torch.randperm(len(fit), generator=order)]
            for index in shuffled.split(opt['batch_size']):
                logits = model(*batch_inputs(index))
                loss = F.cross_entropy(logits, targets[index].cuda())
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True); loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), opt['grad_clip'])
                assert torch.isfinite(norm)
                optimizer.step(); epoch_loss += float(loss.detach()) * len(index)
            losses.append(epoch_loss / len(fit))
            print(json.dumps(dict(variant=variant, epoch=epoch+1, loss=losses[-1], elapsed=time.time()-start)), flush=True)
        checkpoint = root / (variant + '_final.pth')
        torch.save(dict(model=model.state_dict(), variant=variant, spec_sha256=sha(root / 'spec.json'),
                        base_checkpoint_sha256=spec['checkpoint_sha256'], epochs=opt['epochs'],
                        trainer_sha256=sha(__file__)), checkpoint)
        fit_stats, _ = evaluate(model, fit)
        holdout_stats, logits = evaluate(model, heldout)
        loaded = LocalSpatialAssociation(spatial=variant == 'spatial').cuda()
        loaded.load_state_dict(torch.load(checkpoint)['model'], strict=True)
        loaded.eval()
        with torch.no_grad():
            restored = torch.cat([loaded(*batch_inputs(b)).cpu() for b in heldout.split(opt['batch_size'])])
        assert torch.equal(logits, restored)
        results[variant] = dict(parameters=sum(p.numel() for p in model.parameters()), checkpoint_sha256=sha(checkpoint),
            checkpoint_bytes=checkpoint.stat().st_size, reload_exact=True, losses=losses, fit=fit_stats, development_holdout=holdout_stats)
        (root / (variant + '_result.json')).write_text(json.dumps(results[variant], indent=2)+'\n')
        print(json.dumps(dict(variant=variant, development_holdout={k: v for k, v in holdout_stats.items() if k not in ['rows', 'sequence_stats']})), flush=True)
    s = results['spatial']['development_holdout']; g = results['pooled']['development_holdout']
    gates = dict(mean_iou=s['mean_iou'] > g['mean_iou'] and s['mean_iou'] > s['default_mean_iou'],
                 alternative_fixes=s['rescues'] > s['breaks'], sequence_coverage=s['positive_sequences'] >= 3)
    result = dict(status='complete', variants=results, information_gate=gates, information_gate_pass=all(gates.values()),
         spec_sha256=sha(root / 'spec.json'), trainer_sha256=sha(__file__), elapsed_seconds=time.time()-start,
         fit_events=len(fit), development_holdout_events=len(heldout),
         target_counts_fit=dict(Counter(targets[fit].tolist())), target_counts_holdout=dict(Counter(targets[heldout].tolist())),
         claim='Static Train development information test only; no recursive rescue or public metric claim.',
         next='paired prediction-crop recursive development evaluation' if all(gates.values()) else 'stop this frozen M42 variant; inspect failure evidence without threshold/epoch sweeps')
    (root / 'training_result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status='complete', gate=gates, information_gate_pass=all(gates.values()))), flush=True)


if __name__ == '__main__':
    main()
