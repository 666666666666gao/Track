"""Fit the frozen two-view reader on sealed DepthTrack Train observations."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from tools.sttrack_m54_common import check_sources, event_frames, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    plan, parent, spec = check_sources(root)
    from lib.models.sttrack.lachtt_template_reader import TemplateReader
    from lib.test.tracker.sttrack_template_reader import READER_FIELDS
    from tools.train_sttrack_m42 import overlaps
    assert (root / 'collection.exit').read_text().strip() == '0'
    receipt = json.loads((root / 'collection_receipt.json').read_text())
    assert receipt['status'] == 'complete' and receipt['spec_sha256'] == sha(root / 'spec.json')
    assert receipt['events'] == 10615 and not receipt['labels_opened']
    cases = {c['sequence']: c for c in json.loads((parent / 'inference_inputs.json').read_text()) if c['split'] == 'fit'}
    assert len(receipt['sequences']) == len(cases) == 63
    assert {r['sequence'] for r in receipt['sequences']} == set(cases)
    packets = []
    for item in sorted(receipt['sequences'], key=lambda x: x['sequence']):
        path = root / 'features' / (item['sequence'] + '.pt')
        assert sha(path) == item['feature_sha256']
        data = torch.load(path, map_location='cpu')
        case = cases[item['sequence']]
        assert data['sequence'] == case['sequence'] and data['split'] == 'fit' and data['fold'] == case['fold']
        assert data['spec_sha256'] == sha(root / 'spec.json')
        assert [r['frame'] for r in data['records']] == event_frames(case)
        assert all(r['key'] == f'{case["sequence"]}@{r["frame"]}' and r['previous_frame'] == r['frame'] - 1 for r in data['records'])
        assert all(torch.isfinite(data[k]).all() for k in READER_FIELDS + ['boxes'])
        packets.append(data)

    # Only now open GT: all feature packets and event identities are sealed.
    arrays = {k: [] for k in READER_FIELDS}
    keys, ious, valid, gt_hashes = [], [], [], {}
    for data in packets:
        name = data['sequence']
        path = Path(spec['dataset_root']) / name / 'groundtruth.txt'
        gt_hashes[name] = sha(path)
        gt = np.loadtxt(path, delimiter=',')
        for key in READER_FIELDS:
            arrays[key].append(data[key])
        for i, row in enumerate(data['records']):
            target = gt[row['frame']]
            ok = bool(np.isfinite(target).all() and (target[2:] > 0).all())
            valid.append(ok)
            ious.append(overlaps(data['boxes'][i], torch.tensor(target)) if ok else torch.zeros(2, dtype=torch.float64))
            keys.append(row['key'])
    tensors = {k: torch.cat(v) for k, v in arrays.items()}
    del arrays, packets, data
    ious = torch.stack(ious)
    fit = torch.tensor([i for i, ok in enumerate(valid) if ok], dtype=torch.long)
    target = ((ious[:, 0] < .5) & (ious[:, 1] >= .5)).long()
    assert len(keys) == 10615 and len(set(keys)) == len(keys)
    assert target[fit].unique().tolist() == [0, 1]
    opt = plan['optimization']
    torch.set_num_threads(1)
    torch.manual_seed(opt['seed'])
    torch.cuda.manual_seed_all(opt['seed'])
    model = TemplateReader().cuda()
    initial = {k: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt['lr'], weight_decay=opt['weight_decay'])
    order = torch.Generator().manual_seed(opt['seed'])
    order_hash = hashlib.sha256()
    losses, steps = [], 0
    started = time.time()

    def inputs(index):
        return [tensors[k][index].cuda().float() for k in READER_FIELDS]

    for epoch in range(opt['epochs']):
        model.train()
        shuffled = fit[torch.randperm(len(fit), generator=order)]
        order_hash.update(shuffled.numpy().tobytes())
        total = 0.
        for batch in shuffled.split(opt['batch_size']):
            logits = model(*inputs(batch))
            loss = F.cross_entropy(logits, target[batch].cuda())
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), opt['grad_clip']))
            optimizer.step()
            steps += 1
            total += float(loss.detach()) * len(batch)
        row = dict(epoch=epoch + 1, loss=total / len(fit), optimizer_steps=steps, elapsed_seconds=time.time() - started)
        losses.append(row)
        print(json.dumps(row), flush=True)
    assert steps == opt['epochs'] * ((len(fit) + opt['batch_size'] - 1) // opt['batch_size'])
    checkpoint = root / 'reader_final.pth'
    torch.save(dict(model=model.state_dict(), base_checkpoint_sha256=spec['checkpoint_sha256'],
        spec_sha256=sha(root / 'spec.json'), collection_receipt_sha256=sha(root / 'collection_receipt.json'),
        epochs=opt['epochs'], optimizer_steps=steps, target_rule='alternate_only_if_native_below_half_and_alternate_at_least_half'), checkpoint)

    def evaluate(net):
        net.eval()
        with torch.no_grad():
            return torch.cat([net(*inputs(b)).cpu() for b in fit.split(opt['batch_size'])])

    logits = evaluate(model)
    reloaded = TemplateReader().cuda()
    reloaded.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'], strict=True)
    assert torch.equal(logits, evaluate(reloaded))
    chosen = logits.argmax(1)
    default = ious[fit, 0]
    selected = ious[fit].gather(1, chosen[:, None]).flatten()
    changed = [k for k, v in model.state_dict().items() if hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() != initial[k]]
    check_sources(root)
    result = dict(status='complete', scope='Fitting-only training; recursive development evaluation still required',
        spec_sha256=sha(root / 'spec.json'), collection_receipt_sha256=sha(root / 'collection_receipt.json'),
        checkpoint_sha256=sha(checkpoint), base_checkpoint_sha256=spec['checkpoint_sha256'],
        parameters=sum(p.numel() for p in model.parameters()), events=len(keys), valid_events=len(fit),
        invalid_events=len(keys) - len(fit), alternate_targets=int(target[fit].sum()), optimizer_steps=steps,
        epochs=opt['epochs'], sample_order_sha256=order_hash.hexdigest(), initial_state_sha256=initial,
        changed_tensors=changed, strict_reload_exact=True, labels_read_after_sealed_inputs_verified=True,
        gt_sha256=gt_hashes, losses=losses, default_mean_iou=float(default.mean()), selected_mean_iou=float(selected.mean()),
        selected_alternate=int(chosen.sum()), severe_rescues=int(((default <= .1) & (selected >= .5)).sum()),
        severe_harms=int(((default >= .5) & (selected <= .1)).sum()), development_used_for_fitting=False,
        public_evaluation=False, elapsed_seconds=time.time() - started)
    (root / 'training_result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['losses', 'gt_sha256', 'initial_state_sha256']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
