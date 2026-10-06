"""Matched M118 human, generic, and visual-query training on actual Train IoU."""
import argparse, hashlib, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from analyze_train_states import sha
from dense_region_decoder import DenseRegionDecoder, sample_regions
from region_patch_evidence import region_samples
from m118_region_inputs import load_panel, inputs
from train_ab_visual_control import summarize
from train_m113_human_initialization import paired_vs_empty


CONDITIONS = ['empty', 'generic', 'visual_query', 'human_text']


def quality_order(logits, target):
    difference = (target[:, :, None] - target[:, None, :]).clamp_min(0.)
    gap = logits[:, :, None] - logits[:, None, :]
    loss = (F.relu(difference - gap) * difference).sum() / difference.sum().clamp_min(1.)
    return loss, (difference > 0).sum()


def parent_preservation(logits, teacher, target):
    index = teacher.argmax(-1, keepdim=True)
    parent_iou = target.gather(-1, index)
    eligible = (parent_iou >= .5) & (target < parent_iou)
    parent_gap = teacher.gather(-1, index) - teacher
    gap = logits.gather(-1, index) - logits
    return (F.relu(parent_gap - gap) * eligible).sum() / eligible.sum().clamp_min(1), eligible.sum()


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def evaluate(model, panel, initial, bank, condition, device):
    rows = []; model.eval()
    with torch.no_grad():
        for start in range(0, len(panel['key']), 64):
            ids = torch.arange(start, min(start + 64, len(panel['key'])))
            data = inputs(panel, ids, initial, bank, condition, device)
            out = model(data); selected = out['selected_index']
            assert all(bool(torch.isfinite(v).all()) for v in out.values())
            assert torch.equal(out['quality_logits'], data['parent_quality'])
            if condition == 'empty':
                assert torch.equal(out['selection_logits'], data['parent_scores'])
                assert not bool(out['semantic_score_delta'].any())
            row = torch.arange(len(ids), device=device)
            assert torch.equal(out['selected_box'], data['boxes'][row, selected])
            assert torch.equal(out['selected_score'], out['selection_logits'][row, selected])
            assert torch.equal(out['selected_quality'], out['quality_logits'][row, selected])
            assert torch.equal(out['selected_feature'], out['candidate_features'][row, selected])
            for j, i in enumerate(ids.tolist()):
                iou = panel['iou'][i]; pick = int(selected[j]); parent = int(panel['parent_scores'][i].argmax())
                rows.append(dict(key=panel['key'][i], strata=panel['strata'][i],
                    native_iou=float(iou[0]), selected_iou=float(iou[pick]), oracle_iou=float(iou.max()),
                    selected=pick, parent_selected=parent, parent_iou=float(iou[parent]),
                    candidate_iou=iou.tolist(), parent_scores=panel['parent_scores'][i].tolist(),
                    scores=out['selection_logits'][j].cpu().tolist(),
                    semantic_score_delta=out['semantic_score_delta'][j].cpu().tolist()))
    return summarize(rows), rows


def spatial_sanity(model, fit, initial, bank, device):
    data = inputs(fit, torch.arange(3), initial, bank, 'human_text', device)
    for ring in [False, True]:
        xy = (model.roi_xy[[0,1,2,3,4,7,8,11,12,13,14,15]] + 1.) * .5 if ring else (model.roi_xy + 1.) * .5
        actual, mask = sample_regions(data['dense_grid'], data['dense_valid'], data['boxes'], data['dense_origin'], xy, ring)
        for i in range(3):
            expected, expected_mask = region_samples(data['dense_grid'][i], data['dense_valid'][i],
                                data['boxes'][i], data['dense_origin'][i].tolist(), ring)
            torch.testing.assert_close(actual[i], expected)
            torch.testing.assert_close(mask[i], expected_mask)
    return dict(original_M117_region_ring_reader_default_close=True, tested_states=3)


def main():
    parser = argparse.ArgumentParser()
    for key in ['cache','contexts','origins','dense','bank','labels','parent','parent-result','output']:
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--arm', choices=['human_text','generic','visual_query'], required=True)
    parser.add_argument('--mode', choices=['sanity','train'], required=True)
    args = parser.parse_args(); assert not args.output.exists()
    torch.set_num_threads(1); random.seed(2027); np.random.seed(2027)
    torch.manual_seed(2027); torch.cuda.manual_seed_all(2027); device = torch.device('cuda')
    bank = torch.load(args.bank, map_location='cpu'); labels = json.loads(args.labels.read_text())
    assert bank['human_confirmed'] and labels['human_confirmed']
    assert bank['labels_sha256'] == sha(args.labels) and bank['dataset'] == labels['dataset'] == 'depthtrack'
    assert bank['sequences'] == [r['sequence'] for r in labels['initial']]
    assert bank['splits'] == [r['split'] for r in labels['initial']]
    assert len(bank['sequences']) == 152
    dense_receipt = json.loads((args.dense/'result.json').read_text())
    assert all(r['bank_sha256'] == sha(args.bank) and r['labels_sha256'] == sha(args.labels)
               and r['encoder_sha256'] == bank['encoder_sha256'] for r in dense_receipt['full'])
    started = time.time(); panel, initial = load_panel(args, bank, device)
    for split, part in panel.items():
        assert all(bank['splits'][bank['sequences'].index(key.rsplit('@',1)[0])] == split for key in part['key'])
    torch.manual_seed(2027); torch.cuda.manual_seed_all(2027)
    model = DenseRegionDecoder().to(device).eval()
    assert sum(p.numel() for p in model.parameters()) == 150528
    initial_sha = state_digest(model)
    buffers = {k: v.cpu().clone() for k, v in model.named_buffers()}
    coordinates = spatial_sanity(model, panel['fit'], initial, bank, device)
    start_rows = {}
    for split, part in panel.items():
        _, start_rows[split] = evaluate(model, part, initial, bank, 'empty', device)
    fit = panel['fit']; optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    history = []; gradients = []; epochs = 12 if args.mode == 'train' else 1
    for epoch in range(epochs):
        permutation = torch.randperm(len(fit['key']), generator=torch.Generator().manual_seed(2027+epoch))
        batches = permutation.split(64) if args.mode == 'train' else [permutation[:64]] * 3
        terms = []; quality_pairs = preserve_pairs = 0
        for ids in batches:
            optimizer.zero_grad(set_to_none=True)
            data = inputs(fit, ids, initial, bank, args.arm, device); out = model(data)
            target = fit['iou'][ids].to(device)
            order, pairs = quality_order(out['selection_logits'], target)
            keep, protected = parent_preservation(out['selection_logits'], data['parent_scores'], target)
            loss = order + keep
            assert bool(torch.isfinite(loss)); loss.backward()
            assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
            if args.mode == 'sanity':
                gradients.append({name: sum(float(p.grad.norm()) for k,p in model.named_parameters() if k.startswith(name+'.') and p.grad is not None)
                    for name in ['region_projection','query_projection','query_read','visual_read','decision']})
            optimizer.step(); terms.append([float(x.detach()) for x in [loss, order, keep]])
            quality_pairs += int(pairs); preserve_pairs += int(protected)
        history.append(dict(epoch=epoch+1, optimizer_steps=len(terms), mean_losses=np.mean(terms,axis=0).tolist(),
                            quality_pairs=quality_pairs, preserve_pairs=preserve_pairs, seconds=time.time()-started))
        print(json.dumps(history[-1]), flush=True)
    assert all(torch.equal(v.cpu(),buffers[k]) for k,v in model.named_buffers())
    conditions = {}; summaries = {}
    if args.mode == 'train':
        for condition in CONDITIONS:
            summaries[condition], conditions[condition] = evaluate(model, panel['development'], initial, bank, condition, device)
    for split, part in panel.items():
        _, final_empty = evaluate(model, part, initial, bank, 'empty', device)
        assert final_empty == start_rows[split]
    result = dict(status='complete_M118_regional_training' if args.mode=='train' else 'complete_M118_gpu_sanity',
        arm=args.arm, mode=args.mode, seed=2027, epochs=epochs, batch_size=64, learning_rate=3e-4,
        optimized_parameters=150528, optimizer_steps=sum(r['optimizer_steps'] for r in history),
        initial_state_sha256=initial_sha, human_confirmed_initialization=True, current_candidate_semantic_labels=0,
        bank_sha256=sha(args.bank), labels_sha256=sha(args.labels), parent_sha256=sha(args.parent),
        dense_driver_sha256=sha(args.dense/'result.json'), source_sha256=sha(__file__),
        decoder_sha256=sha(Path(__file__).with_name('dense_region_decoder.py')),
        inputs_sha256=sha(Path(__file__).with_name('m118_region_inputs.py')),
        loss_components=['GT_IoU_weighted_candidate_quality_order','reliable_Parent_gap_vs_worse_geometry'],
        metric_definitions=dict(summarize_native='rescues/breaks cross IoU0.5 relative to original native candidate0 (<0.5 versus >=0.5).',
            paired_vs_own_empty='rescue/harm count severe changes between <=0.1 and >=0.5; qualifying counts and mean IoU differences are separate fields.'),
        history=history, coordinates=coordinates, frozen_parent_actual_forward=True,
        empty_scores_quality_index_box_all3039_exact=True, buffers_exact=True, nonempty_quality_exact=True,
        selected_feature_boundary='The learned277-dimensional feature matches this forward selected candidate; it is not frozen or identical to the64-dimensional Parent feature. No tracker consumes it here.',
        visual_query_control='Initial local dense RGB representation repeated across the same valid query slots; no category/attribute strings.',
        no_recursive_or_public_evaluation=True, elapsed_seconds=time.time()-started,
        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    args.output.mkdir()
    if args.mode == 'sanity':
        assert all(v>0 for v in gradients[-1].values()), gradients
        result.update(gradient_norms=gradients, checkpoint_saved=False, development_metrics_reported=False)
    else:
        result['fit'], fit_rows = evaluate(model, fit, initial, bank, args.arm, device)
        (args.output/'fit_events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in fit_rows))
        for condition, rows in conditions.items():
            (args.output/(condition+'_development.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        result.update(content_conditions=summaries, development=summaries[args.arm],
            paired_vs_own_empty={name:paired_vs_empty(rows,conditions['empty']) for name,rows in conditions.items() if name!='empty'})
        pair = result['paired_vs_own_empty'][args.arm]
        result['fixed_state_checks']=dict(correct_exceeds_own_empty=pair['all']['condition_iou50']>pair['all']['empty_iou50'],
            mean_iou_exceeds_own_empty=pair['all']['mean_iou_delta']>0,
            healthy_correct_not_lower=pair['healthy']['condition_iou50']>=pair['healthy']['empty_iou50'],
            healthy_mean_not_lower=pair['healthy']['mean_iou_delta']>=0,
            transition_correct_not_lower=pair['transition']['condition_iou50']>=pair['transition']['empty_iou50'])
        torch.save(model.state_dict(),args.output/'final.pt')
        reloaded=torch.load(args.output/'final.pt',map_location='cpu')
        assert all(torch.equal(v.cpu(),reloaded[k]) for k,v in model.state_dict().items())
        result.update(final_state_roundtrip_exact=True, final_sha256=sha(args.output/'final.pt'))
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],arm=args.arm,steps=result['optimizer_steps'])),flush=True)


if __name__ == '__main__':
    main()
