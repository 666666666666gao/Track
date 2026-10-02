"""M106 selected-geometry preservation; same M104 visual refiner and frozen parent."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from analyze_train_states import overlaps, sha
from instance_ab_prototype import InstanceCandidatePrototype
from train_ab_visual_control import load_inputs, batch, summarize


class LocalVisualRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(64, 16)
        self.regression = nn.Sequential(nn.Linear(56 * 16 + 13, 64), nn.GELU(), nn.Linear(64, 4))
        nn.init.zeros_(self.regression[-1].weight)
        nn.init.zeros_(self.regression[-1].bias)

    def forward(self, tokens, valid, geometry):
        local = self.projection(tokens) * valid[..., None]
        return self.regression(torch.cat((local.flatten(-2), geometry), -1))


def decode(boxes, delta, image_shape):
    increase = boxes[..., 2:] * torch.expm1(delta[..., 2:])
    shift = delta[..., :2] * boxes[..., 2:] - .5 * increase
    raw = boxes + torch.cat((shift, increase), -1)
    height, width = image_shape[:, 0, None], image_shape[:, 1, None]
    left = torch.minimum(raw[..., 0].clamp_min(0), width - 10)
    top = torch.minimum(raw[..., 1].clamp_min(0), height - 10)
    raw_right = raw[..., 0]+raw[..., 2]
    raw_bottom = raw[..., 1]+raw[..., 3]
    right = torch.minimum(raw_right.clamp_min(10), width)
    bottom = torch.minimum(raw_bottom.clamp_min(10), height)
    corrected_width = (raw[..., 2]+(right-raw_right)-(left-raw[..., 0])).clamp_min(10)
    corrected_height = (raw[..., 3]+(bottom-raw_bottom)-(top-raw[..., 1])).clamp_min(10)
    return torch.stack((left, top, corrected_width, corrected_height), -1)


def normalized_target(boxes, target):
    center = boxes[:, :2] + .5 * boxes[:, 2:]
    target_center = target[:, :2] + .5 * target[:, 2:]
    return torch.cat(((target_center-center) / boxes[:, 2:], torch.log(target[:, 2:] / boxes[:, 2:])), -1)


def metadata(panel, cache, geometry_probe):
    labels = json.loads((cache/'training_labels.json').read_text())
    probes = {r['key']:r for r in (json.loads(s) for s in (geometry_probe/'events.jsonl').read_text().splitlines())}
    shapes = {}
    for shard in (0, 1):
        for item in json.loads((cache/f'collect_shard{shard}.json').read_text())['sequences']:
            data = torch.load(cache/'features'/f"{item['sequence']}.pt", map_location='cpu')
            for index, frame in enumerate(data['event_frames']):
                shapes[f"{item['sequence']}@{frame}"] = data['image_shape'][index]
    return {split:dict(target=torch.tensor([labels[k]['current'] for k in group['key']], dtype=torch.float32),
                       eligible=torch.tensor([probes[k]['refinement_eligible'] for k in group['key']]),
                       image_shape=torch.stack([shapes[k] for k in group['key']]).float())
            for split,group in panel.items()}


def encoded_inputs(parent, panel, empty, device):
    fields = defaultdict(list)
    with torch.no_grad():
        for start in range(0, len(panel['key']), 64):
            indices = torch.arange(start, min(start+64, len(panel['key'])))
            inputs = batch(panel, indices, empty, device)
            current, valid, _ = parent.evidence.encode(inputs['candidate_rois'], inputs['candidate_mask'], inputs['depth_valid'], 2)
            context, context_valid, _ = parent.evidence.encode(inputs['contexts'], inputs['context_mask'], inputs['depth_valid'], 3, True)
            output = parent(inputs)
            assert torch.equal(output['selection_logits'], output['visual_selection_logits'])
            fields['tokens'].append(torch.cat((current, context), -2).cpu())
            fields['valid'].append(torch.cat((valid, context_valid), -1).cpu())
            fields['selected'].append(output['selected_index'].cpu())
            fields['selection_logits'].append(output['selection_logits'].cpu())
    return {key:torch.cat(values) for key,values in fields.items()}


def evaluate(model, panel, encoded, meta, device, reference_only=False):
    rows = []
    with torch.no_grad():
        for start in range(0, len(panel['key']), 64):
            indices = torch.arange(start, min(start+64, len(panel['key'])))
            boxes = panel['boxes'][indices].to(device).float()
            if reference_only:
                refined = boxes
            else:
                delta = model(encoded['tokens'][indices].to(device), encoded['valid'][indices].to(device), panel['geometry'][indices].to(device).float())
                refined = decode(boxes, delta, meta['image_shape'][indices].to(device))
                assert bool(torch.isfinite(refined).all())
            refined = refined.cpu()
            for offset,index in enumerate(indices.tolist()):
                old = panel['iou'][index]
                after = overlaps(refined[offset], meta['target'][index])
                selected = int(encoded['selected'][index])
                rows.append(dict(key=panel['key'][index], strata=panel['strata'][index],
                                 selected=selected, native_iou=float(old[0]), parent_iou=float(old[selected]),
                                 selected_iou=float(after[selected]), oracle_iou=float(after.max()),
                                 native_top10_iou=float(old.max())))
    summary = summarize(rows)
    for tag, values in [('all', rows)] + [(tag, [r for r in rows if tag in r['strata']]) for tag in summary if tag != 'all']:
        summary[tag].update(parent_correct=sum(r['parent_iou']>=.5 for r in values),
                            parent_mean_iou=sum(r['parent_iou'] for r in values)/len(values),
                            parent_rescues=sum(r['parent_iou']<.5 and r['selected_iou']>=.5 for r in values),
                            parent_breaks=sum(r['parent_iou']>=.5 and r['selected_iou']<.5 for r in values),
                            original_top10_correct=sum(r['native_top10_iou']>=.5 for r in values),
                            original_miss_refined_hit=sum(r['native_top10_iou']<.5 and r['oracle_iou']>=.5 for r in values))
    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    for name in ('cache','contexts','origins','empty-bank','geometry-probe','parent-weight','parent-result','repository','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--mode', choices=('sanity','train','reference'), required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--selected-preservation-weight', type=int, choices=(0,1), required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    sys.path.insert(0, str(args.repository))
    from lib.utils.box_ops import giou_loss, box_xywh_to_xyxy
    torch.set_num_threads(1)
    random.seed(2027); np.random.seed(2027); torch.manual_seed(2027); torch.cuda.manual_seed_all(2027)
    started = time.monotonic()
    parent_report = json.loads(args.parent_result.read_text())
    assert parent_report['status']=='complete_visual_control' and parent_report['native_preservation_weight']==1
    assert sha(args.parent_weight)==parent_report['final_weights_sha256']
    panel = load_inputs(args.cache, args.contexts, args.origins, False)
    meta = metadata(panel, args.cache, args.geometry_probe)
    assert int(meta['fit']['eligible'].sum())==2033 and int(meta['development']['eligible'].sum())==346
    device = torch.device(args.device)
    empty = torch.load(args.empty_bank, map_location='cpu')['empty'].to(device).float()
    parent = InstanceCandidatePrototype().to(device)
    parent.load_state_dict(torch.load(args.parent_weight, map_location=device))
    parent.eval().requires_grad_(False)
    frozen = {name:value.detach().cpu().clone() for name,value in parent.state_dict().items()}
    encoded = {split:encoded_inputs(parent, group, empty, device) for split,group in panel.items()}
    model = LocalVisualRefiner().to(device)
    zero_checked = 0
    with torch.no_grad():
        for split,group in panel.items():
            for start in range(0, len(group['key']), 64):
                at = torch.arange(start, min(start+64, len(group['key'])))
                delta = model(encoded[split]['tokens'][at].to(device), encoded[split]['valid'][at].to(device), group['geometry'][at].to(device).float())
                assert bool((delta==0).all())
                output = decode(group['boxes'][at].to(device).float(), delta, meta[split]['image_shape'][at].to(device))
                assert torch.equal(output, group['boxes'][at].to(device).float())
                zero_checked += len(at)
    assert zero_checked==3039
    reference = {r['key']:r for r in parent_report['development_rows']}
    for index,key in enumerate(panel['development']['key']):
        choice = int(encoded['development']['selected'][index])
        assert choice==reference[key]['selected'] and float(panel['development']['iou'][index,choice])==reference[key]['selected_iou']
    fit = panel['fit']; fit_meta = meta['fit']; fit_encoded = encoded['fit']
    eligible = fit_meta['eligible'].nonzero().flatten()
    best = fit['iou'].argmax(-1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    history = []; gradients = {}
    if args.mode != 'reference':
        epochs = 12 if args.mode=='train' else 1
        for epoch in range(epochs):
            order = eligible[torch.randperm(len(eligible), generator=torch.Generator().manual_seed(2027+epoch))]
            batches = order.split(64) if args.mode=='train' else [order[:64], order[:64]]
            losses=[]; preservation_losses=[]; reliable_counts=[]; epoch_start=time.monotonic()
            for step,indices in enumerate(batches):
                choice = best[indices]
                tokens = fit_encoded['tokens'][indices,choice].to(device)
                valid = fit_encoded['valid'][indices,choice].to(device)
                geometry = fit['geometry'][indices,choice].to(device).float()
                boxes = fit['boxes'][indices,choice].to(device).float()
                target = fit_meta['target'][indices].to(device)
                optimizer.zero_grad(set_to_none=True)
                delta = model(tokens, valid, geometry)
                decoded = decode(boxes[:,None], delta[:,None], fit_meta['image_shape'][indices].to(device))[:,0]
                giou, _ = giou_loss(box_xywh_to_xyxy(decoded), box_xywh_to_xyxy(target))
                loss = F.smooth_l1_loss(delta, normalized_target(boxes, target)) + 2*giou
                preservation = loss.new_zeros(())
                reliable_count = 0
                if args.selected_preservation_weight == 1:
                    selected = fit_encoded['selected'][indices]
                    before = fit['iou'][indices,selected].to(device)
                    reliable = before >= .5
                    selected_delta = model(fit_encoded['tokens'][indices,selected].to(device),
                                           fit_encoded['valid'][indices,selected].to(device),
                                           fit['geometry'][indices,selected].to(device).float())
                    selected_boxes = decode(fit['boxes'][indices,selected].to(device).float()[:,None],
                                            selected_delta[:,None], fit_meta['image_shape'][indices].to(device))[:,0]
                    _, after = giou_loss(box_xywh_to_xyxy(selected_boxes), box_xywh_to_xyxy(target))
                    preservation = ((before-after).clamp_min(0) * reliable).mean()
                    reliable_count = int(reliable.sum())
                    loss = loss + preservation
                preservation_losses.append(float(preservation.detach()))
                reliable_counts.append(reliable_count)
                assert bool(torch.isfinite(loss))
                loss.backward()
                assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
                if args.mode=='sanity':
                    gradients[f'step{step+1}_final'] = float(model.regression[-1].weight.grad.norm())
                    if step==1:
                        gradients['step2_projection'] = float(model.projection.weight.grad.norm())
                        assert gradients['step2_projection']>0
                    assert gradients[f'step{step+1}_final']>0
                optimizer.step(); losses.append(float(loss.detach()))
            row=dict(epoch=epoch+1,optimizer_steps=len(losses),mean_loss=sum(losses)/len(losses),seconds=time.monotonic()-epoch_start)
            row.update(selected_preservation_mean=sum(preservation_losses)/len(preservation_losses),
                       reliable_selected_fit_calls=sum(reliable_counts))
            history.append(row); print(json.dumps(row),flush=True)
    assert all(torch.equal(value.detach().cpu(), frozen[name]) for name,value in parent.state_dict().items())
    result=dict(status='complete_selected_geometry_sanity' if args.mode=='sanity' else 'complete_selected_geometry_preservation',
                mode=args.mode,seed=2027,learning_rate=3e-4,batch_size=64,eligible_fit_events=len(eligible),
                selected_preservation_weight=args.selected_preservation_weight,
                selected_preservation_scope='Fitting current GT IoU only; no identity labels or inference gate',
                optimized_parameters=sum(p.numel() for p in model.parameters()) if args.mode!='reference' else 0,
                optimizer_steps=sum(r['optimizer_steps'] for r in history),history=history,
                actual3039zero_geometry_exact=True,actual495parent_selection_exact=True,parent_parameters_buffers_unchanged=True,
                no_GT_forward_gate=True,no_semantic_identity_labels_or_public_or_recursive_action=True,
                parent_weight_sha256=sha(args.parent_weight),source_sha256=sha(__file__))
    args.output.mkdir()
    if args.mode=='sanity':
        result.update(gradient_norms=gradients,development_evaluated=False,checkpoint_saved=False)
    else:
        for split in ('fit','development'):
            result[split],rows = evaluate(model,panel[split],encoded[split],meta[split],device,args.mode=='reference')
            (args.output/(split+'_events.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        dev=result['development']
        result['fixed_state_checks']=dict(correct_exceeds_parent=dev['all']['selected_iou50']>dev['all']['parent_correct'],
            mean_iou_exceeds_parent=dev['all']['selected_mean_iou']>dev['all']['parent_mean_iou'],
            healthy_new_breaks_zero=dev['healthy']['parent_breaks']==0,
            transition_at_least_parent=dev['transition']['selected_iou50']>=dev['transition']['parent_correct'])
        if args.mode=='train':
            torch.save(model.state_dict(),args.output/'final.pt')
            reloaded=LocalVisualRefiner().to(device)
            reloaded.load_state_dict(torch.load(args.output/'final.pt',map_location=device))
            for split in ('fit','development'):
                summary,actual=evaluate(reloaded,panel[split],encoded[split],meta[split],device)
                saved=[json.loads(s) for s in (args.output/(split+'_events.jsonl')).read_text().splitlines()]
                assert actual==saved and summary==result[split]
            result['final_reload_all3039rows_exact']=True
            result['final_weight_sha256']=sha(args.output/'final.pt')
    result['elapsed_seconds']=time.monotonic()-started
    result['gpu_peak_allocated_bytes']=torch.cuda.max_memory_allocated(device)
    result['gpu_peak_reserved_bytes']=torch.cuda.max_memory_reserved(device)
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)


if __name__=='__main__':
    main()
