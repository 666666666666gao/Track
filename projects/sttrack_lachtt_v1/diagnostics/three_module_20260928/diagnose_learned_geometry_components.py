"""M105 same-final center/size readouts; no training or tracker actions."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import torch

from analyze_train_states import overlaps, sha
from instance_ab_prototype import InstanceCandidatePrototype
from train_ab_visual_control import load_inputs
from train_local_visual_geometry import LocalVisualRefiner, decode, encoded_inputs, metadata


VARIANTS = ('parent', 'full', 'center_only', 'size_only')


def summarize(rows):
    base = [row['selected_iou']['parent'] for row in rows]
    return {name:dict(
        events=len(rows),
        selected_correct=sum(row['selected_iou'][name] >= .5 for row in rows),
        selected_mean_iou=sum(row['selected_iou'][name] for row in rows) / len(rows),
        parent_rescues=sum(old < .5 and row['selected_iou'][name] >= .5 for old, row in zip(base, rows)),
        parent_breaks=sum(old >= .5 and row['selected_iou'][name] < .5 for old, row in zip(base, rows)),
        oracle_correct=sum(max(row['candidate_iou'][name]) >= .5 for row in rows),
        original_miss_hit=sum(max(row['candidate_iou']['parent']) < .5 and max(row['candidate_iou'][name]) >= .5 for row in rows),
        original_hit_miss=sum(max(row['candidate_iou']['parent']) >= .5 and max(row['candidate_iou'][name]) < .5 for row in rows),
    ) for name in VARIANTS}


def main():
    parser = argparse.ArgumentParser()
    for name in ('cache', 'contexts', 'origins', 'empty-bank', 'geometry-probe',
                 'parent-weight', 'parent-result', 'm104-directory', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--split', choices=('fit', 'development'), required=True)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(1)
    started = time.monotonic()
    parent_report = json.loads(args.parent_result.read_text())
    original_result = json.loads((args.m104_directory / 'result.json').read_text())
    original_rows = [json.loads(line) for line in (args.m104_directory / (args.split + '_events.jsonl')).read_text().splitlines()]
    assert parent_report['status'] == 'complete_visual_control' and parent_report['native_preservation_weight'] == 1
    assert original_result['status'] == 'complete_fixed_state_visual_geometry' and original_result['mode'] == 'train'
    assert sha(args.parent_weight) == parent_report['final_weights_sha256'] == original_result['parent_weight_sha256']
    weights = args.m104_directory / 'final.pt'
    assert sha(weights) == original_result['final_weight_sha256']
    panel = load_inputs(args.cache, args.contexts, args.origins, False)
    meta = metadata(panel, args.cache, args.geometry_probe)[args.split]
    group = panel[args.split]
    assert len(group['key']) == (2544 if args.split == 'fit' else 495)
    assert [row['key'] for row in original_rows] == group['key']
    device = torch.device(args.device)
    empty = torch.load(args.empty_bank, map_location='cpu')['empty'].to(device).float()
    parent = InstanceCandidatePrototype().to(device)
    parent.load_state_dict(torch.load(args.parent_weight, map_location=device))
    parent.eval().requires_grad_(False)
    model = LocalVisualRefiner().to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval().requires_grad_(False)
    frozen_parent = {name:value.detach().cpu().clone() for name,value in parent.state_dict().items()}
    frozen_model = {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    encoded = encoded_inputs(parent, group, empty, device)
    rows = []
    with torch.no_grad():
        for start in range(0, len(group['key']), 64):
            at = torch.arange(start, min(start + 64, len(group['key'])))
            boxes = group['boxes'][at].to(device).float()
            delta = model(encoded['tokens'][at].to(device), encoded['valid'][at].to(device), group['geometry'][at].to(device).float())
            center_delta = torch.cat((delta[..., :2], torch.zeros_like(delta[..., 2:])), -1)
            size_delta = torch.cat((torch.zeros_like(delta[..., :2]), delta[..., 2:]), -1)
            shapes = meta['image_shape'][at].to(device)
            decoded = {
                'parent': boxes,
                'full': decode(boxes, delta, shapes),
                'center_only': decode(boxes, center_delta, shapes),
                'size_only': decode(boxes, size_delta, shapes),
            }
            assert torch.equal(decode(boxes, torch.zeros_like(delta), shapes), boxes)
            assert all(bool(torch.isfinite(value).all()) for value in decoded.values())
            decoded = {name:value.cpu() for name,value in decoded.items()}
            for offset, index in enumerate(at.tolist()):
                selection = int(encoded['selected'][index])
                candidate_iou = {name:overlaps(value[offset], meta['target'][index]).tolist() for name,value in decoded.items()}
                selected_iou = {name:values[selection] for name,values in candidate_iou.items()}
                expected = original_rows[index]
                assert group['strata'][index] == expected['strata']
                assert selection == expected['selected']
                assert float(group['iou'][index, 0]) == expected['native_iou']
                assert selected_iou['parent'] == expected['parent_iou']
                assert selected_iou['full'] == expected['selected_iou']
                assert max(candidate_iou['parent']) == expected['native_top10_iou']
                assert max(candidate_iou['full']) == expected['oracle_iou']
                row = dict(key=group['key'][index], strata=group['strata'][index], selected=selection,
                           native_iou=expected['native_iou'], selected_iou=selected_iou, candidate_iou=candidate_iou,
                           selected_boxes={name:value[offset, selection].tolist() for name,value in decoded.items()},
                           selected_delta=delta[offset, selection].cpu().tolist())
                rows.append(row)
    assert all(torch.equal(value.detach().cpu(), frozen_parent[name]) for name,value in parent.state_dict().items())
    assert all(torch.equal(value.detach().cpu(), frozen_model[name]) for name,value in model.state_dict().items())
    groups = defaultdict(list)
    profiles = defaultdict(list)
    for row in rows:
        groups['all'].append(row)
        for tag in row['strata']:
            groups[tag].append(row)
        profiles['|'.join(sorted(row['strata']))].append(row)
    summary = {tag:summarize(values) for tag,values in groups.items()}
    joints = {profile:summarize(values) for profile,values in profiles.items()}
    assert sum(values['parent']['events'] for values in joints.values()) == len(rows)
    for tag, values in summary.items():
        expected = original_result[args.split][tag]
        members = groups[tag]
        actual = dict(valid_gt=len(members),
                      native_iou50=sum(row['native_iou'] >= .5 for row in members),
                      selected_iou50=values['full']['selected_correct'],
                      oracle_iou50=values['full']['oracle_correct'],
                      rescues=sum(row['native_iou'] < .5 and row['selected_iou']['full'] >= .5 for row in members),
                      breaks=sum(row['native_iou'] >= .5 and row['selected_iou']['full'] < .5 for row in members),
                      native_mean_iou=sum(row['native_iou'] for row in members) / len(members),
                      selected_mean_iou=values['full']['selected_mean_iou'],
                      parent_correct=values['parent']['selected_correct'],
                      parent_mean_iou=values['parent']['selected_mean_iou'],
                      parent_rescues=values['full']['parent_rescues'],
                      parent_breaks=values['full']['parent_breaks'],
                      original_top10_correct=values['parent']['oracle_correct'],
                      original_miss_refined_hit=values['full']['original_miss_hit'])
        assert actual == expected, tag
    result = dict(status='complete_same_final_geometry_components', split=args.split, events=len(rows),
                  variants=list(VARIANTS), summaries=summary, joint_profiles=joints,
                  full_replay_original_rows_and_summaries_exact=True, actual_all_zero_geometry_exact=True,
                  parent_and_refiner_parameters_buffers_unchanged=True, optimizer_steps=0, checkpoint_saved=False,
                  no_GT_in_geometry_forward=True, no_tracker_memory_semantic_or_public_action=True,
                  interpretation='Read-only component intervention on the same final; center/size clipping may interact. No retrained ablation or official metric.',
                  parent_weight_sha256=sha(args.parent_weight), refiner_weight_sha256=sha(weights),
                  source_sha256=sha(__file__), elapsed_seconds=time.monotonic() - started,
                  gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
    args.output.mkdir()
    (args.output / 'events.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], split=args.split, events=len(rows),
                          all=summary['all'], healthy=summary['healthy'], seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
