"""M103 cached native geometry arithmetic, never a deployed GT correction."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import torch

from analyze_train_states import overlaps, sha


def clipped(boxes, image_shape):
    height, width = [float(x) for x in image_shape]
    left = boxes[:, 0].clamp(0, width - 10)
    top = boxes[:, 1].clamp(0, height - 10)
    right = (boxes[:, 0] + boxes[:, 2]).clamp(10, width)
    bottom = (boxes[:, 1] + boxes[:, 3]).clamp(10, height)
    return torch.stack((left, top, (right-left).clamp_min(10),
                        (bottom-top).clamp_min(10)), dim=-1)


def interventions(boxes, target, image_shape):
    center = boxes[:, :2] + .5 * boxes[:, 2:]
    gt_center = target[:2] + .5 * target[2:]
    center_only = torch.cat((gt_center[None] - .5 * boxes[:, 2:], boxes[:, 2:]), dim=-1)
    size_only = torch.cat((center - .5 * target[2:], target[2:].expand(len(boxes), -1)), dim=-1)
    return (float(overlaps(clipped(center_only, image_shape), target).max()),
            float(overlaps(clipped(size_only, image_shape), target).max()))


def center_supported(boxes, center, factor):
    half = .5 * factor * boxes[:, 2:]
    origins = boxes[:, :2] + .5 * boxes[:, 2:]
    return bool(((center >= origins-half) & (center <= origins+half)).all(-1).any())


def summarize(rows):
    local = [r for r in rows if r['center_inside_crop'] and r['full256_iou'] < .5]
    joint_fields = ['top10_partial_overlap', 'full_GT_inside', 'center_in_candidate',
                    'center_in_context', 'GT_side_below10', 'top10_center_only_hit',
                    'top10_size_only_hit']
    joint_profiles = Counter(''.join('1' if value else '0' for value in (
        r['top10_iou'] >= .1, r['full_gt_inside_crop'], r['center_in_candidate'],
        r['center_in_context'], r['GT_side_below10'],
        r['top10_GT_center_iou'] >= .5, r['top10_GT_size_iou'] >= .5)) for r in local)
    return dict(events=len(rows), center_inside=sum(r['center_inside_crop'] for r in rows),
                full_gt_inside=sum(r['full_gt_inside_crop'] for r in rows),
                native_correct=sum(r['native_iou'] >= .5 for r in rows),
                top10_correct=sum(r['top10_iou'] >= .5 for r in rows),
                full256_correct=sum(r['full256_iou'] >= .5 for r in rows),
                eligible_local_refinement=sum(r['refinement_eligible'] for r in rows),
                local_dense_misses=dict(
                    events=len(local),
                    top10_center_only_hit=sum(r['top10_GT_center_iou'] >= .5 for r in local),
                    top10_size_only_hit=sum(r['top10_GT_size_iou'] >= .5 for r in local),
                    full256_center_only_hit=sum(r['full256_GT_center_iou'] >= .5 for r in local),
                    full256_size_only_hit=sum(r['full256_GT_size_iou'] >= .5 for r in local),
                    clipped_GT_hit=sum(r['clipped_GT_iou'] >= .5 for r in local),
                    top10_partial_overlap=sum(r['top10_iou'] >= .1 for r in local),
                    target_center_in_candidate=sum(r['center_in_candidate'] for r in local),
                    target_center_in_context=sum(r['center_in_context'] for r in local),
                    GT_side_below10=sum(r['GT_side_below10'] for r in local),
                    full_GT_inside=sum(r['full_gt_inside_crop'] for r in local),
                    refinement_eligible=sum(r['refinement_eligible'] for r in local),
                    joint_fields=joint_fields, joint_profiles=dict(joint_profiles)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--m101', type=Path, required=True)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    sys.path.insert(0, str(args.repository))
    from lib.test.tracker.sttrack_lachtt_observation import _search_origin
    torch.set_num_threads(1)
    prep = json.loads((args.cache/'preparation.json').read_text())
    label_path = args.cache/'training_labels.json'
    assert sha(label_path) == prep['training_labels_sha256']
    labels = json.loads(label_path.read_text())
    m101 = json.loads(args.m101.read_text())
    assert m101['status'] == 'complete_visual_control' and m101['mode'] == 'train'
    reference = {r['key']:r for r in m101['development_rows']}
    assert len(reference) == 495
    groups = defaultdict(list); rows = []; matched = set()
    for shard in (0, 1):
        receipt = json.loads((args.cache/f'collect_shard{shard}.json').read_text())
        assert receipt['status'] == 'complete' and not receipt['smoke']
        for item in receipt['sequences']:
            data = torch.load(args.cache/'features'/f"{item['sequence']}.pt", map_location='cpu')
            assert not data['labels_loaded']
            for index, frame in enumerate(data['event_frames']):
                key = f"{item['sequence']}@{frame}"
                label = labels[key]
                assert label['split'] == data['split']
                if label['current'] is None:
                    continue
                target = torch.tensor(label['current'], dtype=torch.float32)
                boxes = data['boxes'][index].float()
                dense = data['dense_boxes'][index].float()
                native = float(overlaps(boxes[:1], target)[0])
                top = float(overlaps(boxes, target).max())
                full = float(overlaps(dense, target).max())
                if data['split'] == 'development':
                    assert native == reference[key]['native_iou'] and top == reference[key]['oracle_iou']
                    matched.add(key)
                left, up, side = _search_origin(data['prior_bbox'][index].tolist(), 256,
                                                float(data['resize_factor'][index]))
                cx, cy = [float(x) for x in target[:2]+.5*target[2:]]
                inside = left <= cx < left+side and up <= cy < up+side
                full_inside = (float(target[0]) >= left and float(target[1]) >= up and
                               float(target[0]+target[2]) <= left+side and
                               float(target[1]+target[3]) <= up+side)
                top_center, top_size = interventions(boxes, target, data['image_shape'][index])
                full_center, full_size = interventions(dense, target, data['image_shape'][index])
                row = dict(key=key, split=data['split'], strata=list(label['strata']),
                           native_iou=native, top10_iou=top,
                           full256_iou=full, center_inside_crop=inside, full_gt_inside_crop=full_inside,
                           top10_GT_center_iou=top_center, top10_GT_size_iou=top_size,
                           full256_GT_center_iou=full_center, full256_GT_size_iou=full_size,
                           clipped_GT_iou=float(overlaps(clipped(target[None], data['image_shape'][index]), target)[0]),
                           center_in_candidate=center_supported(boxes, target[:2]+.5*target[2:], 1.),
                           center_in_context=center_supported(boxes, target[:2]+.5*target[2:], 2.),
                           GT_side_below10=bool((target[2:] < 10).any()),
                           refinement_eligible=full_inside and top >= .1)
                rows.append(row); groups[data['split']].append(row)
                for tag in row['strata']:
                    groups[data['split']+':'+tag].append(row)
    assert len(rows) == 3039 and len({r['key'] for r in rows}) == 3039
    assert len(groups['fit']) == 2544 and len(groups['development']) == 495
    assert matched == set(reference)
    result = dict(status='complete_read_only_cached_geometry_probe', events=len(rows),
                  summary={name:summarize(values) for name,values in groups.items()},
                  actual_all495native_top10_fields_exact=True,
                  stratum_groups_overlap_do_not_sum=True,
                  source_training_labels_sha256=prep['training_labels_sha256'],
                  source_m101_result_sha256=sha(args.m101), source_sha256=sha(__file__),
                  counterfactuals_use_GT_not_deployable=True,
                  crop_reconstructed_from_cached_float32_prior_resize=True,
                  no_GPU_network_training_checkpoint_tracker_action=True,
                  no_semantic_identity_or_public_evaluation=True)
    args.output.mkdir()
    (args.output/'events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
