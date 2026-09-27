"""Read-only candidate capacity at selected native Train failure onsets."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_train_states import overlaps, sha
from prepare_train_states import overlap


def read(path):
    return json.loads(Path(path).read_text())


def summarize(rows):
    return dict(
        onsets=len(rows),
        center_outside=sum(not row['center_inside'] for row in rows),
        center_inside_top10=sum(row['center_inside'] and row['top10_correct'] for row in rows),
        center_inside_full_only=sum(row['center_inside'] and not row['top10_correct']
                                    and row['full256_correct'] for row in rows),
        center_inside_no_full=sum(row['center_inside'] and not row['full256_correct']
                                  for row in rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--full152-spec', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    preparation = read(root / 'preparation.json')
    assert preparation['source_spec_sha256'] == sha(args.full152_spec)
    assert preparation['inference_inputs_sha256'] == sha(root / 'inference_inputs.json')
    assert preparation['training_labels_sha256'] == sha(root / 'training_labels.json')
    labels = read(root / 'training_labels.json')
    spec = read(args.full152_spec)
    plans = read(root / 'inference_inputs.json')
    receipts = [read(root / f'collect_shard{shard}.json') for shard in (0, 1)]
    by_name = {item['sequence']: item for receipt in receipts
               for item in receipt['sequences']}
    assert all(receipt['status'] == 'complete' for receipt in receipts)
    assert len(by_name) == len(plans) == 152
    source_rows = {row['sequence']: row for row in spec['sequence_order']}
    records = []
    for plan in plans:
        name = plan['sequence']
        source = source_rows[name]
        gt_path = Path(spec['dataset_root']) / name / 'groundtruth.txt'
        assert sha(gt_path) == source['groundtruth_sha256']
        gt = np.loadtxt(gt_path, delimiter=',').reshape(-1, 4)[:plan['frames']]
        assert len(gt) == plan['frames']
        valid = np.isfinite(gt).all(axis=1) & (gt[:, 2] > 0) & (gt[:, 3] > 0)
        public = plan['expected_rows']
        assert len(public) == max(plan['event_frames']) + 1
        low = np.zeros(len(public), dtype=bool)
        for frame in range(1, len(public)):
            if valid[frame]:
                low[frame] = overlap(public[frame]['bbox'], gt[frame]) <= .1
        selected = [frame for frame in plan['event_frames']
                    if 'transition' in labels[f'{name}@{frame}']['strata']
                    and low[frame] and not low[frame - 1]]
        feature = root / 'features' / f'{name}.pt'
        assert sha(feature) == by_name[name]['feature_sha256']
        data = torch.load(feature, map_location='cpu')
        assert not data['labels_loaded'] and data['event_frames'] == plan['event_frames']
        indexes = {frame: index for index, frame in enumerate(data['event_frames'])}
        for frame in selected:
            index = indexes[frame]
            target = torch.tensor(gt[frame], dtype=torch.float32)
            ten = overlaps(data['boxes'][index].float(), target)
            full = overlaps(data['dense_boxes'][index].float(), target)
            assert ten[0] <= .1
            previous = data['prior_bbox'][index].numpy()
            side = 256. / float(data['resize_factor'][index])
            crop_left = previous[0] + .5 * previous[2] - .5 * side
            crop_top = previous[1] + .5 * previous[3] - .5 * side
            center = gt[frame, :2] + .5 * gt[frame, 2:]
            inside = bool(crop_left <= center[0] < crop_left + side
                          and crop_top <= center[1] < crop_top + side)
            records.append(dict(sequence=name, split=plan['split'], frame=frame,
                                center_inside=inside, native_iou=float(ten[0]),
                                top10_best_iou=float(ten.max()),
                                full256_best_iou=float(full.max()),
                                top10_correct=bool(ten.max() >= .5),
                                full256_correct=bool(full.max() >= .5)))
    summary = {split: summarize([row for row in records if row['split'] == split])
               for split in ('fit', 'development')}
    summary['all'] = summarize(records)
    assert all(sum(value[key] for key in ('center_outside', 'center_inside_top10',
                                         'center_inside_full_only',
                                         'center_inside_no_full')) == value['onsets']
               for value in summary.values())
    result = dict(status='complete_train_only',
                  scope='Selected native STTrack >=10-frame IoU<=0.1 run starts on DepthTrack Train; own-history states, no public tests, no model update.',
                  preparation_sha256=sha(root / 'preparation.json'),
                  full152_spec_sha256=sha(args.full152_spec),
                  receipt_sha256={str(shard): sha(root / f'collect_shard{shard}.json')
                                  for shard in (0, 1)},
                  summary=summary, rows=records)
    destination = root / 'selected_failure_onsets.json'
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
