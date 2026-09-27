"""Measure native, top-10, and full-grid candidate coverage on Train states."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import torch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def overlaps(boxes, target):
    left = torch.maximum(boxes[:, :2], target[:2])
    right = torch.minimum(boxes[:, :2] + boxes[:, 2:], target[:2] + target[2:])
    intersection = (right - left).clamp_min(0).prod(1)
    union = boxes[:, 2:].prod(1) + target[2:].prod() - intersection
    return intersection / union


def summarize(rows):
    valid = [row for row in rows if row['gt_valid']]
    return dict(events=len(rows), valid_gt=len(valid),
                native_iou50=sum(row['native_iou'] >= .5 for row in valid),
                top10_oracle_iou50=sum(row['top10_iou'] >= .5 for row in valid),
                full256_oracle_iou50=sum(row['full256_iou'] >= .5 for row in valid),
                native_miss_top10_hit=sum(row['native_iou'] < .5 and row['top10_iou'] >= .5 for row in valid),
                top10_miss_full256_hit=sum(row['top10_iou'] < .5 and row['full256_iou'] >= .5 for row in valid),
                full256_miss=sum(row['full256_iou'] < .5 for row in valid),
                native_mean_iou=sum(row['native_iou'] for row in valid) / len(valid) if valid else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    preparation = json.loads((root / 'preparation.json').read_text())
    assert sha(root / 'training_labels.json') == preparation['training_labels_sha256']
    labels = json.loads((root / 'training_labels.json').read_text())
    groups = defaultdict(list)
    sequence_rows = defaultdict(list)
    seen = set()
    for shard in (0, 1):
        receipt = json.loads((root / f'collect_shard{shard}.json').read_text())
        assert receipt['status'] == 'complete' and not receipt['smoke']
        assert receipt['preparation_sha256'] == sha(root / 'preparation.json')
        for item in receipt['sequences']:
            path = root / 'features' / f"{item['sequence']}.pt"
            assert sha(path) == item['feature_sha256']
            data = torch.load(path, map_location='cpu')
            assert not data['labels_loaded']
            assert data['feature_candidate_count'] == 10 and data['dense_candidate_count'] == 256
            assert data['preparation_sha256'] == receipt['preparation_sha256']
            for index, frame in enumerate(data['event_frames']):
                key = f"{item['sequence']}@{frame}"
                label = labels[key]
                assert key not in seen and label['split'] == data['split']
                seen.add(key)
                target = label['current']
                if target is None:
                    row = dict(key=key, gt_valid=False)
                else:
                    target = torch.tensor(target, dtype=torch.float32)
                    candidates = data['boxes'][index].float()
                    dense = data['dense_boxes'][index].float()
                    native = float(overlaps(candidates[:1], target)[0])
                    row = dict(key=key, gt_valid=True, native_iou=native,
                               top10_iou=float(overlaps(candidates, target).max()),
                               full256_iou=float(overlaps(dense, target).max()))
                groups[data['split']].append(row)
                sequence_rows[item['sequence']].append(row)
                for tag in label['strata']:
                    groups[f"{data['split']}:{tag}"].append(row)
    assert seen == set(labels)
    result = dict(status='complete_train_only', preparation_sha256=sha(root / 'preparation.json'),
                  event_count=len(seen), group={name: summarize(rows) for name, rows in sorted(groups.items())},
                  sequence={name: summarize(rows) for name, rows in sorted(sequence_rows.items())},
                  no_public_evaluation=True, no_model_training=True)
    (root / 'candidate_coverage.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({name: value for name, value in result.items() if name != 'sequence'}, indent=2))


if __name__ == '__main__':
    main()
