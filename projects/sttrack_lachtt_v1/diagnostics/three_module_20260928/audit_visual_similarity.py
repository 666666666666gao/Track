"""Train-only, fixed-state visual identity similarity diagnostic; no fitting."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from analyze_train_states import overlaps, sha


def read(path):
    return json.loads(Path(path).read_text())


def summarize(rows):
    valid = [row for row in rows if row['gt_valid']]
    return dict(events=len(rows), valid_gt=len(valid),
                native_iou50=sum(row['native_iou'] >= .5 for row in valid),
                visual_iou50=sum(row['visual_iou'] >= .5 for row in valid),
                top10_oracle_iou50=sum(row['top10_iou'] >= .5 for row in valid),
                rescues=sum(row['native_iou'] < .5 and row['visual_iou'] >= .5 for row in valid),
                breaks=sum(row['native_iou'] >= .5 and row['visual_iou'] < .5 for row in valid),
                native_mean_iou=sum(row['native_iou'] for row in valid) / len(valid) if valid else None,
                visual_mean_iou=sum(row['visual_iou'] for row in valid) / len(valid) if valid else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    preparation = read(root / 'preparation.json')
    assert preparation['training_labels_sha256'] == sha(root / 'training_labels.json')
    labels = read(root / 'training_labels.json')
    grouped = defaultdict(list)
    rows = []
    for shard in (0, 1):
        receipt = read(root / f'collect_shard{shard}.json')
        assert receipt['status'] == 'complete' and not receipt['smoke']
        for item in receipt['sequences']:
            path = root / 'features' / f"{item['sequence']}.pt"
            assert sha(path) == item['feature_sha256']
            data = torch.load(path, map_location='cpu')
            assert not data['labels_loaded']
            candidate = data['candidate_rois'].float().mean(dim=-2)
            initial = data['initial_rois'].float().mean(dim=-2)
            similarity = F.cosine_similarity(candidate, initial[None, None], dim=-1).mean(dim=-1)
            selected = similarity.argmax(dim=1)
            assert len(selected) == len(data['event_frames'])
            for index, frame in enumerate(data['event_frames']):
                key = f"{item['sequence']}@{frame}"
                label = labels[key]
                target = label['current']
                if target is None:
                    row = dict(key=key, split=data['split'], gt_valid=False)
                else:
                    iou = overlaps(data['boxes'][index].float(),
                                   torch.tensor(target, dtype=torch.float32))
                    row = dict(key=key, split=data['split'], gt_valid=True,
                               native_iou=float(iou[0]),
                               visual_iou=float(iou[selected[index]]),
                               top10_iou=float(iou.max()),
                               visual_choice=int(selected[index]),
                               native_visual_same=bool(selected[index] == 0))
                rows.append(row)
                grouped[data['split']].append(row)
                for tag in label['strata']:
                    grouped[f"{data['split']}:{tag}"].append(row)
    assert len(rows) == len(labels) == 3502
    report = dict(status='complete_train_only',
                  scope='Pure first-frame RGB-D RoI mean-feature cosine on fixed native Train states; no learned parameters, no public test, no recursive action.',
                  preparation_sha256=sha(root / 'preparation.json'),
                  receipt_sha256={str(shard): sha(root / f'collect_shard{shard}.json')
                                  for shard in (0, 1)},
                  group={name: summarize(values) for name, values in sorted(grouped.items())},
                  rows=rows)
    path = root / 'visual_similarity_diagnostic.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['group'], indent=2))


if __name__ == '__main__':
    main()
