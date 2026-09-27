"""Prepare Train-only, blinded A/B candidate images for instance review."""

import argparse
import json
from pathlib import Path
import random

from PIL import Image, ImageDraw
import torch

from analyze_train_states import overlaps, sha


def read(path):
    return json.loads(Path(path).read_text())


def draw_box(draw, box, color, label):
    x, y, width, height = [float(value) for value in box]
    draw.rectangle((x, y, x + width, y + height), outline=color, width=3)
    draw.text((x + 3, max(0, y - 17)), label, fill=color)


def crop_with_context(image, box):
    x, y, width, height = [float(value) for value in box]
    margin = .15 * max(width, height)
    bounds = (max(0, int(x - margin)), max(0, int(y - margin)),
              min(image.width, int(x + width + margin)),
              min(image.height, int(y + height + margin)))
    return image.crop(bounds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=24)
    args = parser.parse_args()
    preparation = read(args.root / 'preparation.json')
    assert sha(args.spec) == preparation['source_spec_sha256']
    assert sha(args.root / 'training_labels.json') == preparation['training_labels_sha256']
    dataset_root = Path(read(args.spec)['dataset_root'])
    labels = read(args.root / 'training_labels.json')
    eligible = []
    for shard in (0, 1):
        receipt = read(args.root / f'collect_shard{shard}.json')
        assert receipt['status'] == 'complete' and not receipt['smoke']
        for item in receipt['sequences']:
            if item['split'] != 'fit':
                continue
            path = args.root / 'features' / f"{item['sequence']}.pt"
            assert sha(path) == item['feature_sha256']
            data = torch.load(path, map_location='cpu')
            best = None
            for index, frame in enumerate(data['event_frames']):
                key = f"{item['sequence']}@{frame}"
                target = labels[key]['current']
                if target is None:
                    continue
                boxes = data['boxes'][index].float()
                ious = overlaps(boxes, torch.tensor(target, dtype=torch.float32))
                positive = int(ious.argmax())
                native_iou = float(ious[0])
                positive_iou = float(ious[positive])
                if native_iou > .1 or positive_iou < .6:
                    continue
                cross_iou = float(overlaps(boxes[:1], boxes[positive])[0])
                if cross_iou > .1:
                    continue
                row = dict(sequence=item['sequence'], frame=frame,
                           native_index=0, positive_index=positive,
                           native_box=boxes[0].tolist(),
                           positive_box=boxes[positive].tolist(),
                           gt_box=target, native_iou=native_iou,
                           positive_iou=positive_iou, cross_iou=cross_iou)
                if best is None or row['positive_iou'] > best['positive_iou']:
                    best = row
            if best is not None:
                eligible.append(best)
    selected = sorted(eligible, key=lambda row: (-row['positive_iou'], row['sequence']))[:args.count]
    assert len(selected) == args.count
    rng = random.Random(2027)
    swapped = set(rng.sample(range(args.count), args.count // 2))
    images = args.output / 'images'
    images.mkdir(parents=True, exist_ok=True)
    for position, row in enumerate(selected):
        audit_id = f'{position + 1:03d}'
        source = dataset_root / row['sequence'] / 'color' / f"{row['frame'] + 1:08d}.jpg"
        image = Image.open(source).convert('RGB')
        a_index = row['positive_index'] if position in swapped else 0
        b_index = 0 if position in swapped else row['positive_index']
        a_box = row['positive_box'] if position in swapped else row['native_box']
        b_box = row['native_box'] if position in swapped else row['positive_box']
        current = image.copy()
        draw = ImageDraw.Draw(current)
        draw_box(draw, a_box, '#ff2d43', 'A')
        draw_box(draw, b_box, '#24b5ff', 'B')
        current.save(images / f'{audit_id}_current.jpg', quality=92)
        crop_with_context(image, a_box).save(images / f'{audit_id}_A.png')
        crop_with_context(image, b_box).save(images / f'{audit_id}_B.png')
        row.update(audit_id=audit_id, a_index=a_index, b_index=b_index,
                   source_image=str(source))
    result = dict(status='prepared_train_only', seed=2027, candidate_count=args.count,
                  eligible_sequence_count=len(eligible),
                  selection='One high-IoU, disjoint native-miss versus Top-10-hit event per fit sequence; balanced blinded A/B',
                  preparation_sha256=sha(args.root / 'preparation.json'),
                  labels_sha256=preparation['training_labels_sha256'],
                  rows=selected)
    (args.output / 'private_selection.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'eligible_sequences': len(eligible),
                      'selected': len(selected), 'a_positive': len(swapped),
                      'b_positive': args.count - len(swapped)}))


if __name__ == '__main__':
    main()
