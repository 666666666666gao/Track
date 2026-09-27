"""Reissue the same private cases without a publicly seeded answer order."""

import argparse
import json
from pathlib import Path
import random

from PIL import Image, ImageDraw

from prepare_candidate_review import crop_with_context, draw_box


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    original = json.loads(args.source_manifest.read_text())
    rows = original['rows']
    assert len(rows) == 24 and not args.output.exists()
    rng = random.SystemRandom()
    rng.shuffle(rows)
    positive_a = set(rng.sample(range(24), 12))
    images = args.output / 'images'
    images.mkdir(parents=True)
    for index, row in enumerate(rows):
        old_id = row['audit_id']
        audit_id = f'{index + 1:03d}'
        a_positive = index in positive_a
        a_box = row['positive_box'] if a_positive else row['native_box']
        b_box = row['native_box'] if a_positive else row['positive_box']
        image = Image.open(row['source_image']).convert('RGB')
        display = image.copy()
        draw = ImageDraw.Draw(display)
        draw_box(draw, a_box, '#ff2d43', 'A')
        draw_box(draw, b_box, '#24b5ff', 'B')
        display.save(images / f'{audit_id}_current.jpg', quality=92)
        crop_with_context(image, a_box).save(images / f'{audit_id}_A.png')
        crop_with_context(image, b_box).save(images / f'{audit_id}_B.png')
        row.update(source_audit_id=old_id, audit_id=audit_id,
                   a_index=row['positive_index'] if a_positive else 0,
                   b_index=0 if a_positive else row['positive_index'])
    result = dict(status='prepared_train_only', candidate_count=24,
                  selection='Same 24 M92 cases; privately reordered and A/B reassigned',
                  assignment='Unseeded private presentation order; not a training RNG',
                  rows=rows)
    (args.output / 'private_selection.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'cases': 24,
                      'positive_A': 12, 'positive_B': 12,
                      'old_results_changed': False}))


if __name__ == '__main__':
    main()
