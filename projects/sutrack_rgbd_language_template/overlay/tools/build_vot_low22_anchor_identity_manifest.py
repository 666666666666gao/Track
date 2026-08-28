#!/usr/bin/env python3
"""Build the preregistered low-22 VOT anchor identity-text manifest.

The text is deliberately limited to category and stable visual identity.  A
record is emitted for every multi-start initialization anchor, but never for a
non-anchor frame.  No future frame, trajectory result, depth state, motion,
occlusion state, spatial position, or distractor statement is used.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


PROFILES = {
    'ball06_indoor_2': ('ball', 'a yellow spherical ball'),
    'bandlight_indoor_1': ('band light', 'a green band-shaped light'),
    'cube02_indoor_1': ('cube', 'a black cube'),
    'cube02_indoor_2': ('cube', 'a dark cube with a fabric-like surface'),
    'cube05_indoor_1': ('cube', 'a dark rectangular cube'),
    'cube05_indoor_2': ('cube', 'a cube'),
    'cube05_indoor_4': ('cube', 'a light-colored cube'),
    'cube05_indoor_5': ('cube', 'a wooden cube with a printed number'),
    'cube05_indoor_6': ('cube', 'a white cube'),
    'cup02_indoor_1': ('cup', 'a red cup with a white interior'),
    'duck03_wild_1': ('duck', 'a dark-feathered duck'),
    'duck03_wild_2': ('duck', 'a dark-feathered duck'),
    'earphone01_indoor_1': (
        'headphones', 'black over-ear headphones with padded earcups'),
    'humans_shirts_room_occ_1_A_2': (
        'person', 'a person wearing a patterned shirt and jeans'),
    'humans_shirts_room_occ_1_B_1': (
        'person', 'a person wearing a patterned long-sleeved collared shirt'),
    'robot_human_corridor_noocc_1_B_1': (
        'person', 'a person wearing a black shirt and blue pants'),
    'shoes02_indoor_1': ('shoe', 'a black laced shoe'),
    'shoes02_indoor_2': ('shoe', 'a black laced shoe'),
    'squirrel_wild_1': ('squirrel', 'a brown squirrel'),
    'toy09_indoor_1': ('toy', 'a rectangular metallic-looking toy'),
    'two_tennis_balls_3': ('tennis ball', 'a yellow tennis ball'),
    'yogurt_indoor_1': ('yogurt cup', 'a yogurt cup with a printed label'),
}


FORBIDDEN_LANGUAGE = (
    'closer', 'farther', 'background', 'occlusion', 'occluded', 'visible',
    'left', 'right', 'center', 'moving', 'motion', 'blurred', 'depth',
    'distractor', 'currently', 'surrounding',
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-{}'.format(os.getpid()))
    with open(temporary, 'x', encoding='utf-8', newline='\n') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sequences-root', type=Path,
        default=Path('/root/autodl-tmp/VOT-RGBD2022/sequences'))
    parser.add_argument(
        '--source-manifest', type=Path,
        default=Path(
            '/home/OSTrack_RGBD_L_dataset_modified/annotations_cleaned/'
            'votrgbd2022_language.jsonl'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    source_records = {}
    for raw_line in args.source_manifest.read_text(encoding='utf-8').splitlines():
        record = json.loads(raw_line)
        source_records[record['sequence_name']] = record
    missing = sorted(set(PROFILES) - set(source_records))
    if missing:
        raise KeyError('Source language manifest lacks {}'.format(missing))

    records = []
    anchor_counts = {}
    corrected_categories = {}
    for sequence_name in sorted(PROFILES):
        category, identity = PROFILES[sequence_name]
        lowered = identity.lower()
        forbidden = [term for term in FORBIDDEN_LANGUAGE if term in lowered]
        if forbidden:
            raise ValueError(
                '{} identity text contains transient terms {}'.format(
                    sequence_name, forbidden))
        source_category = str(
            source_records[sequence_name].get('target_tokens', {}).get(
                'category', '')).strip()
        if source_category.lower() != category.lower():
            corrected_categories[sequence_name] = {
                'source': source_category,
                'corrected': category,
            }

        anchor_path = args.sequences_root / sequence_name / 'anchor.value'
        values = [float(value) for value in anchor_path.read_text(
            encoding='utf-8').splitlines()]
        count = 0
        for anchor_index, value in enumerate(values):
            if value == 0:
                continue
            count += 1
            records.append({
                'dataset': 'votrgbd2022',
                'sequence_name': sequence_name,
                'anchor_index': anchor_index,
                'direction': 'forward' if value > 0 else 'backward',
                'category': category,
                'language': identity,
                'annotation_method': 'anchor_keyed_identity_only_v1',
                'annotation_quality': {
                    'is_valid': True,
                    'identity_only': True,
                    'uses_future_frame': False,
                    'has_bbox_leak': False,
                    'has_absolute_path': False,
                    'transient_state_removed': True,
                },
            })
        if count == 0:
            raise RuntimeError('{} has no VOT anchors'.format(sequence_name))
        anchor_counts[sequence_name] = count

    output_text = ''.join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n'
        for record in records)
    atomic_text(args.output, output_text)
    receipt = {
        'schema': 'votrgbd2022_low22_anchor_identity_manifest_receipt_v1',
        'status': 'complete',
        'dataset': 'votrgbd2022',
        'selection_rule': 'ACC < 0.70 OR ROB < 0.75 on frozen formal result',
        'full_dataset_evaluation_authorized': False,
        'sequence_count': len(PROFILES),
        'anchor_record_count': len(records),
        'anchor_counts': anchor_counts,
        'corrected_categories': corrected_categories,
        'source_manifest': str(args.source_manifest.resolve()),
        'source_manifest_sha256': sha256_file(args.source_manifest),
        'manifest': str(args.output.resolve()),
        'manifest_sha256': sha256_file(args.output),
        'annotation_contract': {
            'per_anchor_record': True,
            'per_frame_text': False,
            'category_and_stable_identity_only': True,
            'future_frame_used': False,
            'transient_depth_motion_occlusion_position_removed': True,
        },
    }
    atomic_text(
        args.receipt,
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
