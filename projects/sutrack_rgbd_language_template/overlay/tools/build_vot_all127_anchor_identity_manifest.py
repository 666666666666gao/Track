#!/usr/bin/env python3
"""Build audited identity-only text for every VOT-RGBD2022 anchor."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from build_vot_low22_anchor_identity_manifest import PROFILES as LOW22_PROFILES


CATEGORY_ALIASES = {
    'bandlight': 'band light',
    'boxes': 'box',
    'cartman': 'Cartman plush toy',
    'colacan': 'soda can',
    'developmentboard': 'development board',
    'dumbbells': 'dumbbell',
    'earphone': 'headphones',
    'human': 'person',
    'humans': 'person',
    'mobilephone': 'mobile phone',
    'paperpunch': 'paper punch',
    'shoes': 'shoe',
    'toiletpaper': 'toilet paper roll',
    'trashcans': 'trash can',
    'trendnet': 'TRENDnet product box',
    'trendnetbag': 'TRENDnet bag',
    'xmg': 'XMG product box',
}


TEXT_OVERRIDES = {
    'adapter01_indoor_1': (
        'adapter', 'a white rectangular adapter with rounded corners'),
    'backpack_blue_1': (
        'backpack', 'a black rectangular backpack with multiple zippers and straps'),
    'backpack_indoor_1': (
        'backpack', 'a black rectangular backpack with an attached keychain'),
    'backpack_robotarm_lab_occ_1': (
        'backpack', 'a dark blue rectangular backpack with straps and zippers'),
    'backpack_room_noocc_1_1': (
        'backpack', 'a green rectangular backpack with multiple straps and compartments'),
    'bag01_indoor_2': ('bag', 'a red and blue bag'),
    'bag02_indoor_1': ('bag', 'a dark bag'),
    'bag02_indoor_2': ('bag', 'a dark rectangular bag'),
    'bag_outside_2': ('bag', 'a dark rectangular bag'),
    'bag_outside_3': ('bag', 'a black rectangular bag with straps'),
    'ball06_indoor_1': ('ball', 'a red round ball with a square pattern'),
    'ball11_wild_2': ('ball', 'a red ball'),
    'ball11_wild_5': ('ball', 'a red spherical ball'),
    'ball20_indoor_4': ('ball', 'a ball'),
    'box1_outside_1': ('box', 'a black-and-red rectangular box'),
    'box_darkroom_noocc_10_1': (
        'box', 'a dark rectangular box with green text or symbols'),
    'box_darkroom_noocc_2_1': (
        'box', 'a black rectangular box with a purple label'),
    'box_darkroom_noocc_3_1': (
        'box', "a black box with green 'XMG' text"),
    'box_darkroom_noocc_4_1': ('box', 'a rectangular box with text'),
    'box_darkroom_noocc_5_1': (
        'box', 'a black rectangular box with text and graphics'),
    'box_darkroom_noocc_6_1': (
        'box', 'a black rectangular box with text'),
    'box_darkroom_noocc_7_1': (
        'box', 'a black rectangular box with a circular mark'),
    'box_humans_room_occ_1_1': (
        'box', "a black box with green 'XMG' text"),
    'box_room_noocc_3_1': (
        'box', 'a black rectangular box with white text and symbols'),
    'box_room_noocc_4_1': (
        'box', 'a rectangular box with text and a controller image'),
    'box_room_noocc_7_1': (
        'box', "a black box with green 'XMG' text and green stripes"),
    'box_room_occ_2_1': (
        'box', 'a black rectangular box with white text'),
    'box_room_occ_2_2': (
        'box', 'a product box with a red border and printed text'),
    'boxes_backpack_room_occ_1_2': (
        'box', 'a medium green rectangular box'),
    'boxes_humans_room_occ_1_1': ('box', 'a box in a room'),
    'boxes_humans_room_occ_1_4': (
        'box', 'a box in a row of stacked boxes'),
    'boxes_humans_room_occ_1_5': (
        'person', 'a person wearing a blue sweater and jeans'),
    'boxes_room_occ_1_1': (
        'box', 'a black rectangular box with blue and white text'),
    'boxes_room_occ_1_2': (
        'box', "a black rectangular box with 'TRENDnet' text"),
    'cartman_robotarm_lab_noocc_1': (
        'Cartman plush toy',
        'a Cartman plush toy with a blue hat and yellow shirt'),
    'case_1': ('case', 'a black rectangular case'),
    'colacan03_indoor_1': ('soda can', 'a red rectangular soda can'),
    'colacan03_indoor_3': ('soda can', 'a red rectangular soda can'),
    'colacan03_indoor_7': ('soda can', 'a rectangular metallic soda can'),
    'container_room_noocc_1_1': (
        'container', 'an orange plastic container with handles'),
    'cube05_indoor_3': ('cube', 'a rectangular metallic cube'),
    'cup04_indoor_1': ('cup', 'a black cylindrical cup with red markings'),
    'developmentboard_indoor_4': (
        'development board', 'a rectangular development board'),
    'dumbbells01_indoor_2': ('dumbbell', 'a dark dumbbell'),
    'file01_indoor_1': (
        'file', 'a rectangular file folder with black dots'),
    'flag_indoor_1': (
        'flag', 'a white rectangular flag with a blue cross'),
    'human02_indoor_1': (
        'person', 'a person wearing glasses and a white shirt'),
    'human02_indoor_2': (
        'person', 'a man wearing glasses and a white shirt'),
    'human02_indoor_3': (
        'person', 'a person wearing a white T-shirt and black shorts'),
    'humans_corridor_occ_1_1': (
        'person', 'a man wearing a sweater and jeans'),
    'humans_shirts_room_occ_1_A_1': (
        'person', 'a man wearing a dark blue sweater and jeans'),
    'jug_1': ('jug', 'a transparent glass jug with a handle'),
    'lamp02_indoor_1': ('lamp', 'an illuminated lamp'),
    'mobilephone03_indoor_2': (
        'mobile phone', 'a black rectangular mobile phone with a screen'),
    'notebook01_indoor_1': ('notebook', 'a black rectangular notebook'),
    'paperpunch_3': ('paper punch', 'a dark rectangular paper punch'),
    'person_outside_1': (
        'person', 'a person wearing a dark fur-lined coat and light pants'),
    'pigeon01_wild_1': ('pigeon', 'a dark pigeon'),
    'pigeon02_wild_1': ('pigeon', 'a dark pigeon'),
    'pigeon04_wild_1': ('pigeon', 'a dark pigeon'),
    'pot_indoor_1': ('pot', 'a dark pot'),
    'pot_indoor_2': ('pot', 'a dark pot'),
    'pot_indoor_3': ('pot', 'a dark pot'),
    'robot_lab_occ_1': (
        'robot',
        'a rectangular robot with electronic and mechanical components'),
    'roller_indoor_1': ('roller', 'a dark roller'),
    'roller_indoor_2': ('roller', 'a dark roller with a speckled surface'),
    'roller_indoor_4': ('roller', 'a dark roller'),
    'squirrel_wild_2': ('squirrel', 'a brown squirrel'),
    'squirrel_wild_3': ('squirrel', 'a brown squirrel'),
    'stick_indoor_1': ('stick', 'a blue-and-white stick'),
    'toiletpaper01_indoor_1': (
        'toilet paper roll', 'a white cylindrical toilet paper roll'),
    'toiletpaper01_indoor_2': (
        'toilet paper roll', 'a light gray toilet paper roll'),
    'toy02_indoor_1': ('toy', 'a brown textured toy'),
    'toy_office_noocc_1_1': (
        'toy', 'a cartoon-character toy with large eyes and a red hat'),
    'trashcan_room_occ_1_2': (
        'trash can', 'a green trash can with a lid'),
    'trashcans_room_occ_1_A_1': (
        'trash can', "a green trash can marked 'SULO'"),
    'trashcans_room_occ_1_B_1': (
        'trash can', "a green trash can marked 'SULO'"),
    'trashcans_room_occ_1_B_3': (
        'trash can', 'a green trash can with wheels'),
    'trendNet_outside_1': (
        'TRENDnet product box', 'a dark rectangular TRENDnet product box'),
    'trendNet_outside_2': (
        'TRENDnet product box', 'a dark rectangular TRENDnet product box'),
    'trendNetBag_outside_1': (
        'TRENDnet bag', 'a dark rectangular TRENDnet bag'),
    'two_mugs_1': ('mug', 'a striped mug with a logo'),
    'two_tennis_balls_1': ('tennis ball', 'a yellow tennis ball'),
    'XMG_outside_2': (
        'XMG product box',
        'a black rectangular XMG product box with yellow text'),
}


FORBIDDEN = re.compile(
    r'\b(?:closer|farther|background|occlusion|occluded|visible|currently|'
    r'moving|motion|depth|distractor|surrounding|blurred|blurry|'
    r'indistinct|standing upright)\b', re.IGNORECASE)


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


def normalize(value):
    return ' '.join(str(value or '').split()).strip(' ,.;')


def clean_appearance(value):
    value = normalize(value)
    value = re.sub(
        r'^the target appears to be\s+', '', value, flags=re.IGNORECASE)
    value = re.sub(r'\bpossibly\s+', '', value, flags=re.IGNORECASE)
    value = re.sub(r'\bsmall\s*,?\s*', '', value, flags=re.IGNORECASE)
    value = re.sub(r'\b(?:blurred|blurry|indistinct)\b\s*,?\s*', '', value,
                   flags=re.IGNORECASE)
    value = re.sub(r'\bstanding upright\b\s*,?\s*', '', value,
                   flags=re.IGNORECASE)
    value = re.sub(r'\bvisible\s+', '', value, flags=re.IGNORECASE)
    value = re.sub(r'\bbackground\b', 'surface', value,
                   flags=re.IGNORECASE)
    value = re.sub(r'\bdark and\s*$', 'dark', value, flags=re.IGNORECASE)
    value = re.sub(r'\s+,', ',', value)
    value = re.sub(r',\s*,+', ', ', value)
    value = normalize(value)
    if value.casefold() in {'shape', 'dark shape', 'dark', 'light gray'}:
        return ''
    return value


def auto_profile(record):
    sequence = record['sequence_name']
    if sequence in LOW22_PROFILES:
        return LOW22_PROFILES[sequence], 'low22_preregistered_manual'
    if sequence in TEXT_OVERRIDES:
        return TEXT_OVERRIDES[sequence], 'manual_semantic_correction'
    target = record.get('target_tokens') or {}
    source_category = normalize(target.get('category')).casefold()
    category = CATEGORY_ALIASES.get(source_category, source_category)
    if not category:
        raise ValueError('{} lacks a category'.format(sequence))
    appearance = clean_appearance(target.get('appearance'))
    if not appearance:
        return (category, 'a {}'.format(category)), 'category_only_fallback'
    lowered = appearance.casefold()
    category_words = [word.casefold() for word in category.split()]
    contains_category = any(
        re.search(r'\b{}\b'.format(re.escape(word)), lowered)
        for word in category_words if len(word) >= 3)
    if contains_category:
        language = appearance
        if not re.match(r'^(?:a|an|the)\b', language, flags=re.IGNORECASE):
            language = 'a ' + language
    else:
        language = 'a {} with {}'.format(category, appearance)
    language = normalize(language)
    return (category, language), 'structured_identity_cleanup'


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
    source_rows = [
        json.loads(line) for line in args.source_manifest.read_text(
            encoding='utf-8').splitlines()]
    if len(source_rows) != 127:
        raise ValueError('Expected 127 source sequences, found {}'.format(
            len(source_rows)))
    by_name = {row['sequence_name']: row for row in source_rows}
    if len(by_name) != len(source_rows):
        raise ValueError('Source manifest contains duplicate sequences')

    records = []
    method_counts = {}
    sequence_profiles = {}
    category_corrections = {}
    anchor_counts = {}
    for sequence in sorted(by_name):
        row = by_name[sequence]
        (category, language), method = auto_profile(row)
        category = normalize(category)
        language = normalize(language)
        if not category or not language or FORBIDDEN.search(language):
            raise ValueError(
                '{} produced unsafe identity text {!r}'.format(
                    sequence, language))
        method_counts[method] = method_counts.get(method, 0) + 1
        sequence_profiles[sequence] = {
            'category': category,
            'language': language,
            'method': method,
        }
        source_category = normalize(
            (row.get('target_tokens') or {}).get('category'))
        if source_category.casefold() != category.casefold():
            category_corrections[sequence] = {
                'source': source_category,
                'corrected': category,
            }
        values = [
            float(value) for value in (
                args.sequences_root / sequence / 'anchor.value').read_text(
                    encoding='utf-8').splitlines()]
        count = 0
        for anchor_index, value in enumerate(values):
            if value == 0:
                continue
            count += 1
            records.append({
                'dataset': 'votrgbd2022',
                'sequence_name': sequence,
                'anchor_index': anchor_index,
                'direction': 'forward' if value > 0 else 'backward',
                'category': category,
                'language': language,
                'annotation_method': 'anchor_keyed_identity_only_all127_v1',
                'profile_method': method,
                'annotation_quality': {
                    'is_valid': True,
                    'identity_only': True,
                    'uses_future_frame': False,
                    'has_bbox_leak': False,
                    'has_absolute_path': False,
                    'transient_state_removed': True,
                },
            })
        anchor_counts[sequence] = count

    low22_manifest = {}
    for record in records:
        if record['sequence_name'] in LOW22_PROFILES:
            low22_manifest.setdefault(
                record['sequence_name'], record['language'])
    for sequence, (_, expected_language) in LOW22_PROFILES.items():
        if low22_manifest.get(sequence) != expected_language:
            raise RuntimeError('{} changed from the promoted low22 text'.format(
                sequence))

    output_text = ''.join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n'
        for record in records)
    atomic_text(args.output, output_text)
    receipt = {
        'schema': 'votrgbd2022_all127_anchor_identity_manifest_receipt_v1',
        'status': 'complete',
        'dataset': 'votrgbd2022',
        'sequence_count': len(sequence_profiles),
        'anchor_record_count': len(records),
        'anchor_counts': anchor_counts,
        'method_counts': method_counts,
        'category_corrections': category_corrections,
        'sequence_profiles': sequence_profiles,
        'low22_promoted_text_preserved_exactly': True,
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
    if len(sequence_profiles) != 127 or len(records) != 1765:
        raise RuntimeError(
            'Expected 127 sequences/1765 anchors, found {}/{}'.format(
                len(sequence_profiles), len(records)))
    atomic_text(
        args.receipt,
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    print(json.dumps({
        'sequence_count': len(sequence_profiles),
        'anchor_record_count': len(records),
        'method_counts': method_counts,
        'category_corrections': category_corrections,
        'manifest_sha256': receipt['manifest_sha256'],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
