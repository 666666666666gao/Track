#!/usr/bin/env python3
"""Create disjoint VOT multi-start anchor shards for the six ROB failures."""

import argparse
import hashlib
import json
import os
from pathlib import Path


SEQUENCES = (
    'cup02_indoor_1',
    'earphone01_indoor_1',
    'toy09_indoor_1',
    'glass01_indoor_2',
    'shoes02_indoor_1',
    'bag02_indoor_2',
)
TRACKER = 'sutrack_l384_rgbd_language_safe_template'


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-{}'.format(os.getpid()))
    with open(temporary, 'x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source-sequences', type=Path,
        default=Path('/root/autodl-tmp/VOT-RGBD2022/sequences'))
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--shards', type=int, default=10)
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument(
        '--sequence', action='append', default=[],
        help='Optional sequence name. Repeat to override the six-sequence default.')
    parser.add_argument(
        '--all-sequences', action='store_true',
        help='Use every sequence in source-sequences/list.txt.')
    parser.add_argument(
        '--tracker', default=TRACKER,
        help='Tracker identifier and matching lib.test.vot module name.')
    parser.add_argument(
        '--anchor', action='append', default=[],
        help='Optional exact trajectory name, e.g. cup02_indoor_1_00000100. '
             'Repeat to build a strict subset.')
    return parser.parse_args()


def link_entry(source, destination):
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(str(destination))
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def main():
    args = parse_args()
    if args.shards <= 0 or args.gpus <= 0:
        raise ValueError('Shard and GPU counts must be positive')
    source_root = args.source_sequences.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError('Output root is not empty: {}'.format(output_root))
    output_root.mkdir(parents=True, exist_ok=True)

    if args.all_sequences and args.sequence:
        raise ValueError('--all-sequences and --sequence are mutually exclusive')
    if args.all_sequences:
        sequence_order = [
            line.strip()
            for line in (source_root / 'list.txt').read_text(
                encoding='utf-8').splitlines()
            if line.strip()
        ]
    elif args.sequence:
        sequence_order = list(args.sequence)
    else:
        sequence_order = list(SEQUENCES)
    if not sequence_order or len(sequence_order) != len(set(sequence_order)):
        raise ValueError('Sequence selection must be non-empty and unique')

    anchors = []
    source = {}
    for sequence in sequence_order:
        sequence_root = source_root / sequence
        anchor_path = sequence_root / 'anchor.value'
        values = anchor_path.read_text(encoding='utf-8').splitlines()
        frame_count = len(values)
        if frame_count <= 1:
            raise ValueError('Invalid anchor file for {}'.format(sequence))
        parsed = [float(value.strip()) for value in values]
        source[sequence] = {
            'root': str(sequence_root),
            'frame_count': frame_count,
            'anchor_sha256': sha256_file(anchor_path),
            'values': parsed,
        }
        for index, value in enumerate(parsed):
            if value > 0:
                cost = frame_count - index
            elif value < 0:
                cost = index + 1
            else:
                continue
            anchors.append({
                'sequence': sequence,
                'index': index,
                'direction': 'forward' if value > 0 else 'backward',
                'value': value,
                'estimated_frames': cost,
            })

    requested_anchors = set(args.anchor)
    if requested_anchors:
        available = {
            '{}_{:08d}'.format(row['sequence'], row['index'])
            for row in anchors
        }
        missing = sorted(requested_anchors - available)
        if missing:
            raise ValueError('Unknown requested anchors: {}'.format(missing))
        anchors = [
            row for row in anchors
            if '{}_{:08d}'.format(row['sequence'], row['index'])
            in requested_anchors
        ]

    loads = [0] * args.shards
    assignments = [[] for _ in range(args.shards)]
    for anchor in sorted(
            anchors,
            key=lambda row: (-row['estimated_frames'], row['sequence'], row['index'])):
        shard_index = min(range(args.shards), key=lambda i: (loads[i], i))
        assignments[shard_index].append(anchor)
        loads[shard_index] += anchor['estimated_frames']

    config_text = (
        'registry:\n'
        '- ./trackers.ini\n'
        'sequences: ./sequences\n'
        'stack: vot2022/rgbd\n')
    shard_records = []
    for shard_index, rows in enumerate(assignments):
        shard_root = output_root / 'shard-{:02d}'.format(shard_index)
        sequences_root = shard_root / 'sequences'
        sequences_root.mkdir(parents=True)
        gpu = shard_index % args.gpus
        trackers_text = (
            '[{tracker}]\n'
            'label = {tracker}\n'
            'protocol = traxpython\n'
            'command = lib.test.vot.{tracker}\n'
            'paths = /home/SUTrack_RGBD_L\n'
            'env_CUDA_VISIBLE_DEVICES = {gpu}\n'
            'env_PYTHONPATH = /home/SUTrack_RGBD_L\n'
            'env_TOKENIZERS_PARALLELISM = false\n'
            'timeout = 600\n'
            'restart = false\n').format(tracker=args.tracker, gpu=gpu)
        atomic_text(shard_root / 'config.yaml', config_text)
        atomic_text(shard_root / 'trackers.ini', trackers_text)

        by_sequence = {}
        for row in rows:
            by_sequence.setdefault(row['sequence'], []).append(row)
        ordered_sequences = [name for name in sequence_order if name in by_sequence]
        atomic_text(
            sequences_root / 'list.txt',
            ''.join(name + '\n' for name in ordered_sequences))
        expected = []
        for sequence in ordered_sequences:
            source_sequence = source_root / sequence
            shard_sequence = sequences_root / sequence
            shard_sequence.mkdir()
            for entry in source_sequence.iterdir():
                if entry.name == 'anchor.value':
                    continue
                link_entry(entry, shard_sequence / entry.name)
            assigned = {row['index']: row['value'] for row in by_sequence[sequence]}
            filtered = [
                str(assigned.get(index, 0)) for index in range(source[sequence]['frame_count'])
            ]
            atomic_text(shard_sequence / 'anchor.value', '\n'.join(filtered) + '\n')
            for row in by_sequence[sequence]:
                expected.append('{}_{:08d}'.format(sequence, row['index']))

        shard_records.append({
            'index': shard_index,
            'gpu': gpu,
            'root': str(shard_root),
            'estimated_frames': loads[shard_index],
            'anchor_count': len(rows),
            'expected_trajectories': sorted(expected),
            'anchors': sorted(rows, key=lambda row: (row['sequence'], row['index'])),
            'config_sha256': sha256_file(shard_root / 'config.yaml'),
            'trackers_sha256': sha256_file(shard_root / 'trackers.ini'),
            'list_sha256': sha256_file(sequences_root / 'list.txt'),
        })

    flattened = [
        (row['sequence'], row['index'])
        for shard in shard_records for row in shard['anchors']
    ]
    expected_pairs = [(row['sequence'], row['index']) for row in anchors]
    if sorted(flattened) != sorted(expected_pairs) or len(flattened) != len(set(flattened)):
        raise RuntimeError('Anchor assignment is not an exact disjoint cover')

    manifest = {
        'schema': 'sutrack_vot_failure_family_anchor_shards_v1',
        'tracker': args.tracker,
        'source_sequences_root': str(source_root),
        'sequences': sequence_order,
        'shard_count': args.shards,
        'gpu_count': args.gpus,
        'total_anchor_count': len(anchors),
        'total_estimated_frames': sum(row['estimated_frames'] for row in anchors),
        'source': {
            name: {key: value for key, value in record.items() if key != 'values'}
            for name, record in source.items()
        },
        'shards': shard_records,
    }
    atomic_text(
        output_root / 'shard_manifest.json',
        json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({
        'output_root': str(output_root),
        'total_anchor_count': len(anchors),
        'estimated_frame_loads': loads,
        'anchor_counts': [len(rows) for rows in assignments],
        'manifest_sha256': sha256_file(output_root / 'shard_manifest.json'),
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
