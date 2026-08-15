#!/usr/bin/env python3
"""Create a deterministic two-GPU plan for the remaining Train152 trace."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


SCHEMA = 'sutrack-depthtrack-train-state-trace-plan/v1'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--language-manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--fixed-sequences', required=True)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + '.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def frame_names(path):
    return [item.stem for item in sorted(path.iterdir()) if item.is_file()]


def main():
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    manifest_path = args.language_manifest.resolve()
    sequences = []
    with manifest_path.open('r', encoding='utf-8') as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            record = json.loads(raw_line)
            sequence = str(record.get('sequence_name', '')).strip()
            if not sequence or sequence in sequences:
                raise ValueError('malformed language row {}'.format(line_number))
            sequences.append(sequence)
    if len(sequences) != 152:
        raise ValueError('expected exactly 152 sequences')
    fixed = [name.strip() for name in args.fixed_sequences.split(',')
             if name.strip()]
    if (len(fixed) != 6 or len(fixed) != len(set(fixed)) or
            not set(fixed).issubset(sequences)):
        raise ValueError('fixed sequence contract failed')
    frame_counts = {}
    for sequence in sequences:
        root = dataset_root / sequence
        rgb = frame_names(root / 'color')
        depth = frame_names(root / 'depth')
        if not rgb or rgb != depth:
            raise ValueError('RGB/depth alignment failed for {}'.format(sequence))
        frame_counts[sequence] = len(rgb)
    remaining = sorted(
        (sequence for sequence in sequences if sequence not in fixed),
        key=lambda name: (-frame_counts[name], name))
    shards = [[], []]
    totals = [0, 0]
    for sequence in remaining:
        index = min(range(2), key=lambda item: (totals[item], item))
        shards[index].append(sequence)
        totals[index] += frame_counts[sequence]
    if set(shards[0]).intersection(shards[1]):
        raise ValueError('overlapping shards')
    if set(fixed + shards[0] + shards[1]) != set(sequences):
        raise ValueError('plan does not cover Train152 exactly')
    output = {
        'schema': SCHEMA,
        'complete': True,
        'dataset': 'DepthTrack Train only',
        'dataset_root': str(dataset_root),
        'sequence_count': len(sequences),
        'frame_count': sum(frame_counts.values()),
        'fixed6_reused': fixed,
        'remaining_sequence_count': len(remaining),
        'remaining_frame_count': sum(frame_counts[name] for name in remaining),
        'shards': [
            {'gpu': index, 'sequences': shard,
             'sequence_count': len(shard), 'frame_count': totals[index]}
            for index, shard in enumerate(shards)
        ],
        'all_sequences_language_manifest_order': sequences,
        'analysis_sequence_order': fixed + shards[0] + shards[1],
        'frame_counts': frame_counts,
        'ground_truth_available_to_tracker': False,
        'future_frame_text_used': False,
        'public_evaluation': False,
        'language_manifest': {
            'path': str(manifest_path),
            'sha256': sha256_file(manifest_path),
        },
        'config': {
            'path': str(args.config.resolve()),
            'sha256': sha256_file(args.config.resolve()),
        },
        'checkpoint': {
            'path': str(args.checkpoint.resolve()),
            'sha256': sha256_file(args.checkpoint.resolve()),
        },
    }
    atomic_write(
        args.output.resolve(),
        (json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2) +
         '\n').encode('utf-8'))
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
