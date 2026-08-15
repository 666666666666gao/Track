#!/usr/bin/env python3
"""Seed exact completed trajectories into a new disjoint VOT shard layout."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


SUFFIXES = ('.bin', '_confidence.value', '_time.value')


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-{}'.format(os.getpid()))
    with open(temporary, 'x', encoding='utf-8', newline='\n') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-master', type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root.resolve()
    source_master = args.source_master.resolve()
    manifest_path = root / 'shard_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    tracker = manifest['tracker']
    source = source_master / 'results' / tracker / 'baseline'
    if not source.is_dir():
        raise FileNotFoundError(str(source))
    seeded = {}
    for shard in manifest['shards']:
        shard_root = Path(shard['root'])
        for trajectory in shard['expected_trajectories']:
            sequence = trajectory.rsplit('_', 1)[0]
            source_paths = [
                source / sequence / (trajectory + suffix)
                for suffix in SUFFIXES
            ]
            present = [path.is_file() and path.stat().st_size > 0
                       for path in source_paths]
            if any(present) and not all(present):
                raise RuntimeError(
                    'Partial source trajectory {}'.format(trajectory))
            if not all(present):
                continue
            destination = (
                shard_root / 'results' / tracker / 'baseline' / sequence)
            destination.mkdir(parents=True, exist_ok=True)
            for source_path in source_paths:
                target = destination / source_path.name
                if target.exists():
                    if sha256_file(target) != sha256_file(source_path):
                        raise RuntimeError(
                            'Conflicting seeded result {}'.format(target))
                else:
                    shutil.copy2(source_path, target)
                seeded[str(target.relative_to(root))] = sha256_file(target)

    receipt = {
        'schema': 'sutrack_vot_shard_preseed_v1',
        'tracker': tracker,
        'root': str(root),
        'shard_manifest': str(manifest_path),
        'shard_manifest_sha256': sha256_file(manifest_path),
        'source_master': str(source_master),
        'source_file_count': len(seeded),
        'source_trajectory_count': len(seeded) // len(SUFFIXES),
        'seeded_sha256': dict(sorted(seeded.items())),
    }
    receipt_path = root / 'preseed_receipt.json'
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text(encoding='utf-8'))
        if existing != receipt:
            raise RuntimeError('Existing preseed receipt does not match')
    else:
        atomic_json(receipt_path, receipt)
    print(json.dumps({
        'trajectory_count': receipt['source_trajectory_count'],
        'file_count': receipt['source_file_count'],
        'receipt_sha256': sha256_file(receipt_path),
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
