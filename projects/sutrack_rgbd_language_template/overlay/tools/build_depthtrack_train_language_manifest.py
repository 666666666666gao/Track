#!/usr/bin/env python3
"""Materialize a path-free DepthTrack Train language manifest.

Only first-frame, sequence-level descriptions are copied.  Bounding boxes,
absolute paths, raw model output and all future-frame information stay out of
the deployable manifest.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


SCHEMA = 'sutrack-depthtrack-train-language-materialization/v1'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--expected-sequences', type=int, default=152)
    return parser.parse_args()


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


def main():
    args = parse_args()
    if args.expected_sequences <= 0:
        raise ValueError('--expected-sequences must be positive')
    source = args.source.resolve()
    records = []
    seen = set()
    with source.open('r', encoding='utf-8') as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                raise ValueError('blank source row {}'.format(line_number))
            source_record = json.loads(raw_line)
            if source_record.get('dataset') != 'DepthTrack':
                raise ValueError('unexpected dataset at row {}'.format(line_number))
            if source_record.get('split') != 'train':
                raise ValueError('unexpected split at row {}'.format(line_number))
            sequence = str(source_record.get('sequence', '')).strip()
            language = str(source_record.get('final_description', '')).strip()
            annotation = source_record.get('annotation')
            if (not sequence or sequence in seen or not language or
                    not isinstance(annotation, dict)):
                raise ValueError('malformed source row {}'.format(line_number))
            provenance = source_record.get('provenance')
            if (not isinstance(provenance, dict) or
                    provenance.get('uses_initial_gt_bbox') is not True):
                raise ValueError(
                    'row {} is not bound to first-frame initialization'.format(
                        line_number))
            forbidden = ('\\', '/root/', '/home/', 'D:', 'bbox',
                         'groundtruth.txt', '.jpg', '.png')
            if any(token.lower() in language.lower() for token in forbidden):
                raise ValueError(
                    'deployable language leaks a path/bbox token at row {}'.format(
                        line_number))
            clean = {
                'dataset': 'depthtrack_train',
                'sequence_name': sequence,
                'language': language,
                'annotation_quality': {
                    'is_valid': True,
                    'has_bbox_leak': False,
                    'has_absolute_path': False,
                    'source_scope': 'first_frame_initialization_only',
                },
            }
            records.append(clean)
            seen.add(sequence)
    if len(records) != args.expected_sequences:
        raise ValueError(
            'expected {} sequences, observed {}'.format(
                args.expected_sequences, len(records)))
    records.sort(key=lambda record: record['sequence_name'])
    output_payload = ''.join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n'
        for record in records).encode('utf-8')
    atomic_write(args.output.resolve(), output_payload)
    receipt = {
        'schema': SCHEMA,
        'source_path': str(source),
        'source_sha256': sha256_file(source),
        'output_path': str(args.output.resolve()),
        'output_sha256': sha256_file(args.output.resolve()),
        'sequence_count': len(records),
        'dataset': 'depthtrack_train',
        'ground_truth_scope': 'first_frame_initialization_only',
        'future_frame_text_used': False,
        'bbox_or_absolute_path_materialized': False,
    }
    atomic_write(
        args.receipt.resolve(),
        (json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) +
         '\n').encode('utf-8'))
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
