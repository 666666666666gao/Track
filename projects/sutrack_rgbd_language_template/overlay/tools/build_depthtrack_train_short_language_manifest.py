#!/usr/bin/env python3
"""Materialize short appearance+category prompts for DepthTrack Train.

The deployable prompt uses only reviewed first-frame appearance and category.
Depth relations, occlusion state, distractors, paths, boxes and future frames are
excluded so the frozen tracker receives a short caption-like input.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


SCHEMA = 'sutrack-depthtrack-train-short-language-materialization/v1'
PROMPT_STRATEGY = 'appearance_category_v1'
REVIEW_STATUSES = {
    'accepted_visual',
    'corrected_after_cross_frame_identity_audit',
    'corrected_after_independent_gt_crop_adjudication',
    'corrected_after_full_corpus_low_information_audit',
}


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


def normalized_words(value):
    return re.findall(r'[a-z0-9]+', value.lower())


def short_prompt(appearance, category):
    appearance = appearance.strip().strip(' ;,.')
    category = category.strip().strip(' ;,.')
    if not appearance or not category:
        raise ValueError('appearance and category must be non-empty')
    appearance_words = normalized_words(appearance)
    category_words = normalized_words(category)
    if not appearance_words or not category_words:
        raise ValueError('appearance and category must contain alphanumeric words')
    width = len(category_words)
    category_present = any(
        appearance_words[index:index + width] == category_words
        for index in range(len(appearance_words) - width + 1))
    if category_present:
        return appearance
    return '{} {}'.format(appearance, category)


def main():
    args = parse_args()
    if args.expected_sequences <= 0:
        raise ValueError('--expected-sequences must be positive')
    source = args.source.resolve()
    records = []
    seen = set()
    prompt_lengths = []
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
            annotation = source_record.get('annotation')
            provenance = source_record.get('provenance')
            if (not sequence or sequence in seen or
                    not isinstance(annotation, dict)):
                raise ValueError('malformed source row {}'.format(line_number))
            if (not isinstance(provenance, dict) or
                    provenance.get('uses_initial_gt_bbox') is not True or
                    provenance.get('generated_from_target_crop') is not True or
                    provenance.get('review_status') not in REVIEW_STATUSES):
                raise ValueError(
                    'row {} lacks accepted first-frame provenance'.format(
                        line_number))
            category = str(annotation.get('category', '')).strip()
            appearance = str(annotation.get('appearance', '')).strip()
            category_source = str(
                annotation.get('category_source', '')).strip()
            if (not category or category.lower() in
                    {'unknown', 'unknown object', 'object'} or
                    not appearance or category_source != 'sequence_hint_confirmed'):
                raise ValueError(
                    'row {} lacks a reviewed category/appearance'.format(
                        line_number))
            language = short_prompt(appearance, category)
            forbidden = ('\\', '/root/', '/home/', 'd:', 'bbox',
                         'groundtruth.txt', '.jpg', '.png')
            if any(token in language.lower() for token in forbidden):
                raise ValueError(
                    'deployable prompt leaks a path/bbox token at row {}'.format(
                        line_number))
            clean = {
                'dataset': 'depthtrack_train_short',
                'sequence_name': sequence,
                'language': language,
                'annotation_quality': {
                    'is_valid': True,
                    'has_bbox_leak': False,
                    'has_absolute_path': False,
                    'source_scope': 'first_frame_appearance_category_only',
                    'prompt_strategy': PROMPT_STRATEGY,
                },
            }
            records.append(clean)
            seen.add(sequence)
            prompt_lengths.append(len(language.split()))
    if len(records) != args.expected_sequences:
        raise ValueError(
            'expected {} sequences, observed {}'.format(
                args.expected_sequences, len(records)))
    records.sort(key=lambda record: record['sequence_name'])
    output = args.output.resolve()
    payload = ''.join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n'
        for record in records).encode('utf-8')
    atomic_write(output, payload)
    receipt = {
        'schema': SCHEMA,
        'prompt_strategy': PROMPT_STRATEGY,
        'source_path': str(source),
        'source_sha256': sha256_file(source),
        'output_path': str(output),
        'output_sha256': sha256_file(output),
        'sequence_count': len(records),
        'dataset': 'depthtrack_train_short',
        'fields_used': ['annotation.appearance', 'annotation.category'],
        'excluded_fields': [
            'depth_relation', 'depth_quality', 'occlusion_state',
            'distractor_relation', 'motion_or_state'],
        'ground_truth_scope': 'first_frame_initialization_only',
        'future_frame_text_used': False,
        'bbox_or_absolute_path_materialized': False,
        'minimum_prompt_words': min(prompt_lengths),
        'maximum_prompt_words': max(prompt_lengths),
        'mean_prompt_words': sum(prompt_lengths) / len(prompt_lengths),
    }
    atomic_write(
        args.receipt.resolve(),
        (json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) +
         '\n').encode('utf-8'))
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
