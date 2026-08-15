#!/usr/bin/env python3
"""Post-hoc OFF/structured/short language oracle capacity audit.

This analyzer consumes two already-frozen Train-only paired reports. Ground
truth is never used by a tracker or runtime decision. The oracle is an upper
bound for deciding whether a learned language-action gate is worth training;
it is not a deployable policy and not a VOT metric.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


SCHEMA = 'sutrack-depthtrack-train-language-multiaction-capacity/v1'
PAIR_SCHEMA = 'sutrack-depthtrack-train-language-ablation-analysis/v1'
FAILURE_IOU = 0.1
FAILURE_GRACE = 10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--structured-analysis', type=Path, required=True)
    parser.add_argument('--short-analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_jsonl(path):
    rows = []
    with Path(path).open('r', encoding='utf-8') as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                raise ValueError('blank row {} in {}'.format(line_number, path))
            rows.append(json.loads(raw))
    return rows


def atomic_json(path, payload):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError('refusing existing output {}'.format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + '.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2,
                      sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def finite_iou(value):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError('invalid IoU {}'.format(value))
    return value


def failure_episodes(values):
    streak = 0
    episodes = 0
    for value in values:
        if value is None:
            streak = 0
        elif value <= FAILURE_IOU:
            streak += 1
            if streak == FAILURE_GRACE:
                episodes += 1
        else:
            streak = 0
    return episodes


def mean(values):
    return sum(values) / len(values)


def validate_analysis(path, expected_manifest=None):
    report = load_json(path)
    if (report.get('schema') != PAIR_SCHEMA or
            report.get('complete') is not True or
            report.get('public_evaluation') is not False or
            report.get('dataset') != 'DepthTrack Train only' or
            report.get('metric_scope') !=
            'single-start Train-only proxy; not VOT anchor multi-start EAO/ROB'):
        raise ValueError('invalid frozen analysis {}'.format(path))
    paired = report.get('paired_rows')
    if not isinstance(paired, dict):
        raise ValueError('missing paired rows in {}'.format(path))
    paired_path = Path(paired['path']).resolve()
    if (sha256_file(paired_path) != paired.get('sha256') or
            len(load_jsonl(paired_path)) != int(paired.get('rows', -1))):
        raise ValueError('paired rows mismatch in {}'.format(path))
    if (expected_manifest is not None and
            report.get('language_manifest_sha256') != expected_manifest):
        raise ValueError('unexpected language manifest in {}'.format(path))
    return report, paired_path


def main():
    args = parse_args()
    structured_path = args.structured_analysis.resolve()
    short_path = args.short_analysis.resolve()
    structured, structured_rows_path = validate_analysis(structured_path)
    short, short_rows_path = validate_analysis(short_path)
    if (structured.get('checkpoint_sha256') != short.get('checkpoint_sha256') or
            structured.get('clip_checkpoint_sha256') !=
            short.get('clip_checkpoint_sha256') or
            structured.get('implementation_sha256') !=
            short.get('implementation_sha256') or
            structured.get('sequences') != short.get('sequences') or
            structured.get('frame_count') != short.get('frame_count')):
        raise ValueError('structured/short model or coverage differs')

    structured_rows = load_jsonl(structured_rows_path)
    short_rows = load_jsonl(short_rows_path)
    if len(structured_rows) != len(short_rows):
        raise ValueError('paired row counts differ')

    series = defaultdict(lambda: defaultdict(list))
    valid_values = defaultdict(list)
    actions = Counter()
    actions_by_sequence = defaultdict(Counter)
    severe = Counter()
    for structured_row, short_row in zip(structured_rows, short_rows):
        key_fields = ('sequence', 'frame_index', 'frame_name',
                      'initialization', 'gt_valid')
        if any(structured_row.get(key) != short_row.get(key)
               for key in key_fields):
            raise ValueError('row identity differs')
        if (structured_row.get('off_bbox') != short_row.get('off_bbox') or
                structured_row.get('off_iou') != short_row.get('off_iou')):
            raise ValueError('OFF reference is not byte/value identical')
        if structured_row.get('initialization'):
            continue
        sequence = structured_row['sequence']
        values = {
            'off': finite_iou(structured_row.get('off_iou')),
            'structured': finite_iou(structured_row.get('on_iou')),
            'short': finite_iou(short_row.get('on_iou')),
        }
        if structured_row.get('gt_valid') is not True:
            if any(value is not None for value in values.values()):
                raise ValueError('invalid GT row contains IoU')
            for action in ('off', 'structured', 'short', 'oracle'):
                series[sequence][action].append(None)
            continue
        if any(value is None for value in values.values()):
            raise ValueError('valid GT row lacks IoU')
        # Ties fail closed to OFF, then structured, then short.
        priority = {'off': 2, 'structured': 1, 'short': 0}
        action = max(values, key=lambda name: (values[name], priority[name]))
        oracle = values[action]
        values['oracle'] = oracle
        actions[action] += 1
        actions_by_sequence[sequence][action] += 1
        for name, value in values.items():
            valid_values[name].append(value)
            series[sequence][name].append(value)
            if value <= FAILURE_IOU:
                severe[name] += 1

    metrics = {}
    per_sequence = []
    for name in ('off', 'structured', 'short', 'oracle'):
        metrics[name] = {
            'mean_iou': mean(valid_values[name]),
            'severe_rows': int(severe[name]),
            'failure_episodes_proxy': int(sum(
                failure_episodes(series[sequence][name])
                for sequence in sorted(series))),
        }
    for sequence in sorted(series):
        item = {'sequence': sequence,
                'oracle_actions': dict(actions_by_sequence[sequence])}
        for name in ('off', 'structured', 'short', 'oracle'):
            sequence_values = [
                value for value in series[sequence][name]
                if value is not None]
            item[name] = {
                'mean_iou': mean(sequence_values),
                'failure_episodes_proxy': failure_episodes(
                    series[sequence][name]),
            }
        per_sequence.append(item)

    mean_delta = metrics['oracle']['mean_iou'] - metrics['off']['mean_iou']
    failure_delta = (metrics['oracle']['failure_episodes_proxy'] -
                     metrics['off']['failure_episodes_proxy'])
    checks = {
        'oracle_mean_iou_delta_at_least_0_01': mean_delta >= 0.01,
        'oracle_severe_rows_not_increased': (
            metrics['oracle']['severe_rows'] <= metrics['off']['severe_rows']),
        'oracle_failure_episodes_strictly_reduced': failure_delta < 0,
        'all_three_actions_used': all(actions[name] > 0 for name in
                                      ('off', 'structured', 'short')),
    }
    if (checks['oracle_mean_iou_delta_at_least_0_01'] and
            checks['oracle_severe_rows_not_increased'] and
            checks['all_three_actions_used'] and
            not checks['oracle_failure_episodes_strictly_reduced']):
        decision = 'mean_overlap_capacity_without_robustness_capacity'
    elif all(checks.values()):
        decision = 'multiaction_capacity_supported'
    else:
        decision = 'multiaction_capacity_not_supported'

    result = {
        'schema': SCHEMA,
        'complete': True,
        'dataset': 'DepthTrack Train only',
        'public_evaluation': False,
        'deployable_policy': False,
        'oracle_uses_full_train_gt_post_hoc': True,
        'decision': decision,
        'checks': checks,
        'frame_count': int(structured['frame_count']),
        'valid_noninitialization_rows': len(valid_values['off']),
        'metrics': metrics,
        'oracle_mean_iou_delta_over_off': mean_delta,
        'oracle_failure_episode_delta_over_off': failure_delta,
        'oracle_actions': dict(actions),
        'per_sequence': per_sequence,
        'sources': {
            'structured_analysis': {
                'path': str(structured_path),
                'sha256': sha256_file(structured_path),
                'paired_rows_path': str(structured_rows_path),
                'paired_rows_sha256': sha256_file(structured_rows_path),
                'language_manifest_sha256':
                    structured['language_manifest_sha256'],
            },
            'short_analysis': {
                'path': str(short_path),
                'sha256': sha256_file(short_path),
                'paired_rows_path': str(short_rows_path),
                'paired_rows_sha256': sha256_file(short_rows_path),
                'language_manifest_sha256': short['language_manifest_sha256'],
            },
        },
        'metric_scope': (
            'post-hoc per-frame Train-only GT oracle upper bound; '
            'not a runtime gate and not VOT EAO/ACC/ROB'),
    }
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
