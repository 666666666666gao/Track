#!/usr/bin/env python3
"""Post-inference paired analysis for SUTrack language ON/OFF traces."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import yaml


TRACE_SCHEMA = 'sutrack-depthtrack-train-language-ablation-trace/v1'
SCHEMA = 'sutrack-depthtrack-train-language-ablation-analysis/v1'
ALLOWED_CONFIG_DIFFS = {
    ('TEST', 'USE_NLP', 'DEPTHTRACK'): (True, False),
    ('TEST', 'RGBD_LANGUAGE', 'USE'): (True, False),
}
MIN_MEAN_IOU_DELTA = 0.0
MIN_NONNEGATIVE_SEQUENCES = 4
FAILURE_IOU = 0.10
FAILURE_GRACE = 10
PAIR_MARGIN = 0.05
CATASTROPHIC_GOOD_IOU = 0.50
CATASTROPHIC_BAD_IOU = 0.10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--on-shard-dir', action='append', type=Path,
                        required=True)
    parser.add_argument('--off-shard-dir', action='append', type=Path,
                        required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--expected-sequences', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
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


def load_json(path):
    with Path(path).open('r', encoding='utf-8') as stream:
        return json.load(stream)


def load_jsonl(path):
    records = []
    with Path(path).open('r', encoding='utf-8') as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                raise ValueError('blank row {}:{}'.format(path, line_number))
            record = json.loads(raw_line)
            if not isinstance(record, dict):
                raise ValueError('non-object row {}:{}'.format(
                    path, line_number))
            records.append(record)
    return records


def validate_file_record(record):
    if not isinstance(record, dict):
        raise ValueError('malformed file record')
    path = Path(record['path']).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != record.get('sha256'):
        raise ValueError('SHA mismatch {}'.format(path))
    if path.stat().st_size != record.get('bytes'):
        raise ValueError('byte-count mismatch {}'.format(path))
    return path


def finite_bbox(values):
    try:
        bbox = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    if (len(bbox) != 4 or not all(math.isfinite(value) for value in bbox) or
            bbox[2] <= 0.0 or bbox[3] <= 0.0):
        return None
    return bbox


def iou_xywh(first, second):
    ax0, ay0, aw, ah = first
    bx0, by0, bw, bh = second
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    intersection = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(
        0.0, min(ay1, by1) - max(ay0, by0))
    union = aw * ah + bw * bh - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def read_gt(path):
    boxes = []
    with Path(path).open('r', encoding='utf-8') as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            values = raw_line.strip().replace('\t', ',').split(',')
            if len(values) != 4:
                raise ValueError('malformed GT {}:{}'.format(
                    path, line_number))
            try:
                raw_bbox = [float(value) for value in values]
            except ValueError as error:
                raise ValueError('malformed GT {}:{}'.format(
                    path, line_number)) from error
            boxes.append(finite_bbox(raw_bbox))
    if not boxes:
        raise ValueError('empty GT {}'.format(path))
    return boxes


def flatten(value, prefix=()):
    if isinstance(value, dict):
        output = {}
        for key, nested in value.items():
            output.update(flatten(nested, prefix + (str(key),)))
        return output
    if isinstance(value, list):
        output = {}
        for index, nested in enumerate(value):
            output.update(flatten(nested, prefix + (str(index),)))
        return output
    return {prefix: value}


def validate_config_pair(on_path, off_path):
    with on_path.open('r', encoding='utf-8') as stream:
        on_config = yaml.safe_load(stream)
    with off_path.open('r', encoding='utf-8') as stream:
        off_config = yaml.safe_load(stream)
    on_flat = flatten(on_config)
    off_flat = flatten(off_config)
    if set(on_flat) != set(off_flat):
        raise ValueError('ON/OFF config key sets differ')
    observed = {
        key: (on_flat[key], off_flat[key])
        for key in on_flat if on_flat[key] != off_flat[key]
    }
    if observed != ALLOWED_CONFIG_DIFFS:
        raise ValueError('unexpected ON/OFF config differences: {}'.format(
            observed))
    return {
        '.'.join(key): {'on': values[0], 'off': values[1]}
        for key, values in observed.items()
    }


def load_branch(shard_dirs, mode, dataset_root, expected_sequences):
    repository_root = Path('/home/SUTrack_RGBD_L')
    manifests = []
    predictions = []
    observed_sequences = []
    source_snapshot = {}
    common = None
    for shard_dir in shard_dirs:
        shard_dir = shard_dir.resolve()
        manifest_path = shard_dir / 'manifest.json'
        manifest = load_json(manifest_path)
        source_snapshot[str(manifest_path)] = sha256_file(manifest_path)
        if (manifest.get('schema') != TRACE_SCHEMA or
                manifest.get('complete') is not True or
                manifest.get('dataset') != 'DepthTrack Train only' or
                Path(manifest.get('dataset_root', '')).resolve() != dataset_root or
                manifest.get('language_mode') != mode or
                manifest.get('language_enabled') is not (mode == 'on') or
                manifest.get('ground_truth_consumption') !=
                'first_frame_initialization_only' or
                manifest.get('ground_truth_available_to_tracker_after_initialization')
                is not False or
                manifest.get('future_frame_text_used') is not False or
                manifest.get('public_evaluation') is not False):
            raise ValueError('invalid {} manifest {}'.format(mode, manifest_path))
        predictions_path = validate_file_record(manifest['predictions'])
        source_snapshot[str(predictions_path)] = sha256_file(predictions_path)
        rows = load_jsonl(predictions_path)
        if (len(rows) != manifest.get('prediction_row_count') or
                len(rows) != manifest.get('frame_count')):
            raise ValueError('row-count mismatch {}'.format(shard_dir))
        config_path = validate_file_record(manifest['config'])
        checkpoint_path = validate_file_record(manifest['checkpoint'])
        clip_path = validate_file_record(manifest['clip_checkpoint'])
        language_record = manifest.get('language_manifest')
        if mode == 'on':
            language_path = validate_file_record(language_record)
        elif language_record is not None:
            raise ValueError('OFF branch unexpectedly binds language manifest')
        else:
            language_path = None
        contract = {
            'config_path': config_path,
            'checkpoint_path': checkpoint_path,
            'checkpoint_sha256': manifest['checkpoint']['sha256'],
            'clip_path': clip_path,
            'clip_sha256': manifest['clip_checkpoint']['sha256'],
            'language_path': language_path,
            'language_sha256': (
                language_record['sha256'] if language_record else None),
            'implementation_sha256': manifest['implementation_sha256'],
        }
        if common is None:
            common = contract
        elif contract != common:
            raise ValueError('{} shards do not share one contract'.format(mode))
        for relative, expected_sha in manifest['implementation_sha256'].items():
            path = repository_root / relative
            if sha256_file(path) != expected_sha:
                raise ValueError('implementation drift {}'.format(relative))
        observed_sequences.extend(manifest['sequences'])
        predictions.extend(rows)
        manifests.append(manifest)
    if (observed_sequences != expected_sequences or
            len(observed_sequences) != len(set(observed_sequences))):
        raise ValueError('{} sequence scope/order mismatch'.format(mode))
    if common is None:
        raise ValueError('empty {} branch'.format(mode))
    return {
        'contract': common,
        'manifests': manifests,
        'predictions': predictions,
        'source_snapshot': source_snapshot,
    }


def failure_episode_count(ious):
    streak = 0
    episodes = 0
    for value in ious:
        if value is None:
            streak = 0
        elif value <= FAILURE_IOU:
            streak += 1
            if streak == FAILURE_GRACE:
                episodes += 1
        else:
            streak = 0
    return episodes


def main():
    args = parse_args()
    expected_sequences = [
        item.strip() for item in args.expected_sequences.split(',')
        if item.strip()]
    if (not expected_sequences or
            len(expected_sequences) != len(set(expected_sequences))):
        raise ValueError('expected sequences must be non-empty and unique')
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('refusing non-empty output {}'.format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate and freeze both inference branches before opening full GT.
    on = load_branch(
        args.on_shard_dir, 'on', dataset_root, expected_sequences)
    off = load_branch(
        args.off_shard_dir, 'off', dataset_root, expected_sequences)
    if (on['contract']['checkpoint_sha256'] !=
            off['contract']['checkpoint_sha256'] or
            on['contract']['clip_sha256'] != off['contract']['clip_sha256'] or
            on['contract']['implementation_sha256'] !=
            off['contract']['implementation_sha256']):
        raise ValueError('ON/OFF model or implementation differs')
    config_differences = validate_config_pair(
        on['contract']['config_path'], off['contract']['config_path'])
    if on['contract']['language_path'] is None or off['contract']['language_path'] is not None:
        raise ValueError('language provenance mismatch')

    on_rows = {}
    off_rows = {}
    for mode, branch, destination in (
            ('on', on, on_rows), ('off', off, off_rows)):
        for record in branch['predictions']:
            if (record.get('schema') != TRACE_SCHEMA or
                    record.get('language_mode') != mode or
                    record.get('future_frame_text_used') is not False):
                raise ValueError('invalid {} prediction row'.format(mode))
            key = (record['sequence'], int(record['frame_index']))
            if key in destination:
                raise ValueError('duplicate {} row {}'.format(mode, key))
            bbox = finite_bbox(record.get('bbox'))
            if bbox is None:
                raise ValueError('non-finite {} bbox {}'.format(mode, key))
            destination[key] = record
    if set(on_rows) != set(off_rows):
        raise ValueError('ON/OFF frame coverage differs')

    # Full Train GT is consumed only here, after both branches are immutable.
    gt_by_sequence = {}
    source_snapshot = {}
    source_snapshot.update(on['source_snapshot'])
    source_snapshot.update(off['source_snapshot'])
    for sequence in expected_sequences:
        gt_path = dataset_root / sequence / 'groundtruth.txt'
        gt_by_sequence[sequence] = read_gt(gt_path)
        source_snapshot[str(gt_path)] = sha256_file(gt_path)

    paired_rows = []
    sequence_ious = defaultdict(lambda: {'on': [], 'off': []})
    sequence_failure_series = defaultdict(lambda: {'on': [], 'off': []})
    sequence_counts = defaultdict(Counter)
    aggregate = Counter()
    for sequence in expected_sequences:
        frame_count = len(gt_by_sequence[sequence])
        for frame_index in range(frame_count):
            key = (sequence, frame_index)
            if key not in on_rows or key not in off_rows:
                raise ValueError('prediction/GT coverage mismatch {}'.format(key))
            on_record = on_rows[key]
            off_record = off_rows[key]
            if (on_record['frame_name'] != off_record['frame_name'] or
                    bool(on_record['initialization']) != (frame_index == 0) or
                    bool(off_record['initialization']) != (frame_index == 0)):
                raise ValueError('paired frame metadata mismatch {}'.format(key))
            gt = gt_by_sequence[sequence][frame_index]
            on_iou = iou_xywh(on_record['bbox'], gt) if gt else None
            off_iou = iou_xywh(off_record['bbox'], gt) if gt else None
            row = {
                'sequence': sequence,
                'frame_index': frame_index,
                'frame_name': on_record['frame_name'],
                'initialization': frame_index == 0,
                'gt_valid': gt is not None,
                'on_bbox': on_record['bbox'],
                'off_bbox': off_record['bbox'],
                'on_iou': on_iou,
                'off_iou': off_iou,
                'iou_delta_on_minus_off': (
                    on_iou - off_iou if gt is not None else None),
            }
            paired_rows.append(row)
            if frame_index == 0:
                continue
            sequence_failure_series[sequence]['on'].append(on_iou)
            sequence_failure_series[sequence]['off'].append(off_iou)
            if gt is None:
                continue
            sequence_ious[sequence]['on'].append(on_iou)
            sequence_ious[sequence]['off'].append(off_iou)
            sequence_counts[sequence]['valid_rows'] += 1
            aggregate['valid_rows'] += 1
            delta = on_iou - off_iou
            if delta >= PAIR_MARGIN:
                sequence_counts[sequence]['beneficial_rows'] += 1
                aggregate['beneficial_rows'] += 1
            if delta <= -PAIR_MARGIN:
                sequence_counts[sequence]['harmful_rows'] += 1
                aggregate['harmful_rows'] += 1
            if off_iou >= CATASTROPHIC_GOOD_IOU and on_iou <= CATASTROPHIC_BAD_IOU:
                sequence_counts[sequence]['catastrophic_rows'] += 1
                aggregate['catastrophic_rows'] += 1
            if off_iou <= CATASTROPHIC_BAD_IOU and on_iou >= CATASTROPHIC_GOOD_IOU:
                sequence_counts[sequence]['rescue_rows'] += 1
                aggregate['rescue_rows'] += 1
            if on_iou <= FAILURE_IOU:
                sequence_counts[sequence]['on_severe_rows'] += 1
                aggregate['on_severe_rows'] += 1
            if off_iou <= FAILURE_IOU:
                sequence_counts[sequence]['off_severe_rows'] += 1
                aggregate['off_severe_rows'] += 1

    per_sequence = []
    total_on_iou = 0.0
    total_off_iou = 0.0
    nonnegative_sequences = 0
    for sequence in expected_sequences:
        on_values = sequence_ious[sequence]['on']
        off_values = sequence_ious[sequence]['off']
        if not on_values or len(on_values) != len(off_values):
            raise ValueError('empty paired valid rows {}'.format(sequence))
        on_mean = sum(on_values) / len(on_values)
        off_mean = sum(off_values) / len(off_values)
        delta = on_mean - off_mean
        if delta >= 0.0:
            nonnegative_sequences += 1
        on_failures = failure_episode_count(
            sequence_failure_series[sequence]['on'])
        off_failures = failure_episode_count(
            sequence_failure_series[sequence]['off'])
        aggregate['on_failure_episodes'] += on_failures
        aggregate['off_failure_episodes'] += off_failures
        total_on_iou += sum(on_values)
        total_off_iou += sum(off_values)
        counts = sequence_counts[sequence]
        per_sequence.append({
            'sequence': sequence,
            'valid_rows': len(on_values),
            'on_mean_iou': on_mean,
            'off_mean_iou': off_mean,
            'mean_iou_delta_on_minus_off': delta,
            'on_failure_episodes_proxy': on_failures,
            'off_failure_episodes_proxy': off_failures,
            'failure_episode_delta_on_minus_off': on_failures - off_failures,
            'on_severe_rows': counts['on_severe_rows'],
            'off_severe_rows': counts['off_severe_rows'],
            'beneficial_rows': counts['beneficial_rows'],
            'harmful_rows': counts['harmful_rows'],
            'catastrophic_rows': counts['catastrophic_rows'],
            'rescue_rows': counts['rescue_rows'],
        })

    valid_rows = int(aggregate['valid_rows'])
    on_mean_iou = total_on_iou / valid_rows
    off_mean_iou = total_off_iou / valid_rows
    mean_delta = on_mean_iou - off_mean_iou
    checks = {
        'mean_iou_nonnegative': mean_delta >= MIN_MEAN_IOU_DELTA,
        'failure_episodes_not_increased': (
            aggregate['on_failure_episodes'] <=
            aggregate['off_failure_episodes']),
        'severe_rows_not_increased': (
            aggregate['on_severe_rows'] <= aggregate['off_severe_rows']),
        'catastrophic_rows_not_exceed_rescues': (
            aggregate['catastrophic_rows'] <= aggregate['rescue_rows']),
        'sequence_breadth': nonnegative_sequences >= MIN_NONNEGATIVE_SEQUENCES,
    }
    supported = all(checks.values())
    result = {
        'schema': SCHEMA,
        'complete': True,
        'decision': (
            'structured_language_supported_on_fixed6'
            if supported else 'structured_language_not_supported_on_fixed6'),
        'language_supported': supported,
        'dataset': 'DepthTrack Train only',
        'public_evaluation': False,
        'sequence_count': len(expected_sequences),
        'sequences': expected_sequences,
        'frame_count': len(paired_rows),
        'valid_noninitialization_rows': valid_rows,
        'on_mean_iou': on_mean_iou,
        'off_mean_iou': off_mean_iou,
        'mean_iou_delta_on_minus_off': mean_delta,
        'on_failure_episodes_proxy': int(aggregate['on_failure_episodes']),
        'off_failure_episodes_proxy': int(aggregate['off_failure_episodes']),
        'failure_episode_delta_on_minus_off': int(
            aggregate['on_failure_episodes'] -
            aggregate['off_failure_episodes']),
        'on_severe_rows': int(aggregate['on_severe_rows']),
        'off_severe_rows': int(aggregate['off_severe_rows']),
        'beneficial_rows': int(aggregate['beneficial_rows']),
        'harmful_rows': int(aggregate['harmful_rows']),
        'catastrophic_rows': int(aggregate['catastrophic_rows']),
        'rescue_rows': int(aggregate['rescue_rows']),
        'nonnegative_sequences': nonnegative_sequences,
        'checks': checks,
        'thresholds': {
            'minimum_mean_iou_delta': MIN_MEAN_IOU_DELTA,
            'minimum_nonnegative_sequences': MIN_NONNEGATIVE_SEQUENCES,
            'failure_iou': FAILURE_IOU,
            'failure_grace_frames': FAILURE_GRACE,
            'pair_margin': PAIR_MARGIN,
            'catastrophic_good_iou': CATASTROPHIC_GOOD_IOU,
            'catastrophic_bad_iou': CATASTROPHIC_BAD_IOU,
        },
        'metric_scope': (
            'single-start Train-only proxy; not VOT anchor multi-start EAO/ROB'),
        'config_differences': config_differences,
        'checkpoint_sha256': on['contract']['checkpoint_sha256'],
        'clip_checkpoint_sha256': on['contract']['clip_sha256'],
        'language_manifest_sha256': on['contract']['language_sha256'],
        'implementation_sha256': on['contract']['implementation_sha256'],
        'analyzer_sha256': sha256_file(Path(__file__).resolve()),
        'source_snapshot': source_snapshot,
        'per_sequence': per_sequence,
    }
    rows_path = output_dir / 'paired_rows.jsonl'
    atomic_write(rows_path, ''.join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n'
        for row in paired_rows).encode('utf-8'))
    result['paired_rows'] = {
        'path': str(rows_path),
        'sha256': sha256_file(rows_path),
        'bytes': rows_path.stat().st_size,
        'rows': len(paired_rows),
    }
    result_path = output_dir / 'analysis.json'
    atomic_write(result_path, (
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) +
        '\n').encode('utf-8'))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
