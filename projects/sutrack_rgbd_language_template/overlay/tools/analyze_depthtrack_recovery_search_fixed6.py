#!/usr/bin/env python3
"""Strictly post-inference OFF/ON audit for fixed6 recovery search."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


SCHEMA = 'sutrack-depthtrack-recovery-search-fixed6-analysis/v1'
REFERENCE_SCHEMA = 'sutrack-depthtrack-train-state-trace/v1'
CANDIDATE_SCHEMA = 'sutrack-depthtrack-train-recovery-search-trace/v1'
EXPECTED_SEQUENCES = (
    'bottle03_indoor',
    'ball16_indoor',
    'bag04_indoor',
    'flower03_indoor',
    'pigeon05_wild',
    'toy03_indoor',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-shard-dir', action='append', type=Path,
                        required=True)
    parser.add_argument('--off-shard-dir', action='append', type=Path,
                        required=True)
    parser.add_argument('--on-shard-dir', action='append', type=Path,
                        required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--expected-factor', type=float, default=6.0)
    parser.add_argument('--expected-maximum-consecutive', type=int, default=1)
    parser.add_argument('--expected-cooldown-frames', type=int, default=2)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    with Path(path).open('r', encoding='utf-8') as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError('{} is not an object'.format(path))
    return value


def load_jsonl(path):
    rows = []
    with Path(path).open('r', encoding='utf-8') as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                raise ValueError('blank row {}:{}'.format(path, line_number))
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError('non-object row {}:{}'.format(
                    path, line_number))
            rows.append(row)
    return rows


def finite_bbox(values):
    try:
        bbox = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    if (len(bbox) != 4 or not all(math.isfinite(value) for value in bbox) or
            bbox[2] <= 0.0 or bbox[3] <= 0.0):
        return None
    return bbox


def read_gt(path):
    rows = []
    with Path(path).open('r', encoding='utf-8') as stream:
        for raw_line in stream:
            values = raw_line.strip().replace('\t', ',').split(',')
            try:
                raw_bbox = [float(value) for value in values]
            except ValueError:
                raw_bbox = []
            rows.append(finite_bbox(raw_bbox))
    if not rows or rows[0] is None:
        raise ValueError('invalid initialization GT {}'.format(path))
    return rows


def iou_xywh(first, second):
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    intersection = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0.0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def failure_stats(values, threshold=0.10, minimum_run=10):
    runs = []
    current = 0
    for value in values:
        if value is not None and value <= threshold:
            current += 1
        else:
            if current:
                runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return {
        'ten_frame_failure_starts': sum(
            int(length >= minimum_run) for length in runs),
        'longest_low_iou_run': max(runs) if runs else 0,
        'low_iou_frames': sum(runs),
        'low_iou_run_count': len(runs),
    }


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + '.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True,
                      indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_file_record(record, source_snapshot):
    if not isinstance(record, dict):
        raise ValueError('malformed file record')
    path = Path(record.get('path', '')).resolve()
    expected_sha = record.get('sha256')
    if (not path.is_file() or sha256_file(path) != expected_sha or
            path.stat().st_size != int(record.get('bytes', -1))):
        raise ValueError('file record mismatch {}'.format(path))
    source_snapshot[str(path)] = expected_sha
    return path


def load_shards(shard_dirs, role, dataset_root, source_snapshot):
    if len(shard_dirs) != 2:
        raise ValueError('{} requires exactly two shards'.format(role))
    rows = []
    sequences = []
    contract = None
    manifests = []
    for shard_dir in shard_dirs:
        manifest_path = shard_dir.resolve() / 'manifest.json'
        manifest = load_json(manifest_path)
        expected_schema = (REFERENCE_SCHEMA if role == 'reference' else
                           CANDIDATE_SCHEMA)
        if (manifest.get('schema') != expected_schema or
                manifest.get('complete') is not True or
                Path(manifest.get('dataset_root', '')).resolve() !=
                dataset_root or
                manifest.get('ground_truth_consumption') !=
                'first_frame_initialization_only' or
                manifest.get('ground_truth_available_to_tracker') is not False or
                manifest.get('future_frame_text_used') is not False or
                manifest.get('public_evaluation') is not False):
            raise ValueError('{} manifest contract failed {}'.format(
                role, manifest_path))
        if role == 'off':
            recovery = manifest.get('recovery_search', {})
            if (manifest.get('role') !=
                    'source_identical_recovery_search_off' or
                    recovery.get('enabled') is not False or
                    int(recovery.get('second_pass_count', -1)) != 0 or
                    int(recovery.get('recovery_selected_count', -1)) != 0):
                raise ValueError('OFF path was not inert')
        elif role == 'on':
            recovery = manifest.get('recovery_search', {})
            if (manifest.get('role') != 'recovery_search_on' or
                    recovery.get('enabled') is not True or
                    recovery.get('selection_uses_ground_truth') is not False):
                raise ValueError('ON path contract failed')
        prediction_path = validate_file_record(
            manifest['predictions'], source_snapshot)
        for key in ('config', 'checkpoint', 'language_manifest'):
            validate_file_record(manifest[key], source_snapshot)
        source_snapshot[str(manifest_path)] = sha256_file(manifest_path)
        implementation = manifest.get('implementation_sha256')
        if not isinstance(implementation, dict):
            raise ValueError('missing implementation snapshot')
        for relative, expected_sha in implementation.items():
            path = Path('/home/SUTrack_RGBD_L') / relative
            if sha256_file(path) != expected_sha:
                raise ValueError('implementation drift {}'.format(relative))
        current_contract = {
            'config': manifest['config'],
            'checkpoint': manifest['checkpoint'],
            'language_manifest': manifest['language_manifest'],
        }
        if contract is None:
            contract = current_contract
        elif current_contract != contract:
            raise ValueError('{} shard source mismatch'.format(role))
        shard_rows = load_jsonl(prediction_path)
        if len(shard_rows) != int(manifest.get('frame_count', -1)):
            raise ValueError('{} frame count mismatch'.format(role))
        rows.extend(shard_rows)
        sequences.extend(manifest.get('sequences', []))
        manifests.append(manifest)
    if (len(sequences) != len(set(sequences)) or
            set(sequences) != set(EXPECTED_SEQUENCES)):
        raise ValueError('{} fixed6 coverage mismatch'.format(role))
    return rows, contract, manifests


def index_predictions(rows, role):
    indexed = {}
    selected = defaultdict(int)
    for row in rows:
        key = (row.get('sequence'), int(row.get('frame_index', -1)))
        bbox = finite_bbox(row.get('deployed_bbox'))
        if (bbox is None or key in indexed or
                key[0] not in set(EXPECTED_SEQUENCES) or key[1] < 0):
            raise ValueError('malformed {} prediction {}'.format(role, key))
        recovery_selected = bool(row.get('recovery_selected', False))
        if role == 'off' and recovery_selected:
            raise ValueError('OFF prediction selected recovery')
        selected[key[0]] += int(recovery_selected)
        indexed[key] = {
            'bbox': bbox,
            'recovery_selected': recovery_selected,
        }
    return indexed, selected


def main():
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if (float(args.expected_factor) <= 4.0 or
            int(args.expected_maximum_consecutive) != 1 or
            int(args.expected_cooldown_frames) != 2):
        raise ValueError('fixed6 recovery protocol is frozen at factor>4, 1/2')
    plan_path = args.plan.resolve()
    plan = load_json(plan_path)
    frozen_gate = {
        'minimum_recovery_action_rows': 5,
        'minimum_recovery_action_sequences': 3,
        'minimum_action_precision': 0.8,
        'maximum_action_harm_rate': 0.05,
        'maximum_catastrophic_action_rows': 0,
        'minimum_mean_iou_delta': 0.0,
        'maximum_ten_frame_failure_starts_delta': -1,
        'maximum_catastrophic_sequence_regressions': 0,
    }
    if (plan.get('schema') != 'sutrack-recovery-search-fixed6-plan/v1' or
            plan.get('complete') is not True or
            plan.get('created_before_off_or_on_inference') is not True or
            plan.get('dataset') != 'DepthTrack Train fixed6 only' or
            Path(plan.get('dataset_root', '')).resolve() != dataset_root or
            plan.get('public_evaluation') is not False or
            plan.get('future_frame_text_used') is not False or
            plan.get('source_identical_off_must_pass_before_on') is not True or
            float(plan.get('off_maximum_bbox_difference', -1.0)) != 0.0 or
            float(plan.get('factor', -1.0)) != float(args.expected_factor) or
            int(plan.get('maximum_consecutive_second_passes', -1)) !=
            int(args.expected_maximum_consecutive) or
            int(plan.get('cooldown_frames', -1)) !=
            int(args.expected_cooldown_frames) or
            plan.get('fixed6_gate') != frozen_gate):
        raise ValueError('frozen recovery-search plan mismatch')
    for record in plan.get('reference_manifests', []):
        path = Path(record.get('path', '')).resolve()
        if sha256_file(path) != record.get('sha256'):
            raise ValueError('frozen reference manifest drift {}'.format(path))
    if len(plan.get('reference_manifests', [])) != 2:
        raise ValueError('plan must bind exactly two reference manifests')
    for path_string, expected_sha in plan.get(
            'implementation_sha256', {}).items():
        if sha256_file(path_string) != expected_sha:
            raise ValueError('planned implementation drift {}'.format(
                path_string))
    source_snapshot = {
        str(plan_path): sha256_file(plan_path),
        str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve())}
    reference_rows, reference_contract, reference_manifests = load_shards(
        args.reference_shard_dir, 'reference', dataset_root, source_snapshot)
    off_rows, off_contract, off_manifests = load_shards(
        args.off_shard_dir, 'off', dataset_root, source_snapshot)
    on_rows, on_contract, on_manifests = load_shards(
        args.on_shard_dir, 'on', dataset_root, source_snapshot)
    if not (reference_contract == off_contract == on_contract):
        raise ValueError('OFF/ON do not use the frozen reference source')
    for manifest in off_manifests + on_manifests:
        recovery = manifest['recovery_search']
        if (float(recovery['factor']) != float(args.expected_factor) or
                int(recovery['maximum_consecutive_second_passes']) !=
                int(args.expected_maximum_consecutive) or
                int(recovery['cooldown_frames']) !=
                int(args.expected_cooldown_frames)):
            raise ValueError('recovery protocol mismatch')

    reference, _ = index_predictions(reference_rows, 'reference')
    off, _ = index_predictions(off_rows, 'off')
    on, selected_by_sequence = index_predictions(on_rows, 'on')
    if not (set(reference) == set(off) == set(on)):
        raise ValueError('OFF/ON prediction coverage mismatch')
    maximum_off_difference = 0.0
    for key in reference:
        difference = max(abs(a - b) for a, b in zip(
            reference[key]['bbox'], off[key]['bbox']))
        maximum_off_difference = max(maximum_off_difference, difference)
        if difference != 0.0:
            raise ValueError('OFF path perturbed source prediction {}'.format(key))

    # Strictly post-inference GT join begins here.
    groundtruth = {}
    for sequence in EXPECTED_SEQUENCES:
        path = dataset_root / sequence / 'groundtruth.txt'
        groundtruth[sequence] = read_gt(path)
        source_snapshot[str(path)] = sha256_file(path)
    expected_keys = {
        (sequence, frame_index)
        for sequence in EXPECTED_SEQUENCES
        for frame_index in range(len(groundtruth[sequence]))
    }
    if set(reference) != expected_keys:
        raise ValueError('fixed6 frame/GT coverage mismatch')

    sequence_metrics = {}
    all_reference = []
    all_on = []
    action_rows = 0
    beneficial_actions = 0
    harmful_actions = 0
    catastrophic_actions = 0
    reference_failure_total = 0
    on_failure_total = 0
    catastrophic_sequences = []
    for sequence in EXPECTED_SEQUENCES:
        reference_values = []
        on_values = []
        sequence_action_deltas = []
        for frame_index, gt in enumerate(groundtruth[sequence]):
            key = (sequence, frame_index)
            if gt is None:
                reference_values.append(None)
                on_values.append(None)
                continue
            reference_iou = iou_xywh(reference[key]['bbox'], gt)
            on_iou = iou_xywh(on[key]['bbox'], gt)
            reference_values.append(reference_iou)
            on_values.append(on_iou)
            all_reference.append(reference_iou)
            all_on.append(on_iou)
            if on[key]['recovery_selected']:
                delta = on_iou - reference_iou
                sequence_action_deltas.append(delta)
                action_rows += 1
                beneficial_actions += int(delta >= 0.02)
                harmful_actions += int(delta <= -0.02)
                catastrophic_actions += int(
                    reference_iou >= 0.50 and on_iou <= 0.10)
        valid_reference = [v for v in reference_values if v is not None]
        valid_on = [v for v in on_values if v is not None]
        reference_mean = sum(valid_reference) / len(valid_reference)
        on_mean = sum(valid_on) / len(valid_on)
        reference_failure = failure_stats(reference_values)
        on_failure = failure_stats(on_values)
        failure_delta = (on_failure['ten_frame_failure_starts'] -
                         reference_failure['ten_frame_failure_starts'])
        mean_delta = on_mean - reference_mean
        catastrophic_sequence = bool(mean_delta < -0.05 and failure_delta > 0)
        if catastrophic_sequence:
            catastrophic_sequences.append(sequence)
        reference_failure_total += reference_failure[
            'ten_frame_failure_starts']
        on_failure_total += on_failure['ten_frame_failure_starts']
        sequence_metrics[sequence] = {
            'valid_gt_frames': len(valid_reference),
            'reference_mean_iou': reference_mean,
            'on_mean_iou': on_mean,
            'mean_iou_delta': mean_delta,
            'reference_failure_stats': reference_failure,
            'on_failure_stats': on_failure,
            'ten_frame_failure_starts_delta': failure_delta,
            'recovery_selected': selected_by_sequence[sequence],
            'selected_valid_gt_rows': len(sequence_action_deltas),
            'selected_net_iou_gain': sum(sequence_action_deltas),
            'catastrophic_sequence_regression': catastrophic_sequence,
        }

    reference_mean = sum(all_reference) / len(all_reference)
    on_mean = sum(all_on) / len(all_on)
    mean_delta = on_mean - reference_mean
    failure_delta = on_failure_total - reference_failure_total
    selected_sequences = sum(
        int(value > 0) for value in selected_by_sequence.values())
    action_precision = (
        beneficial_actions / action_rows if action_rows else 0.0)
    action_harm_rate = harmful_actions / action_rows if action_rows else 0.0
    checks = {
        'off_source_identical_zero_difference': maximum_off_difference == 0.0,
        'at_least_5_recovery_actions': action_rows >= 5,
        'recovery_actions_cover_at_least_3_sequences': selected_sequences >= 3,
        'action_precision_at_least_0p80': action_precision >= 0.80,
        'action_harm_rate_at_most_0p05': action_harm_rate <= 0.05,
        'zero_catastrophic_actions': catastrophic_actions == 0,
        'mean_iou_non_decreasing': mean_delta >= 0.0,
        'failure_starts_reduced_by_at_least_1': failure_delta <= -1,
        'zero_catastrophic_sequence_regressions':
            len(catastrophic_sequences) == 0,
    }
    passed = all(checks.values())
    result = {
        'schema': SCHEMA,
        'complete': True,
        'decision': ('fixed6_recovery_search_supported' if passed else
                     'fixed6_recovery_search_rejected'),
        'fixed6_recovery_search_supported': passed,
        'eligible_for_full152_recovery_trace': passed,
        'scope': 'DepthTrack Train fixed6 only',
        'ground_truth_join': 'strictly_post_inference',
        'future_frame_text_used': False,
        'public_evaluation': False,
        'expected_sequences': list(EXPECTED_SEQUENCES),
        'expected_factor': float(args.expected_factor),
        'maximum_consecutive_second_passes': int(
            args.expected_maximum_consecutive),
        'cooldown_frames': int(args.expected_cooldown_frames),
        'valid_gt_frames': len(all_reference),
        'off_maximum_bbox_difference': maximum_off_difference,
        'reference_mean_iou': reference_mean,
        'on_mean_iou': on_mean,
        'mean_iou_delta': mean_delta,
        'reference_ten_frame_failure_starts': reference_failure_total,
        'on_ten_frame_failure_starts': on_failure_total,
        'ten_frame_failure_starts_delta': failure_delta,
        'recovery_action_rows': action_rows,
        'recovery_action_sequences': selected_sequences,
        'beneficial_action_rows': beneficial_actions,
        'harmful_action_rows': harmful_actions,
        'catastrophic_action_rows': catastrophic_actions,
        'action_precision': action_precision,
        'action_harm_rate': action_harm_rate,
        'catastrophic_sequence_regressions': catastrophic_sequences,
        'checks': checks,
        'sequence_metrics': sequence_metrics,
        'source_contract': reference_contract,
        'source_snapshot_sha256': source_snapshot,
        'analyzer_path': str(Path(__file__).resolve()),
        'analyzer_sha256': source_snapshot[str(Path(__file__).resolve())],
        'plan_path': str(plan_path),
        'plan_sha256': source_snapshot[str(plan_path)],
    }
    for path_string, expected_sha in source_snapshot.items():
        if sha256_file(path_string) != expected_sha:
            raise ValueError('source drift during fixed6 analysis {}'.format(
                path_string))
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('refusing non-empty output {}'.format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / 'fixed6_recovery_result.json', result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2,
                     allow_nan=False))


if __name__ == '__main__':
    main()
