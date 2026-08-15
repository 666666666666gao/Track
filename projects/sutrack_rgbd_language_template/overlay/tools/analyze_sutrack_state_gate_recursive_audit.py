#!/usr/bin/env python3
"""Evaluate one frozen learned gate on the held-out recursive Train audit."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


SCHEMA = 'sutrack-state-gate-recursive-audit/v1'
TRACE_SCHEMA = 'sutrack-state-gate-recursive-audit-trace/v1'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard-dir', action='append', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--capacity-result', type=Path, required=True)
    parser.add_argument('--split-plan', type=Path, required=True)
    parser.add_argument('--training-result', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
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


def validate_file_record(record):
    if not isinstance(record, dict):
        raise ValueError('malformed file record')
    path = Path(record['path']).resolve()
    if (not path.is_file() or sha256_file(path) != record.get('sha256') or
            path.stat().st_size != int(record.get('bytes', -1))):
        raise ValueError('file record mismatch {}'.format(path))
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


def close_bbox(first, second, tolerance=1e-6):
    return max(abs(a - b) for a, b in zip(first, second)) <= tolerance


def read_gt(path):
    boxes = []
    with Path(path).open('r', encoding='utf-8') as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            values = raw_line.strip().replace('\t', ',').split(',')
            if len(values) != 4:
                raise ValueError('malformed GT {}:{}'.format(path, line_number))
            try:
                raw_bbox = [float(value) for value in values]
            except ValueError as error:
                raise ValueError('malformed GT {}:{}'.format(
                    path, line_number)) from error
            boxes.append(finite_bbox(raw_bbox))
    if not boxes or boxes[0] is None:
        raise ValueError('invalid initialization GT {}'.format(path))
    return boxes


def iou_xywh(first, second):
    ax0, ay0, aw, ah = first
    bx0, by0, bw, bh = second
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    intersection = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(
        0.0, min(ay1, by1) - max(ay0, by0))
    union = aw * ah + bw * bh - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def failure_stats(values, threshold=0.10, minimum_run=10):
    runs = []
    current = 0
    low_frames = 0
    for value in values:
        if value is not None and value <= threshold:
            current += 1
            low_frames += 1
        else:
            if current:
                runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return {
        'low_iou_frames': low_frames,
        'ten_frame_failure_starts': sum(
            int(length >= minimum_run) for length in runs),
        'longest_low_iou_run': max(runs) if runs else 0,
        'low_iou_run_count': len(runs),
    }


def main():
    args = parse_args()
    if len(args.shard_dir) != 2:
        raise ValueError('recursive audit requires exactly two frozen shards')
    dataset_root = args.dataset_root.resolve()
    capacity_path = args.capacity_result.resolve()
    split_path = args.split_plan.resolve()
    training_path = args.training_result.resolve()
    capacity = load_json(capacity_path)
    split = load_json(split_path)
    training = load_json(training_path)
    if (capacity.get('schema') != 'sutrack-depthtrack-train-state-capacity/v1' or
            capacity.get('complete') is not True or
            capacity.get('decision') != 'capacity_supported' or
            capacity.get('capacity_supported') is not True or
            len(capacity.get('expected_sequences', [])) != 152 or
            capacity.get('ground_truth_join') != 'strictly_post_inference' or
            capacity.get('public_evaluation') is not False):
        raise ValueError('full152 capacity contract failed')
    if (split.get('schema') != 'sutrack-state-gate-split-plan/v1' or
            split.get('complete') is not True or
            split.get('audit_consumption_limit') != 1 or
            split.get('public_evaluation') is not False):
        raise ValueError('split contract failed')
    if (training.get('schema') != 'sutrack-state-gate-training/v1' or
            training.get('complete') is not True or
            training.get('ready_for_recursive_audit') is not True or
            training.get('all_seeds_oof_passed') is not True or
            training.get('immediate_audit_policies_evaluated') != 1 or
            training.get('immediate_audit_passed') is not True or
            training.get('seed_selection_used_audit') is not False or
            training.get('backbone_frozen') is not True or
            training.get('public_evaluation') is not False):
        raise ValueError('training result is not eligible')
    capacity_rows_path = Path(capacity['rows_path']).resolve()
    if sha256_file(capacity_rows_path) != capacity.get('rows_sha256'):
        raise ValueError('capacity rows SHA mismatch')
    if (Path(training.get('capacity_result_path', '')).resolve() != capacity_path or
            training.get('capacity_result_sha256') != sha256_file(capacity_path) or
            Path(training.get('capacity_rows_path', '')).resolve() !=
            capacity_rows_path or
            training.get('capacity_rows_sha256') !=
            sha256_file(capacity_rows_path) or
            Path(training.get('split_plan_path', '')).resolve() != split_path or
            training.get('split_plan_sha256') != sha256_file(split_path) or
            int(training.get('audit_consumption_limit', -1)) != 1):
        raise ValueError('training source binding mismatch')
    audit_sequences = list(split['audit_sequences'])
    calibration_sequences = list(split['calibration_sequences'])
    if (len(audit_sequences) != 30 or
            len(audit_sequences) != len(set(audit_sequences)) or
            len(calibration_sequences) != len(set(calibration_sequences)) or
            set(audit_sequences).intersection(calibration_sequences) or
            set(audit_sequences + calibration_sequences) !=
            set(capacity['expected_sequences'])):
        raise ValueError('capacity/split sequence binding mismatch')
    audit_set = set(audit_sequences)
    source_snapshot = {
        str(capacity_path): sha256_file(capacity_path),
        str(capacity_rows_path): sha256_file(capacity_rows_path),
        str(split_path): sha256_file(split_path),
        str(training_path): sha256_file(training_path),
        str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
    }

    common_contract = None
    observed_sequences = []
    predictions = []
    claimed_frame_counts = {}
    for shard_dir in args.shard_dir:
        manifest_path = shard_dir.resolve() / 'manifest.json'
        manifest = load_json(manifest_path)
        if (manifest.get('schema') != TRACE_SCHEMA or
                manifest.get('complete') is not True or
                manifest.get('role') !=
                'single_frozen_policy_recursive_audit_shard' or
                manifest.get('dataset') != 'DepthTrack Train audit only' or
                Path(manifest.get('dataset_root', '')).resolve() != dataset_root or
                manifest.get('ground_truth_consumption') !=
                'first_frame_initialization_only' or
                manifest.get('ground_truth_available_to_tracker') is not False or
                manifest.get('future_frame_text_used') is not False or
                manifest.get('public_evaluation') is not False or
                manifest.get('policy_evaluations_on_audit') != 1):
            raise ValueError('candidate manifest contract failed {}'.format(
                manifest_path))
        manifest_sequences = manifest.get('sequences')
        if (not isinstance(manifest_sequences, list) or
                len(manifest_sequences) != len(set(manifest_sequences)) or
                int(manifest.get('sequence_count', -1)) !=
                len(manifest_sequences) or
                not set(manifest_sequences).issubset(audit_set)):
            raise ValueError('candidate shard sequence contract failed')
        sequence_records = manifest.get('sequence_records')
        if (not isinstance(sequence_records, list) or
                {record.get('sequence') for record in sequence_records} !=
                set(manifest_sequences) or
                len(sequence_records) != len(manifest_sequences)):
            raise ValueError('candidate sequence records mismatch')
        for record in sequence_records:
            sequence = record['sequence']
            frame_count = int(record.get('frame_count', -1))
            if frame_count <= 0 or sequence in claimed_frame_counts:
                raise ValueError('invalid candidate frame claim {}'.format(sequence))
            claimed_frame_counts[sequence] = frame_count
        for key in ('split_plan', 'training_result', 'artifact', 'config',
                    'checkpoint', 'language_manifest', 'predictions'):
            path = validate_file_record(manifest[key])
            source_snapshot[str(path)] = manifest[key]['sha256']
        if (manifest['split_plan']['sha256'] != sha256_file(split_path) or
                manifest['training_result']['sha256'] !=
                sha256_file(training_path)):
            raise ValueError('candidate source binding mismatch')
        contract = {
            'artifact': manifest['artifact'],
            'config': manifest['config'],
            'checkpoint': manifest['checkpoint'],
            'language_manifest': manifest['language_manifest'],
            'implementation_sha256': manifest['implementation_sha256'],
        }
        if common_contract is None:
            common_contract = contract
        elif contract != common_contract:
            raise ValueError('candidate shards use different policies')
        for relative, expected_sha in manifest['implementation_sha256'].items():
            path = Path('/home/SUTrack_RGBD_L') / relative
            if sha256_file(path) != expected_sha:
                raise ValueError('candidate implementation drift {}'.format(
                    relative))
        prediction_path = Path(manifest['predictions']['path']).resolve()
        rows = load_jsonl(prediction_path)
        if len(rows) != int(manifest['frame_count']):
            raise ValueError('candidate prediction row count mismatch')
        observed_sequences.extend(manifest_sequences)
        predictions.extend(rows)
        source_snapshot[str(manifest_path)] = sha256_file(manifest_path)
        source_snapshot[str(prediction_path)] = sha256_file(prediction_path)
    if (len(observed_sequences) != len(set(observed_sequences)) or
            set(observed_sequences) != set(audit_sequences)):
        raise ValueError('candidate audit coverage mismatch')
    if common_contract['artifact']['sha256'] not in {
            record['sha256'] for record in training['artifacts']
            if int(record['seed']) == int(training['deployment_seed'])}:
        raise ValueError('candidate did not use the deployment artifact')

    prediction_by_key = {}
    action_by_sequence = defaultdict(int)
    for row in predictions:
        if row.get('schema') != TRACE_SCHEMA:
            raise ValueError('candidate prediction schema mismatch')
        sequence = row['sequence']
        frame_index = int(row['frame_index'])
        key = (sequence, frame_index)
        if key in prediction_by_key or sequence not in audit_set:
            raise ValueError('duplicate/out-of-scope candidate row {}'.format(key))
        bbox = finite_bbox(row['deployed_bbox'])
        if bbox is None:
            raise ValueError('malformed candidate bbox {}'.format(key))
        rollback = row.get('rollback_state')
        initialization = row.get('initialization')
        if not isinstance(rollback, bool) or not isinstance(initialization, bool):
            raise ValueError('malformed candidate decision {}'.format(key))
        if frame_index == 0:
            if (initialization is not True or rollback is not False or
                    row.get('probability') is not None):
                raise ValueError('malformed initialization row {}'.format(key))
        else:
            decision = row.get('gate_decision')
            if (initialization is not False or not isinstance(decision, dict) or
                    bool(decision.get('rollback_state')) != rollback or
                    row.get('ground_truth_available_to_tracker') is not False or
                    row.get('future_frame_text_used') is not False):
                raise ValueError('candidate gate row contract failed {}'.format(key))
            probability = row.get('probability')
            if probability is not None and not math.isfinite(float(probability)):
                raise ValueError('non-finite candidate probability {}'.format(key))
        prediction_by_key[key] = {'bbox': bbox, 'row': row}
        action_by_sequence[sequence] += int(rollback)

    capacity_rows = load_jsonl(capacity_rows_path)
    baseline_by_key = {}
    for row in capacity_rows:
        sequence = row['sequence']
        if sequence not in audit_set:
            continue
        key = (sequence, int(row['frame_index']))
        if key in baseline_by_key:
            raise ValueError('duplicate baseline audit row {}'.format(key))
        baseline_by_key[key] = row

    # Strictly post-inference GT join begins here.
    gt_by_sequence = {}
    for sequence in audit_sequences:
        gt_path = dataset_root / sequence / 'groundtruth.txt'
        gt_by_sequence[sequence] = read_gt(gt_path)
        source_snapshot[str(gt_path)] = sha256_file(gt_path)
        if claimed_frame_counts.get(sequence) != len(gt_by_sequence[sequence]):
            raise ValueError('candidate/GT frame-count mismatch {}'.format(sequence))
    expected_prediction_keys = {
        (sequence, frame_index)
        for sequence in audit_sequences
        for frame_index in range(len(gt_by_sequence[sequence]))
    }
    if set(prediction_by_key) != expected_prediction_keys:
        raise ValueError('candidate recursive audit frame coverage mismatch')
    for sequence in audit_sequences:
        initialization = prediction_by_key[(sequence, 0)]['bbox']
        if not close_bbox(initialization, gt_by_sequence[sequence][0]):
            raise ValueError('candidate initialization mismatch {}'.format(sequence))

    sequence_metrics = {}
    all_baseline = []
    all_candidate = []
    baseline_failures = 0
    candidate_failures = 0
    catastrophic_sequences = []
    for sequence in audit_sequences:
        groundtruth = gt_by_sequence[sequence]
        baseline_values = []
        candidate_values = []
        for frame_index in range(1, len(groundtruth)):
            key = (sequence, frame_index)
            if key not in prediction_by_key or key not in baseline_by_key:
                raise ValueError('incomplete audit frame {}'.format(key))
            gt = groundtruth[frame_index]
            baseline_row = baseline_by_key[key]
            if gt is None:
                if (baseline_row.get('label_available') is not False or
                        baseline_row.get('candidate_iou') is not None):
                    raise ValueError('absent-GT contract mismatch {}'.format(key))
                baseline_values.append(None)
                candidate_values.append(None)
                continue
            baseline_iou = float(baseline_row['candidate_iou'])
            candidate_iou = iou_xywh(prediction_by_key[key]['bbox'], gt)
            if not all(math.isfinite(value) for value in (
                    baseline_iou, candidate_iou)):
                raise ValueError('non-finite audit IoU {}'.format(key))
            baseline_values.append(baseline_iou)
            candidate_values.append(candidate_iou)
            all_baseline.append(baseline_iou)
            all_candidate.append(candidate_iou)
        valid_baseline = [value for value in baseline_values if value is not None]
        valid_candidate = [value for value in candidate_values if value is not None]
        if not valid_baseline or len(valid_baseline) != len(valid_candidate):
            raise ValueError('empty/misaligned sequence audit {}'.format(sequence))
        baseline_mean = sum(valid_baseline) / len(valid_baseline)
        candidate_mean = sum(valid_candidate) / len(valid_candidate)
        baseline_failure = failure_stats(baseline_values)
        candidate_failure = failure_stats(candidate_values)
        failure_delta = (candidate_failure['ten_frame_failure_starts'] -
                         baseline_failure['ten_frame_failure_starts'])
        mean_delta = candidate_mean - baseline_mean
        catastrophic = bool(mean_delta < -0.05 and failure_delta > 0)
        if catastrophic:
            catastrophic_sequences.append(sequence)
        baseline_failures += baseline_failure['ten_frame_failure_starts']
        candidate_failures += candidate_failure['ten_frame_failure_starts']
        sequence_metrics[sequence] = {
            'valid_gt_frames': len(valid_baseline),
            'rollback_actions': action_by_sequence[sequence],
            'baseline_mean_iou': baseline_mean,
            'candidate_mean_iou': candidate_mean,
            'mean_iou_delta': mean_delta,
            'baseline_failure_stats': baseline_failure,
            'candidate_failure_stats': candidate_failure,
            'ten_frame_failure_starts_delta': failure_delta,
            'catastrophic_regression': catastrophic,
        }

    baseline_mean = sum(all_baseline) / len(all_baseline)
    candidate_mean = sum(all_candidate) / len(all_candidate)
    mean_delta = candidate_mean - baseline_mean
    failure_delta = candidate_failures - baseline_failures
    gate = split['recursive_audit_gate']
    checks = {
        'mean_iou_non_decreasing': (
            mean_delta >= float(gate['mean_iou_delta_minimum'])),
        'ten_frame_failure_starts_reduced': (
            failure_delta <= int(
                gate['ten_frame_failure_starts_delta_maximum'])),
        'no_catastrophic_sequence_regressions': (
            len(catastrophic_sequences) <= int(
                gate['catastrophic_sequence_regressions_maximum'])),
        'single_policy_evaluation': all(
            load_json(shard.resolve() / 'manifest.json').get(
                'policy_evaluations_on_audit') == 1
            for shard in args.shard_dir),
    }
    passed = all(checks.values())
    result = {
        'schema': SCHEMA,
        'complete': True,
        'decision': ('recursive_audit_passed' if passed else
                     'recursive_audit_rejected'),
        'recursive_audit_passed': passed,
        'eligible_for_public_evaluation': passed,
        'public_evaluation_started': False,
        'scope': 'DepthTrack Train held-out recursive audit only',
        'metric_note': (
            'ten-frame low-IoU runs are a Train-only ROB surrogate, not the '
            'public VOT EAO/ROB metric'),
        'policy_evaluations_on_audit': 1,
        'audit_sequences': audit_sequences,
        'audit_sequence_count': len(audit_sequences),
        'valid_gt_frames': len(all_baseline),
        'rollback_actions': sum(action_by_sequence.values()),
        'baseline_mean_iou': baseline_mean,
        'candidate_mean_iou': candidate_mean,
        'mean_iou_delta': mean_delta,
        'baseline_ten_frame_failure_starts': baseline_failures,
        'candidate_ten_frame_failure_starts': candidate_failures,
        'ten_frame_failure_starts_delta': failure_delta,
        'catastrophic_regression_sequences': catastrophic_sequences,
        'checks': checks,
        'sequence_metrics': sequence_metrics,
        'capacity_result_path': str(capacity_path),
        'capacity_result_sha256': source_snapshot[str(capacity_path)],
        'capacity_rows_path': str(capacity_rows_path),
        'capacity_rows_sha256': source_snapshot[str(capacity_rows_path)],
        'split_plan_path': str(split_path),
        'split_plan_sha256': source_snapshot[str(split_path)],
        'training_result_path': str(training_path),
        'training_result_sha256': source_snapshot[str(training_path)],
        'artifact': common_contract['artifact'],
        'candidate_contract': common_contract,
        'source_snapshot_sha256': source_snapshot,
        'analyzer_path': str(Path(__file__).resolve()),
        'analyzer_sha256': source_snapshot[str(Path(__file__).resolve())],
        'future_frame_text_used': False,
    }
    for path_string, expected_sha in source_snapshot.items():
        if sha256_file(path_string) != expected_sha:
            raise ValueError('source drift during recursive audit {}'.format(
                path_string))
    for relative, expected_sha in common_contract['implementation_sha256'].items():
        if sha256_file(Path('/home/SUTrack_RGBD_L') / relative) != expected_sha:
            raise ValueError('implementation drift during recursive audit {}'.format(
                relative))
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('refusing non-empty output {}'.format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / 'recursive_audit_result.json', result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2,
                     allow_nan=False))


if __name__ == '__main__':
    main()
