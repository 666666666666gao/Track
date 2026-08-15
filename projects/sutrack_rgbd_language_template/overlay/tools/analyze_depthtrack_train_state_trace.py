#!/usr/bin/env python3
"""Join frozen SUTrack traces with Train GT and test rollback capacity.

The tracker never imports this module.  Ground truth is opened only after all
inference artifacts have been loaded and provenance-checked.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


SCHEMA = 'sutrack-depthtrack-train-state-capacity/v1'
TRACE_SCHEMA = 'sutrack-depthtrack-train-state-trace/v1'
HARD_REASONS = (
    'large_center_jump',
    'low_static_rgb_identity',
    'large_depth_change',
    'temporal_identity_rejected',
)
FEATURE_NAMES = (
    'confidence',
    'response_margin',
    'identity_similarity',
    'identity_missing',
    'center_jump',
    'log_depth_change',
    'depth_change_missing',
    'log_area_ratio',
    'log_aspect_ratio',
    'dynamic_active',
    'checked',
    'stable_frames_log1p',
    'low_confidence_reason',
    'small_response_margin_reason',
    'large_center_jump_reason',
    'low_identity_reason',
    'large_depth_change_reason',
    'missing_depth_reason',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard-dir', action='append', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--expected-sequences', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--benefit-margin', type=float, default=0.05)
    parser.add_argument('--minimum-useful-iou', type=float, default=0.10)
    parser.add_argument('--catastrophic-good-iou', type=float, default=0.50)
    parser.add_argument('--catastrophic-bad-iou', type=float, default=0.10)
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
                raise ValueError('non-object row {}:{}'.format(path, line_number))
            records.append(record)
    return records


def validate_file_record(record):
    if not isinstance(record, dict):
        raise ValueError('malformed file record')
    path = Path(record['path']).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != record.get('sha256'):
        raise ValueError('SHA mismatch for {}'.format(path))
    if path.stat().st_size != record.get('bytes'):
        raise ValueError('byte-count mismatch for {}'.format(path))
    return path, observed


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
                raise ValueError('malformed GT {}:{}'.format(path, line_number))
            try:
                raw_bbox = [float(value) for value in values]
            except ValueError as error:
                raise ValueError(
                    'malformed GT {}:{}'.format(path, line_number)) from error
            # DepthTrack marks target-absent frames with NaN boxes.  Retain
            # their rows and online features, but never manufacture a label.
            bbox = finite_bbox(raw_bbox)
            boxes.append(bbox)
    if not boxes:
        raise ValueError('empty GT {}'.format(path))
    return boxes


def close_bbox(first, second, tolerance=1.0e-9):
    return max(abs(a - b) for a, b in zip(first, second)) <= tolerance


def safe_scalar(value, missing_value=0.0):
    if value is None:
        return float(missing_value), 1.0
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('non-finite online feature')
    return value, 0.0


def features(record):
    evidence = record['online_evidence']
    reasons = set(evidence['reasons'])
    prior = record['prior_bbox']
    candidate = record['candidate_bbox']
    identity, identity_missing = safe_scalar(
        evidence.get('identity_similarity'))
    center_jump, center_jump_missing = safe_scalar(
        evidence.get('normalized_center_jump'))
    depth_change, depth_missing = safe_scalar(
        evidence.get('log_depth_change'))
    if center_jump_missing and bool(evidence.get('checked')):
        raise ValueError('checked online evidence is missing center jump')
    area_ratio = (candidate[2] * candidate[3]) / (prior[2] * prior[3])
    aspect_ratio = ((candidate[2] / candidate[3]) /
                    (prior[2] / prior[3]))
    values = (
        float(evidence['confidence']),
        float(evidence['response_margin']),
        identity,
        identity_missing,
        center_jump,
        depth_change,
        depth_missing,
        math.log(max(area_ratio, 1.0e-12)),
        math.log(max(aspect_ratio, 1.0e-12)),
        float(bool(evidence['dynamic_active'])),
        float(bool(evidence['checked'])),
        math.log1p(int(evidence['stable_frames'])),
        float('low_confidence' in reasons),
        float('small_response_margin' in reasons),
        float('large_center_jump' in reasons),
        float('low_static_rgb_identity' in reasons),
        float('large_depth_change' in reasons),
        float('missing_or_unreliable_depth' in reasons),
    )
    if len(values) != len(FEATURE_NAMES) or not all(
            math.isfinite(value) for value in values):
        raise ValueError('malformed feature vector')
    return list(values)


def main():
    args = parse_args()
    scalars = (args.benefit_margin, args.minimum_useful_iou,
               args.catastrophic_good_iou, args.catastrophic_bad_iou)
    if (not all(math.isfinite(value) for value in scalars) or
            args.benefit_margin <= 0.0 or
            not all(0.0 <= value <= 1.0 for value in scalars[1:])):
        raise ValueError('invalid thresholds')
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

    source_snapshot = {}
    common_contract = None
    records = []
    observed_sequences = []
    manifests = []
    for shard_dir in args.shard_dir:
        shard_dir = shard_dir.resolve()
        manifest_path = shard_dir / 'manifest.json'
        manifest_sha = sha256_file(manifest_path)
        manifest = load_json(manifest_path)
        if (manifest.get('schema') != TRACE_SCHEMA or
                manifest.get('complete') is not True or
                manifest.get('dataset') != 'DepthTrack Train only' or
                Path(manifest.get('dataset_root', '')).resolve() != dataset_root or
                manifest.get('ground_truth_consumption') !=
                'first_frame_initialization_only' or
                manifest.get('ground_truth_available_to_tracker') is not False or
                manifest.get('future_frame_text_used') is not False or
                manifest.get('public_evaluation') is not False):
            raise ValueError('invalid trace contract {}'.format(manifest_path))
        trace_path, trace_sha = validate_file_record(manifest['trace'])
        predictions_path, predictions_sha = validate_file_record(
            manifest['predictions'])
        trace_rows = load_jsonl(trace_path)
        prediction_rows = load_jsonl(predictions_path)
        if (len(trace_rows) != manifest.get('trace_row_count') or
                len(prediction_rows) != manifest.get('frame_count')):
            raise ValueError('manifest row-count mismatch {}'.format(shard_dir))
        contract = {
            'config': manifest['config'],
            'checkpoint': manifest['checkpoint'],
            'language_manifest': manifest['language_manifest'],
            'implementation_sha256': manifest['implementation_sha256'],
        }
        for key in ('config', 'checkpoint', 'language_manifest'):
            validate_file_record(contract[key])
        if common_contract is None:
            common_contract = contract
        elif contract != common_contract:
            raise ValueError('shards do not share one frozen implementation')
        for relative, expected_sha in contract['implementation_sha256'].items():
            path = Path('/home/SUTrack_RGBD_L') / relative
            if sha256_file(path) != expected_sha:
                raise ValueError('implementation drift {}'.format(relative))
        observed_sequences.extend(manifest['sequences'])
        records.extend(trace_rows)
        source_snapshot[str(manifest_path)] = manifest_sha
        source_snapshot[str(trace_path)] = trace_sha
        source_snapshot[str(predictions_path)] = predictions_sha
        manifests.append(manifest)
    if (observed_sequences != expected_sequences or
            len(observed_sequences) != len(set(observed_sequences))):
        raise ValueError(
            'trace sequence order/scope differs from pre-registration')
    records.sort(key=lambda record: (
        expected_sequences.index(record['sequence']), record['frame_index']))

    # Strictly post-inference GT join begins here.
    gt_by_sequence = {}
    for sequence in expected_sequences:
        gt_path = dataset_root / sequence / 'groundtruth.txt'
        gt_by_sequence[sequence] = read_gt(gt_path)
        source_snapshot[str(gt_path)] = sha256_file(gt_path)

    reason_counts = Counter()
    sequence_stats = defaultdict(lambda: Counter())
    rows = []
    seen_keys = set()
    for record in records:
        if (record.get('schema') != TRACE_SCHEMA or
                record.get('ground_truth_available_to_tracker') is not False or
                record.get('future_frame_text_used') is not False):
            raise ValueError('invalid trace row contract')
        sequence = record['sequence']
        frame_index = int(record['frame_index'])
        key = (sequence, frame_index)
        if key in seen_keys:
            raise ValueError('duplicate trace row {}'.format(key))
        seen_keys.add(key)
        if frame_index <= 0 or frame_index >= len(gt_by_sequence[sequence]):
            raise ValueError('trace/GT frame mismatch {}'.format(key))
        prior = finite_bbox(record['prior_bbox'])
        candidate = finite_bbox(record['candidate_bbox'])
        deployed = finite_bbox(record['deployed_bbox'])
        if prior is None or candidate is None or deployed is None:
            raise ValueError('non-finite trace bbox {}'.format(key))
        if not close_bbox(candidate, deployed):
            raise ValueError(
                'safe-v1 trace unexpectedly changed the recursive state {}'.format(
                    key))
        reasons = tuple(record['online_evidence']['reasons'])
        reason_counts.update(reasons)
        hard_conflict = any(reason in reasons for reason in HARD_REASONS)
        gt = gt_by_sequence[sequence][frame_index]
        label_available = gt is not None
        candidate_iou = None
        prior_iou = None
        delta = None
        beneficial = None
        harmful = None
        catastrophic_harm = None
        if label_available:
            candidate_iou = iou_xywh(candidate, gt)
            prior_iou = iou_xywh(prior, gt)
            delta = prior_iou - candidate_iou
            beneficial = bool(
                hard_conflict and delta >= args.benefit_margin and
                prior_iou >= args.minimum_useful_iou)
            harmful = bool(delta <= -args.benefit_margin)
            catastrophic_harm = bool(
                candidate_iou >= args.catastrophic_good_iou and
                prior_iou <= args.catastrophic_bad_iou)
        row = {
            'schema': SCHEMA,
            'sequence': sequence,
            'frame_index': frame_index,
            'features': features(record),
            'feature_names': list(FEATURE_NAMES),
            'hard_conflict': hard_conflict,
            'reasons': list(reasons),
            'label_available': label_available,
            'candidate_iou': candidate_iou,
            'rollback_iou': prior_iou,
            'rollback_delta_iou': delta,
            'rollback_beneficial': beneficial,
            'rollback_harmful': harmful,
            'rollback_catastrophic_harm': catastrophic_harm,
            'label_source': 'post_inference_depthtrack_train_gt',
        }
        rows.append(row)
        stats = sequence_stats[sequence]
        stats['rows'] += 1
        stats['absent_gt_rows'] += int(not label_available)
        stats['labeled_rows'] += int(label_available)
        stats['hard_conflict_rows'] += int(
            hard_conflict and label_available)
        stats['beneficial_rows'] += int(bool(beneficial))
        stats['harmful_rows'] += int(bool(harmful))
        stats['catastrophic_harm_rows'] += int(bool(catastrophic_harm))

    labeled_rows = [row for row in rows if row['label_available']]
    hard_rows = [row for row in labeled_rows if row['hard_conflict']]
    beneficial_rows = [row for row in hard_rows if row['rollback_beneficial']]
    beneficial_sequences = sorted(set(
        row['sequence'] for row in beneficial_rows))
    minimum_beneficial_rows = max(
        10, math.ceil(len(labeled_rows) * 0.001))
    oracle_gain = sum(max(0.0, row['rollback_delta_iou'])
                      for row in beneficial_rows)
    capacity_checks = {
        'at_least_20_hard_conflict_rows': len(hard_rows) >= 20,
        'beneficial_rows_at_least_max_10_or_0p1pct': (
            len(beneficial_rows) >= minimum_beneficial_rows),
        'beneficial_sequences_at_least_3': len(beneficial_sequences) >= 3,
        'oracle_immediate_gain_at_least_1': oracle_gain >= 1.0,
        'all_features_finite': all(
            all(math.isfinite(value) for value in row['features'])
            for row in rows),
    }
    capacity_supported = all(capacity_checks.values())
    result = {
        'schema': SCHEMA,
        'complete': True,
        'decision': ('capacity_supported' if capacity_supported else
                     'capacity_rejected'),
        'capacity_supported': capacity_supported,
        'capacity_checks': capacity_checks,
        'scope': 'DepthTrack Train fixed6 immediate-action diagnostic only',
        'recursive_rollout_claim_supported': False,
        'public_evaluation': False,
        'ground_truth_join': 'strictly_post_inference',
        'expected_sequences': expected_sequences,
        'row_count': len(rows),
        'labeled_row_count': len(labeled_rows),
        'absent_gt_row_count': len(rows) - len(labeled_rows),
        'hard_conflict_rows': len(hard_rows),
        'beneficial_rows': len(beneficial_rows),
        'minimum_beneficial_rows': minimum_beneficial_rows,
        'beneficial_sequences': beneficial_sequences,
        'oracle_immediate_gain_sum': oracle_gain,
        'baseline_candidate_mean_iou': (
            sum(row['candidate_iou'] for row in labeled_rows) /
            len(labeled_rows)),
        'immediate_oracle_mean_iou': (
            sum(max(row['candidate_iou'], row['rollback_iou'])
                if row['rollback_beneficial'] else row['candidate_iou']
                for row in labeled_rows) / len(labeled_rows)),
        'reason_counts': dict(sorted(reason_counts.items())),
        'sequence_stats': {
            sequence: dict(sequence_stats[sequence])
            for sequence in expected_sequences
        },
        'feature_names': list(FEATURE_NAMES),
        'thresholds': {
            'benefit_margin': args.benefit_margin,
            'minimum_useful_iou': args.minimum_useful_iou,
            'catastrophic_good_iou': args.catastrophic_good_iou,
            'catastrophic_bad_iou': args.catastrophic_bad_iou,
        },
        'source_snapshot_sha256': source_snapshot,
        'trace_contract': common_contract,
        'analyzer_path': str(Path(__file__).resolve()),
        'analyzer_sha256': sha256_file(Path(__file__).resolve()),
    }
    rows_path = output_dir / 'capacity_rows.jsonl'
    result_path = output_dir / 'capacity_result.json'
    atomic_write(rows_path, ''.join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n'
        for row in rows).encode('utf-8'))
    result['rows_path'] = str(rows_path)
    result['rows_sha256'] = sha256_file(rows_path)
    # Close the source TOCTOU window before publishing the decision.
    for path_string, expected_sha in source_snapshot.items():
        if sha256_file(Path(path_string)) != expected_sha:
            raise ValueError('source drift during analysis {}'.format(path_string))
    for relative, expected_sha in common_contract['implementation_sha256'].items():
        if sha256_file(Path('/home/SUTrack_RGBD_L') / relative) != expected_sha:
            raise ValueError('implementation drift during analysis {}'.format(
                relative))
    atomic_write(
        result_path,
        (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) +
         '\n').encode('utf-8'))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
