#!/usr/bin/env python3
"""Freeze an outcome-independent calibration/audit split for SUTrack gate."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


SCHEMA = 'sutrack-state-gate-split-plan/v1'
SALT = 'sutrack-state-gate-depthtrack-train152-audit-v1'
FIXED6 = (
    'bottle03_indoor',
    'ball16_indoor',
    'bag04_indoor',
    'flower03_indoor',
    'pigeon05_wild',
    'toy03_indoor',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace-plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit-sequences', type=int, default=30)
    parser.add_argument('--folds', type=int, default=5)
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


def rank_key(sequence):
    return hashlib.sha256((SALT + '\0' + sequence).encode('utf-8')).hexdigest()


def main():
    args = parse_args()
    if args.audit_sequences <= 0 or args.folds < 2:
        raise ValueError('invalid split dimensions')
    trace_plan_path = args.trace_plan.resolve()
    trace_plan = json.loads(trace_plan_path.read_text(encoding='utf-8'))
    sequences = list(trace_plan['all_sequences_language_manifest_order'])
    frame_counts = {
        key: int(value) for key, value in trace_plan['frame_counts'].items()
    }
    if (len(sequences) != 152 or len(set(sequences)) != 152 or
            set(frame_counts) != set(sequences) or
            not set(FIXED6).issubset(sequences)):
        raise ValueError('Trace152 plan contract failed')

    never_preflighted = sorted(
        (sequence for sequence in sequences if sequence not in FIXED6),
        key=lambda sequence: (rank_key(sequence), sequence))
    audit = never_preflighted[:args.audit_sequences]
    calibration = [sequence for sequence in sequences if sequence not in audit]
    if set(audit).intersection(FIXED6):
        raise ValueError('fixed6 leaked into audit')
    if (set(audit).intersection(calibration) or
            set(audit + calibration) != set(sequences)):
        raise ValueError('split is not a partition')

    # Balance calibration folds by frame count without using any labels.
    fold_sequences = [[] for _ in range(args.folds)]
    fold_frames = [0 for _ in range(args.folds)]
    for sequence in sorted(
            calibration,
            key=lambda item: (-frame_counts[item], rank_key(item), item)):
        fold = min(range(args.folds),
                   key=lambda item: (fold_frames[item], item))
        fold_sequences[fold].append(sequence)
        fold_frames[fold] += frame_counts[sequence]

    plan = {
        'schema': SCHEMA,
        'complete': True,
        'created_before_full152_gt_join': True,
        'dataset': 'DepthTrack Train only',
        'outcome_fields_used_for_split': [],
        'split_salt': SALT,
        'sequence_count': len(sequences),
        'frame_count': sum(frame_counts.values()),
        'preflight_fixed6': list(FIXED6),
        'preflight_fixed6_allowed_roles': ['calibration'],
        'audit_sequences': audit,
        'audit_sequence_count': len(audit),
        'audit_frame_count': sum(frame_counts[name] for name in audit),
        'audit_consumption_limit': 1,
        'audit_role': 'single_immediate_then_recursive_safety_audit',
        'calibration_sequences': calibration,
        'calibration_sequence_count': len(calibration),
        'calibration_frame_count': sum(
            frame_counts[name] for name in calibration),
        'folds': [
            {'fold': index, 'sequences': names,
             'sequence_count': len(names), 'frame_count': fold_frames[index]}
            for index, names in enumerate(fold_sequences)
        ],
        'training_seeds': [2026, 2027, 2028],
        'deployment_seed': 2026,
        'model_family': 'linear_logit_over_current_delta_and_two_frame_mean',
        'backbone_frozen': True,
        'threshold_selection': {
            'source': 'calibration_sequence_group_oof_only',
            'minimum_precision': 0.85,
            'maximum_harm_rate': 0.02,
            'maximum_catastrophic_harm_rows': 0,
            'minimum_action_rows': 20,
            'minimum_action_sequences': 10,
            'minimum_net_iou_gain': 1.0,
            'all_seeds_must_pass': True,
            'seed_selection_uses_audit': False,
        },
        'immediate_audit_gate': {
            'minimum_precision': 0.85,
            'maximum_harm_rate': 0.02,
            'maximum_catastrophic_harm_rows': 0,
            'minimum_action_rows': 5,
            'minimum_action_sequences': 3,
            'minimum_net_iou_gain_exclusive': 0.0,
            'policy_evaluations_on_audit': 1,
        },
        'recursive_audit_gate': {
            'mean_iou_delta_minimum': 0.0,
            'ten_frame_failure_starts_delta_maximum': -1,
            'catastrophic_sequence_regressions_maximum': 0,
            'maximum_consecutive_gate_rollbacks': 1,
            'cooldown_frames_after_rollback': 2,
            'policy_evaluations_on_audit': 1,
        },
        'trace_plan_path': str(trace_plan_path),
        'trace_plan_sha256': sha256_file(trace_plan_path),
        'public_evaluation': False,
        'future_frame_text_used': False,
    }
    atomic_write(
        args.output.resolve(),
        (json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2) +
         '\n').encode('utf-8'))
    print(json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
