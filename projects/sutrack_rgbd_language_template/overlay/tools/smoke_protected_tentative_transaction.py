#!/usr/bin/env python3
"""CPU smoke for the dormant protected/tentative state transaction."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
MODULE_SOURCE = (REPO_ROOT / 'lib' / 'test' / 'tracker' /
                 'protected_tentative_transaction.py')
if MODULE_SOURCE.is_file():
    sys.path.insert(0, str(REPO_ROOT))
    from lib.test.tracker.protected_tentative_transaction import (
        BranchEvidence,
        ProtectedTentativeTemplateTransaction,
        TrackerRecursiveSnapshot,
    )
else:
    # Permit review from a temporary directory containing the two sibling
    # files. Canonical evidence is still produced only from the repo layout.
    MODULE_SOURCE = SCRIPT_PATH.with_name(
        'protected_tentative_transaction.py')
    if not MODULE_SOURCE.is_file():
        raise RuntimeError('transaction module is not beside or below script')
    sys.path.insert(0, str(SCRIPT_PATH.parent))
    from protected_tentative_transaction import (  # noqa: E402
        BranchEvidence,
        ProtectedTentativeTemplateTransaction,
        TrackerRecursiveSnapshot,
    )


ANCHOR = 'votrgbd2022:smoke@frame-1'


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-{}'.format(os.getpid()))
    with open(temporary, 'x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def snapshot(value, x=10.0, annotation_slots=1, template_slots=2,
             dtype=torch.float32):
    annotations = [
        torch.full((1, 4), value + 0.50 + index * 0.25, dtype=dtype)
        for index in range(annotation_slots)
    ]
    return TrackerRecursiveSnapshot.capture(
        [x, 20.0, 30.0, 40.0],
        [torch.full((1, 2, 2), value + index * 0.25, dtype=dtype)
         for index in range(template_slots)],
        annotations,
        {
            'track_query': [torch.full((1, 3), value + 1.0, dtype=dtype)],
            'policy': {
                'stable_frames': int(value),
                'dynamic_active': value > 1.0,
                'nested': [np.asarray([value, value + 1.0])],
            },
            'text_src': torch.full((1, 2), value + 2.0, dtype=dtype),
        })


def evidence(confidence, margin, identity, depth, temporal, conflict=False,
             identity_anchor=ANCHOR):
    return BranchEvidence(
        confidence=confidence, response_margin=margin,
        identity_similarity=identity, depth_consistency=depth,
        temporal_continuity=temporal, identity_anchor=identity_anchor,
        hard_conflict=conflict)


def expect_error(function, exception_type):
    try:
        function()
    except exception_type:
        return True
    return False


def current_full127_is_unwired(repo_root):
    needles = (
        'protected_tentative_transaction',
        'ProtectedTentativeTemplateTransaction',
        'sutrack_transaction',
    )
    references = []
    paths = (
        repo_root / 'lib' / 'test' / 'tracker' / 'sutrack.py',
        repo_root / 'lib' / 'test' / 'vot' /
        'sutrack_l384_rgbd_anchor_identity_all127.py',
        repo_root / 'experiments' / 'sutrack' /
        'sutrack_l384_rgbd_anchor_identity_all127.yaml',
    )
    for path in paths:
        if not path.is_file():
            references.append('missing:' + str(path))
            continue
        text = path.read_text(encoding='utf-8')
        if any(needle in text for needle in needles):
            references.append(str(path))
    return not references, references


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args()

    checks = {}
    checks['confirmation_count_is_fixed_at_two'] = expect_error(
        lambda: ProtectedTentativeTemplateTransaction(confirm_frames=1),
        ValueError)
    transaction = ProtectedTentativeTemplateTransaction(
        confirm_frames=2, max_shadow_frames=2,
        min_confidence=0.65, min_response_margin=0.10,
        min_identity=0.75, min_depth_consistency=0.70,
        min_temporal_continuity=0.70,
        min_utility_advantage=0.02)

    protected = snapshot(1.0)
    tentative = snapshot(2.0, x=11.0, annotation_slots=2)
    checks['different_valid_annotation_layouts_are_supported'] = (
        len(protected.templates) == 2 and
        len(protected.template_annotations) == 1 and
        len(tentative.template_annotations) == 2)
    started = transaction.begin(1, protected, tentative, ANCHOR)
    checks['begin_is_hold'] = (
        started.action == 'hold' and transaction.active and
        started.consecutive_confirmations == 0)
    checks['concurrent_begin_rejected'] = expect_error(
        lambda: transaction.begin(2, protected, tentative, ANCHOR),
        RuntimeError)
    exposed_protected, exposed_tentative = transaction.active_snapshots()
    exposed_protected.templates[0].add_(200.0)
    exposed_tentative.template_annotations[0].add_(200.0)
    checks['active_snapshots_return_isolated_storage'] = bool(
        exposed_protected.templates[0].data_ptr() !=
        protected.templates[0].data_ptr() and
        exposed_tentative.template_annotations[0].data_ptr() !=
        tentative.template_annotations[0].data_ptr())
    protected.templates[0].add_(100.0)
    protected.template_annotations[0].add_(100.0)
    protected.auxiliary['text_src'].add_(100.0)
    protected.auxiliary['policy']['nested'][0].fill(100.0)
    cancelled = transaction.cancel(2, reason='smoke_cancel')
    checks['cancel_rolls_back'] = (
        cancelled.action == 'rollback' and not transaction.active and
        cancelled.reasons == ('smoke_cancel',))
    checks['begin_snapshot_is_deep_clone'] = bool(
        torch.equal(cancelled.resolved_snapshot.templates[0],
                    torch.full((1, 2, 2), 1.0)) and
        torch.equal(cancelled.resolved_snapshot.template_annotations[0],
                    torch.full((1, 4), 1.5)) and
        torch.equal(cancelled.resolved_snapshot.auxiliary['text_src'],
                    torch.full((1, 2), 3.0)) and
        np.array_equal(
            cancelled.resolved_snapshot.auxiliary['policy']['nested'][0],
            np.asarray([1.0, 2.0])))

    transaction.begin(10, snapshot(1.0), snapshot(2.0, x=11.0), ANCHOR)
    protected_11 = snapshot(1.1, x=10.5)
    tentative_11 = snapshot(2.1, x=10.8)
    first = transaction.observe(
        11, protected_11, tentative_11,
        evidence(0.80, 0.20, 0.82, 0.90, 0.90),
        evidence(0.86, 0.30, 0.91, 0.92, 0.93))
    checks['first_confirmation_holds'] = (
        first.action == 'hold' and first.consecutive_confirmations == 1 and
        first.age == 1 and transaction.active)
    protected_12 = snapshot(1.2, x=10.7)
    tentative_12 = snapshot(2.2, x=10.9)
    promoted = transaction.observe(
        12, protected_12, tentative_12,
        evidence(0.81, 0.21, 0.83, 0.90, 0.91),
        evidence(0.88, 0.32, 0.93, 0.94, 0.94))
    checks['second_confirmation_promotes'] = (
        promoted.action == 'promote' and
        promoted.consecutive_confirmations == 2 and
        promoted.age == 2 and not transaction.active)
    checks['promotion_is_atomic_tentative_snapshot'] = bool(
        promoted.resolved_snapshot.state == tentative_12.state and
        torch.equal(promoted.resolved_snapshot.templates[0],
                    torch.full((1, 2, 2), 2.2)) and
        promoted.resolved_snapshot.auxiliary['policy']['dynamic_active'])
    checks['promotion_has_independent_storage'] = bool(
        promoted.resolved_snapshot.templates[0].data_ptr() !=
        tentative_12.templates[0].data_ptr())
    materialized = promoted.resolved_snapshot.materialize()
    materialized['template_list'][0].add_(50.0)
    materialized['template_anno_list'][0].add_(50.0)
    materialized['auxiliary']['policy']['nested'][0].fill(50.0)
    checks['materialize_returns_fresh_storage'] = bool(
        torch.equal(promoted.resolved_snapshot.templates[0],
                    torch.full((1, 2, 2), 2.2)) and
        torch.equal(promoted.resolved_snapshot.template_annotations[0],
                    torch.full((1, 4), 2.7)) and
        np.array_equal(
            promoted.resolved_snapshot.auxiliary['policy']['nested'][0],
            np.asarray([2.2, 3.2])))

    transaction.begin(20, snapshot(3.0), snapshot(4.0, x=12.0), ANCHOR)
    protected_21 = snapshot(3.1, x=10.2)
    conflict = transaction.observe(
        21, protected_21, snapshot(4.1, x=30.0),
        evidence(0.70, 0.10, 0.80, 0.85, 0.80),
        evidence(0.95, 0.50, 0.90, 0.90, 0.90, conflict=True))
    checks['hard_conflict_rolls_back_immediately'] = (
        conflict.action == 'rollback' and
        conflict.reasons == ('tentative_hard_conflict',) and
        conflict.resolved_snapshot.state == protected_21.state and
        not transaction.active)

    transaction.begin(30, snapshot(5.0), snapshot(6.0), ANCHOR)
    first_bad = transaction.observe(
        31, snapshot(5.1), snapshot(6.1),
        evidence(0.85, 0.30, 0.90, 0.90, 0.90),
        evidence(0.70, 0.10, 0.60, 0.65, 0.60))
    expired = transaction.observe(
        32, snapshot(5.2), snapshot(6.2),
        evidence(0.80, 0.20, 0.82, 0.88, 0.88),
        evidence(0.88, 0.31, 0.92, 0.93, 0.93))
    checks['bad_frame_resets_confirmation'] = (
        first_bad.action == 'hold' and
        first_bad.consecutive_confirmations == 0 and
        expired.consecutive_confirmations == 1)
    checks['horizon_uses_elapsed_frames_and_rolls_back'] = (
        expired.action == 'rollback' and expired.age == 2 and
        'shadow_horizon_expired' in expired.reasons and
        expired.resolved_snapshot.state == snapshot(5.2).state and
        not transaction.active)

    transaction.begin(40, snapshot(7.0), snapshot(8.0), ANCHOR)
    malformed_tentative = snapshot(8.1)
    malformed_tentative.auxiliary['bad'] = np.asarray(
        [object()], dtype=object)
    failed_observe_is_rejected = expect_error(
        lambda: transaction.observe(
            41, snapshot(7.1), malformed_tentative,
            evidence(0.80, 0.20, 0.82, 0.88, 0.88),
            evidence(0.88, 0.31, 0.92, 0.93, 0.93)),
        TypeError)
    recovered_same_frame = transaction.observe(
        41, snapshot(7.1), snapshot(8.1),
        evidence(0.80, 0.20, 0.82, 0.88, 0.88),
        evidence(0.88, 0.31, 0.92, 0.93, 0.93))
    checks['failed_observe_is_atomic'] = (
        failed_observe_is_rejected and recovered_same_frame.action == 'hold' and
        recovered_same_frame.age == 1 and
        recovered_same_frame.consecutive_confirmations == 1)
    transaction.cancel(42, 'cleanup_after_atomic_observe')

    malformed_begin = snapshot(9.0)
    malformed_begin.auxiliary['bad'] = np.asarray([object()], dtype=object)
    failed_begin_is_rejected = expect_error(
        lambda: transaction.begin(
            50, snapshot(8.0), malformed_begin, ANCHOR), TypeError)
    same_frame_begin = transaction.begin(
        50, snapshot(8.0), snapshot(9.0), ANCHOR)
    checks['failed_begin_is_atomic'] = (
        failed_begin_is_rejected and same_frame_begin.event_id == 6 and
        transaction.last_frame_id == 50)
    transaction.cancel(51, 'cleanup_after_atomic_begin')

    transaction.begin(60, snapshot(10.0), snapshot(11.0), ANCHOR)
    skipped = transaction.observe(
        62, snapshot(10.2), snapshot(11.2),
        evidence(0.80, 0.20, 0.82, 0.88, 0.88),
        evidence(0.90, 0.35, 0.95, 0.95, 0.95))
    checks['skipped_frame_fails_closed'] = (
        skipped.action == 'rollback' and skipped.age == 2 and
        skipped.consecutive_confirmations == 0 and
        skipped.reasons == ('nonconsecutive_shadow_frame',) and
        not transaction.active)

    mismatched_templates = snapshot(12.0, template_slots=1)
    failed_slot_begin = expect_error(
        lambda: transaction.begin(
            70, snapshot(11.0), mismatched_templates, ANCHOR), ValueError)
    valid_same_frame = transaction.begin(
        70, snapshot(11.0, annotation_slots=1),
        snapshot(12.0, annotation_slots=2), ANCHOR)
    checks['cross_branch_slot_mismatch_is_atomic'] = (
        failed_slot_begin and valid_same_frame.event_id == 8)
    checks['nonmonotonic_observation_rejected'] = expect_error(
        lambda: transaction.observe(
            70, snapshot(11.1, annotation_slots=2), snapshot(12.1),
            evidence(0.80, 0.20, 0.82, 0.88, 0.88),
            evidence(0.90, 0.35, 0.95, 0.95, 0.95)), ValueError)
    wrong_anchor_rejected = expect_error(
        lambda: transaction.observe(
            71, snapshot(11.1, annotation_slots=2), snapshot(12.1),
            evidence(0.80, 0.20, 0.82, 0.88, 0.88),
            evidence(0.90, 0.35, 0.95, 0.95, 0.95,
                     identity_anchor='different-anchor')), ValueError)
    valid_after_wrong_anchor = transaction.observe(
        71, snapshot(11.1, annotation_slots=2), snapshot(12.1),
        evidence(0.80, 0.20, 0.82, 0.88, 0.88),
        evidence(0.90, 0.35, 0.95, 0.95, 0.95))
    checks['identity_anchor_is_immutable_and_atomic'] = (
        wrong_anchor_rejected and valid_after_wrong_anchor.age == 1 and
        valid_after_wrong_anchor.consecutive_confirmations == 1)
    checks['valid_annotation_layout_transition_is_supported'] = (
        valid_after_wrong_anchor.action == 'hold' and transaction.active)
    transaction.cancel(72, 'cleanup_after_identity_check')

    transaction.begin(80, snapshot(14.0), snapshot(15.0), ANCHOR)
    protected_conflict = transaction.observe(
        81, snapshot(14.1), snapshot(15.1),
        evidence(0.90, 0.30, 0.90, 0.90, 0.90, conflict=True),
        evidence(0.95, 0.40, 0.95, 0.95, 0.95))
    checks['protected_hard_conflict_rolls_back'] = (
        protected_conflict.action == 'rollback' and
        protected_conflict.reasons == ('protected_hard_conflict',) and
        protected_conflict.resolved_snapshot.state == snapshot(14.1).state)

    transaction.begin(90, snapshot(16.0), snapshot(17.0), ANCHOR)
    absolute_bad = transaction.observe(
        91, snapshot(16.1), snapshot(17.1),
        evidence(0.80, 0.20, 0.90, 0.90, 0.90),
        evidence(0.64, 0.09, 0.95, 0.95, 0.95))
    absolute_expired = transaction.observe(
        92, snapshot(16.2), snapshot(17.2),
        evidence(0.80, 0.20, 0.90, 0.90, 0.90),
        evidence(0.90, 0.30, 0.95, 0.95, 0.95))
    checks['absolute_confidence_margin_gate_is_required'] = (
        absolute_bad.action == 'hold' and
        absolute_bad.consecutive_confirmations == 0 and
        'tentative_confidence_below_minimum' in absolute_bad.reasons and
        'tentative_margin_below_minimum' in absolute_bad.reasons and
        absolute_expired.action == 'rollback')

    transaction.begin(100, snapshot(18.0), snapshot(19.0), ANCHOR)
    current_protected = snapshot(18.1, x=10.2)
    explicit_rollback = transaction.rollback_current(
        101, current_protected, 'malformed_tentative_branch')
    checks['explicit_current_protected_rollback_is_atomic'] = (
        explicit_rollback.action == 'rollback' and
        explicit_rollback.resolved_snapshot.state == current_protected.state and
        explicit_rollback.reasons == ('malformed_tentative_branch',) and
        not transaction.active)

    checks['object_array_rejected'] = expect_error(
        lambda: TrackerRecursiveSnapshot.capture(
            [0.0, 0.0, 1.0, 1.0], [torch.zeros(1)],
            [torch.zeros(1)], {'bad': np.asarray([object()], dtype=object)}),
        TypeError)
    checks['nonfinite_evidence_rejected'] = expect_error(
        lambda: evidence(float('nan'), 0.1, 0.8, 0.8, 0.8), ValueError)
    checks['boolean_evidence_rejected'] = expect_error(
        lambda: evidence(True, 0.1, 0.8, 0.8, 0.8), ValueError)
    checks['numeric_string_evidence_rejected'] = expect_error(
        lambda: evidence('0.8', 0.1, 0.8, 0.8, 0.8), ValueError)

    float16_snapshot = snapshot(13.0, dtype=torch.float16)
    float16_clone = float16_snapshot.clone()
    checks['cpu_tensor_dtype_and_device_preserved'] = bool(
        float16_clone.templates[0].dtype == torch.float16 and
        float16_clone.templates[0].device == float16_snapshot.templates[0].device)

    repo_root = REPO_ROOT
    source = MODULE_SOURCE
    wiring_scan_exercised = (repo_root / 'lib' / 'test').is_dir()
    unwired, references = current_full127_is_unwired(repo_root)
    isolated_tracker = (
        repo_root / 'lib' / 'test' / 'tracker' / 'sutrack_transaction.py')
    isolated_connected = bool(
        isolated_tracker.is_file() and
        'ProtectedTentativeTemplateTransaction' in
        isolated_tracker.read_text(encoding='utf-8'))
    checks['module_unwired_from_current_full127'] = (
        wiring_scan_exercised and unwired)
    checks['isolated_low22_tracker_is_connected'] = isolated_connected

    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise AssertionError('transaction smoke failed: {}'.format(failed))

    payload = {
        'schema': 'sutrack_protected_tentative_transaction_smoke_v2',
        'status': 'passed',
        'current_full127_tracker_connected': not unwired,
        'current_full127_reference_paths': references,
        'isolated_low22_tracker_connected': isolated_connected,
        'public_tracker_wiring_scan_exercised': wiring_scan_exercised,
        'public_vot_metrics_changed': False,
        'low22_vot_started': False,
        'cuda_device_preservation_exercised': False,
        'cuda_note': 'Deferred while both GPUs run the frozen full127 VOT job.',
        'checks': checks,
        'events_started': transaction.event_count,
        'source': {'path': str(source), 'sha256': sha256_file(source)},
        'smoke_script': {
            'path': str(Path(__file__).resolve()),
            'sha256': sha256_file(Path(__file__).resolve()),
        },
    }
    atomic_json(args.output_json.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
