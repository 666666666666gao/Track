#!/usr/bin/env python3
"""Seal the transaction low22 result and emit a machine-readable gate."""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import finalize_vot_full127 as common


TRACKER = 'sutrack_l384_rgbd_anchor_identity_template_transaction_low22'
BASELINE = {
    'eao': 0.43274104354018916,
    'acc': 0.7206551125207067,
    'rob': 0.5438802182117735,
}
BASELINE_FAILURES = 195
EXPECTED_SEQUENCES = 22
EXPECTED_ANCHORS = 303
ACC_TOLERANCE = 0.001
GATE_RESULT_SCHEMA = 'sutrack_transaction_low22_gate_v1'
SOURCE_SNAPSHOT_SCHEMA = 'sutrack_transaction_low22_sources_v1'
EXPECTED_CHECKPOINT_SHA256 = (
    '2a686e8b55091d3396886de0c9e2d7a46794a5773581b96e37006f851e9dacd4')
TRANSACTION_SOURCE_FILES = (
    'experiments/sutrack/'
    'sutrack_l384_rgbd_anchor_identity_template_transaction_low22.yaml',
    'lib/test/parameter/sutrack_transaction.py',
    'lib/test/tracker/protected_tentative_transaction.py',
    'lib/test/tracker/sutrack_transaction.py',
    'lib/test/vot/'
    'sutrack_l384_rgbd_anchor_identity_template_transaction_low22.py',
    'lib/test/vot/sutrack_transaction_class.py',
    'tools/prepare_vot_transaction_low22.py',
    'tools/launch_vot_transaction_low22.sh',
    'tools/finalize_vot_transaction_low22.py',
    'tools/finalize_vot_transaction_low22_diagnostics.py',
    'tools/smoke_sutrack_transaction_integration.py',
    'tools/smoke_sutrack_transaction_gpu.py',
    'tools/smoke_sutrack_template_transaction_parity.py',
    'tools/diagnose_vot_transaction_failure.py',
)
SOURCE_FILES = tuple(dict.fromkeys(
    tuple(common.IMPLEMENTATION_FILES) + TRANSACTION_SOURCE_FILES))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument(
        '--repo-root', type=Path, default=Path('/home/SUTrack_RGBD_L'))
    parser.add_argument(
        '--python', default='/root/miniconda3/envs/mplt/bin/python')
    parser.add_argument('--poll-seconds', type=float, default=60.0)
    parser.add_argument(
        '--checkpoint', type=Path,
        default=Path(
            '/root/autodl-tmp/sutrack_assets/weights/'
            'SUTRACK_ep0180_l384.pth.tar'))
    parser.add_argument('--analysis-name', default='transaction_low22_analysis')
    parser.add_argument('--expected-anchor-count', type=int,
                        default=EXPECTED_ANCHORS)
    parser.add_argument('--expected-sequence-count', type=int,
                        default=EXPECTED_SEQUENCES)
    parser.add_argument('--expected-tracker', default=TRACKER)
    parser.add_argument('--expected-toolkit', default='0.7.1')
    parser.add_argument('--expected-manifest-sha256', default='')
    return parser.parse_args()


def expected_gate():
    return {
        'baseline_metrics_fraction': BASELINE,
        'baseline_confirmed_failures': BASELINE_FAILURES,
        'rule': {
            'eao': 'candidate > baseline',
            'rob': 'candidate > baseline',
            'acc': 'candidate >= baseline - 0.001 fraction',
            'confirmed_failures': 'candidate <= baseline',
        },
        'expected_sequence_count': EXPECTED_SEQUENCES,
        'expected_anchor_count': EXPECTED_ANCHORS,
        'full127_allowed_only_if_low22_improves': True,
    }


def validate_gate_manifest(manifest):
    gate = manifest.get('selection_gate')
    if not isinstance(gate, dict):
        raise ValueError('Selection gate is missing')
    for key, value in expected_gate().items():
        if gate.get(key) != value:
            raise ValueError('Selection gate differs at {}'.format(key))
    for prefix in ('baseline_manifest', 'baseline_report'):
        path = common.require_file(gate[prefix]).resolve()
        if common.sha256_file(path) != gate[prefix + '_sha256']:
            raise ValueError('{} changed after gate freeze'.format(prefix))
    trace_root = Path(manifest.get('transaction_trace_root', '')).resolve()
    if trace_root.parent != Path(manifest['shards'][0]['root']).parents[0]:
        run_root = Path(manifest['shards'][0]['root']).parent.resolve()
        if trace_root.parent != run_root:
            raise ValueError('Transaction trace root escapes run root')
    return trace_root


def source_snapshot(args, root, manifest_sha):
    records = {}
    for relative in SOURCE_FILES:
        path = common.require_file(args.repo_root.resolve() / relative)
        records[relative] = {
            'path': str(path.resolve()),
            'size': path.stat().st_size,
            'sha256': common.sha256_file(path),
        }
    checkpoint = common.require_file(args.checkpoint)
    checkpoint_sha = common.sha256_file(checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError('DepthTrack-trained checkpoint SHA differs')
    records['checkpoint'] = {
        'path': str(checkpoint.resolve()),
        'size': checkpoint.stat().st_size,
        'sha256': checkpoint_sha,
    }
    path = root / 'transaction_source_snapshot.json'
    payload = {
        'schema': SOURCE_SNAPSHOT_SCHEMA,
        'manifest_sha256': manifest_sha,
        'sources': records,
    }
    if path.exists():
        if common.load_json(path) != payload:
            raise ValueError('Existing transaction source snapshot differs')
    else:
        common.atomic_json(path, payload)
    return path, payload


def validate_sources(payload):
    for name, record in payload['sources'].items():
        path = common.require_file(record['path'])
        if (path.stat().st_size != record['size'] or
                common.sha256_file(path) != record['sha256']):
            raise ValueError('Frozen source changed: {}'.format(name))


def analysis_settings(experiment):
    eao_score = None
    ar_average = None
    for analysis in experiment.analyses:
        name = analysis.__class__.__name__
        if name == 'EAOScore':
            eao_score = analysis
        elif name == 'AverageAccuracyRobustness':
            ar_average = analysis
    if eao_score is None or ar_average is None:
        raise RuntimeError('Official EAO or AR analysis is missing')
    curves = eao_score.eaocurve.curves
    ar_partial = ar_average.analysis
    for name in ('burnin', 'grace', 'threshold', 'ignore_masks'):
        if getattr(curves, name) != getattr(ar_partial, name):
            raise RuntimeError('Official EAO/AR settings disagree')
    return {
        'burnin': int(ar_partial.burnin),
        'grace': int(ar_partial.grace),
        'threshold': float(ar_partial.threshold),
        'ignore_masks': str(ar_partial.ignore_masks),
    }


def failure_progress(overlaps, proxy, grace_frames, threshold):
    remaining = grace_frames
    progress = len(proxy)
    for index, overlap in enumerate(overlaps):
        if overlap <= threshold and not proxy.groundtruth(index).is_empty():
            remaining -= 1
            if remaining == 0:
                progress = index + 1 - grace_frames
                break
        else:
            remaining = grace_frames
    return progress


def collect_confirmed_failure_outcomes(
        master, tracker_id, expected_anchors=EXPECTED_ANCHORS):
    from vot.dataset.proxy import FrameMapSequence
    from vot.experiment.multistart import find_anchors
    from vot.region import calculate_overlaps
    from vot.tracker import Trajectory
    from vot.workspace import Workspace

    workspace = Workspace.load(str(master))
    trackers = workspace.registry.resolve(
        tracker_id, storage=workspace.storage.substorage('results'),
        skip_unknown=False)
    if len(trackers) != 1:
        raise RuntimeError('Expected exactly one resolved tracker')
    tracker = trackers[0]
    experiment = workspace.stack.experiments['baseline']
    settings = analysis_settings(experiment)
    outcomes = {}
    failures = 0
    per_sequence = {}
    for sequence in experiment.transform(workspace.dataset):
        results = experiment.results(tracker, sequence)
        forward, backward = find_anchors(sequence, experiment.anchor)
        anchors = [(index, False) for index in forward]
        anchors.extend((index, True) for index in backward)
        sequence_failures = 0
        for anchor, reverse in anchors:
            name = '{}_{:08d}'.format(sequence.name, anchor)
            if not Trajectory.exists(results, name):
                raise RuntimeError('Missing trajectory {}'.format(name))
            frame_map = (
                list(reversed(range(0, anchor + 1))) if reverse else
                list(range(anchor, len(sequence))))
            proxy = FrameMapSequence(sequence, frame_map)
            trajectory = Trajectory.read(results, name)
            if len(trajectory) != len(proxy):
                raise RuntimeError('Incomplete trajectory {}'.format(name))
            masks = proxy.object(settings['ignore_masks'])
            overlaps = list(calculate_overlaps(
                trajectory.regions(), proxy.groundtruth(),
                proxy.size if settings['burnin'] else None, ignore=masks))
            progress = failure_progress(
                overlaps, proxy, settings['grace'], settings['threshold'])
            failed = progress < len(overlaps)
            key = '{}@{}{}'.format(
                sequence.name, anchor, 'B' if reverse else 'F')
            if key in outcomes:
                raise RuntimeError('Duplicate failure outcome {}'.format(key))
            outcomes[key] = {
                'anchor_key': key,
                'sequence': sequence.name,
                'anchor': int(anchor),
                'direction': 'backward' if reverse else 'forward',
                'failed': bool(failed),
                'progress': int(progress),
                'run_length': len(overlaps),
            }
            sequence_failures += int(failed)
            failures += int(failed)
        per_sequence[sequence.name] = {
            'anchors': len(anchors),
            'confirmed_failures': sequence_failures,
        }
    if len(outcomes) != expected_anchors:
        raise RuntimeError('Failure audit anchor count differs')
    return outcomes, failures, per_sequence, settings


def count_confirmed_failures(master, tracker_id):
    _, failures, per_sequence, settings = (
        collect_confirmed_failure_outcomes(master, tracker_id))
    return failures, per_sequence, settings


def validate_traces(trace_root, trajectories):
    expected = {
        '{}__anchor-{:06d}.jsonl'.format(
            trajectory.rsplit('_', 1)[0], int(trajectory.rsplit('_', 1)[1]))
        for trajectory in trajectories
    }
    present = {path.name for path in trace_root.glob('*.jsonl')}
    if present != expected:
        raise RuntimeError(
            'Trace coverage differs: missing={}, extra={}'.format(
                len(expected - present), len(present - expected)))
    for name in expected:
        path = trace_root / name
        first = path.read_text(encoding='utf-8').splitlines()[0]
        if json.loads(first).get('type') != 'initialize':
            raise ValueError('Trace lacks initialize record: {}'.format(name))
    return {
        'root': str(trace_root),
        'file_count': len(present),
    }


def build_gate_result(metrics, failures, per_sequence, settings, trace,
                      manifest_sha, merge_path, analysis_path, snapshot_path):
    checks = {
        'eao_strictly_improved': metrics['eao'] > BASELINE['eao'],
        'rob_strictly_improved': metrics['rob'] > BASELINE['rob'],
        'acc_within_minus_0_10_pp': (
            metrics['acc'] >= BASELINE['acc'] - ACC_TOLERANCE),
        'confirmed_failures_not_increased': failures <= BASELINE_FAILURES,
    }
    gate_passed = all(checks.values())
    return {
        'schema': GATE_RESULT_SCHEMA,
        'status': 'complete',
        'tracker': TRACKER,
        'toolkit': '0.7.1',
        'sequence_count': EXPECTED_SEQUENCES,
        'anchor_count': EXPECTED_ANCHORS,
        'baseline': {
            'metrics_fraction': BASELINE,
            'metrics_percent': {
                key: value * 100.0 for key, value in BASELINE.items()},
            'confirmed_failures': BASELINE_FAILURES,
        },
        'candidate': {
            'metrics_fraction': metrics,
            'metrics_percent': {
                key: value * 100.0 for key, value in metrics.items()},
            'confirmed_failures': failures,
        },
        'delta_percent_points': {
            key: (metrics[key] - BASELINE[key]) * 100.0
            for key in BASELINE
        },
        'failure_delta': failures - BASELINE_FAILURES,
        'gate_checks': checks,
        'gate_passed': gate_passed,
        'full127_authorized': gate_passed,
        'automatic_full127_launch': False,
        'official_failure_settings': settings,
        'per_sequence_failures': per_sequence,
        'transaction_traces': trace,
        'shard_manifest_sha256': manifest_sha,
        'merge_result': {
            'path': str(merge_path),
            'sha256': common.sha256_file(merge_path),
        },
        'analysis': {
            'path': str(analysis_path),
            'sha256': common.sha256_file(analysis_path),
        },
        'source_snapshot': {
            'path': str(snapshot_path),
            'sha256': common.sha256_file(snapshot_path),
        },
    }


def main():
    args = parse_args()
    if args.poll_seconds <= 0:
        raise ValueError('poll-seconds must be positive')
    root, manifest_path, manifest_sha, manifest, trajectories = (
        common.load_manifest(args))
    trace_root = validate_gate_manifest(manifest)
    snapshot_path, snapshot = source_snapshot(args, root, manifest_sha)
    lock = open(root / 'transaction_finalizer.lock', 'a+', encoding='utf-8')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError('Another transaction finalizer owns the lock') from error

    while not (root / 'merge_result.json').exists():
        complete = common.completed_count(manifest)
        common.write_status(
            root / 'transaction_finalizer_status.json',
            'waiting_for_merge', complete, EXPECTED_ANCHORS)
        print('WAIT {}/{}'.format(complete, EXPECTED_ANCHORS), flush=True)
        time.sleep(args.poll_seconds)

    validate_sources(snapshot)
    merge_path, _, master = common.validate_merge(
        root, manifest_path, manifest_sha, manifest, trajectories)
    analysis_path = common.run_analysis(
        args, root, master, manifest['tracker'])
    _, metrics = common.parse_analysis(args, analysis_path, manifest)
    failures, per_sequence, settings = count_confirmed_failures(
        master, manifest['tracker'])
    trace = validate_traces(trace_root, trajectories)
    validate_sources(snapshot)
    result = build_gate_result(
        metrics, failures, per_sequence, settings, trace, manifest_sha,
        merge_path, analysis_path, snapshot_path)
    result_path = root / 'low22_gate_result.json'
    if result_path.exists():
        if common.load_json(result_path) != result:
            raise ValueError('Existing low22 gate result differs')
    else:
        common.atomic_json(result_path, result)
    common.write_status(
        root / 'transaction_finalizer_status.json',
        'complete', EXPECTED_ANCHORS, EXPECTED_ANCHORS, {
            'gate_passed': result['gate_passed'],
            'full127_authorized': result['full127_authorized'],
            'automatic_full127_launch': False,
            'gate_result_sha256': common.sha256_file(result_path),
        })
    print(json.dumps({
        'status': 'complete',
        'candidate_metrics_percent': result['candidate']['metrics_percent'],
        'candidate_confirmed_failures': failures,
        'gate_checks': result['gate_checks'],
        'gate_passed': result['gate_passed'],
        'automatic_full127_launch': False,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
