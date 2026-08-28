#!/usr/bin/env python3
"""Prepare an auditable 22-sequence VOT workspace for transaction testing."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKER = 'sutrack_l384_rgbd_anchor_identity_transaction_low22'
EXPECTED_SEQUENCE_COUNT = 22
EXPECTED_ANCHOR_COUNT = 303
BASELINE_METRICS = {
    'eao': 0.43274104354018916,
    'acc': 0.7206551125207067,
    'rob': 0.5438802182117735,
}
BASELINE_FAILURES = 195


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    temporary = path.with_name(path.name + '.tmp-{}'.format(os.getpid()))
    with open(temporary, 'x', encoding='utf-8', newline='\n') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--baseline-manifest', type=Path, required=True)
    parser.add_argument('--baseline-report', type=Path, required=True)
    parser.add_argument('--trace-root', type=Path, required=True)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    baseline_path = args.baseline_manifest.resolve()
    baseline_report_path = args.baseline_report.resolve()
    trace_root = args.trace_root.resolve()
    if trace_root.parent != output_root:
        raise ValueError('Trace root must be a direct child of output root')
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    baseline_report = json.loads(
        baseline_report_path.read_text(encoding='utf-8'))
    report_candidate = baseline_report['candidate']
    sequences = list(baseline['sequences'])
    if (len(sequences) != EXPECTED_SEQUENCE_COUNT or
            baseline['total_anchor_count'] != EXPECTED_ANCHOR_COUNT):
        raise ValueError('Baseline manifest is not the frozen low22/303 gate')
    report_metrics = {
        'eao': report_candidate['aggregate']['eao'],
        'acc': report_candidate['aggregate']['accuracy'],
        'rob': report_candidate['aggregate']['robustness'],
    }
    if (report_metrics != BASELINE_METRICS or
            report_candidate['failed_anchors'] != BASELINE_FAILURES or
            report_candidate['total_anchors'] != EXPECTED_ANCHOR_COUNT):
        raise ValueError('Baseline report differs from the frozen gate')

    command = [
        sys.executable,
        str(REPO_ROOT / 'tools' / 'create_vot_failure_family_shards.py'),
        '--output-root', str(output_root),
        '--shards', '4', '--gpus', '2', '--tracker', TRACKER,
    ]
    for sequence in sequences:
        command.extend(('--sequence', sequence))
    subprocess.run(command, check=True, cwd=str(REPO_ROOT))

    manifest_path = output_root / 'shard_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    baseline_trajectories = sorted(
        trajectory
        for shard in baseline['shards']
        for trajectory in shard['expected_trajectories'])
    candidate_trajectories = sorted(
        trajectory
        for shard in manifest['shards']
        for trajectory in shard['expected_trajectories'])
    if (manifest['sequences'] != sequences or
            manifest['total_anchor_count'] != EXPECTED_ANCHOR_COUNT or
            candidate_trajectories != baseline_trajectories):
        raise RuntimeError('Candidate workspace differs from low22 gate set')

    trace_line = (
        'env_SUTRACK_TRANSACTION_TRACE_ROOT = {}\n'.format(trace_root))
    for shard in manifest['shards']:
        tracker_path = Path(shard['root']) / 'trackers.ini'
        tracker_text = tracker_path.read_text(encoding='utf-8')
        if 'env_SUTRACK_TRANSACTION_TRACE_ROOT' in tracker_text:
            raise RuntimeError('Trace environment is already present')
        with open(tracker_path, 'a', encoding='utf-8', newline='\n') as stream:
            stream.write(trace_line)
            stream.flush()
            os.fsync(stream.fileno())
        shard['trackers_sha256'] = sha256_file(tracker_path)

    manifest['selection_gate'] = {
        'baseline_manifest': str(baseline_path),
        'baseline_manifest_sha256': sha256_file(baseline_path),
        'baseline_report': str(baseline_report_path),
        'baseline_report_sha256': sha256_file(baseline_report_path),
        'baseline_metrics_fraction': BASELINE_METRICS,
        'baseline_confirmed_failures': BASELINE_FAILURES,
        'rule': {
            'eao': 'candidate > baseline',
            'rob': 'candidate > baseline',
            'acc': 'candidate >= baseline - 0.001 fraction',
            'confirmed_failures': 'candidate <= baseline',
        },
        'expected_sequence_count': EXPECTED_SEQUENCE_COUNT,
        'expected_anchor_count': EXPECTED_ANCHOR_COUNT,
        'full127_allowed_only_if_low22_improves': True,
    }
    manifest['transaction_trace_root'] = str(trace_root)
    atomic_json(manifest_path, manifest)
    print(json.dumps({
        'status': 'prepared',
        'root': str(output_root),
        'trace_root': str(trace_root),
        'sequence_count': len(sequences),
        'anchor_count': len(candidate_trajectories),
        'manifest_sha256': sha256_file(manifest_path),
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
