#!/usr/bin/env python3
"""Seal the pre-registered worst-5 state-rollback diagnostic."""

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time


TRACKER = 'sutrack_l384_rgbd_language_safe_template_state_rollback_v3'
BASELINE_TRACKER = 'sutrack_l384_rgbd_language_safe_template'
SEQUENCES = (
    'bandlight_indoor_1',
    'box_room_noocc_4_1',
    'cube05_indoor_5',
    'cube05_indoor_6',
    'yogurt_indoor_1',
)
SUFFIXES = ('.bin', '_confidence.value', '_time.value')
GATES_PP = {
    'minimum_eao_gain': 0.50,
    'minimum_rob_gain': 3.00,
    'maximum_acc_loss': 1.00,
}
IMPLEMENTATION_FILES = (
    'lib/config/sutrack/config.py',
    'lib/models/sutrack/encoder.py',
    'lib/test/evaluation/local.py',
    'lib/test/parameter/sutrack.py',
    'lib/test/tracker/rgbd_frame.py',
    'lib/test/tracker/rgbd_language_manifest.py',
    'lib/test/tracker/safe_template_update.py',
    'lib/test/tracker/temporal_depth_identity.py',
    'lib/test/tracker/sutrack.py',
    'lib/test/vot/sutrack_class.py',
    'lib/test/vot/vot.py',
    'lib/test/vot/sutrack_l384_rgbd_language_safe_template_state_rollback_v3.py',
    'tools/create_vot_failure_family_shards.py',
    'tools/run_vot_failure_family_shards.py',
    'tools/finalize_state_rollback_worst5.py',
)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-{}'.format(os.getpid()))
    with open(temporary, 'x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + '\n')


def load_json(path):
    with open(path, 'r', encoding='utf-8') as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError('{} is not a JSON object'.format(path))
    return value


def require_file(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(str(path))
    return path


def source_record(path):
    path = require_file(path).resolve()
    return {
        'path': str(path),
        'size': path.stat().st_size,
        'sha256': sha256_file(path),
    }


def current_sources(args):
    records = {
        'implementation/' + relative: source_record(args.repo / relative)
        for relative in IMPLEMENTATION_FILES
    }
    external = {
        'configuration': args.repo / 'experiments/sutrack' /
            'sutrack_l384_rgbd_language_safe_template_state_rollback_v3.yaml',
        'checkpoint': Path(
            '/root/autodl-tmp/sutrack_assets/weights/SUTRACK_ep0180_l384.pth.tar'),
        'clip_checkpoint': Path(
            '/root/autodl-tmp/sutrack_assets/weights/ViT-L-14.pt'),
        'language_manifest': Path(
            '/home/OSTrack_RGBD_L_dataset_modified/annotations_cleaned/'
            'votrgbd2022_language.jsonl'),
        'baseline_view_manifest': args.root / 'baseline_safe_v1_view/view_manifest.json',
        'baseline_analysis': args.root / (
            'baseline_safe_v1_view/analysis/baseline_safe_v1_worst5.json'),
        'formal_safe_v1_full_result': Path(
            '/root/autodl-tmp/sutrack_rgbd_language_safe_template_vot_full127_v1/'
            'full_result.json'),
    }
    for name, path in external.items():
        records[name] = source_record(path)
    return records


def ensure_snapshot(args, manifest_sha):
    path = args.root / 'diagnostic_source_snapshot.json'
    sources = current_sources(args)
    if path.exists():
        value = load_json(path)
        if (value.get('schema') != 'sutrack_state_rollback_worst5_snapshot_v1'
                or value.get('shard_manifest_sha256') != manifest_sha
                or value.get('sources') != sources):
            raise ValueError('Source snapshot mismatch')
    else:
        value = {
            'schema': 'sutrack_state_rollback_worst5_snapshot_v1',
            'created_at': utc_now(),
            'shard_manifest_sha256': manifest_sha,
            'sources': sources,
        }
        atomic_json(path, value)
    return path, value


def validate_snapshot(path, expected):
    if load_json(path) != expected:
        raise ValueError('Snapshot file changed')
    for name, record in expected['sources'].items():
        path = require_file(record['path'])
        if (path.stat().st_size != record['size'] or
                sha256_file(path) != record['sha256']):
            raise ValueError('Frozen source changed: {}'.format(name))


def validate_manifest(args):
    path = require_file(args.root / 'shard_manifest.json')
    digest = sha256_file(path)
    if digest != args.expected_manifest_sha256:
        raise ValueError('Shard manifest SHA mismatch')
    value = load_json(path)
    if (value.get('schema') != 'sutrack_vot_failure_family_anchor_shards_v1'
            or value.get('tracker') != TRACKER
            or value.get('sequences') != list(SEQUENCES)
            or value.get('total_anchor_count') != 112
            or value.get('shard_count') != 10
            or value.get('gpu_count') != 2):
        raise ValueError('Shard manifest contract mismatch')
    trajectories = []
    for shard in value['shards']:
        shard_root = Path(shard['root']).resolve()
        if args.root.resolve() not in shard_root.parents:
            raise ValueError('Shard escapes run root')
        for name, key in (
                ('config.yaml', 'config_sha256'),
                ('trackers.ini', 'trackers_sha256'),
                ('sequences/list.txt', 'list_sha256')):
            if sha256_file(require_file(shard_root / name)) != shard[key]:
                raise ValueError('Shard source changed: {}'.format(shard_root / name))
        trajectories.extend(shard['expected_trajectories'])
    if len(trajectories) != 112 or len(set(trajectories)) != 112:
        raise ValueError('Trajectory cover mismatch')
    return path, digest, value, trajectories


def completed_count(manifest):
    count = 0
    for shard in manifest['shards']:
        root = Path(shard['root']) / 'results' / TRACKER / 'baseline'
        for trajectory in shard['expected_trajectories']:
            sequence = trajectory.rsplit('_', 1)[0]
            if all((root / sequence / (trajectory + suffix)).is_file()
                   for suffix in SUFFIXES):
                count += 1
    return count


def wait_for_merge(args, manifest):
    path = args.root / 'merge_result.json'
    last = None
    while not path.exists():
        current = completed_count(manifest)
        if current != last:
            atomic_json(args.root / 'diagnostic_status.json', {
                'schema': 'sutrack_state_rollback_worst5_status_v1',
                'updated_at': utc_now(),
                'stage': 'waiting_for_merge',
                'completed_anchors': current,
                'total_anchors': 112,
            })
            last = current
        time.sleep(args.poll_seconds)
    return require_file(path)


def validate_merge(args, manifest_path, manifest_sha, manifest, trajectories):
    path = require_file(args.root / 'merge_result.json')
    value = load_json(path)
    master = (args.root / 'master').resolve()
    if (value.get('schema') != 'sutrack_vot_failure_family_anchor_merge_v1'
            or value.get('status') != 'complete'
            or value.get('tracker') != TRACKER
            or Path(value.get('master_workspace', '')).resolve() != master
            or Path(value.get('source_manifest', '')).resolve() != manifest_path.resolve()
            or value.get('source_manifest_sha256') != manifest_sha
            or value.get('anchor_count') != 112
            or value.get('result_file_count') != 336):
        raise ValueError('Merge receipt contract mismatch')
    expected = set()
    for trajectory in trajectories:
        sequence = trajectory.rsplit('_', 1)[0]
        for suffix in SUFFIXES:
            expected.add(str(Path('results') / TRACKER / 'baseline' / sequence /
                             (trajectory + suffix)))
    if set(value.get('result_sha256', {})) != expected:
        raise ValueError('Merge result coverage mismatch')
    for relative, digest in value['result_sha256'].items():
        if sha256_file(require_file(master / relative)) != digest:
            raise ValueError('Merged result changed: {}'.format(relative))
    if (master / 'sequences/list.txt').read_text(encoding='utf-8').splitlines() != list(SEQUENCES):
        raise ValueError('Master sequence order mismatch')
    return path, value, master


def run_analysis(args, master):
    path = master / 'analysis/state_rollback_v3_worst5.json'
    if not path.exists():
        environment = dict(os.environ)
        environment['PYTHONPATH'] = str(args.repo)
        command = [
            args.python, '-m', 'vot', 'analysis', '--workspace', str(master),
            '--format', 'json', '--name', 'state_rollback_v3_worst5', TRACKER,
        ]
        with open(args.root / 'diagnostic_analysis.log', 'ab', buffering=0) as log:
            result = subprocess.run(
                command, cwd=str(args.repo), env=environment,
                stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0:
            raise RuntimeError('VOT analysis failed: {}'.format(result.returncode))
    return require_file(path)


def metrics(path, tracker):
    value = load_json(path)
    if (value.get('toolkit') != '0.7.1'
            or set(value.get('sequences', {})) != set(SEQUENCES)
            or tracker not in value.get('trackers', {})):
        raise ValueError('Analysis contract mismatch: {}'.format(path))
    results = value['results']['baseline']['results']
    output = {
        'eao': float(results[0][0][0]),
        'acc': float(results[2][0][0]),
        'rob': float(results[2][0][1]),
    }
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in output.values()):
        raise ValueError('Invalid metric')
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path('/home/SUTrack_RGBD_L'))
    parser.add_argument('--python', default='/root/miniconda3/envs/mplt/bin/python')
    parser.add_argument('--poll-seconds', type=float, default=30.0)
    parser.add_argument('--expected-manifest-sha256', required=True)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.repo = args.repo.resolve()

    manifest_path, manifest_sha, manifest, trajectories = validate_manifest(args)
    snapshot_path, snapshot = ensure_snapshot(args, manifest_sha)
    wait_for_merge(args, manifest)
    merge_path, _merge, master = validate_merge(
        args, manifest_path, manifest_sha, manifest, trajectories)
    validate_snapshot(snapshot_path, snapshot)
    analysis_path = run_analysis(args, master)
    validate_snapshot(snapshot_path, snapshot)

    baseline_path = args.root / (
        'baseline_safe_v1_view/analysis/baseline_safe_v1_worst5.json')
    baseline = metrics(baseline_path, BASELINE_TRACKER)
    candidate = metrics(analysis_path, TRACKER)
    delta_pp = {
        name: (candidate[name] - baseline[name]) * 100.0
        for name in ('eao', 'acc', 'rob')
    }
    checks = {
        'eao_gain_at_least_0_50pp':
            delta_pp['eao'] >= GATES_PP['minimum_eao_gain'],
        'rob_gain_at_least_3_00pp':
            delta_pp['rob'] >= GATES_PP['minimum_rob_gain'],
        'acc_loss_at_most_1_00pp':
            delta_pp['acc'] >= -GATES_PP['maximum_acc_loss'],
    }
    result = {
        'schema': 'sutrack_state_rollback_worst5_result_v1',
        'status': 'complete',
        'generated_at': utc_now(),
        'scope': 'post_hoc_public_worst5_diagnostic',
        'claim_limit': (
            'Selected after the safe-v1 full-127 result; this is a capacity '
            'diagnostic, not an unbiased estimate and not a formal benchmark result.'),
        'tracker': TRACKER,
        'baseline_tracker': BASELINE_TRACKER,
        'sequences': list(SEQUENCES),
        'anchor_count': 112,
        'baseline_metrics_percent': {
            name: value * 100.0 for name, value in baseline.items()},
        'candidate_metrics_percent': {
            name: value * 100.0 for name, value in candidate.items()},
        'delta_pp': delta_pp,
        'preregistered_capacity_gates_pp': GATES_PP,
        'capacity_checks': checks,
        'eligible_for_full127_evaluation': all(checks.values()),
        'provenance': {
            'shard_manifest': {
                'path': str(manifest_path), 'sha256': manifest_sha},
            'source_snapshot': {
                'path': str(snapshot_path), 'sha256': sha256_file(snapshot_path)},
            'merge_result': {
                'path': str(merge_path), 'sha256': sha256_file(merge_path)},
            'baseline_view_manifest': {
                'path': str(args.root / 'baseline_safe_v1_view/view_manifest.json'),
                'sha256': sha256_file(args.root / 'baseline_safe_v1_view/view_manifest.json')},
            'baseline_analysis': {
                'path': str(baseline_path), 'sha256': sha256_file(baseline_path)},
            'candidate_analysis': {
                'path': str(analysis_path), 'sha256': sha256_file(analysis_path)},
        },
    }
    output = args.root / 'diagnostic_result.json'
    if output.exists():
        existing = load_json(output)
        comparable = dict(result)
        comparable['generated_at'] = existing.get('generated_at')
        if existing != comparable:
            raise ValueError('Existing diagnostic result differs')
        result = existing
    else:
        atomic_json(output, result)
    atomic_json(args.root / 'diagnostic_status.json', {
        'schema': 'sutrack_state_rollback_worst5_status_v1',
        'updated_at': utc_now(),
        'stage': 'complete',
        'completed_anchors': 112,
        'total_anchors': 112,
        'result_sha256': sha256_file(output),
    })
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
