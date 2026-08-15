#!/usr/bin/env python3
"""Wait for frozen Train152 traces, then analyze and train the small gate."""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback


SCHEMA = 'sutrack-state-gate-controller/v1'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=int, default=30)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_json(path):
    with Path(path).open('r', encoding='utf-8') as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError('{} is not an object'.format(path))
    return value


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    args = parse_args()
    if args.poll_seconds <= 0 or args.poll_seconds > 60:
        raise ValueError('--poll-seconds must lie in [1, 60]')
    root = args.root.resolve()
    repository = args.repository.resolve()
    dataset_root = args.dataset_root.resolve()
    status_path = root / 'state_gate_controller_status.json'
    lock_path = root / 'state_gate_controller.lock'
    root.mkdir(parents=True, exist_ok=True)
    lock_stream = lock_path.open('a+')
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError('another state-gate controller holds the lock') from error

    trace_plan_path = root / 'full152_trace_plan.json'
    split_plan_path = root / 'source' / 'state_gate_split_plan.json'
    analyzer_path = repository / 'tools' / (
        'analyze_depthtrack_train_state_trace_full152.py')
    trainer_path = repository / 'tools' / 'train_sutrack_state_gate.py'
    controller_path = Path(__file__).resolve()
    sources = {
        str(trace_plan_path): sha256_file(trace_plan_path),
        str(split_plan_path): sha256_file(split_plan_path),
        str(analyzer_path): sha256_file(analyzer_path),
        str(trainer_path): sha256_file(trainer_path),
        str(controller_path): sha256_file(controller_path),
    }
    status = {
        'schema': SCHEMA,
        'state': 'waiting_for_traces',
        'started_at': now(),
        'updated_at': now(),
        'root': str(root),
        'repository': str(repository),
        'dataset_root': str(dataset_root),
        'source_sha256': sources,
        'public_evaluation': False,
        'backbone_training_started': False,
        'small_gate_training_started': False,
        'future_frame_text_used': False,
    }
    atomic_json(status_path, status)
    print(json.dumps(status, sort_keys=True), flush=True)
    try:
        exit_paths = [
            root / 'logs' / 'full152_shard0.exit',
            root / 'logs' / 'full152_shard1.exit',
        ]
        while not all(path.is_file() for path in exit_paths):
            time.sleep(args.poll_seconds)
        exits = [int(path.read_text(encoding='utf-8').strip())
                 for path in exit_paths]
        if exits != [0, 0]:
            raise RuntimeError('trace shard exits are {}'.format(exits))

        shard_dirs = [
            root / 'fixed6_trace' / 'shard0',
            root / 'fixed6_trace' / 'shard1',
            root / 'full152_trace' / 'shard0',
            root / 'full152_trace' / 'shard1',
        ]
        for shard_dir in shard_dirs:
            manifest_path = shard_dir / 'manifest.json'
            manifest = load_json(manifest_path)
            if (manifest.get('complete') is not True or
                    manifest.get('public_evaluation') is not False or
                    manifest.get('ground_truth_available_to_tracker') is not
                    False or manifest.get('future_frame_text_used') is not
                    False):
                raise ValueError('trace manifest contract failed {}'.format(
                    manifest_path))
        for path_string, expected_sha in sources.items():
            if sha256_file(path_string) != expected_sha:
                raise ValueError('controller source drift {}'.format(path_string))

        trace_plan = load_json(trace_plan_path)
        expected_sequences = ','.join(
            trace_plan['analysis_sequence_order'])
        analysis_dir = root / 'full152_analysis'
        analysis_log = root / 'logs' / 'full152_analysis.log'
        status.update({
            'state': 'analyzing_full152',
            'updated_at': now(),
            'trace_exits': exits,
        })
        atomic_json(status_path, status)
        print(json.dumps(status, sort_keys=True), flush=True)
        analyzer_command = [
            sys.executable, str(analyzer_path),
            '--shard-dir', str(shard_dirs[0]),
            '--shard-dir', str(shard_dirs[1]),
            '--shard-dir', str(shard_dirs[2]),
            '--shard-dir', str(shard_dirs[3]),
            '--dataset-root', str(dataset_root),
            '--expected-sequences', expected_sequences,
            '--trace-plan', str(trace_plan_path),
            '--output-dir', str(analysis_dir),
        ]
        with analysis_log.open('x', encoding='utf-8') as stream:
            subprocess.run(
                analyzer_command, cwd=str(repository), stdout=stream,
                stderr=subprocess.STDOUT, check=True,
                env=dict(os.environ, PYTHONPATH=str(repository)))
        capacity_path = analysis_dir / 'capacity_result.json'
        capacity = load_json(capacity_path)
        if (capacity.get('complete') is not True or
                capacity.get('capacity_supported') is not True or
                len(capacity.get('expected_sequences', [])) != 152):
            status.update({
                'state': 'complete_capacity_rejected',
                'updated_at': now(),
                'capacity_result_path': str(capacity_path),
                'capacity_result_sha256': sha256_file(capacity_path),
                'small_gate_training_started': False,
            })
            atomic_json(status_path, status)
            print(json.dumps(status, sort_keys=True), flush=True)
            return

        training_root = root / 'gate_training'
        training_log = root / 'logs' / 'gate_training.log'
        status.update({
            'state': 'training_small_gate',
            'updated_at': now(),
            'capacity_result_path': str(capacity_path),
            'capacity_result_sha256': sha256_file(capacity_path),
            'small_gate_training_started': True,
        })
        atomic_json(status_path, status)
        print(json.dumps(status, sort_keys=True), flush=True)
        training_command = [
            sys.executable, str(trainer_path),
            '--capacity-result', str(capacity_path),
            '--split-plan', str(split_plan_path),
            '--output-root', str(training_root),
        ]
        with training_log.open('x', encoding='utf-8') as stream:
            subprocess.run(
                training_command, cwd=str(repository), stdout=stream,
                stderr=subprocess.STDOUT, check=True,
                env=dict(os.environ, PYTHONPATH=str(repository)))
        training_result_path = training_root / 'training_result.json'
        training_result = load_json(training_result_path)
        status.update({
            'state': 'complete',
            'updated_at': now(),
            'training_result_path': str(training_result_path),
            'training_result_sha256': sha256_file(training_result_path),
            'training_decision': training_result['decision'],
            'ready_for_recursive_audit': training_result[
                'ready_for_recursive_audit'],
        })
        atomic_json(status_path, status)
        print(json.dumps(status, sort_keys=True), flush=True)
    except BaseException as error:
        status.update({
            'state': 'failed',
            'updated_at': now(),
            'error_type': type(error).__name__,
            'error': str(error),
            'traceback': traceback.format_exc(),
        })
        atomic_json(status_path, status)
        print(json.dumps(status, sort_keys=True), flush=True)
        raise


if __name__ == '__main__':
    main()
