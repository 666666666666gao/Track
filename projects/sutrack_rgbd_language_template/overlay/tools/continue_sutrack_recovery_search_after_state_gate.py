#!/usr/bin/env python3
"""Conditionally run source-identical OFF then fixed6 recovery-search ON."""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import traceback


SCHEMA = 'sutrack-recovery-search-controller/v1'
UPSTREAM_TERMINAL = {
    'complete',
    'complete_capacity_rejected',
    'failed',
}


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
            value = json.loads(raw_line)
            if not isinstance(value, dict):
                raise ValueError('non-object row {}:{}'.format(
                    path, line_number))
            rows.append(value)
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


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def validate_plan(plan_path, repository, dataset_root):
    plan = load_json(plan_path)
    if (plan.get('schema') != 'sutrack-recovery-search-fixed6-plan/v1' or
            plan.get('complete') is not True or
            plan.get('created_before_off_or_on_inference') is not True or
            Path(plan.get('dataset_root', '')).resolve() != dataset_root or
            plan.get('public_evaluation') is not False or
            plan.get('future_frame_text_used') is not False or
            plan.get('source_identical_off_must_pass_before_on') is not True or
            float(plan.get('off_maximum_bbox_difference', -1.0)) != 0.0 or
            len(plan.get('shards', [])) != 2):
        raise ValueError('recovery-search plan contract failed')
    for record in plan.get('reference_manifests', []):
        path = Path(record.get('path', '')).resolve()
        if sha256_file(path) != record.get('sha256'):
            raise ValueError('reference manifest drift {}'.format(path))
    if len(plan.get('reference_manifests', [])) != 2:
        raise ValueError('plan must bind two reference manifests')
    for path_string, expected_sha in plan.get(
            'implementation_sha256', {}).items():
        path = Path(path_string).resolve()
        if (not path.is_file() or sha256_file(path) != expected_sha or
                repository not in path.parents):
            raise ValueError('planned implementation drift {}'.format(path))
    return plan


def validate_planned_gpus_free(plan, maximum_used_mib=500):
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,memory.used',
        '--format=csv,noheader,nounits'], text=True)
    memory_by_device = {}
    for raw_line in output.splitlines():
        if not raw_line.strip():
            continue
        index, used = [item.strip() for item in raw_line.split(',')]
        memory_by_device[int(index)] = int(used)
    planned_devices = {int(shard['device']) for shard in plan['shards']}
    busy = {
        device: memory_by_device.get(device)
        for device in planned_devices
        if (device not in memory_by_device or
            memory_by_device[device] >= maximum_used_mib)
    }
    if busy:
        raise RuntimeError('planned GPUs are not free: {}'.format(busy))


def terminate_groups(processes, pgids):
    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 10.0
    while time.time() < deadline:
        alive = False
        for pgid in pgids:
            try:
                os.killpg(pgid, 0)
                alive = True
            except ProcessLookupError:
                pass
        if not alive:
            break
        time.sleep(0.2)
    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def complete_phase_exists(output_dirs, enabled):
    if not all((path / 'manifest.json').is_file() for path in output_dirs):
        return False
    expected_role = ('recovery_search_on' if enabled else
                     'source_identical_recovery_search_off')
    for path in output_dirs:
        manifest = load_json(path / 'manifest.json')
        recovery = manifest.get('recovery_search', {})
        if (manifest.get('complete') is not True or
                manifest.get('role') != expected_role or
                recovery.get('enabled') is not enabled or
                manifest.get('public_evaluation') is not False):
            raise ValueError('existing phase manifest failed {}'.format(path))
    return True


def run_phase(
        role, enabled, plan, root, repository, dataset_root, runner_path):
    output_dirs = [
        root / 'recovery_search_fixed6_v1' / role / 'shard0',
        root / 'recovery_search_fixed6_v1' / role / 'shard1',
    ]
    if complete_phase_exists(output_dirs, enabled):
        return output_dirs
    for path in output_dirs:
        if path.exists() and any(path.iterdir()):
            raise FileExistsError('partial recovery phase {}'.format(path))
        path.mkdir(parents=True, exist_ok=True)
    processes = []
    pgids = []
    streams = []
    try:
        for shard in plan['shards']:
            index = int(shard['shard'])
            device = int(shard['device'])
            log_path = root / 'logs' / (
                'recovery_search_{}_shard{}.log'.format(role, index))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stream = log_path.open('x', encoding='utf-8')
            streams.append(stream)
            command = [
                sys.executable, str(runner_path),
                '--dataset-root', str(dataset_root),
                '--config', str(plan['config']),
                '--sequences', ','.join(shard['sequences']),
                '--output-dir', str(output_dirs[index]),
                '--device', '0',
                '--recovery-search-factor', str(plan['factor']),
                '--maximum-consecutive-second-passes',
                str(plan['maximum_consecutive_second_passes']),
                '--cooldown-frames', str(plan['cooldown_frames']),
            ]
            if not enabled:
                command.append('--disable-recovery-search')
            environment = dict(
                os.environ, PYTHONPATH=str(repository),
                CUDA_VISIBLE_DEVICES=str(device))
            process = subprocess.Popen(
                command, cwd=str(repository), stdout=stream,
                stderr=subprocess.STDOUT, env=environment,
                start_new_session=True)
            processes.append(process)
            pgids.append(os.getpgid(process.pid))
        while True:
            codes = [process.poll() for process in processes]
            if any(code is not None and code != 0 for code in codes):
                raise RuntimeError('{} shard exits {}'.format(role, codes))
            if all(code == 0 for code in codes):
                break
            time.sleep(2.0)
    except BaseException:
        terminate_groups(processes, pgids)
        raise
    finally:
        for stream in streams:
            stream.close()
    if not complete_phase_exists(output_dirs, enabled):
        raise ValueError('{} did not publish complete manifests'.format(role))
    return output_dirs


def indexed_predictions(shard_dirs, expected_role):
    indexed = {}
    contract = None
    for shard_dir in shard_dirs:
        manifest = load_json(shard_dir / 'manifest.json')
        if manifest.get('role') != expected_role:
            raise ValueError('unexpected manifest role')
        current_contract = {
            key: manifest[key] for key in (
                'config', 'checkpoint', 'language_manifest')}
        if contract is None:
            contract = current_contract
        elif current_contract != contract:
            raise ValueError('phase source contract mismatch')
        prediction_record = manifest['predictions']
        path = Path(prediction_record['path']).resolve()
        if (sha256_file(path) != prediction_record['sha256'] or
                path.stat().st_size != int(prediction_record['bytes'])):
            raise ValueError('prediction record mismatch')
        for row in load_jsonl(path):
            key = (row['sequence'], int(row['frame_index']))
            bbox = [float(value) for value in row['deployed_bbox']]
            if key in indexed or len(bbox) != 4 or not all(
                    value == value and abs(value) != float('inf')
                    for value in bbox):
                raise ValueError('malformed prediction {}'.format(key))
            indexed[key] = bbox
    return indexed, contract


def validate_off_parity(plan, off_dirs):
    reference_dirs = [
        Path(record['path']).resolve().parent
        for record in plan['reference_manifests']]
    reference, reference_contract = indexed_predictions(
        reference_dirs, expected_role=None)
    off, off_contract = indexed_predictions(
        off_dirs, 'source_identical_recovery_search_off')
    if reference_contract != off_contract or set(reference) != set(off):
        raise ValueError('OFF/reference source or coverage mismatch')
    maximum_difference = 0.0
    for key in reference:
        difference = max(abs(a - b) for a, b in zip(
            reference[key], off[key]))
        maximum_difference = max(maximum_difference, difference)
        if difference != 0.0:
            raise ValueError('OFF perturbed prediction {}'.format(key))
    return maximum_difference, len(reference)


def main():
    args = parse_args()
    if args.poll_seconds <= 0 or args.poll_seconds > 60:
        raise ValueError('--poll-seconds must lie in [1, 60]')
    root = args.root.resolve()
    repository = args.repository.resolve()
    dataset_root = args.dataset_root.resolve()
    status_path = root / 'recovery_search_controller_status.json'
    lock_path = root / 'recovery_search_controller.lock'
    upstream_status_path = root / 'state_gate_controller_status.json'
    plan_path = root / 'source' / 'recovery_search_fixed6_plan.json'
    runner_path = repository / 'tools' / (
        'run_depthtrack_train_recovery_search_trace.py')
    analyzer_path = repository / 'tools' / (
        'analyze_depthtrack_recovery_search_fixed6.py')
    lock_stream = lock_path.open('a+')
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError('another recovery controller holds the lock') from error

    plan = validate_plan(plan_path, repository, dataset_root)
    status = {
        'schema': SCHEMA,
        'state': 'waiting_for_state_gate_terminal',
        'started_at': now(),
        'updated_at': now(),
        'root': str(root),
        'repository': str(repository),
        'dataset_root': str(dataset_root),
        'plan_path': str(plan_path),
        'plan_sha256': sha256_file(plan_path),
        'controller_path': str(Path(__file__).resolve()),
        'controller_sha256': sha256_file(Path(__file__).resolve()),
        'public_evaluation': False,
        'future_frame_text_used': False,
        'gpu_experiment_started': False,
    }
    atomic_json(status_path, status)
    print(json.dumps(status, sort_keys=True), flush=True)

    def interrupted(signum, _frame):
        raise RuntimeError('controller received signal {}'.format(signum))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        while True:
            upstream = load_json(upstream_status_path)
            if upstream.get('state') in UPSTREAM_TERMINAL:
                break
            time.sleep(args.poll_seconds)
        if (upstream.get('public_evaluation') is not False or
                upstream.get('backbone_training_started') is not False):
            raise ValueError('upstream safety contract failed')
        status.update({
            'upstream_state': upstream['state'],
            'upstream_status_sha256': sha256_file(upstream_status_path),
        })
        if upstream['state'] == 'failed':
            status.update({
                'state': 'stopped_upstream_failed',
                'updated_at': now(),
            })
            atomic_json(status_path, status)
            print(json.dumps(status, sort_keys=True), flush=True)
            return
        if (upstream['state'] == 'complete' and
                upstream.get('ready_for_recursive_audit') is True):
            status.update({
                'state': 'skipped_state_gate_ready_for_recursive_audit',
                'updated_at': now(),
                'gpu_experiment_started': False,
            })
            atomic_json(status_path, status)
            print(json.dumps(status, sort_keys=True), flush=True)
            return

        # Revalidate every frozen source immediately before consuming GPUs.
        plan = validate_plan(plan_path, repository, dataset_root)
        validate_planned_gpus_free(plan)
        status.update({
            'state': 'running_source_identical_off',
            'updated_at': now(),
            'gpu_experiment_started': True,
        })
        atomic_json(status_path, status)
        print(json.dumps(status, sort_keys=True), flush=True)
        off_dirs = run_phase(
            'off', False, plan, root, repository, dataset_root, runner_path)
        maximum_difference, frame_count = validate_off_parity(plan, off_dirs)
        status.update({
            'state': 'off_source_identical_passed',
            'updated_at': now(),
            'off_maximum_bbox_difference': maximum_difference,
            'off_frame_count': frame_count,
        })
        atomic_json(status_path, status)
        print(json.dumps(status, sort_keys=True), flush=True)

        plan = validate_plan(plan_path, repository, dataset_root)
        validate_planned_gpus_free(plan)
        status.update({
            'state': 'running_recovery_search_on',
            'updated_at': now(),
        })
        atomic_json(status_path, status)
        print(json.dumps(status, sort_keys=True), flush=True)
        on_dirs = run_phase(
            'on', True, plan, root, repository, dataset_root, runner_path)

        analysis_dir = root / 'recovery_search_fixed6_v1' / 'analysis'
        analysis_log = root / 'logs' / 'recovery_search_fixed6_analysis.log'
        if analysis_dir.exists() and any(analysis_dir.iterdir()):
            raise FileExistsError('recovery analysis output already exists')
        command = [
            sys.executable, str(analyzer_path),
            '--reference-shard-dir', str(Path(
                plan['reference_manifests'][0]['path']).parent),
            '--reference-shard-dir', str(Path(
                plan['reference_manifests'][1]['path']).parent),
            '--off-shard-dir', str(off_dirs[0]),
            '--off-shard-dir', str(off_dirs[1]),
            '--on-shard-dir', str(on_dirs[0]),
            '--on-shard-dir', str(on_dirs[1]),
            '--dataset-root', str(dataset_root),
            '--plan', str(plan_path),
            '--output-dir', str(analysis_dir),
            '--expected-factor', str(plan['factor']),
            '--expected-maximum-consecutive',
            str(plan['maximum_consecutive_second_passes']),
            '--expected-cooldown-frames', str(plan['cooldown_frames']),
        ]
        with analysis_log.open('x', encoding='utf-8') as stream:
            subprocess.run(
                command, cwd=str(repository), stdout=stream,
                stderr=subprocess.STDOUT, check=True,
                env=dict(os.environ, PYTHONPATH=str(repository)))
        result_path = analysis_dir / 'fixed6_recovery_result.json'
        result = load_json(result_path)
        status.update({
            'state': 'complete',
            'updated_at': now(),
            'result_path': str(result_path),
            'result_sha256': sha256_file(result_path),
            'decision': result['decision'],
            'eligible_for_full152_recovery_trace': result[
                'eligible_for_full152_recovery_trace'],
            'public_evaluation': False,
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
            'public_evaluation': False,
        })
        atomic_json(status_path, status)
        print(json.dumps(status, sort_keys=True), flush=True)
        raise


if __name__ == '__main__':
    main()
