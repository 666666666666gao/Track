#!/usr/bin/env python3
"""Run, validate and merge disjoint VOT multi-start anchor shards."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


SUFFIXES = ('.bin', '_confidence.value', '_time.value')


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument(
        '--python', default='/root/miniconda3/envs/mplt/bin/python')
    parser.add_argument('--poll-seconds', type=float, default=30.0)
    return parser.parse_args()


def trajectory_files(shard_root, tracker, trajectory):
    sequence = trajectory.rsplit('_', 1)[0]
    result_root = shard_root / 'results' / tracker / 'baseline' / sequence
    return [result_root / (trajectory + suffix) for suffix in SUFFIXES]


def completed_count(shard, tracker):
    root = Path(shard['root'])
    return sum(
        all(path.is_file() and path.stat().st_size > 0
            for path in trajectory_files(root, tracker, trajectory))
        for trajectory in shard['expected_trajectories'])


def terminate_groups(processes):
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 15.0
    while time.time() < deadline:
        if all(process.poll() is not None for process in processes):
            break
        time.sleep(0.25)
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def main():
    args = parse_args()
    root = args.root.resolve()
    manifest_path = root / 'shard_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    tracker = manifest['tracker']
    logs_root = root / 'controller_logs'
    logs_root.mkdir(exist_ok=True)
    processes = []
    logs = []
    try:
        for shard in manifest['shards']:
            shard_root = Path(shard['root'])
            log_path = logs_root / 'shard-{:02d}.log'.format(shard['index'])
            log = open(log_path, 'ab', buffering=0)
            command = [
                args.python, '-m', 'vot', 'evaluate', '--workspace', '.', tracker]
            environment = dict(os.environ)
            environment['PYTHONPATH'] = '/home/SUTrack_RGBD_L'
            process = subprocess.Popen(
                command, cwd=str(shard_root), env=environment,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            logs.append(log)

        last_progress = None
        while True:
            progress = [
                completed_count(shard, tracker) for shard in manifest['shards']]
            totals = [shard['anchor_count'] for shard in manifest['shards']]
            if progress != last_progress:
                print('PROGRESS {}/{} {}'.format(
                    sum(progress), sum(totals),
                    ' '.join('{}/{}'.format(done, total)
                             for done, total in zip(progress, totals))),
                    flush=True)
                last_progress = progress
            failed = []
            all_exited = True
            for process, shard, done in zip(processes, manifest['shards'], progress):
                code = process.poll()
                if code is None:
                    all_exited = False
                elif code != 0 or done != shard['anchor_count']:
                    failed.append((shard['index'], code, done, shard['anchor_count']))
            if failed:
                raise RuntimeError('Shard failure: {}'.format(failed))
            if all_exited:
                break
            time.sleep(args.poll_seconds)
    except BaseException:
        terminate_groups(processes)
        raise
    finally:
        for log in logs:
            log.close()

    master = root / 'master'
    if master.exists() and any(master.iterdir()):
        raise FileExistsError('Master workspace is not empty: {}'.format(master))
    sequences_root = master / 'sequences'
    sequences_root.mkdir(parents=True)
    atomic_text(
        master / 'config.yaml',
        'registry:\n- ./trackers.ini\nsequences: ./sequences\nstack: vot2022/rgbd\n')
    shutil.copy2(Path(manifest['shards'][0]['root']) / 'trackers.ini', master / 'trackers.ini')
    atomic_text(
        sequences_root / 'list.txt',
        ''.join(sequence + '\n' for sequence in manifest['sequences']))
    source_root = Path(manifest['source_sequences_root'])
    for sequence in manifest['sequences']:
        (sequences_root / sequence).symlink_to(
            (source_root / sequence).resolve(), target_is_directory=True)

    destination = master / 'results' / tracker / 'baseline'
    copied = {}
    for shard in manifest['shards']:
        shard_root = Path(shard['root'])
        for trajectory in shard['expected_trajectories']:
            sequence = trajectory.rsplit('_', 1)[0]
            sequence_destination = destination / sequence
            sequence_destination.mkdir(parents=True, exist_ok=True)
            for source_path in trajectory_files(shard_root, tracker, trajectory):
                target_path = sequence_destination / source_path.name
                if target_path.exists() or target_path.name in copied:
                    raise RuntimeError('Duplicate merged result {}'.format(target_path))
                shutil.copy2(source_path, target_path)
                copied[str(target_path.relative_to(master))] = sha256_file(target_path)

    expected_file_count = manifest['total_anchor_count'] * len(SUFFIXES)
    if len(copied) != expected_file_count:
        raise RuntimeError(
            'Merged result count mismatch: {} != {}'.format(
                len(copied), expected_file_count))
    result = {
        'schema': 'sutrack_vot_failure_family_anchor_merge_v1',
        'status': 'complete',
        'tracker': tracker,
        'master_workspace': str(master),
        'source_manifest': str(manifest_path),
        'source_manifest_sha256': sha256_file(manifest_path),
        'anchor_count': manifest['total_anchor_count'],
        'result_file_count': len(copied),
        'result_sha256': dict(sorted(copied.items())),
    }
    atomic_text(
        root / 'merge_result.json',
        json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({
        'status': 'complete',
        'master_workspace': str(master),
        'anchor_count': manifest['total_anchor_count'],
        'result_file_count': len(copied),
        'merge_result_sha256': sha256_file(root / 'merge_result.json'),
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
