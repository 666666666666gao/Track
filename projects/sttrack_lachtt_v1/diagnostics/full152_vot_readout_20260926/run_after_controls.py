"""Wait for the existing controls, then preflight and run the two replay shards."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path('/root/autodl-tmp/sttrack_full152_vot_readout_20260926')
CONTROLS = Path('/root/autodl-tmp/sttrack_full152_content_20260926')
MODEL_PYTHON = '/root/autodl-tmp/envs/sttrack/bin/python'
METRIC_PYTHON = '/root/miniconda3/envs/mplt/bin/python'


def run_step(name, command, environment):
    with (ROOT / 'logs' / (name + '.log')).open('w') as log:
        result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    (ROOT / (name + '.exit')).write_text(str(result.returncode) + '\n')
    assert result.returncode == 0, name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--not-before', required=True)
    parser.add_argument('--controls-pid', type=int, required=True)
    args = parser.parse_args()
    ready_time = datetime.fromisoformat(args.not_before).timestamp()
    print(json.dumps(dict(state='waiting_until', not_before=args.not_before, controls_pid=args.controls_pid)), flush=True)
    time.sleep(max(0, ready_time - time.time()))
    while not (CONTROLS / 'controls.exit').exists():
        command = Path(f'/proc/{args.controls_pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        assert 'rgbd_full152_content_20260926' in command, 'Existing control process is missing or changed'
        print(json.dumps(dict(state='controls_running', observed_utc=datetime.now(timezone.utc).isoformat())), flush=True)
        time.sleep(300)
    assert (CONTROLS / 'controls.exit').read_text().strip() == '0'
    for model in ('M82', 'M67'):
        for variant in ('empty', 'swapped'):
            for dataset in ('depthtrack', 'cdtb'):
                metric = json.loads((CONTROLS / model / dataset / variant / 'predictions/metrics.json').read_text())
                assert metric['status'] == 'complete'
                assert metric['metrics']['sequences'] == (50 if dataset == 'depthtrack' else 80)
    memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()
    assert len(memory) == 2 and all(int(value) < 500 for value in memory), 'Both GPUs must be free before replay'
    base_env = os.environ.copy()
    base_env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    started = time.time()
    run_step('preflight', [MODEL_PYTHON, '-u', str(ROOT / 'readout.py'), '--shard', '0', '--preflight'],
             dict(base_env, CUDA_VISIBLE_DEVICES='0'))
    preflight = json.loads((ROOT / 'preflight/0/receipt.json').read_text())
    assert preflight['status'] == 'complete' and len(preflight['cases']) == 1
    seconds = time.time() - started
    spec = json.loads((ROOT / 'spec.json').read_text())
    estimate = max(spec['shard_replay_calls']) * seconds / preflight['cases'][0]['replay_calls']
    print(json.dumps(dict(state='preflight_passed', preflight_seconds=seconds, estimated_two_shard_seconds=estimate)), flush=True)
    jobs, handles = [], []
    for shard in (0, 1):
        log = (ROOT / 'logs' / f'shard{shard}.log').open('w')
        handles.append(log)
        process = subprocess.Popen([MODEL_PYTHON, '-u', str(ROOT / 'readout.py'), '--shard', str(shard)],
                                   env=dict(base_env, CUDA_VISIBLE_DEVICES=str(shard)), stdout=log, stderr=subprocess.STDOUT)
        jobs.append(process)
    (ROOT / 'replay_launch.json').write_text(json.dumps(dict(
        started_utc=datetime.now(timezone.utc).isoformat(), pids=[p.pid for p in jobs],
        gpu_indices=[0, 1], spec_sha256=hashlib.sha256((ROOT / 'spec.json').read_bytes()).hexdigest(),
        estimated_seconds=estimate), indent=2) + '\n')
    statuses = [job.wait() for job in jobs]
    for shard, (status, handle) in enumerate(zip(statuses, handles)):
        handle.close()
        (ROOT / f'shard{shard}.exit').write_text(str(status) + '\n')
    assert statuses == [0, 0], statuses
    run_step('analysis', [METRIC_PYTHON, '-u', str(ROOT / 'analyze.py')], base_env)
    (ROOT / 'controller.exit').write_text('0\n')
    print(json.dumps(dict(state='complete', elapsed_seconds=time.time() - started)), flush=True)


if __name__ == '__main__':
    main()
