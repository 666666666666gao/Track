"""Run one fixed control/candidate pair on the two free GPUs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path('/root/autodl-tmp/sttrack_native_candidate_preservation_20260926')
PYTHON = '/root/autodl-tmp/envs/sttrack/bin/python'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('preflight', 'training'), required=True)
    phase = parser.parse_args().phase
    memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()
    assert len(memory) == 2 and all(int(v) < 500 for v in memory)
    assert not (ROOT / phase).exists()
    logs = ROOT / 'logs'
    logs.mkdir(exist_ok=True)
    jobs, handles = [], []
    for gpu, name in enumerate(('control', 'candidate')):
        command = [PYTHON, '-u', str(ROOT / 'train_candidate.py'), '--arm', 'category', '--weight', str(gpu)]
        if phase == 'preflight':
            command.append('--preflight')
        handle = (logs / f'{phase}_{name}.log').open('w')
        handles.append(handle)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        jobs.append(subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT))
    launch = dict(phase=phase, started_utc=datetime.now(timezone.utc).isoformat(), pids=[p.pid for p in jobs],
                  gpu_indices=[0, 1], controller_pid=os.getpid(),
                  experiment_spec_sha256=hashlib.sha256((ROOT / 'experiment_spec.json').read_bytes()).hexdigest())
    (ROOT / f'{phase}_launch.json').write_text(json.dumps(launch, indent=2) + '\n')
    print(json.dumps(launch), flush=True)
    statuses = [p.wait() for p in jobs]
    for name, status, handle in zip(('control', 'candidate'), statuses, handles):
        handle.close()
        (ROOT / f'{phase}_{name}.exit').write_text(str(status) + '\n')
    assert statuses == [0, 0], statuses
    (ROOT / f'{phase}.exit').write_text('0\n')
    print(json.dumps(dict(phase=phase, status='complete', finished_utc=datetime.now(timezone.utc).isoformat())), flush=True)


if __name__ == '__main__':
    main()
