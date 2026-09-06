"""Two serial queues of independent complete M57 development trajectories."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from run_recursive import bound
from train import sha, write


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, choices=[0, 1], required=True)
gpu = parser.parse_args().gpu
root = Path('/root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906')
spec, frozen, training, cases = bound(root)
arms = {0: ['initial_text', 'initial_visual'], 1: ['candidate_text', 'candidate_visual']}[gpu]
environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1')
for arm in arms:
    assert not (root / (arm + '_recursive_launch.json')).exists()
    memory = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used',
        '--format=csv,noheader,nounits'], text=True).strip())
    assert memory < 500, (gpu, memory)
    command = ['/root/autodl-tmp/envs/sttrack/bin/python', str(root / 'run_recursive.py'),
               '--root', str(root), '--arm', arm]
    with (root / (arm + '_recursive.log')).open('w') as stream:
        process = subprocess.Popen(command, cwd=spec['repository'], env=environment,
                                   stdout=stream, stderr=subprocess.STDOUT)
        (root / (arm + '_recursive.pid')).write_text(str(process.pid) + '\n')
        record = dict(started_utc=datetime.now(timezone.utc).isoformat(), variant=arm,
            physical_gpu=gpu, memory_mib_before=memory, pid=process.pid, command=command,
            runner_sha256=frozen['runner_sha256'], queue_sha256=sha(__file__),
            recursive_spec_sha256=sha(root / 'recursive_spec.json'),
            checkpoint_sha256=training['variants'][arm]['checkpoint_sha256'])
        write(root / (arm + '_recursive_launch.json'), record)
        print(json.dumps(record), flush=True)
        code = process.wait()
    (root / (arm + '_recursive.exit')).write_text(str(code) + '\n')
    assert code == 0, (arm, code)
    receipt = json.loads((root / (arm + '_recursive_receipt.json')).read_text())
    assert receipt['status'] == 'complete' and receipt['frames'] == 33130
print(json.dumps(dict(status='complete_queue', physical_gpu=gpu, arms=arms)), flush=True)
