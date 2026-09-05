"""Launch one frozen full recursive comparison after its state contract passes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    root = p.parse_args().root
    plan = json.loads((root / 'spec.json').read_text())
    spec = json.loads((Path(plan['source_root']) / 'spec.json').read_text())
    repo = Path(spec['repository'])
    receipt = json.loads((root / 'runtime_contract.json').read_text())
    assert receipt['status'] == 'PASS' and receipt['spec_sha256'] == sha(root / 'spec.json')
    assert receipt['checkpoint_sha256'] == plan['checkpoint_sha256']
    assert not (root / 'controller.sh').exists()
    for name, digest in plan['source_sha256'].items():
        assert sha(repo / name) == digest
    memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    assert all(int(x) < 500 for x in memory.splitlines()), memory
    free = shutil.disk_usage(root).free
    assert free > 500_000_000
    py = '/root/autodl-tmp/envs/sttrack/bin/python'
    for shard in [0, 1]:
        (root / f'recursive_s{shard}.sh').write_text(f'''#!/bin/bash
cd {repo}
CUDA_VISIBLE_DEVICES={shard} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 {py} {repo}/tools/run_sttrack_m48.py --root {root} --shard {shard} > {root}/recursive_s{shard}.log 2>&1 &
job=$!
printf '%s\\n' "$job" > {root}/recursive_s{shard}.pid
wait "$job"
status=$?
printf '%s\\n' "$status" > {root}/recursive_s{shard}.exit
exit "$status"
''')
    (root / 'pipeline.sh').write_text(f'''#!/bin/bash
cd {repo}
bash {root}/recursive_s0.sh &
first=$!
bash {root}/recursive_s1.sh &
second=$!
wait "$first"
first_status=$?
wait "$second"
second_status=$?
if [ "$first_status" -ne 0 ] || [ "$second_status" -ne 0 ]; then exit 1; fi
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' {py} {repo}/tools/run_sttrack_m48.py --root {root} --analyze > {root}/analysis.log 2>&1
''')
    (root / 'controller.sh').write_text(f'''#!/bin/bash
bash {root}/pipeline.sh &
job=$!
printf '%s\\n' "$job" > {root}/controller.pid
wait "$job"
status=$?
printf '%s\\n' "$status" > {root}/controller.exit
exit "$status"
''')
    screen = 'sttrack_m48_native_continuity_20260905'
    subprocess.run(['screen', '-dmS', screen, 'bash', str(root / 'controller.sh')], check=True)
    time.sleep(2)
    pid = int((root / 'controller.pid').read_text())
    os.kill(pid, 0)
    launch = dict(status='started', pid=pid, screen=screen, started_unix=time.time(),
        spec_sha256=sha(root / 'spec.json'), contract_sha256=sha(root / 'runtime_contract.json'),
        checkpoint_sha256=plan['checkpoint_sha256'], shard_frames=plan['shard_frames'],
        initial_free_bytes=free, new_training=False, additional_optimizer_steps=0)
    (root / 'launch.json').write_text(json.dumps(launch, indent=2) + '\n')
    print(json.dumps(launch, indent=2))


if __name__ == '__main__':
    main()
