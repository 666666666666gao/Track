"""Wait near GPU release, then run fixed M56 development arms in dedicated screens."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--gpu', type=int, choices=[0, 1], required=True)
parser.add_argument('--not-before', required=True, help='Timezone-aware ISO datetime')
args = parser.parse_args()
root = args.root
python = '/root/autodl-tmp/envs/sttrack/bin/python'
tasks = ['attributes', 'empty'] if args.gpu == 0 else ['pooled']
not_before = datetime.datetime.fromisoformat(args.not_before)
assert not_before.tzinfo is not None
plan = dict(gpu=args.gpu, tasks=tasks, not_before=args.not_before, poll_seconds=240,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    recursive_spec_sha256=hashlib.sha256((root / 'recursive_spec.json').read_bytes()).hexdigest())
(root / ('queue_gpu%d_plan.json' % args.gpu)).write_text(json.dumps(plan, indent=2) + '\n')
print(json.dumps(dict(status='waiting_until_expected_gpu_release', **plan)), flush=True)
remaining = not_before.timestamp() - time.time()
if remaining > 0:
    time.sleep(remaining)
assert (root / 'train.exit').read_text().strip() == '0'
assert json.loads((root / 'training_result.json').read_text())['status'] == 'complete_train'
while True:
    used = int(subprocess.check_output(['nvidia-smi', '--id=' + str(args.gpu),
        '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
    if used < 500:
        break
    print(json.dumps(dict(status='waiting_for_free_gpu', gpu=args.gpu, used_mib=used, utc=datetime.datetime.now(datetime.timezone.utc).isoformat())), flush=True)
    time.sleep(240)
environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
for arm in tasks:
    assert not (root / (arm + '_recursive.exit')).exists()
    with (root / (arm + '_recursive.log')).open('w') as stream:
        status = subprocess.run([python, str(root / 'run_recursive.py'), '--root', str(root), '--arm', arm],
            stdout=stream, stderr=subprocess.STDOUT, env=environment).returncode
    (root / (arm + '_recursive.exit')).write_text(str(status) + '\n')
    print(json.dumps(dict(variant=arm, exit_code=status, utc=datetime.datetime.now(datetime.timezone.utc).isoformat())), flush=True)
    if status != 0:
        raise SystemExit(status)
if args.gpu == 0:
    while not (root / 'pooled_recursive.exit').exists():
        controller = root / 'queue_gpu1_controller.exit'
        assert not controller.exists(), 'GPU1 controller exited before producing its pooled result'
        time.sleep(240)
    assert (root / 'pooled_recursive.exit').read_text().strip() == '0'
    with (root / 'recursive_analysis.log').open('w') as stream:
        status = subprocess.run([python, str(root / 'run_recursive.py'), '--root', str(root), '--analyze'],
            stdout=stream, stderr=subprocess.STDOUT).returncode
    (root / 'recursive_analysis.exit').write_text(str(status) + '\n')
    if status != 0:
        raise SystemExit(status)
(root / ('queue_gpu%d.exit' % args.gpu)).write_text('0\n')
