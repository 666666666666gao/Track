"""Run paired M55 final-weight recursion after the scheduled M56 GPU work."""
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
parser.add_argument('--predecessor', type=Path, required=True)
parser.add_argument('--gpu', type=int, choices=[0, 1], required=True)
parser.add_argument('--not-before', required=True)
args = parser.parse_args()
root = args.root
python = '/root/autodl-tmp/envs/sttrack/bin/python'
arm = ['control', 'clone'][args.gpu]
not_before = datetime.datetime.fromisoformat(args.not_before)
assert not_before.tzinfo is not None
dependency = args.predecessor / ('queue_gpu%d_controller.exit' % args.gpu)
plan = dict(gpu=args.gpu, arm=arm, not_before=args.not_before, poll_seconds=240,
    predecessor_controller_exit=str(dependency), predecessor_success_required=False,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    recursive_spec_sha256=hashlib.sha256((root / 'recursive_spec.json').read_bytes()).hexdigest(),
    weight_selection='Both completed final epoch 15 weights, verified by the frozen runner',
    public_evaluation=False)
(root / ('recursive_queue_gpu%d_plan.json' % args.gpu)).write_text(json.dumps(plan, indent=2) + '\n')
print(json.dumps(dict(status='waiting_near_predecessor_completion', **plan)), flush=True)
remaining = not_before.timestamp() - time.time()
if remaining > 0:
    time.sleep(remaining)
while not dependency.exists():
    print(json.dumps(dict(status='waiting_for_predecessor', gpu=args.gpu,
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat())), flush=True)
    time.sleep(240)
print(json.dumps(dict(status='predecessor_terminal', gpu=args.gpu,
    exit_code=int(dependency.read_text().strip()))), flush=True)
while True:
    used = int(subprocess.check_output(['nvidia-smi', '--id=' + str(args.gpu),
        '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
    if used < 500:
        break
    print(json.dumps(dict(status='waiting_for_free_gpu', gpu=args.gpu, used_mib=used)), flush=True)
    time.sleep(240)
assert hashlib.sha256((root / 'recursive_spec.json').read_bytes()).hexdigest() == plan['recursive_spec_sha256']
assert not (root / (arm + '_recursive.exit')).exists()
environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
with (root / (arm + '_recursive.log')).open('w') as stream:
    status = subprocess.run([python, str(root / 'run_recursive.py'), '--root', str(root), '--arm', arm],
        stdout=stream, stderr=subprocess.STDOUT, env=environment).returncode
(root / (arm + '_recursive.exit')).write_text(str(status) + '\n')
print(json.dumps(dict(arm=arm, exit_code=status,
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat())), flush=True)
if status != 0:
    raise SystemExit(status)
if args.gpu == 0:
    while not (root / 'clone_recursive.exit').exists():
        assert not (root / 'recursive_queue_gpu1_controller.exit').exists(), 'Clone queue ended before its result'
        time.sleep(240)
    assert (root / 'clone_recursive.exit').read_text().strip() == '0'
    with (root / 'recursive_analysis.log').open('w') as stream:
        status = subprocess.run([python, str(root / 'run_recursive.py'), '--root', str(root), '--analyze'],
            stdout=stream, stderr=subprocess.STDOUT).returncode
    (root / 'recursive_analysis.exit').write_text(str(status) + '\n')
    if status != 0:
        raise SystemExit(status)
(root / ('recursive_queue_gpu%d.exit' % args.gpu)).write_text('0\n')
