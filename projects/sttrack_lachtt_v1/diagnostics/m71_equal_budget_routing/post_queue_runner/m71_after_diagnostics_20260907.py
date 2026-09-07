"""Run the frozen M71 readout once, after the registered diagnostic queue succeeds."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE = Path('/root/autodl-tmp')
PARENT = BASE / 'sttrack_post_m67_diagnostic_queue_20260907'
M70 = BASE / 'sttrack_m70_recovery_window_inventory_20260907/candidate_capacity'
M71 = BASE / 'sttrack_m71_equal_budget_routing_20260907'
ROOT = M71 / 'post_queue_runner'
DEPENDENCIES = {
    BASE / 'post_m67_diagnostic_queue_20260907.py': '4a4bc5debaf8581ede1cec7bd1c2020b3361ee5963509b7c7b18679fab3688cb',
    PARENT / 'spec.json': '5f34ea8f38dfe6f7047d4cf65fba388da076c4c178d0b7e3b3afa3f4c078eddf',
    BASE / 'm71_equal_budget_routing_20260907.py': 'd98dde637536f56468564eaa03f39cc0bb6b6c38bbab61ecff8cf06465b014f3',
    M71 / 'spec.json': '1762f6ea077aa37a035a2ed9a1cc65afa99e6af551ac4bb5bde9f28884cda7a2',
    M71 / 'geometry_accounting.json': 'ae2767a5c4890e0b294aeaca27827a0837a6a60fd85178a034fd62d188438b61',
    BASE / 'm70_candidate_capacity_20260907.py': 'e0f02783e2410f745a40465799407a4cef218993e1d8eee6fc745d0029f53977',
    M70 / 'spec.json': '796a2756db4c2be0b5282d567294a76d494ad031cc4e186ba96deb3228c00fa5',
}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
def now(): return datetime.now(timezone.utc).isoformat()


def identity(pid):
    folder = Path('/proc') / str(pid)
    fields = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
    assert fields[0] != 'Z'
    return dict(pid=pid, start_ticks=fields[19], cwd=str((folder / 'cwd').resolve()),
        argv=[s.decode() for s in (folder / 'cmdline').read_bytes().split(b'\0') if s])


def dependencies():
    for path, expected in DEPENDENCIES.items(): assert sha(path) == expected, str(path)


def prepare():
    dependencies()
    parent = read(PARENT / 'running_identity.json')['identity']
    assert parent == identity(471187)
    assert parent['start_ticks'] == '4578045533'
    assert parent['cwd'] == str(PARENT)
    assert parent['argv'] == [sys.executable, '-u', str(BASE / 'post_m67_diagnostic_queue_20260907.py'), 'run']
    assert not (PARENT / 'controller.exit').exists() and not (M71 / 'result.json').exists()
    ROOT.mkdir()
    launch = ROOT / 'launch.sh'
    launch.write_text('''#!/bin/bash
set -u
cd /root/autodl-tmp/sttrack_m71_equal_budget_routing_20260907/post_queue_runner || exit 1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/root/autodl-tmp/envs/sttrack/bin/python -u /root/autodl-tmp/m71_after_diagnostics_20260907.py run > controller.log 2>&1
status=$?
printf '%s\\n' "$status" > controller.exit
exit "$status"
''')
    subprocess.run(['bash', '-n', str(launch)], check=True)
    capacity = read(M70 / 'spec.json')
    # Two conditions: 256 float64 xywh boxes, two float32 responses, two int16 top10 index lists.
    payload = capacity['shadow_calls_per_text'] * 2 * (256 * 4 * 8 + 2 * 256 * 4 + 2 * 10 * 2)
    spec = dict(status='prepared_after_live_registered_queue', created_utc=now(), source_sha256=sha(__file__),
        launch_sha256=sha(launch), parent_identity=parent, parent_spec_sha256=sha(PARENT / 'spec.json'), poll_seconds=240,
        dependencies={str(p): h for p, h in DEPENDENCIES.items()},
        analysis_argv=[sys.executable, str(BASE / 'm71_equal_budget_routing_20260907.py'), 'analyze'],
        parent_required_status='completed_authorized_post_M67_diagnostic_sequence',
        expected_result_status='completed_equal_window_budget_routing_attribution',
        M70_formal_array_uncompressed_payload_bytes=payload,
        payload_scope='Logical array payload only; excludes NPZ headers, JSON, smoke output and other queued stages. Compression is not assumed.',
        new_model_calls=0, new_optimizer_steps=0, parent_queue_modified=False, automatic_promotion=False,
        automatic_public_evaluation=False, retries=False, goal_completed=False, independent_model_review_pass=False)
    write(ROOT / 'spec.json', spec)
    print(json.dumps(dict(spec_sha256=sha(ROOT / 'spec.json'), source_sha256=sha(__file__), parent_identity=parent,
        logical_M70_array_bytes=payload), indent=2))


def checked():
    dependencies()
    spec = read(ROOT / 'spec.json')
    assert spec['source_sha256'] == sha(__file__) and spec['launch_sha256'] == sha(ROOT / 'launch.sh')
    return spec


def log(event, **values):
    row = dict(time=now(), event=event, **values)
    with (ROOT / 'events.jsonl').open('a') as stream: stream.write(json.dumps(row, allow_nan=False) + '\n')
    write(ROOT / 'latest.json', row)
    print(json.dumps(row), flush=True)


def run():
    spec = checked()
    assert not (ROOT / 'running_identity.json').exists() and not (M71 / 'result.json').exists()
    write(ROOT / 'running_identity.json', dict(observed_utc=now(), identity=identity(os.getpid()),
        source_sha256=sha(__file__), spec_sha256=sha(ROOT / 'spec.json')))
    while not (PARENT / 'controller.exit').exists():
        parent = identity(spec['parent_identity']['pid'])
        assert parent == spec['parent_identity']
        log('waiting_for_registered_diagnostic_queue', parent_identity=parent, poll_seconds=spec['poll_seconds'])
        time.sleep(spec['poll_seconds'])
    assert (PARENT / 'controller.exit').read_text().strip() == '0'
    parent_result = read(PARENT / 'result.json')
    assert parent_result['status'] == spec['parent_required_status']
    assert parent_result['spec_sha256'] == spec['parent_spec_sha256']
    assert parent_result['outcomes']['M70_capacity']['result_sha256'] == sha(M70 / 'result.json')
    checked()
    assert not (M71 / 'result.json').exists()
    log('sealed_parent_completed_starting_M71', parent_result_sha256=sha(PARENT / 'result.json'))
    with (ROOT / 'analysis.log').open('w') as stream:
        completed = subprocess.run(spec['analysis_argv'], cwd=M71, env=os.environ.copy(), stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT)
    (ROOT / 'analysis.exit').write_text(str(completed.returncode) + '\n')
    assert completed.returncode == 0
    result = read(M71 / 'result.json')
    assert result['status'] == spec['expected_result_status'] and result['spec_sha256'] == sha(M71 / 'spec.json')
    assert result['new_tracking_calls'] == result['new_optimizer_steps'] == 0 and not result['public_evaluation_allowed']
    write(ROOT / 'result.json', dict(status='completed_frozen_M71_after_diagnostic_queue', observed_utc=now(),
        source_sha256=sha(__file__), spec_sha256=sha(ROOT / 'spec.json'),
        parent_result_sha256=sha(PARENT / 'result.json'), M71_result_sha256=sha(M71 / 'result.json'),
        new_model_calls=0, new_optimizer_steps=0, public_evaluation_allowed=False, goal_completed=False))
    log('M71_completed', result_sha256=sha(M71 / 'result.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'check', 'run'])
    args = parser.parse_args()
    {'prepare': prepare, 'check': checked, 'run': run}[args.action]()
