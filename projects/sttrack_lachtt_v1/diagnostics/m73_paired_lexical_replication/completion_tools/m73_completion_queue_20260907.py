"""Wait for the registered M73 training queue, audit, then conditionally run content."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

B = Path('/root/autodl-tmp')
ROOT = B / 'sttrack_m73_paired_lexical_replication_20260907'
OUT = ROOT / 'completion_queue'
CONTENT = ROOT / 'content_counterfactuals'
SCRIPT = B / 'm73_content_counterfactuals_20260907.py'
SCRIPT_SHA = 'c2b3422b1c5ad19cbb5231d415244431033f238f2da747b11b053eefcd7c686d'
CONTENT_SPEC_SHA = 'b25a073ad07e2f9e5740a2732f06b8ee4d23472072ccad91bd3a5b37385ee7d5'
AUDITOR = B / 'audit_m73_completed_20260907.py'
AUDITOR_SHA = '4fcaf10e73225ea7bf9f57dbd031c1a6d9f58ebed29fa04341277d0d69fb3ef8'
PYTHON = B / 'envs/sttrack/bin/python'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p, v): Path(p).write_text(json.dumps(v, indent=2, allow_nan=False) + '\n')
def now(): return datetime.now(timezone.utc).isoformat()


def identity(pid):
    p = Path('/proc') / str(pid); fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    assert fields[0] != 'Z'
    return dict(pid=pid, start_ticks=fields[19], cwd=str((p / 'cwd').resolve()),
        argv=[s.decode() for s in (p / 'cmdline').read_bytes().split(b'\0') if s])


def content_checked():
    assert sha(SCRIPT) == SCRIPT_SHA and sha(CONTENT / 'spec.json') == CONTENT_SPEC_SHA
    assert sha(AUDITOR) == AUDITOR_SHA
    loader = importlib.util.spec_from_file_location('m73_content_binding', str(SCRIPT))
    content = importlib.util.module_from_spec(loader); loader.loader.exec_module(content)
    content.checked()


def prepare():
    content_checked(); assert not (ROOT / 'controller.exit').exists()
    parent = identity(482884)
    assert parent['start_ticks'] == '4579784392' and parent['cwd'] == str(ROOT)
    assert parent['argv'] == ['bash', str(ROOT / 'run_m73.sh')]
    OUT.mkdir()
    launch = OUT / 'run_after_training.sh'
    launch.write_text('''#!/bin/bash
set -u
cd /root/autodl-tmp/sttrack_m73_paired_lexical_replication_20260907/completion_queue || exit 1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/root/autodl-tmp/envs/sttrack/bin/python -u /root/autodl-tmp/m73_completion_queue_20260907.py run > controller.log 2>&1
status=$?; printf '%s\\n' "$status" > controller.exit
exit "$status"
''')
    subprocess.run(['bash', '-n', str(launch)], check=True)
    spec = dict(status='frozen_M73_completion_follower_before_parent_results', observed_utc=now(),
        source_sha256=sha(__file__), launch_sha256=sha(launch), content_source_sha256=SCRIPT_SHA,
        content_spec_sha256=CONTENT_SPEC_SHA, auditor_sha256=AUDITOR_SHA,
        parent_spec_sha256=sha(ROOT / 'spec.json'), parent_frozen_sha256=sha(ROOT / 'frozen.json'),
        parent_queue_sha256=sha(ROOT / 'run_m73.sh'), parent_identity=parent, poll_seconds=240,
        behavior='Wait for this exact full two-seed queue; require exit0; audit all four finals and scalar metrics; run fixed-seed content only if both gates pass, otherwise write explicit skipped record.',
        GPU_calls_while_waiting=0, parent_queue_modified=False, public_evaluation_allowed=False,
        independent_model_review_pass=False)
    write(OUT / 'spec.json', spec); print(json.dumps(spec))


def checked():
    content_checked(); spec = read(OUT / 'spec.json')
    assert sha(__file__) == spec['source_sha256'] and sha(OUT / 'run_after_training.sh') == spec['launch_sha256']
    assert sha(ROOT / 'spec.json') == spec['parent_spec_sha256'] and sha(ROOT / 'frozen.json') == spec['parent_frozen_sha256']
    assert sha(ROOT / 'run_m73.sh') == spec['parent_queue_sha256']
    return spec


def run():
    spec = checked(); assert not (OUT / 'running_identity.json').exists()
    write(OUT / 'running_identity.json', dict(observed_utc=now(), identity=identity(os.getpid()), spec_sha256=sha(OUT / 'spec.json')))
    while not (ROOT / 'controller.exit').exists():
        # The terminal exit file is written before the registered parent exits.
        proc = Path('/proc') / str(spec['parent_identity']['pid'])
        if not proc.exists():
            assert (ROOT / 'controller.exit').exists(), 'Registered M73 parent vanished without an exit receipt'
            break
        current = identity(spec['parent_identity']['pid']); assert current == spec['parent_identity']
        event = dict(observed_utc=now(), status='waiting_for_registered_full_M73_training_queue',
            parent_identity=current, poll_seconds=240, new_model_calls=0)
        write(OUT / 'latest.json', event); print(json.dumps(event), flush=True)
        time.sleep(240)
    assert (ROOT / 'controller.exit').read_text().strip() == '0'
    parent = read(ROOT / 'result.json')
    assert parent['status'] == 'completed_two_seed_paired_lexical_replication' and parent['spec_sha256'] == spec['parent_spec_sha256']
    checked()
    write(OUT / 'latest.json', dict(observed_utc=now(), status='auditing_completed_M73', parent_result_sha256=sha(ROOT / 'result.json')))
    with (OUT / 'audit.log').open('w') as f:
        process = subprocess.run([str(PYTHON), '-u', str(AUDITOR), 'completed'], stdout=f, stderr=subprocess.STDOUT)
    (OUT / 'audit.exit').write_text(str(process.returncode) + '\n'); assert process.returncode == 0
    audit_path = ROOT / 'completion_tools/completed.json'; audit = read(audit_path)
    assert audit['auditor_sha256'] == AUDITOR_SHA and audit['parent_result_sha256'] == sha(ROOT / 'result.json')
    if not audit['both_seed_development_gates_pass']:
        result = dict(status='completed_M73_audit_content_skipped_by_frozen_gates', observed_utc=now(),
            spec_sha256=sha(OUT / 'spec.json'), audit_sha256=sha(audit_path), content_started=False,
            reason='At least one prospective seed failed one or more inherited development requirements.',
            gates={s: r['gates'] for s, r in audit['seeds'].items()}, public_evaluation_allowed=False)
        assert not (CONTENT / 'activation.json').exists()
    else:
        write(OUT / 'latest.json', dict(observed_utc=now(), status='running_conditional_seed2027_content', audit_sha256=sha(audit_path)))
        with (OUT / 'content.log').open('w') as f:
            process = subprocess.run(['bash', str(CONTENT / 'run_controls.sh')], cwd=str(CONTENT), stdout=f, stderr=subprocess.STDOUT)
        (OUT / 'content.exit').write_text(str(process.returncode) + '\n'); assert process.returncode == 0
        content = read(CONTENT / 'result.json')
        assert content['status'] == 'completed_M73_predetermined_final_fixed_head_content'
        assert content['M73_audit_sha256'] == sha(audit_path) and content['candidate_seed'] == 2027
        result = dict(status='completed_M73_audit_and_conditional_content', observed_utc=now(),
            spec_sha256=sha(OUT / 'spec.json'), audit_sha256=sha(audit_path), content_started=True,
            content_result_sha256=sha(CONTENT / 'result.json'), lexical_criteria_pass=content['lexical_criteria_pass'],
            candidate_entry_parity_preparation_allowed=content['candidate_entry_parity_preparation_allowed'], public_evaluation_allowed=False)
    write(OUT / 'result.json', result); write(OUT / 'latest.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['prepare', 'check', 'run']); a = p.parse_args()
    {'prepare': prepare, 'check': checked, 'run': run}[a.action]()
