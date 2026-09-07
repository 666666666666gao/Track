"""Adopt the existing seed2027 pair after the user's cancellation of multi-seed work."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime, timezone

B = Path('/root/autodl-tmp')
ROOT = B / 'sttrack_m73_paired_lexical_replication_20260907'
AMEND = ROOT / 'single_seed_amendment'
OUT = AMEND / 'controller'
SEED = ROOT / 'seed2027'
PYTHON = B / 'envs/sttrack/bin/python'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p, value): Path(p).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
def now(): return datetime.now(timezone.utc).isoformat()


def identity(pid):
    p = Path('/proc') / str(pid)
    f = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, start_ticks=f[19], state=f[0],
        argv=[v.decode() for v in (p / 'cmdline').read_bytes().split(b'\0') if v])


def checked():
    spec = read(OUT / 'spec.json'); plan = read(AMEND / 'plan.json')
    assert sha(__file__) == spec['source_sha256']
    assert sha(AMEND / 'plan.json') == spec['amendment_sha256']
    assert sha(AMEND / 'patch_receipt.json') == spec['patch_receipt_sha256']
    assert plan['active_seeds'] == [2027] and plan['cancelled_seeds'] == [2028]
    for p, h in read(AMEND / 'patch_receipt.json')['after_sha256'].items(): assert sha(p) == h
    assert not (ROOT / 'seed2028/training').exists()
    for name, key in [('training_spec.json', 'training_spec_sha256'), ('recursive_spec.json', 'recursive_spec_sha256')]:
        assert sha(SEED / name) == plan['seed2027_frozen'][key]
    return spec


def step(name, command, cwd):
    with (OUT / (name + '.log')).open('w') as log:
        p = subprocess.run(command, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT)
    (OUT / (name + '.exit')).write_text(str(p.returncode) + '\n')
    assert p.returncode == 0, name


def run():
    spec = checked()
    assert not (OUT / 'running_identity.json').exists()
    write(OUT / 'running_identity.json', dict(observed_utc=now(), identity=identity(os.getpid()), spec_sha256=sha(OUT / 'spec.json')))
    while not (SEED / 'controller.exit').exists():
        screen = identity(spec['suspended_screen']['pid'])
        assert screen['start_ticks'] == spec['suspended_screen']['start_ticks'] and screen['state'] == 'T'
        parent = identity(spec['suspended_scheduler']['pid'])
        assert parent['start_ticks'] == spec['suspended_scheduler']['start_ticks'] and parent['state'] == 'T'
        proc = Path('/proc') / str(spec['adopted_pair']['pid'])
        if not proc.exists():
            assert (SEED / 'controller.exit').exists(), 'Existing seed2027 pair disappeared without exit receipt'
            break
        current = identity(spec['adopted_pair']['pid'])
        assert current['start_ticks'] == spec['adopted_pair']['start_ticks'] and current['state'] != 'Z'
        event = dict(status='waiting_for_original_seed2027_pair', observed_utc=now(),
            adopted_pair=current, seed2028_cancelled=True, poll_seconds=240, new_model_calls=0)
        write(OUT / 'latest.json', event); print(json.dumps(event), flush=True)
        time.sleep(240)
    code = (SEED / 'controller.exit').read_text().strip()
    assert code.lstrip('-').isdigit()
    (ROOT / 'seed2027.exit').write_text(code + '\n')
    # The pair is terminal. Kill only the stopped obsolete parent, never resume its loop.
    parent = identity(spec['suspended_scheduler']['pid'])
    assert parent['start_ticks'] == spec['suspended_scheduler']['start_ticks'] and parent['state'] == 'T'
    os.kill(parent['pid'], signal.SIGKILL)
    screen = identity(spec['suspended_screen']['pid'])
    assert screen['start_ticks'] == spec['suspended_screen']['start_ticks'] and screen['state'] == 'T'
    os.kill(screen['pid'], signal.SIGKILL)
    write(AMEND / 'obsolete_scheduler_retired.json', dict(observed_utc=now(), parent=parent,
        screen=screen, signal='SIGKILL management processes after adopted pair exit receipt', seed2027_exit=code, seed2028_started=False))
    if code != '0':
        (ROOT / 'controller.exit').write_text(code + '\n')
        write(OUT / 'result.json', dict(status='seed2027_execution_failed', seed2027_exit=code, seed2028_cancelled=True))
        raise RuntimeError('Original seed2027 pair failed; see its existing logs')
    checked()
    j = read(SEED / 'recursive_result.json'); frozen = read(ROOT / 'frozen.json')['seeds']['2027']
    assert j['training_spec_sha256'] == frozen['training_spec_sha256']
    assert j['recursive_spec_sha256'] == frozen['recursive_spec_sha256']
    result = dict(status='completed_single_seed_paired_lexical_experiment', observed_utc=now(),
        spec_sha256=sha(ROOT / 'spec.json'), source_sha256=sha(__file__), amendment_sha256=sha(AMEND / 'plan.json'),
        results={'2027': dict(result_sha256=sha(SEED / 'recursive_result.json'), aggregates=j['aggregates'], gates=j['gates'], primary_pass=j['primary_pass'])},
        single_seed_development_gates_pass=j['primary_pass'], candidate_seed=2027,
        candidate_head=str(SEED / 'training/category/final.pth'), cancelled_seeds=[2028],
        fixed_head_content_preparation_allowed=j['primary_pass'], public_evaluation_allowed=False,
        independent_model_review_pass=False, goal_completed=False)
    assert not (ROOT / 'result.json').exists()
    write(ROOT / 'result.json', result)
    (ROOT / 'analysis.exit').write_text('0\n'); (ROOT / 'controller.exit').write_text('0\n')
    step('audit', [str(PYTHON), '-u', str(B / 'audit_m73_completed_20260907.py'), 'completed'], ROOT)
    audit_path = ROOT / 'completion_tools/completed.json'; audit = read(audit_path)
    assert audit['status'] == 'completed_M73_single_seed_checkpoint_and_scalar_audit'
    assert audit['amendment_sha256'] == sha(AMEND / 'plan.json')
    content = ROOT / 'content_counterfactuals'
    if not audit['single_seed_development_gates_pass']:
        assert not (content / 'activation.json').exists()
        result = dict(status='completed_M73_audit_content_skipped_by_frozen_gates', observed_utc=now(),
            audit_sha256=sha(audit_path), content_started=False, gates=audit['seeds']['2027']['gates'],
            reason='The retained seed2027 failed one or more unchanged development requirements.', public_evaluation_allowed=False)
    else:
        step('content', ['bash', str(content / 'run_controls.sh')], content)
        r = read(content / 'result.json')
        assert r['status'] == 'completed_M73_predetermined_final_fixed_head_content'
        assert r['M73_audit_sha256'] == sha(audit_path) and r['candidate_seed'] == 2027
        result = dict(status='completed_M73_audit_and_conditional_content', observed_utc=now(),
            audit_sha256=sha(audit_path), content_started=True, content_result_sha256=sha(content / 'result.json'),
            lexical_criteria_pass=r['lexical_criteria_pass'], candidate_entry_parity_preparation_allowed=r['candidate_entry_parity_preparation_allowed'],
            public_evaluation_allowed=False)
    result.update(amendment_sha256=sha(AMEND / 'plan.json'), active_seeds=[2027], cancelled_seeds=[2028])
    write(OUT / 'result.json', result); write(OUT / 'latest.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': run()
