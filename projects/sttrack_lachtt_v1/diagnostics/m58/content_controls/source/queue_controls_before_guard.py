"""Conditional queue: no GPU work unless M58's original recursive gate passes."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

from run_controls import ROOT, PARENT, plans, seal_bundle, sha, write


PYTHON = '/root/autodl-tmp/envs/sttrack/bin/python'


def wait_for(paths, upstream=False):
    while True:
        if upstream:
            observed = [PARENT / n for n in ['training_text.exit', 'training_visual.exit',
                'text_recursive.exit', 'visual_recursive.exit', 'recursive_analysis.exit']]
            failed = {p.name:p.read_text().strip() for p in observed if p.exists() and p.read_text().strip() != '0'}
            if failed:
                write(ROOT / 'queue_result.json', dict(status='not_run_upstream_failed', failures=failed,
                    content_spec_sha256=sha(ROOT / 'spec.json'), actual_control_tracking_calls=0))
                return False
        for path in paths:
            if path.exists():
                assert path.read_text().strip() == '0', str(path)
        if all(p.exists() for p in paths):
            return True
        time.sleep(240)


def free_gpus():
    values = [int(x) for x in subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
        '--format=csv,noheader,nounits']).decode().splitlines()]
    assert len(values) == 2 and all(v < 500 for v in values), values
    return values


def launch(name, argument, gpu):
    script = ROOT / ('run_' + name + '.sh')
    script.write_text('#!/bin/bash\ncd ' + str(ROOT) + '\nCUDA_VISIBLE_DEVICES=' + str(gpu) +
        ' ' + PYTHON + ' -u run_controls.py ' + argument + ' > ' + name + '.log 2>&1\n' +
        'status=$?\nprintf "%s\\n" "$status" > ' + name + '.exit\nexit "$status"\n')
    screen = 'm58_cf_' + name + '_20260906'
    subprocess.run(['screen', '-dmS', screen, 'bash', str(script)], check=True)
    return dict(name=name, gpu=gpu, screen=screen, command_argument=argument, script_sha256=sha(script))


def main():
    spec = plans()
    print('WAIT_FOR_ORIGINAL_M58_RECURSIVE_GATE', datetime.now(timezone.utc).isoformat(), flush=True)
    if not wait_for([PARENT / 'recursive_analysis.exit'], upstream=True):
        return
    parent = json.loads((PARENT / 'recursive_result.json').read_text())
    assert parent['status'] == 'complete_recursive_development'
    assert parent['recursive_spec_sha256'] == spec['parent_recursive_spec_sha256']
    assert parent['primary_pass'] == all(parent['gates'].values())
    if not parent['primary_pass']:
        write(ROOT / 'queue_result.json', dict(status='not_run_parent_gate_failed',
            content_spec_sha256=sha(ROOT / 'spec.json'), parent_result_sha256=sha(PARENT / 'recursive_result.json'),
            parent_gates=parent['gates'], actual_control_tracking_calls=0, candidate_bundle_created=False))
        print('PARENT_GATE_FAILED_NO_CONTENT_RUNS', flush=True)
        return
    seal_bundle()
    memory = free_gpus()
    parity = launch('parity', '--parity', 0)
    write(ROOT / 'parity_launch.json', dict(started_utc=datetime.now(timezone.utc).isoformat(),
        gpu_memory_mib_before=memory, run=parity, content_spec_sha256=sha(ROOT / 'spec.json')))
    wait_for([ROOT / 'parity.exit'])
    memory = free_gpus()
    runs = [launch('worker0', '--worker 0', 0), launch('worker1', '--worker 1', 1)]
    write(ROOT / 'controls_launch.json', dict(started_utc=datetime.now(timezone.utc).isoformat(),
        gpu_memory_mib_before=memory, runs=runs, content_spec_sha256=sha(ROOT / 'spec.json')))
    wait_for([ROOT / 'worker0.exit', ROOT / 'worker1.exit'])
    with (ROOT / 'analysis.log').open('x') as log:
        completed = subprocess.run([PYTHON, '-u', str(ROOT / 'run_controls.py'), '--analyze'],
            cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
    (ROOT / 'analysis.exit').write_text(str(completed.returncode) + '\n')
    assert completed.returncode == 0
    write(ROOT / 'queue_result.json', dict(status='content_controls_complete_no_public_evaluation',
        content_spec_sha256=sha(ROOT / 'spec.json'), result_sha256=sha(ROOT / 'result.json')))


if __name__ == '__main__':
    main()
