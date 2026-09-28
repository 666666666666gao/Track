"""Hourly M98 terminal gate, then one M99 sanity and fixed final-budget run."""

import datetime
import json
import os
from pathlib import Path
import subprocess
import time


SOURCE = Path('/home/SUTrack_RGBD_L/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928')
CONTEXTS = Path('/root/autodl-tmp/sttrack_m98_train_contexts_20260928')
OUTPUT = Path('/root/autodl-tmp/sttrack_m99_ab_visual_control_20260928')
PYTHON = '/root/autodl-tmp/envs/sttrack/bin/python'
FIRST_CHECK = datetime.datetime.fromisoformat('2026-09-28T01:36:15+00:00').timestamp()


def record(status, **fields):
    row = dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               status=status, **fields)
    with (OUTPUT/'queue_events.jsonl').open('a') as stream:
        stream.write(json.dumps(row)+'\n')
    (OUTPUT/'queue_state.json').write_text(json.dumps(row, indent=2)+'\n')
    print(json.dumps(row), flush=True)


def run(mode, directory):
    command = [PYTHON, '-u', str(SOURCE/'train_ab_visual_control.py'),
               '--cache', '/root/autodl-tmp/sttrack_m90_train_states_20260928',
               '--contexts', str(CONTEXTS),
               '--origins', '/root/autodl-tmp/sttrack_m95_initial_origins_20260928',
               '--empty-bank', '/root/autodl-tmp/sttrack_full152_paired_20260925/text_full152.pt',
               '--output', str(OUTPUT/directory), '--mode', mode, '--device', 'cuda']
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    with (OUTPUT/f'{directory}.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        record('running_'+mode, pid=process.pid, command=command)
        code = process.wait()
    (OUTPUT/f'{directory}.exit').write_text(str(code)+'\n')
    if code != 0:
        record('failed_'+mode, exit_code=code, log=str(OUTPUT/f'{directory}.log'))
        (OUTPUT/'queue.exit').write_text(str(code)+'\n')
        raise SystemExit(code)
    return json.loads((OUTPUT/directory/'result.json').read_text())


def main():
    assert (OUTPUT/'cpu_smoke.exit').read_text().strip() == '0'
    assert not (OUTPUT/'queue_state.json').exists()
    record('scheduled_M98_wait', first_check_utc='2026-09-28T01:36:15+00:00',
           poll_seconds=3600, no_GPU_job_started=True)
    next_check = FIRST_CHECK
    while True:
        time.sleep(max(0, next_check-time.time()))
        terminal = CONTEXTS/'queue.exit'
        if terminal.exists():
            code = int(terminal.read_text().strip())
            record('M98_terminal', exit_code=code)
            if code != 0:
                (OUTPUT/'queue.exit').write_text(str(code)+'\n')
                raise SystemExit(code)
            audit = json.loads((CONTEXTS/'input_audit.json').read_text())
            assert audit['status'] == 'complete_context_input_audit_only'
            assert (audit['sequences'], audit['events'], audit['frames']) == (152, 3502, 219194)
            break
        processes = subprocess.run(['ps', '-eo', 'pid=,args='], check=True,
                                   capture_output=True, text=True).stdout.splitlines()
        related = [line for line in processes if 'queue_train_contexts.sh' in line
                   or ('collect_train_contexts.py' in line and str(CONTEXTS) in line)
                   or ('audit_train_contexts.py' in line and str(CONTEXTS) in line)]
        record('verified_M98_wait' if related else 'missing_M98_handle', processes=related)
        assert related, 'M98 has no terminal receipt and no live collection/audit queue handle'
        next_check += 3600
    sanity = run('gpu-sanity', 'gpu_sanity')
    assert sanity['status'] == 'complete_visual_control_sanity_only'
    assert sanity['sanity_examples'] == 64 and sanity['optimizer_steps'] == 2
    assert not sanity['checkpoint_saved'] and not sanity['development_evaluated']
    record('GPU_sanity_passed', optimizer_seconds=sanity['elapsed_seconds'],
           gpu_peak_allocated_bytes=sanity['gpu_peak_allocated_bytes'],
           gpu_peak_reserved_bytes=sanity['gpu_peak_reserved_bytes'],
           full_optimization_seconds_projection=sanity['elapsed_seconds']/2*480,
           projection_excludes_loading_and_evaluation=True)
    result = run('train', 'final_train')
    assert result['status'] == 'complete_visual_control'
    assert result['epochs'] == 12 and result['optimizer_steps'] == 480
    record('complete_fixed_state_visual_control',
           capacity_checks=result['fixed_state_capacity_checks'],
           result=str(OUTPUT/'final_train'/'result.json'),
           no_semantic_recursive_or_official_acceptance=True)
    (OUTPUT/'queue.exit').write_text('0\n')


if __name__ == '__main__':
    main()
