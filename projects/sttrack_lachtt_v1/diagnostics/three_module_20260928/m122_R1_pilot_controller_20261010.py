"""Run the already-reviewed 12-event R1 pilot; no optimizer or full teacher."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).parent
SOURCE = Path('/home/SUTrack_RGBD_L/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928')
PYTHON = '/root/autodl-tmp/envs/sttrack/bin/python'
OUTPUT = ROOT / 'output'


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    gate = json.loads((ROOT / 'gate.json').read_text())
    for row in gate['bound_files']:
        path = Path(row['path'])
        assert path.stat().st_size == row['bytes'], str(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], str(path)
    assert gate['R0_six_complete'] and gate['fresh_result_audit_complete']
    assert gate['fresh_result_audit_blocking_count'] == 0
    assert gate['pilot_source_verdict'] == 'PASS'
    assert not OUTPUT.exists()
    base = [PYTHON, '-B', '-u', str(SOURCE / 'run_template_write_pilot.py')]
    for key, value in gate['arguments'].items():
        base.extend(['--' + key, value])
    base.extend(['--output', str(OUTPUT)])
    launches = []
    environment = dict(os.environ, PYTHONPATH='/home/SUTrack_RGBD_L',
        PYTHONOPTIMIZE='0', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')

    def launch(name, gpu, extra):
        command = base + extra
        with (ROOT / (name + '.log')).open('w') as log:
            child = subprocess.Popen(command, cwd='/home/SUTrack_RGBD_L',
                env=dict(environment, CUDA_VISIBLE_DEVICES=str(gpu)),
                stdout=log, stderr=subprocess.STDOUT)
        row = dict(stage=name, physical_gpu=gpu, pid=child.pid,
            command=command, started=time.time())
        launches.append(row)
        write(ROOT / 'stages.json', launches)
        return child, row

    def finish(child, row):
        row.update(exit=child.wait(), finished=time.time())
        (ROOT / (row['stage'] + '.exit')).write_text(str(row['exit']) + '\n')
        write(ROOT / 'stages.json', launches)
        return row['exit']

    child, row = launch('collect', 0, ['--phase', 'collect'])
    rc = finish(child, row)
    if rc != 0:
        return rc
    workers = [launch('replay' + str(shard), shard,
        ['--phase', 'replay', '--shard', str(shard)]) for shard in [0, 1]]
    exits = [finish(child, row) for child, row in workers]
    if any(exits):
        return next(value for value in exits if value)
    collection = json.loads((OUTPUT / 'pilot_events.json').read_text())
    results = [json.loads((OUTPUT / ('replay_shard%d' % shard) / 'result.json').read_text())
        for shard in [0, 1]]
    assert collection['status'] == 'complete_M122_template_pilot_event_collection_only'
    assert len(collection['events']) == 12
    for shard, result in enumerate(results):
        assert result['status'] == 'complete_M122_template_fork_pilot_shard'
        assert result['shard'] == shard and len(result['events']) == 6
        assert result['same_GPU_for_every_pair'] and result['frozen_before_after_exact']
        assert result['optimizations'] == result['GT_reinitializations_after_first_frame'] == 0
        assert all(event['K_K_per_frame_exact'] for event in result['events'])
    write(ROOT / 'result.json', dict(status='complete_M122_R1_pilot_execution_only',
        completed_at=datetime.now(timezone.utc).isoformat(), stages=launches,
        collection_result=str(OUTPUT / 'pilot_events.json'),
        replay_results=[str(OUTPUT / ('replay_shard%d' % shard) / 'result.json') for shard in [0, 1]],
        optimizations=0, full_teacher_or_C_training=False, independent_result_audit=False))
    return 0


if __name__ == '__main__':
    code = main()
    (ROOT / 'controller.exit').write_text(str(code) + '\n')
    sys.exit(code)
