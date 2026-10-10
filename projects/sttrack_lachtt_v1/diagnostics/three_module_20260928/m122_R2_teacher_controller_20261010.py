"""Run two independent full-Train teacher pipelines, without progress polling."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys


def verify(gate):
    assert gate['R0_six_complete']
    assert gate['R1_runtime_audit_complete'] and gate['R1_runtime_audit_blocking_count'] == 0
    assert gate['R2_source_audit_complete'] and gate['R2_source_audit_blocking_count'] == 0
    for row in gate['bound_files']:
        payload = Path(row['path']).read_bytes()
        assert len(payload) == row['bytes']
        assert hashlib.sha256(payload).hexdigest() == row['sha256'], row['path']


def pipeline(root, gate, shard):
    output = root / ('teacher_shard%d' % shard)
    results = []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), PYTHONUNBUFFERED='1')
    for phase in ['collect', 'replay']:
        name = phase + str(shard)
        command = [gate['python'], gate['teacher_entry'], '--phase', phase, '--shard', str(shard),
                   '--physical-gpu', str(shard), '--output', str(output)] + gate['common_args']
        started = datetime.now(timezone.utc).isoformat()
        launch = dict(stage=name, command=command, physical_gpu=shard, started_at=started)
        (root / (name + '.launch.json')).write_text(json.dumps(launch, indent=2) + '\n')
        with (root / (name + '.log')).open('wb') as log:
            result = subprocess.run(command, cwd=gate['code_root'], env=env, stdout=log, stderr=subprocess.STDOUT)
        (root / (name + '.exit')).write_text(str(result.returncode) + '\n')
        receipt = dict(launch, exit_code=result.returncode, finished_at=datetime.now(timezone.utc).isoformat())
        (root / (name + '.receipt.json')).write_text(json.dumps(receipt, indent=2) + '\n')
        results.append(receipt)
        if result.returncode != 0:
            return dict(shard=shard, complete=False, stages=results)
    return dict(shard=shard, complete=True, stages=results)


def main():
    root = Path(sys.argv[1])
    gate = json.loads((root / 'gate.json').read_text())
    verify(gate)
    assert not (root / 'launch.json').exists()
    launch = dict(started_at=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
                  stage_order='GPU0 collect0->replay0; GPU1 collect1->replay1',
                  progress_polling=False, planned_NN_stages=4,
                  optimizations=0, C_training_or_formal_evaluation=False)
    (root / 'launch.json').write_text(json.dumps(launch, indent=2) + '\n')
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(pipeline, root, gate, shard) for shard in [0, 1]]
        results = [future.result() for future in futures]
    if not all(result['complete'] for result in results):
        (root / 'result.json').write_text(json.dumps(dict(status='R2_teacher_pipeline_failed', pipelines=results), indent=2) + '\n')
        (root / 'controller.exit').write_text('1\n')
        return 1
    collected = [json.loads((root / ('teacher_shard%d' % shard) / 'teacher_events.json').read_text()) for shard in [0, 1]]
    teachers = [json.loads((root / ('teacher_shard%d' % shard) / 'replay/result.json').read_text()) for shard in [0, 1]]
    indices = [index for collection in collected for index in collection['sequence_indices']]
    assert sorted(indices) == list(range(152))
    assert sum(collection['native_track_calls'] for collection in collected) == 219802
    assert collected[0]['inputs'] == collected[1]['inputs'] == teachers[0]['inputs'] == teachers[1]['inputs']
    assert sum(len(collection['events']) for collection in collected) == sum(len(teacher['events']) for teacher in teachers)
    result = dict(status='complete_M122_full_train_template_teacher', pipelines=results,
        inputs=collected[0]['inputs'], sequences=152, native_track_calls=219802,
        events=sum(len(teacher['events']) for teacher in teachers),
        future_tracking_calls=sum(teacher['future_tracking_calls'] for teacher in teachers),
        teacher_shards=['teacher_shard0/replay/result.json', 'teacher_shard1/replay/result.json'],
        optimizations=0, C_training_or_formal_evaluation=False,
        finished_at=datetime.now(timezone.utc).isoformat())
    (root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    (root / 'controller.exit').write_text('0\n')
    print(json.dumps(result), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
