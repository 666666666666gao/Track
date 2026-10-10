"""Run the fixed C preflight, paired fits and development32 without polling."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def run(root, gate, stage, command, gpu):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONOPTIMIZE='0',
        PYTHONUNBUFFERED='1', PYTHONPATH=gate['code_root'] + os.pathsep + gate['base_code_root'])
    launch = dict(stage=stage, command=command, physical_gpu=gpu,
                  started_at=datetime.now(timezone.utc).isoformat())
    write(root / (stage + '.launch.json'), launch)
    with (root / (stage + '.log')).open('xb') as stream:
        result = subprocess.run(command, cwd=gate['code_root'], env=environment,
                                stdout=stream, stderr=subprocess.STDOUT)
    (root / (stage + '.exit')).write_text(str(result.returncode) + '\n')
    receipt = dict(launch, exit_code=result.returncode, finished_at=datetime.now(timezone.utc).isoformat())
    write(root / (stage + '.receipt.json'), receipt)
    return receipt


def failed(root, stages):
    write(root / 'result.json', dict(status='M122_R3_C_stage_failed', stages=stages,
        formal_nine_metrics=False, R4_launched=False))
    (root / 'controller.exit').write_text('1\n')
    return 1


def main():
    assert sys.flags.optimize == 0 and sys.version_info[:2] == (3, 8)
    root = Path(sys.argv[1])
    gate = json.loads((root / 'gate.json').read_text())
    for row in gate['bound_files']:
        path = Path(row['path'])
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], str(path)
    audit = json.loads(Path(gate['teacher_audit']).read_text())
    assert audit['review_call_status'] == 'completed' and audit['blocking_count'] == 0
    assert audit['R2_terminal_raw_audit_complete']
    assert sha(Path(gate['teacher_audit'])) == gate['teacher_audit_sha256']
    teacher_root = Path(gate['teacher_root'])
    assert (teacher_root / 'controller.exit').read_text().strip() == '0'
    assert sha(teacher_root / 'result.json') == audit['audited_teacher_result_sha256']
    for index in [0, 1]:
        assert sha(teacher_root / ('teacher_shard%d/replay/result.json' % index)) == audit['audited_teacher_shards_sha256'][index]
        assert sha(teacher_root / ('teacher_shard%d/native_prefix.jsonl' % index)) == audit['audited_teacher_prefixes_sha256'][index]
    review = json.loads(Path(gate['controller_source_audit']).read_text())
    assert review['review_call_status'] == 'completed' and review['blocking_count'] == 0
    assert review['source_verdict'] == 'PASS'
    assert sha(Path(gate['controller_source_audit'])) == gate['controller_source_audit_sha256']
    assert not (root / 'launch.json').exists()
    launch = dict(pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat(),
        stage_order='GPU0 preflight; paired GPU0 current/GPU1 future fits; paired development32; CPU comparison',
        progress_polling=False, seed=2027, fit_sequences=120, development_sequences=32,
        planned_optimization_steps_per_arm=320, formal_nine_metrics=False, R4_launched=False)
    write(root / 'launch.json', launch)
    code = Path(gate['code_root'])
    program = [gate['python'], '-B', '-u']
    teacher_args = ['--teacher-root', gate['teacher_root'], '--teacher-audit', gate['teacher_audit']]
    preflight = run(root, gate, 'preflight', program + [str(code / 'run_template_write_C_preflight.py')] +
        gate['common_args'] + teacher_args + ['--physical-gpu', '0', '--output', str(root / 'preflight')], 0)
    stages = [preflight]
    if preflight['exit_code'] != 0:
        return failed(root, stages)
    checked = json.loads((root / 'preflight/result.json').read_text())
    assert checked['status'] == 'complete_M122_R3_constant_action_preflight256'
    assert checked['frames_per_history'] == 256 and checked['native_qualified_writes'] > 0
    assert checked['always_write_output_and_submission_exact']
    assert checked['rejection_keeps_current_output_query_and_previous_templates_exact']
    assert checked['optimizations'] == checked['C_MLPs_constructed'] == 0
    assert checked['teacher_result_sha256'] == audit['audited_teacher_result_sha256']
    assert checked['teacher_audit_sha256'] == gate['teacher_audit_sha256']
    fits = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = []
        for gpu, arm in enumerate(['current', 'future']):
            command = program + [str(code / 'train_template_write_C.py')] + teacher_args + [
                '--spec', gate['spec'], '--split', gate['split'], '--arm', arm,
                '--physical-gpu', str(gpu), '--output', str(root / ('fit_' + arm))]
            futures.append(pool.submit(run, root, gate, 'fit_' + arm, command, gpu))
        fits = [future.result() for future in futures]
    stages.extend(fits)
    if any(receipt['exit_code'] != 0 for receipt in fits):
        return failed(root, stages)
    trained = [json.loads((root / ('fit_' + arm) / 'result.json').read_text()) for arm in ['current', 'future']]
    for arm, fitted in zip(['current', 'future'], trained):
        assert fitted['status'] == 'complete_M122_C_fixed_teacher_fit120' and fitted['arm'] == arm
        assert fitted['common_fit_events'] == 993 and fitted['common_development_events'] == 264
        assert fitted['optimizer_steps'] == 320 and fitted['epochs'] == 10 and fitted['seed'] == 2027
        assert fitted['teacher_audit_sha256'] == gate['teacher_audit_sha256']
    for key in ['common_dataset_sha256', 'initial_state_sha256', 'order_sha256']:
        assert trained[0][key] == trained[1][key], key
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = []
        for gpu, arm in enumerate(['current', 'future']):
            fit = root / ('fit_' + arm)
            command = program + [str(code / 'evaluate_template_write_C.py')] + gate['common_args'] + [
                '--C-final', str(fit / 'final.pt'), '--C-result', str(fit / 'result.json'),
                '--split', gate['split'], '--arm', arm, '--physical-gpu', str(gpu),
                '--output', str(root / ('development_' + arm))]
            futures.append(pool.submit(run, root, gate, 'development_' + arm, command, gpu))
        developments = [future.result() for future in futures]
    stages.extend(developments)
    if any(receipt['exit_code'] != 0 for receipt in developments):
        return failed(root, stages)
    command = program + [str(code / 'analyze_template_write_C_dev.py')] + teacher_args + [
        '--current-fit', str(root / 'fit_current'), '--future-fit', str(root / 'fit_future'),
        '--current-predictions', str(root / 'development_current'),
        '--future-predictions', str(root / 'development_future'), '--spec', gate['spec'],
        '--split', gate['split'], '--output', str(root / 'comparison')]
    comparison = run(root, gate, 'comparison', command, '')
    stages.append(comparison)
    if comparison['exit_code'] != 0:
        return failed(root, stages)
    result = dict(status='complete_M122_R3_C_paired_fit120_and_development32', stages=stages,
        teacher_audit_sha256=gate['teacher_audit_sha256'],
        preflight_result_sha256=sha(root / 'preflight/result.json'),
        fit_result_sha256=[sha(root / ('fit_' + arm) / 'result.json') for arm in ['current', 'future']],
        comparison_sha256=sha(root / 'comparison/result.json'),
        future_prescribed_gate_passed=json.loads((root / 'comparison/result.json').read_text())['prescribed_macro_IoU_and_H10_gate_passed'],
        R4_requires_fresh_result_audit=True, R4_launched=False, formal_nine_metrics=False,
        finished_at=datetime.now(timezone.utc).isoformat())
    write(root / 'result.json', result)
    (root / 'controller.exit').write_text('0\n')
    print(json.dumps(result), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
