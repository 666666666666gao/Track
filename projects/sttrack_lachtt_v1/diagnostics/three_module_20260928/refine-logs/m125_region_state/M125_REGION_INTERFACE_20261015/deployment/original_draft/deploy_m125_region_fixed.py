"""Reviewed regional-arm admission on Linux; invoke once per real64/full phase."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import shutil
import subprocess
import sys


NOT_BEFORE = '2026-10-17T12:45:00+08:00'
PREDECESSOR = Path('/root/autodl-tmp/m125_R0_fixed6_20261015')
ARM_FILES = {'roi': 'CANDIDATE_ROI.json', 'language': 'CANDIDATE_R1.json', 'visual': 'CANDIDATE_V.json'}


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify_inputs(folder, arm):
    review = read(folder / 'EXPERIMENT_AUDIT.json')
    assert review['source_verdict'] == 'PASS_SOURCE_ONLY' and review['blocking_findings'] == []
    inventory = read(folder / 'INPUT_MANIFEST.json')
    audited = review['audited_input_hashes']
    for original, digest in inventory.items():
        assert sha(folder / PureWindowsPath(original).name) == digest, original
        assert audited[original] == digest, original
    candidates = [name for name in inventory if PureWindowsPath(name).name == ARM_FILES[arm]]
    assert len(candidates) == 1
    candidate = read(folder / ARM_FILES[arm])
    assert candidate['region_arm'] == arm
    assert candidate['status'] == 'candidate_M125_region_fixed6_source_gate'
    assert candidate['candidate_only_not_deployed']
    assert candidate['fixed_budget_passes'] == candidate['fixed_final_pass'] == 6
    assert candidate['seed'] == 2027 and candidate['no_development_split']
    assert not candidate['external_metric_checkpoint_selection']
    assert not candidate['Test50_CDTB80_used_for_checkpoint_selection']
    assert Path(candidate['required_completed_predecessor']) == PREDECESSOR
    assert Path(candidate['base_code_root']) == PREDECESSOR / 'code'
    assert Path(candidate['root']) == Path('/root/autodl-tmp/m125_region_' + arm + '_fixed6')
    assert Path(candidate['code_root']) == Path(candidate['root']) / 'code'
    return candidate, inventory


def verify_predecessor(candidate):
    root = Path(candidate['required_completed_predecessor'])
    for name in ['controller.supervised.exit', 'controller.exit']:
        assert (root / name).read_text().strip() == '0', name
    terminal = read(root / 'controller.supervisor.terminal.json')
    assert terminal['exit_code'] == 0
    original = read(root / 'actual_launch.json')
    process = Path('/proc') / str(original['supervisor_pid']) / 'cmdline'
    if process.is_file():
        assert process.read_bytes().split(b'\0')[:-1] != [part.encode() for part in original['command']]
    result = read(root / 'result.json')
    assert result['status'] == 'complete_M125_R0_fixed6_and_three_official_datasets'
    assert result['fixed_budget_passes'] == result['fixed_final_pass'] == 6
    assert result['formal_nine_metrics'] and not result['external_metric_checkpoint_selection']
    assert len(result['stages']) == 15 and all(stage['exit_code'] == 0 for stage in result['stages'])
    for epoch in range(1, 7):
        folder = root / ('passes/pass-%02d' % epoch)
        assert (folder / 'train_fixed_pass.exit').read_text().strip() == '0'
        trained = read(folder / 'fit/result.json')
        assert trained['epochs'] == epoch and trained['sequence_runs'] == epoch * 152
        assert trained['track_calls'] == epoch * 219802
    assert trained['fixed_final_after_complete_budget']
    selection = read(root / 'evaluation/selection.json')
    assert selection['status'] == 'M125_R0_fixed6_final_before_metrics' and len(selection['models']) == 1
    final = selection['models'][0]['final_sha256']
    assert sha(root / 'passes/pass-06/fit/final.pt') == final
    combined = read(root / 'evaluation/all_results.json')
    assert combined['status'] == 'three_M125_R0_fixed6_full_evaluations_complete'
    assert combined['selection_sha256'] == sha(root / 'evaluation/selection.json')
    assert len(combined['results']) == 3
    assert {row['dataset'] for row in combined['results']} == {'depthtrack_test', 'cdtb', 'vot'}
    for row in combined['results']:
        assert row['final_sha256'] == final and sha(row['result_path']) == row['result_sha256']
    return dict(root=str(root), result_sha256=sha(root / 'result.json'),
                all_results_sha256=sha(root / 'evaluation/all_results.json'),
                supervisor_terminal_sha256=sha(root / 'controller.supervisor.terminal.json'), final_sha256=final)


def verify_sources(candidate, folder, inventory, deployed):
    code = Path(candidate['code_root'])
    new_names = {PureWindowsPath(name).name for name in inventory if name.endswith('.py')}
    future = {Path(path).name: digest for path, digest in candidate['source_sha256'].items() if Path(path).parent == code}
    assert len(new_names) == 11 and len(future) == 64
    assert new_names <= future.keys()
    for name, digest in future.items():
        source = code / name if deployed else (folder / name if name in new_names else PREDECESSOR / 'code' / name)
        assert sha(source) == digest, str(source)
    for path, digest in candidate['source_sha256'].items():
        if Path(path).parent != code:
            assert sha(path) == digest, path
    for path, digest in candidate['input_sha256'].items():
        assert sha(path) == digest, path
    return future, new_names


def resource_admission(candidate):
    assert shutil.disk_usage(PREDECESSOR).free >= candidate['required_free_disk_bytes']
    query = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    rows = [[int(value.strip()) for value in line.split(',')] for line in query.stdout.splitlines()]
    assert len(rows) == 2 and [row[0] for row in rows] == [0, 1]
    assert all(row[1] < 500 for row in rows), rows
    return rows


def reviewed_gate(candidate, folder):
    root = Path(candidate['root'])
    gate = dict(candidate)
    gate.update(status='reviewed_M125_region_fixed6_source_gate', candidate_only_not_deployed=False,
                source_review_report_path=str(root / 'source_review.json'),
                source_review_report_sha256=sha(folder / 'EXPERIMENT_AUDIT.json'),
                source_review_candidate_sha256=sha(folder / ARM_FILES[candidate['region_arm']]))
    return gate


def launch(root, gate, phase, command, predecessor, gpu_rows):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1',
        PYTHONPATH=os.pathsep.join([gate['code_root'], gate['base_code_root'], '/home/SUTrack_RGBD_L']),
        PYTHONDONTWRITEBYTECODE='1', PYTHONOPTIMIZE='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    if phase == 'real64':
        driver = root / 'real64_supervisor.py'
        program = '\n'.join(['import subprocess,json,sys', 'from pathlib import Path',
            'from datetime import datetime,timezone', 'root=Path(' + repr(str(root)) + ')',
            'command=' + repr(command), 'row=dict(command=command,started_at=datetime.now(timezone.utc).isoformat())',
            'completed=subprocess.run(command,cwd=' + repr(gate['code_root']) + ')',
            'row.update(exit_code=completed.returncode,finished_at=datetime.now(timezone.utc).isoformat())',
            '(root/"real64.exit").write_text(str(completed.returncode)+chr(10))',
            '(root/"real64.receipt.json").write_text(json.dumps(row,indent=2)+chr(10))',
            'sys.exit(completed.returncode)', ''])
        with driver.open('x', encoding='utf-8', newline='\n') as stream: stream.write(program)
        process_command = [gate['python'], '-B', '-u', str(driver)]
        log = root / 'real64.log'
        receipt_path = root / 'real64.actual_launch.json'
    else:
        process_command = command
        log = root / 'controller.log'
        receipt_path = root / 'actual_launch.json'
    with log.open('xb') as stream:
        child = subprocess.Popen(process_command, cwd=gate['code_root'], env=environment,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = dict(status='actual_M125_region_' + phase + '_launched', root=str(root), region_arm=gate['region_arm'],
        started_at=datetime.now(timezone.utc).isoformat(), supervisor_pid=child.pid, command=command,
        process_command=process_command, gate_sha256=sha(root / 'gate.json'),
        source_review_sha256=gate['source_review_report_sha256'], deployment_operator_sha256=sha(__file__),
        completed_predecessor=predecessor, GPU_memory_admission=gpu_rows,
        GPU_query_fields='index,memory.used,memory.total', automatic_retries=0,
        formal_training_launched=phase == 'full', fixed_budget_passes=6, fixed_final_pass=6)
    write_new(receipt_path, receipt)
    print(json.dumps(receipt), flush=True)


def main():
    assert sys.flags.optimize == 0
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--arm', choices=list(ARM_FILES), required=True)
    parser.add_argument('--phase', choices=['real64', 'full'], required=True)
    args = parser.parse_args()
    assert datetime.now(timezone.utc) >= datetime.fromisoformat(NOT_BEFORE)
    candidate, inventory = verify_inputs(args.inputs, args.arm)
    root = Path(candidate['root'])
    predecessor = verify_predecessor(candidate)
    future, new_names = verify_sources(candidate, args.inputs, inventory, args.phase == 'full')
    gate = reviewed_gate(candidate, args.inputs)
    if args.phase == 'real64':
        assert not root.exists()
        gpu_rows = resource_admission(candidate)
        root.mkdir()
        code = root / 'code'
        code.mkdir()
        for name in future:
            source = args.inputs / name if name in new_names else PREDECESSOR / 'code' / name
            with (code / name).open('xb') as stream: stream.write(source.read_bytes())
        for source, destination in [('EXPERIMENT_AUDIT.json', 'source_review.json'),
                                    ('EXPERIMENT_AUDIT.md', 'source_review.md'),
                                    (ARM_FILES[args.arm], 'candidate_gate.json')]:
            with (root / destination).open('xb') as stream: stream.write((args.inputs / source).read_bytes())
        write_new(root / 'gate.json', gate)
        verify_sources(gate, args.inputs, inventory, True)
        command = [gate['python'], '-B', '-u', str(code / 'check_m125_region_real64.py'),
                   '--gate', str(root / 'gate.json'), '--output', str(root / 'real64')]
    else:
        assert read(root / 'gate.json') == gate
        assert sha(root / 'source_review.json') == gate['source_review_report_sha256']
        assert sha(root / 'candidate_gate.json') == gate['source_review_candidate_sha256']
        assert not (root / 'actual_launch.json').exists() and not (root / 'passes').exists()
        assert (root / 'real64.exit').read_text().strip() == '0'
        assert read(root / 'real64.receipt.json')['exit_code'] == 0
        sanity = read(root / 'real64/result.json')
        assert sanity['status'] == 'complete_M125_region_real64_interface_witness'
        assert sanity['region_arm'] == args.arm and sanity['gate_sha256'] == sha(root / 'gate.json')
        assert sanity['source_sha256'] == sha(Path(gate['code_root']) / 'check_m125_region_real64.py')
        assert sanity['track_calls'] == 64 and sanity['AB_optimizer_steps'] == sanity['C_optimizer_steps'] == 2
        for key in ['initial_parameters_equal_common_warm', 'initial_optimizer_empty',
                    'frozen_before_after_exact', 'saved_model_and_optimizer_roundtrip_exact']:
            assert sanity[key], key
        assert not sanity['formal_training_resume_uses_this_sanity']
        assert sanity['warm_sha256'] == sha(gate['warm_final'])
        assert len(sanity['batches']) == 2
        assert all(math.isfinite(value) and value > 0 for value in sanity['batches'][-1]['region_gradient_norms'].values())
        gpu_rows = resource_admission(candidate)
        command = [gate['python'], '-B', '-u', str(Path(gate['code_root']) / 'm125_region_fixed_process_supervisor.py'), str(root)]
    launch(root, gate, args.phase, command, predecessor, gpu_rows)


if __name__ == '__main__':
    main()
