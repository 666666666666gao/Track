"""Train all A+B+C through complete Train152, then evaluate the fixed final."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import subprocess
import sys

from bind_m122_official_initializations import sha
from prepare_m122_evaluation_suite import read, write


def run(root, gate, name, python, arguments, gpu=None):
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join([gate['code_root'], gate['base_code_root'], '/home/SUTrack_RGBD_L']),
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       PYTHONOPTIMIZE='0', PYTHONDONTWRITEBYTECODE='1')
    if gpu is not None:
        environment['CUDA_VISIBLE_DEVICES'] = str(gpu)
    else:
        environment['CUDA_VISIBLE_DEVICES'] = '0,1'
    command = [python, '-B', '-u'] + arguments
    row = dict(stage=name, command=command, gpu=gpu, started_at=datetime.now(timezone.utc).isoformat())
    write(root / (name + '.launch.json'), row)
    with (root / (name + '.log')).open('xb') as stream:
        completed = subprocess.run(command, cwd=gate['code_root'], env=environment,
                                   stdout=stream, stderr=subprocess.STDOUT)
    row.update(exit_code=completed.returncode, finished_at=datetime.now(timezone.utc).isoformat())
    (root / (name + '.exit')).write_text(str(completed.returncode) + '\n')
    write(root / (name + '.receipt.json'), row)
    assert completed.returncode == 0, 'Read this original stage failure; no automatic retry.'
    return row


def main():
    assert sys.flags.optimize == 0
    root = Path(sys.argv[1]); gate = read(root / 'gate.json')
    assert gate['status'] == 'reviewed_M123_ABC_joint_full152_source_gate'
    for name, digest in gate['source_sha256'].items():
        assert sha(name) == digest, name
    for name, digest in gate['input_sha256'].items():
        assert sha(name) == digest, name
    code = Path(gate['code_root']); st = gate['python']; metric = gate['metric_python']
    stages = []; fit = root / 'fit'; evaluation = root / 'evaluation'
    arguments = [str(code / 'train_m123_ABC_joint_full152.py')]
    for name in ['spec', 'repository', 'checkpoint', 'clip_weight', 'bank', 'labels', 'warm_final', 'warm_result']:
        arguments += ['--' + name.replace('_', '-'), gate[name]]
    arguments += ['--output', str(fit)]
    stages.append(run(root, gate, 'train_ABC_complete_all152', st, arguments))
    stages.append(run(root, gate, 'prepare', st, [str(code / 'prepare_m123_joint_evaluation.py'),
        '--training', str(fit), '--base-evaluation', gate['base_evaluation'],
        '--source-gate', str(root / 'gate.json'), '--output', str(evaluation)]))
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(run, root, gate, dataset + '_track', st,
                [str(code / 'run_m123_joint_official.py'), '--mode', 'ope_track',
                 '--plan', str(evaluation / dataset / 'plan.json')], gpu)
                for gpu, dataset in enumerate(['depthtrack_test', 'cdtb'])]
        stages.extend(job.result() for job in jobs)
    for dataset in ['depthtrack_test', 'cdtb']:
        stages.append(run(root, gate, dataset + '_analyze', metric,
                      [str(code / 'run_m123_joint_official.py'), '--mode', 'ope_analyze',
                       '--plan', str(evaluation / dataset / 'plan.json')]))
    vot = evaluation / 'vot'
    stages.append(run(root, gate, 'vot_track', st,
                      [str(code / 'run_m123_joint_vot_shards.py'), '--root', str(vot / 'run')]))
    (evaluation / 'vot.exit').write_text('0\n')
    manifest = read(vot / 'run/shard_manifest.json')
    stages.append(run(root, gate, 'vot_official_analysis', metric,
                  ['-m', 'vot', 'analysis', '--workspace', str(vot / 'run/master'), '--format', 'json',
                   '--name', 'ABC_joint_full152_full127', manifest['tracker']]))
    stages.append(run(root, gate, 'vot_seal', metric,
                  [str(code / 'analyze_m123_joint_vot.py'), '--root', str(evaluation)]))
    stages.append(run(root, gate, 'collect', metric,
                  [str(code / 'collect_m123_joint_results.py'), '--root', str(evaluation)]))
    result = read(evaluation / 'all_results.json')
    assert result['status'] == 'three_M123_ABC_joint_full_evaluations_complete'
    write(root / 'result.json', dict(status='complete_M123_ABC_joint_full152_and_three_official_datasets',
          stages=stages, any_joint_pass=result['any_joint_pass'], automatic_retries=0, progress_queries=0,
          formal_nine_metrics=True, fresh_result_audit_complete=False,
          finished_at=datetime.now(timezone.utc).isoformat()))
    (root / 'controller.exit').write_text('0\n')


if __name__ == '__main__':
    main()
