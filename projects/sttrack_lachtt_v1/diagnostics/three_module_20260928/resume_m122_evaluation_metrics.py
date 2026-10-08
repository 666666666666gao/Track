"""Resume a sealed prediction suite after its recorded missing-vot CPU analysis failure."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from prepare_m122_evaluation_suite import checked_sources, read, write
from bind_m122_official_initializations import sha

ST_PYTHON = '/root/autodl-tmp/envs/sttrack/bin/python'
METRIC_PYTHON = '/root/miniconda3/envs/mplt/bin/python'


def main(args):
    checked_sources(args.source_gate)
    assert not Path('/proc/28100/cmdline').exists(), 'Original evaluation controller must have exited.'
    original = read(args.original_control / 'launches.json')
    assert len(original) == 8 and all(row['exit'] == 0 for row in original[:6])
    assert all(row['stage'] == 'depthtrack_test_ope_analyze' and row['exit'] == 1 for row in original[6:])
    for row in original[6:]:
        log = (args.original_control / (row['name'] + '.log')).read_text()
        assert "ModuleNotFoundError: No module named 'vot'" in log
    selection = read(args.output / 'selection.json')
    for row in selection['models']:
        bundle_path = args.output / row['name'] / 'bundle.json'
        assert sha(bundle_path) == row['bundle_sha256']
        bundle = read(bundle_path)
        assert sha(bundle['final_path']) == row['final_sha256']
        for dataset, sequences, frames in [('depthtrack_test', 50, 76373), ('cdtb', 80, 101956)]:
            plan = read(row['plans'][dataset]['path'])
            receipt = read(Path(plan['output']) / 'receipt.json')
            assert receipt['status'] == 'complete_M122_official_OPE_predictions'
            assert len(receipt['sequences']) == sequences and receipt['frames'] == frames
            assert not (Path(plan['output']) / 'metrics.json').exists()
        assert not (args.output / row['name'] / 'vot/run/launch.json').exists()
    args.control.mkdir(exist_ok=False)
    write(args.control / 'resume_inputs.json', dict(status='M122_recorded_analysis_failure_only_resume',
          original_launches_sha256=sha(args.original_control / 'launches.json'),
          selection_sha256=sha(args.output / 'selection.json'), source_sha256=sha(__file__),
          neural_training_restarted=False, OPE_predictions_repeated=False, environment_rebuilt=False,
          original_sources_modified=False, external_optimizer_steps=0))
    source = Path(__file__).parent
    launches = []

    def run_phase(label, jobs):
        children = []
        for name, command, device in jobs:
            env = dict(os.environ, PYTHONPATH='/home/SUTrack_RGBD_L', OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
            if device is not None:
                env['CUDA_VISIBLE_DEVICES'] = str(device)
            with (args.control / (name + '.log')).open('w') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            row = dict(stage=label, name=name, pid=child.pid, gpu=device, args=command, started=time.time())
            children.append((child, row))
            launches.append(row)
        write(args.control / 'launches.json', launches)
        for child, row in children:
            row.update(exit=child.wait(), finished=time.time())
            (args.control / (row['name'] + '.exit')).write_text(str(row['exit']) + '\n')
        write(args.control / 'launches.json', launches)
        assert all(row['exit'] == 0 for _, row in children), 'Read the actual failed stage; no automatic retry.'

    for dataset in ['depthtrack_test', 'cdtb']:
        jobs = [(row['name'] + '_' + dataset + '_ope_analyze',
                 [METRIC_PYTHON, '-u', str(source / 'run_m122_official.py'), '--plan',
                  row['plans'][dataset]['path'], '--mode', 'ope_analyze'], None)
                for row in selection['models']]
        run_phase(dataset + '_ope_analyze', jobs)
    for row in selection['models']:
        target = args.output / row['name']
        vot = target / 'vot'
        run_phase('vot_track', [(row['name'] + '_vot',
                  [ST_PYTHON, '-u', str(source / 'run_m122_vot_shards.py'), '--root', str(vot / 'run')], None)])
        (target / 'vot.exit').write_text('0\n')
        manifest = read(vot / 'run/shard_manifest.json')
        run_phase('vot_official_analysis', [(row['name'] + '_analysis',
                  [METRIC_PYTHON, '-m', 'vot', 'analysis', '--workspace', str(vot / 'run/master'),
                   '--format', 'json', '--name', row['name'] + '_full127', manifest['tracker']], None)])
    run_phase('vot_metric_seal', [('vot_metric_seal', [METRIC_PYTHON, '-u', str(source / 'analyze_m122_vot.py'),
              '--root', str(args.output)], None)])
    run_phase('collect', [('collect', [ST_PYTHON, '-u', str(source / 'collect_m122_official_results.py'),
              '--root', str(args.output)], None)])
    result = read(args.output / 'all_results.json')
    assert result['status'] == 'six_M122_full_evaluations_complete'
    write(args.control / 'result.json', dict(status='complete_M122_official_full_pair_suite_after_analysis_resume',
          launches=launches, any_joint_pass=result['any_joint_pass'], neural_training_restarted=False,
          OPE_predictions_repeated=False, original_sources_modified=False, external_optimizer_steps=0))
    (args.control / 'driver.exit').write_text('0\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['original-control', 'source-gate', 'output', 'control']:
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
