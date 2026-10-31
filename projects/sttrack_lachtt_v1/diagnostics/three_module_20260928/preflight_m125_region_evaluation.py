"""Check the locked ROI evaluation metadata and CPU entry points without scoring."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root, output):
    gate_path = root / 'gate.json'
    gate = read(gate_path)
    assert gate['status'] == 'reviewed_M125_region_fixed6_source_gate'
    assert gate['region_arm'] == 'roi'
    launch = read(root / 'actual_launch.json')
    assert sha(gate_path) == launch['gate_sha256']
    source_checks = {path: sha(Path(path)) == digest for path, digest in gate['source_sha256'].items()}
    assert all(source_checks.values())
    base = Path(gate['base_evaluation'])
    datasets = {}
    for dataset, count, frames in [('depthtrack_test', 50, 76373), ('cdtb', 80, 101956)]:
        plan_path = base / dataset / 'plan.json'
        plan = read(plan_path)
        assert plan['dataset'] == dataset
        cases_path = Path(plan['cases_path'])
        assert sha(cases_path) == plan['cases_sha256']
        assert sha(Path(plan['metric_source'])) == plan['metric_source_sha256']
        cases = read(cases_path)
        assert len(cases) == len({case['sequence'] for case in cases}) == count
        assert sum(case['frames'] for case in cases) == frames
        gt_hashes = {}
        for case in cases:
            folder = Path(plan['dataset_root']) / case['sequence']
            gt = folder / 'groundtruth.txt'
            gt_hashes[str(gt)] = sha(gt)
            assert gt_hashes[str(gt)] == case['gt_sha256']
            for index in [1, case['frames']]:
                assert (folder / 'color' / ('%08d.jpg' % index)).is_file()
                assert (folder / 'depth' / ('%08d.png' % index)).is_file()
        datasets[dataset] = dict(sequences=count, frames=frames, plan_sha256=sha(plan_path),
                                 cases_sha256=sha(cases_path), gt_sha256=gt_hashes,
                                 first_last_rgb_depth_files_exist=True, interior_frames_checked=False)
    manifest_path = base / 'vot/run/shard_manifest.json'
    manifest = read(manifest_path)
    names = [name for shard in manifest['shards'] for name in shard['expected_trajectories']]
    assert len(manifest['sequences']) == 127
    assert manifest['total_anchor_count'] == len(names) == len(set(names)) == 1765
    code = Path(gate['code_root'])
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONOPTIMIZE='0',
                       PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', PYTHONPATH=os.pathsep.join([str(code), gate['base_code_root'], '/home/SUTrack_RGBD_L']))
    jobs = [
        ('tracking_prepare', gate['python'], [str(code / 'prepare_m125_region_fixed_evaluation.py'), '--help']),
        ('tracking_ope', gate['python'], [str(code / 'run_m125_region_fixed_official.py'), '--help']),
        ('tracking_vot_shards', gate['python'], [str(code / 'run_m123_joint_vot_shards.py'), '--help']),
        ('metric_ope', gate['metric_python'], [str(code / 'run_m125_region_fixed_official.py'), '--help']),
        ('metric_vot_cli', gate['metric_python'], ['-m', 'vot', '--help']),
        ('metric_vot_seal', gate['metric_python'], [str(code / 'analyze_m125_region_fixed_vot.py'), '--help']),
        ('metric_collect', gate['metric_python'], [str(code / 'collect_m125_region_fixed_results.py'), '--help']),
        ('metric_source_import', gate['metric_python'], ['-c',
         'import importlib.util,sys; s=importlib.util.spec_from_file_location("preflight_locked_metric",sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(m.evaluate_depthtrack_results.__name__)',
         plan['metric_source']]),
    ]
    output.mkdir()
    shutil.copyfile(str(Path(__file__)), str(output / Path(__file__).name))
    receipts = []
    for name, python, arguments in jobs:
        command = [python, '-B'] + arguments
        started = datetime.now(timezone.utc).isoformat()
        completed = subprocess.run(command, cwd=str(code), env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output.joinpath(name + '.stdout').write_bytes(completed.stdout)
        output.joinpath(name + '.stderr').write_bytes(completed.stderr)
        receipt = dict(name=name, command=command, started_at_utc=started, exit_code=completed.returncode,
                       stdout_sha256=sha(output / (name + '.stdout')), stderr_sha256=sha(output / (name + '.stderr')))
        receipts.append(receipt)
        output.joinpath('entrypoint_receipts.json').write_text(json.dumps(receipts, indent=2) + '\n')
        assert completed.returncode == 0, name
    summary = dict(status='complete_CPU_entrypoint_and_metadata_preflight',
                   observed_at_utc=datetime.now(timezone.utc).isoformat(), gate_sha256=sha(gate_path),
                   source_checks=source_checks, datasets=datasets, vot_sequences=127, vot_anchors=1765,
                   vot_manifest_sha256=sha(manifest_path), entrypoints=receipts,
                   no_neural_execution=True, no_optimizer_updates=True, no_metric_evaluation=True,
                   no_training_status_query=True, no_training_source_changes=True,
                   readiness_scope='CPU imports/help and sealed OPE metadata only; not proof of final weights, complete image coverage, TraX execution, or benchmark success.',
                   independent_review_status='REVIEW_UNAVAILABLE',
                   independent_review_reason='No callable native Codex reviewer tool in the current client catalog; no reviewer verdict requested or fabricated.')
    output.joinpath('summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(status=summary['status'], entrypoints=len(receipts), datasets=list(datasets))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
