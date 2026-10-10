"""Reuse the sealed P1 datasets/text/protocol; bind a new single composite final."""
import argparse
import json
import shutil
from pathlib import Path

from bind_m122_official_initializations import sha
from m122_official_runtime import checked_plan as checked_base_plan
from m123_joint_official_runtime import checked_plan
from prepare_m122_evaluation_suite import read, write


def main(args):
    gate = read(args.source_gate)
    assert gate['status'] == 'reviewed_M123_ABC_joint_full152_source_gate'
    for name, digest in gate['source_sha256'].items():
        assert sha(name) == digest, name
    trained = read(args.training / 'result.json')
    assert trained['status'] == 'complete_M123_ABC_joint_full152_training'
    assert trained['epochs'] == 3 and trained['training_sequences'] == 152
    assert trained['sequence_runs'] == 456 and trained['track_calls'] == 659406
    assert trained['no_development_split'] and trained['no_cached_feature_fit']
    assert trained['final_sha256'] == sha(args.training / 'final.pt')
    base_path = args.base_evaluation / 'depthtrack_test/plan.json'
    _, base = checked_base_plan(base_path)
    assert base['precision_weight'] == 1
    args.output.mkdir()
    bundle = dict(base, schema='M123_ABC_joint_Train152_official_v1',
        base_final_sha256=base['final_sha256'], final_path=str(args.training / 'final.pt'),
        final_sha256=trained['final_sha256'], training_result_path=str(args.training / 'result.json'),
        training_result_sha256=sha(args.training / 'result.json'), source_sha256=gate['source_sha256'],
        C_arm='current', C_threshold=.5, complete_Train152_joint_training=True,
        no_development_split=True, C_target='current selected-frame real GT IoU',
        source_gate_path=str(args.source_gate), source_gate_sha256=sha(args.source_gate))
    write(args.output / 'bundle.json', bundle)
    plans = {}
    for dataset in ['depthtrack_test', 'cdtb', 'vot']:
        path = args.base_evaluation / dataset / 'plan.json'
        old, old_bundle = checked_base_plan(path)
        assert old_bundle == base
        folder = args.output / dataset; folder.mkdir()
        plan = dict(old, base_plan_path=str(path), base_plan_sha256=sha(path),
                    bundle_path=str(args.output / 'bundle.json'), bundle_sha256=sha(args.output / 'bundle.json'))
        if dataset != 'vot':
            plan['output'] = str(folder / 'predictions')
        write(folder / 'plan.json', plan)
        checked_plan(folder / 'plan.json')
        plans[dataset] = dict(path=str(folder / 'plan.json'), sha256=sha(folder / 'plan.json'))
    source = Path(__file__).parent
    vot = args.output / 'vot'; run = vot / 'run'; run.mkdir()
    original = args.base_evaluation / 'vot/run'
    manifest = read(original / 'shard_manifest.json')
    assert manifest['schema'] == 'M122_full127_frozen_shards_v1'
    assert len(manifest['sequences']) == 127 and manifest['total_anchor_count'] == 1765
    tracker = 'sttrack_m123_ABC_joint_full152'
    wrapper = vot / 'selected_m123_joint_vot.py'
    wrapper.write_text('import sys\nsys.path[:0]=[' + repr(str(source)) + ',' + repr(gate['base_code_root']) + ']\n'
                       'from run_m123_joint_official import vot_track\nvot_track('
                       + repr(str(vot / 'plan.json')) + ')\n')
    shards = []; sealed = [args.output / 'bundle.json', vot / 'plan.json', wrapper]
    old_execution = read(args.base_evaluation / 'vot/execution.json')
    for row in manifest['shards']:
        old = Path(row['root']); target = run / ('shard-%02d' % row['index']); gpu = row['index'] % 2
        shutil.copytree(old / 'sequences', target / 'sequences', symlinks=True)
        shutil.copy2(old / 'config.yaml', target / 'config.yaml')
        ini = (f'[{tracker}]\nlabel = M123 A+B+C joint full Train152\nprotocol = traxpython\n'
               f'command = selected_m123_joint_vot\npaths = {vot}\n'
               f'python = /root/autodl-tmp/envs/sttrack/bin/python\nenv_CUDA_VISIBLE_DEVICES = {gpu}\n'
               f'env_PYTHONPATH = {vot}\nenv_TOKENIZERS_PARALLELISM = false\n'
               'env_PYTHONDONTWRITEBYTECODE = 1\ntimeout = 600\nrestart = false\n')
        (target / 'trackers.ini').write_text(ini)
        shards.append(dict(row, root=str(target), gpu=gpu, trackers_sha256=sha(target / 'trackers.ini')))
        sealed.extend(target / name for name in ['config.yaml', 'trackers.ini', 'sequences/list.txt'])
        for name, digest in old_execution['source_sha256'].items():
            path = Path(name)
            if old / 'sequences' in path.parents:
                copied = target / path.relative_to(old)
                assert sha(copied) == digest
                sealed.append(copied)
    for info in manifest['source'].values():
        path = Path(info['root']) / 'anchor.value'
        assert sha(path) == info['anchor_sha256']; sealed.append(path)
    write(run / 'shard_manifest.json', dict(manifest, tracker=tracker, shards=shards))
    sealed.append(run / 'shard_manifest.json')
    write(vot / 'execution.json', dict(status='frozen_before_M122_VOT_tracking',
          source_sha256={str(path): sha(path) for path in sealed}, anchors=1765, sequences=127,
          result_files=5295, workers=2, reused_prediction_anchors=0,
          original_manifest_sha256=sha(original / 'shard_manifest.json'),
          metadata_source_execution_sha256=sha(args.base_evaluation / 'vot/execution.json')))
    plans['vot']['run'] = str(run)
    write(args.output / 'selection.json', dict(status='M123_ABC_joint_composite_final_fixed_before_metrics',
          final_sha256=bundle['final_sha256'], bundle_sha256=sha(args.output / 'bundle.json'),
          training_result_sha256=sha(args.training / 'result.json'), plans=plans,
          bank_sha256=plan['bank_sha256'], binding_sha256=plan['binding_sha256'],
          models=[dict(name='ABC_joint_full152', final_sha256=bundle['final_sha256'],
                       bundle_sha256=sha(args.output / 'bundle.json'), plans=plans)],
          external_metric_checkpoint_selection=False, external_optimizer_steps=0))
    print(json.dumps(dict(status='prepared_M123_ABC_joint_three_official_plans', plans=plans)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['training', 'base-evaluation', 'source-gate', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
