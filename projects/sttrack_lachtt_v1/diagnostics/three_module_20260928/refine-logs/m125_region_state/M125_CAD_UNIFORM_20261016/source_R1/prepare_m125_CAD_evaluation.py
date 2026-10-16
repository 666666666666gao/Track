"""Bind one immutable R0 final and one profile to three independent full runs."""
import argparse
import json
import shutil
from pathlib import Path
from bind_m122_official_initializations import sha
from m125_R0_fixed_official_runtime import checked_plan as checked_R0_plan
from m125_CAD_official_runtime import checked_plan,make_bundle
from cad_template_profiles import PRIMARY_PROFILES
from prepare_m122_evaluation_suite import read,write


def main(args):
    gate=read(args.source_gate)
    reference_plan,base=checked_R0_plan(args.reference_evaluation/'depthtrack_test/plan.json')
    bundle=make_bundle(reference_plan['bundle_path'],args.source_gate,args.profile)
    args.output.mkdir()
    write(args.output/'bundle.json',bundle)
    plans = {}
    for dataset in ['depthtrack_test', 'cdtb', 'vot']:
        path = args.reference_evaluation / dataset / 'plan.json'
        old, old_bundle = checked_R0_plan(path)
        assert old_bundle == base
        folder = args.output / dataset; folder.mkdir()
        plan = dict(old, reference_R0_plan_path=str(path), reference_R0_plan_sha256=sha(path),
                    bundle_path=str(args.output / 'bundle.json'), bundle_sha256=sha(args.output / 'bundle.json'),
                    inference_profile_id=args.profile,profile_events_root=str(folder/'profile_events'))
        (folder/'profile_events').mkdir()
        if dataset != 'vot':
            plan['output'] = str(folder / 'predictions')
        write(folder / 'plan.json', plan)
        checked_plan(folder / 'plan.json')
        plans[dataset] = dict(path=str(folder / 'plan.json'), sha256=sha(folder / 'plan.json'))
    source = Path(__file__).parent
    vot = args.output / 'vot'; run = vot / 'run'; run.mkdir()
    original = args.reference_evaluation / 'vot/run'
    manifest = read(original / 'shard_manifest.json')
    assert manifest['schema'] == 'M122_full127_frozen_shards_v1'
    assert len(manifest['sequences']) == 127 and manifest['total_anchor_count'] == 1765
    tracker = 'sttrack_m125_CAD_'+args.profile
    wrapper = vot / 'selected_m125_CAD_vot.py'
    wrapper.write_text('import sys\nsys.path[:0]=[' + repr(str(source)) + ',' + repr(gate['base_code_root']) + ']\n'
                       'from run_m125_CAD_official import vot_track\nvot_track('
                       + repr(str(vot / 'plan.json')) + ')\n')
    shards = []; sealed = [args.output / 'bundle.json', vot / 'plan.json', wrapper]
    old_execution = read(args.reference_evaluation / 'vot/execution.json')
    for row in manifest['shards']:
        old = Path(row['root']); target = run / ('shard-%02d' % row['index']); gpu = row['index'] % 2
        shutil.copytree(old / 'sequences', target / 'sequences', symlinks=True)
        shutil.copy2(old / 'config.yaml', target / 'config.yaml')
        ini = (f'[{tracker}]\nlabel = M125 CAD uniform {args.profile}, same R0 fixed-six final\nprotocol = traxpython\n'
               f'command = selected_m125_CAD_vot\npaths = {vot}\n'
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
    write(run / 'shard_manifest.json', dict(manifest, tracker=tracker, shards=shards,
          native_source='Same frozen R0 final with one sealed CAD-inspired template profile for all datasets',
          source_manifest_path=str(original / 'shard_manifest.json'), source_manifest_sha256=sha(original / 'shard_manifest.json')))
    sealed.append(run / 'shard_manifest.json')
    write(vot / 'execution.json', dict(status='frozen_before_M122_VOT_tracking',
          source_sha256={str(path): sha(path) for path in sealed}, anchors=1765, sequences=127,
          result_files=5295, workers=2, reused_prediction_anchors=0,
          original_manifest_sha256=sha(original / 'shard_manifest.json'),
          metadata_source_execution_sha256=sha(args.reference_evaluation / 'vot/execution.json')))
    plans['vot']['run'] = str(run)
    write(args.output / 'selection.json', dict(status='M125_CAD_uniform_profile_before_metrics',
          final_sha256=bundle['final_sha256'], bundle_sha256=sha(args.output / 'bundle.json'),
          training_result_sha256=sha(args.training / 'result.json'), plans=plans,
          bank_sha256=plan['bank_sha256'], binding_sha256=plan['binding_sha256'],
          models=[dict(name='CAD_'+args.profile+'_same_R0_fixed6', final_sha256=bundle['final_sha256'],
                       bundle_sha256=sha(args.output / 'bundle.json'), plans=plans)],
          external_metric_checkpoint_selection=False, Test50_CDTB80_used_for_checkpoint_selection=False,
          fixed_final_pass=6, external_optimizer_steps=0,inference_profile_id=args.profile,inference_profile=bundle['inference_profile'],
          one_profile_for_all_datasets=True,per_dataset_profile_selection=False,new_training_or_weight_selection=False,
          inference_strategy_engineering_comparison=True))
    print(json.dumps(dict(status='prepared_M125_CAD_uniform_profile_three_official_plans', plans=plans)), flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['reference-evaluation','source-gate','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--profile',choices=PRIMARY_PROFILES,required=True)
    main(parser.parse_args())
