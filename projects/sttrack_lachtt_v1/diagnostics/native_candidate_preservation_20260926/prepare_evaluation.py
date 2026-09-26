"""Bind completed M89 final weights to the sealed Full152 evaluation inputs."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

TRAIN = Path('/root/autodl-tmp/sttrack_native_candidate_preservation_20260926')
ROOT = Path('/root/autodl-tmp/sttrack_m89_evaluation_20260926')
OLD = Path('/root/autodl-tmp/sttrack_full152_evaluation_20260925')
SPEC_SHA = 'c06cf35ea16f545a5cbff87708842bd7f3fead7675837d708f496579bf9d2c08'
MODELS = ['control', 'candidate']
MANIFEST = Path('/root/autodl-tmp/sttrack_default_full127_v1_20260905/run/shard_manifest.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def audit_inputs():
    inventory = read(TRAIN / 'evaluation_inputs.json')
    for dataset, entry in inventory.items():
        assert sha(entry['plan']) == entry['sha256'], dataset
        plan = read(entry['plan'])
        assert plan == entry['contents'], dataset
        for key in ['bundle', 'text_bank']:
            assert sha(plan[key + '_path']) == plan[key + '_sha256'], (dataset, key)
        if dataset != 'vot':
            assert sha(plan['cases_path']) == plan['cases_sha256'], dataset
            assert sha(plan['metric_source']) == plan['metric_source_sha256'], dataset
    bundle = read(OLD / 'M82/bundle.json')
    for name, digest in bundle['source_sha256'].items():
        assert sha(Path(bundle['repository']) / name) == digest, name
    for name, digest in bundle['interface_sha256'].items():
        assert sha(OLD / 'interface' / name) == digest, name
    for key in ['base_checkpoint', 'text_protocol_path']:
        digest_key = 'text_protocol_sha256' if key == 'text_protocol_path' else key + '_sha256'
        assert sha(bundle[key]) == bundle[digest_key], key
    return inventory, bundle


def bind():
    import torch
    inventory, old_bundle = audit_inputs()
    assert (TRAIN / 'training.exit').read_text().strip() == '0'
    assert sha(TRAIN / 'experiment_spec.json') == SPEC_SHA
    spec = read(TRAIN / 'experiment_spec.json')
    for name, digest in spec['source_sha256'].items():
        assert sha(TRAIN / name) == digest, name
    models = {}
    initializations = set()
    for arm, weight in [('control', 0), ('candidate', 1)]:
        folder = TRAIN / 'training' / arm
        result = read(folder / 'result.json')
        final = folder / 'final.pth'
        saved = torch.load(final, map_location='cpu')
        assert result['status'] == 'one_full_causal_fit_pass_complete'
        assert result['sequences'] == saved['completed_sequences'] == 152
        assert result['total_track_calls'] == saved['frame_count'] == 219802
        assert result['optimizer_steps'] == saved['optimizer_steps'] == 6807
        assert result['base_parameters_and_buffers_unchanged'] is True
        assert saved['status'] == 'complete' and saved['seed'] == 2027
        assert result['candidate_weight'] == saved['candidate_weight'] == weight
        assert result['experiment_spec_sha256'] == saved['experiment_spec_sha256'] == SPEC_SHA
        assert result['training_spec_sha256'] == saved['training_spec_sha256'] == spec['base_training_spec_sha256']
        assert saved['base_checkpoint_sha256'] == old_bundle['base_checkpoint_sha256']
        for key in ['architecture', 'arm', 'null_support', 'support_loss_weight', 'use_text']:
            assert saved[key] == old_bundle[key], key
        assert sha(final) == result['final_checkpoint_sha256']
        assert sha(folder / 'sequence_log.jsonl') == result['sequence_log_sha256']
        assert sha(folder / 'sampled_state_trace.jsonl') == result['sampled_trace_sha256']
        initializations.add(result['initial_adapter_state_sha256'])
        models[arm] = dict(final=str(final), checkpoint_sha256=sha(final), optimizer_steps=6807,
                           result_sha256=sha(folder / 'result.json'), weight=weight)
    assert initializations == {'56d03acff2972871788dd51f0eff40d0b6d888aad79fe74db4c44cdfb276ca42'}
    ROOT.mkdir(exist_ok=False)
    for arm, model in models.items():
        target = ROOT / arm
        target.mkdir()
        bundle = dict(old_bundle, adapter_checkpoint=model['final'],
                      adapter_checkpoint_sha256=model['checkpoint_sha256'],
                      experiment_spec_sha256=SPEC_SHA, candidate_weight=model['weight'])
        write(target / 'bundle.json', bundle)
        model['bundle_sha256'] = sha(target / 'bundle.json')
        for dataset, entry in inventory.items():
            folder = target / dataset
            folder.mkdir()
            plan = dict(entry['contents'], bundle_path=str(target / 'bundle.json'),
                        bundle_sha256=sha(target / 'bundle.json'))
            if dataset != 'vot':
                plan['output'] = str(folder / 'predictions')
            write(folder / 'plan.json', plan)
    write(ROOT / 'binding.json', dict(status='finals_bound_not_evaluated', models=models,
          experiment_spec_sha256=SPEC_SHA, evaluation_inputs_sha256=sha(TRAIN / 'evaluation_inputs.json'),
          binder_sha256=sha(__file__), checkpoint_selection_from_external_metrics=False))
    write(ROOT / 'selection.json', dict(models=models, checkpoint_selection_from_external_metrics=False))


def checked(arm):
    assert arm in MODELS
    binding = read(ROOT / 'binding.json')
    assert sha(ROOT / arm / 'bundle.json') == binding['models'][arm]['bundle_sha256']
    assert binding['binder_sha256'] == sha(__file__)
    assert binding['evaluation_inputs_sha256'] == sha(TRAIN / 'evaluation_inputs.json')
    bundle = read(ROOT / arm / 'bundle.json')
    assert bundle['adapter_checkpoint_sha256'] == binding['models'][arm]['checkpoint_sha256']
    assert sha(bundle['adapter_checkpoint']) == bundle['adapter_checkpoint_sha256']
    assert bundle['candidate_weight'] == binding['models'][arm]['weight']
    assert bundle['experiment_spec_sha256'] == SPEC_SHA
    return bundle


def bind_vot(arm):
    checked(arm)
    assert sha(MANIFEST) == '8e76256f1c7c135a65a1b262506356769b59557db60eb83bc21ef1392890dd01'
    frozen = read(MANIFEST)
    target = ROOT / arm / 'vot'
    plan = read(target / 'plan.json')
    assert sha(plan['text_bank_path']) == plan['text_bank_sha256']
    tracker = 'sttrack_m89_' + arm + '_full127'
    run = target / 'run'
    run.mkdir()
    wrapper = target / 'selected_semantic_vot.py'
    wrapper.write_text('import sys\nsys.path.insert(0,' + repr(str(OLD / 'interface')) + ')\n'
                       'from run_semantic_vot import run\nrun(' + repr(str(target / 'plan.json')) + ')\n')
    shards = []
    for shard in frozen['shards']:
        src = Path(shard['root'])
        dest = run / ('shard-%02d' % shard['index'])
        gpu = shard['index'] % 2
        assert sha(src / 'config.yaml') == shard['config_sha256']
        assert sha(src / 'sequences/list.txt') == shard['list_sha256']
        shutil.copytree(src / 'sequences', dest / 'sequences', symlinks=True)
        shutil.copyfile(src / 'config.yaml', dest / 'config.yaml')
        (dest / 'trackers.ini').write_text(
            f'[{tracker}]\nlabel = M89 {arm} full127\nprotocol = traxpython\n'
            f'command = selected_semantic_vot\npaths = {target}\n'
            'python = /root/autodl-tmp/envs/sttrack/bin/python\n'
            f'env_CUDA_VISIBLE_DEVICES = {gpu}\nenv_PYTHONPATH = {target}\n'
            'env_TOKENIZERS_PARALLELISM = false\nenv_PYTHONDONTWRITEBYTECODE = 1\n'
            'timeout = 600\nrestart = false\n')
        shards.append(dict(shard, root=str(dest), gpu=gpu, trackers_sha256=sha(dest / 'trackers.ini')))
    write(run / 'shard_manifest.json', dict(frozen, schema='selected_full127_v1',
          tracker=tracker, gpu_count=2, shards=shards))
    files = [ROOT / arm / 'bundle.json', target / 'plan.json', Path(plan['text_bank_path']),
             wrapper, run / 'shard_manifest.json', MANIFEST, OLD / 'run_vot_failure_family_shards.py']
    files += [Path(s['root']) / n for s in shards for n in ['config.yaml', 'trackers.ini', 'sequences/list.txt']]
    metadata = read(OLD / 'vot_inputs/initializations/metadata_sha256.json')
    for path, digest in metadata.items():
        assert sha(path) == digest, path
        files.append(Path(path))
    write(target / 'execution.json', dict(status='frozen_before_tracking', model=arm,
          source_sha256={str(p): sha(p) for p in files}, anchors=1765, sequences=127,
          reused_prediction_anchors=0, planned_frame_positions=1327004, poll_seconds=300,
          workers=len(shards), training_steps_before_external_evaluation=6807))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['audit-inputs', 'bind', 'bind-vot'])
    parser.add_argument('--model', choices=MODELS)
    args = parser.parse_args()
    if args.action == 'bind':
        bind()
    elif args.action == 'bind-vot':
        bind_vot(args.model)
    else:
        audit_inputs()
        print('All sealed evaluation input hashes match; no evaluation launched.')
