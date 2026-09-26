"""Bind completed M89 final weights to the sealed Full152 evaluation inputs."""
import argparse
import hashlib
import json
from pathlib import Path

TRAIN = Path('/root/autodl-tmp/sttrack_native_candidate_preservation_20260926')
ROOT = Path('/root/autodl-tmp/sttrack_m89_evaluation_20260926')
OLD = Path('/root/autodl-tmp/sttrack_full152_evaluation_20260925')
SPEC_SHA = 'c06cf35ea16f545a5cbff87708842bd7f3fead7675837d708f496579bf9d2c08'


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
        models[arm] = dict(final=str(final), checkpoint_sha256=sha(final),
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['audit-inputs', 'bind'])
    args = parser.parse_args()
    if args.action == 'bind':
        bind()
    else:
        audit_inputs()
        print('All sealed evaluation input hashes match; no evaluation launched.')
