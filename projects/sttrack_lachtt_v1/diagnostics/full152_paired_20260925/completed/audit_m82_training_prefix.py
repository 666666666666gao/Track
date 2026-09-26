"""Read-only audit of the old M82 and Full152 training prefix."""
import csv
import hashlib
import json
from pathlib import Path

import torch


OLD = Path('/root/autodl-tmp/sttrack_m82_native_preservation_20260909')
FULL = Path('/root/autodl-tmp/sttrack_full152_paired_20260925/M82')
OUTPUT = Path('/root/autodl-tmp/sttrack_full152_evaluation_20260925/posthoc')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    a, b = read(OLD / 'training_spec.json'), read(FULL / 'training_spec.json')
    old_bank = Path(a['banks']['fit']['category']['path'])
    full_bank = Path(b['banks']['fit']['category']['path'])
    assert sha(old_bank) == a['banks']['fit']['category']['sha256']
    assert sha(full_bank) == b['banks']['fit']['category']['sha256']
    ta, tb = torch.load(old_bank, map_location='cpu'), torch.load(full_bank, map_location='cpu')
    fields = ['seed', 'learning_rate', 'weight_decay', 'gradient_clip', 'gradient_accumulation_frames',
              'native_checkpoint_sha256', 'integration_sha256', 'causal_script_sha256',
              'support_loss_sha256', 'window_loss_sha256', 'preservation_loss_sha256', 'preservation_weight']
    checks = {key: a[key] == b[key] for key in fields}
    checks['category_initial_checkpoint_sha256'] = a['initial_checkpoint_sha256']['category'] == b['initial_checkpoint_sha256']['category']
    checks['all_130_tokens_and_masks'] = all(
        torch.equal(ta['tokens'][i], tb['tokens'][tb['sequences'].index(name)]) and
        torch.equal(ta['mask'][i], tb['mask'][tb['sequences'].index(name)])
        for i, name in enumerate(ta['sequences']))
    checks['empty_text_tensor'] = torch.equal(ta['empty'], tb['empty'])
    old_output, full_output = OLD / 'training/category', FULL / 'training/category'
    ra, rb = read(old_output / 'result.json'), read(full_output / 'result.json')
    for key in ['initial_adapter_state_sha256', 'base_state_before_sha256']:
        checks[key] = ra[key] == rb[key]
    la = [json.loads(line) for line in (old_output / 'sequence_log.jsonl').read_text().splitlines()]
    lb = [json.loads(line) for line in (full_output / 'sequence_log.jsonl').read_text().splitlines()]
    assert len(la) == 130 and len(lb) == 152
    checks['prefix_sequence_order'] = [row['sequence'] for row in la] == [row['sequence'] for row in lb[:130]]
    assert all(checks.values()), checks
    assert sha(old_output / 'sequence_log.jsonl') == ra['sequence_log_sha256']
    assert sha(full_output / 'sequence_log.jsonl') == rb['sequence_log_sha256']
    observed = ['track_calls', 'supervised_frames', 'label_counts', 'mean_training_loss', 'template_writes',
                'maximum_preclip_gradient_norm', 'total_optimizer_steps', 'total_track_calls',
                'cumulative_native_eligible_frames', 'cumulative_preservation_kl_sum']
    different = {key: sum(x[key] != y[key] for x, y in zip(la, lb)) for key in observed}
    rows = [dict(sequence=x['sequence'],
                 old_inside=x['label_counts'].get('centre_inside', 0),
                 full_inside=y['label_counts'].get('centre_inside', 0),
                 old_outside=x['label_counts'].get('centre_outside', 0),
                 full_outside=y['label_counts'].get('centre_outside', 0),
                 old_template_writes=x['template_writes'], full_template_writes=y['template_writes'],
                 old_training_loss=x['mean_training_loss'], full_training_loss=y['mean_training_loss'])
            for x, y in zip(la, lb)]
    paths = [OLD / 'training_spec.json', FULL / 'training_spec.json', old_bank, full_bank,
             old_output / 'result.json', full_output / 'result.json',
             old_output / 'sequence_log.jsonl', full_output / 'sequence_log.jsonl',
             Path(__file__)]
    result = dict(status='complete', checks=checks,
                  source_sha256={str(p): sha(p) for p in paths},
                  prefix_different_sequence_counts=different,
                  old_final_log=la[-1], full_prefix_boundary_log=lb[129],
                  appended_sequences=[row['sequence'] for row in lb[130:]],
                  appended_track_calls=lb[-1]['total_track_calls'] - lb[129]['total_track_calls'],
                  appended_optimizer_steps=lb[-1]['total_optimizer_steps'] - lb[129]['total_optimizer_steps'],
                  interpretation='The first 130 sequences retain the same order, inputs, initialized tensors and training objectives, but observed training paths differ before the appended 22 sequences. The final external difference cannot be attributed solely to those 22 sequences.',
                  limitation='No checkpoint was preserved at Full152 sequence 130. Matching inputs and seed do not establish bitwise execution identity; this audit does not identify the source of the numerical or trajectory differences.')
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / 'M82_training_prefix_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    with (OUTPUT / 'M82_training_prefix_comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: result[key] for key in ['status', 'checks', 'prefix_different_sequence_counts',
                     'appended_track_calls', 'appended_optimizer_steps', 'interpretation', 'limitation']}, indent=2))


if __name__ == '__main__':
    main()
