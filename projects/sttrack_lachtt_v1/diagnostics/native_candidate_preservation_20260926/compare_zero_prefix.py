"""Summarize the old/zero/old short-prefix replay without selecting a model."""
import hashlib
import json
from pathlib import Path


ROOT = Path('/root/autodl-tmp/sttrack_native_candidate_preservation_20260926/probe')
FIELDS = ('bbox', 'previous_bbox', 'best_score', 'resize_factor',
          'template_write', 'label', 'loss')


def compare(left, right):
    assert len(left['frames']) == len(right['frames']) == 64
    first = {}
    for a, b in zip(left['frames'], right['frames']):
        assert a['frame_index'] == b['frame_index']
        for field in FIELDS:
            if field not in first and a[field] != b[field]:
                first[field] = a['frame_index']
    return dict(first_frame_difference=first,
                first_gradient_difference=next((a['frame_index'] for a, b in
                    zip(left['updates'], right['updates'])
                    if a['gradient_sha256'] != b['gradient_sha256']), None),
                first_parameter_difference=next((a['frame_index'] for a, b in
                    zip(left['updates'], right['updates'])
                    if a['adapter_sha256'] != b['adapter_sha256']), None))


def main():
    runs = {name: json.loads((ROOT / (name + '.json')).read_text())
            for name in ('old_a', 'zero', 'old_b')}
    assert [runs[name]['mode'] for name in runs] == ['old', 'zero', 'old']
    for field in ('seed', 'sequence', 'calls', 'training_spec_sha256',
                  'input_frames_sha256', 'initial_adapter_sha256'):
        assert len({run[field] for run in runs.values()}) == 1, field
    report = dict(status='short_prefix_comparison_complete',
                  run_sha256={name: hashlib.sha256((ROOT / (name + '.json')).read_bytes()).hexdigest()
                              for name in runs},
                  common=dict(seed=runs['old_a']['seed'], sequence=runs['old_a']['sequence'],
                              calls=64, input_frames_sha256=runs['old_a']['input_frames_sha256'],
                              initial_adapter_sha256=runs['old_a']['initial_adapter_sha256']),
                  old_repeat=compare(runs['old_a'], runs['old_b']),
                  zero_vs_old=compare(runs['old_a'], runs['zero']),
                  limitation='The same raw RGB-D frames are verified, but crop/query/template '
                             'states are recursively predicted. This is a two-update diagnostic, '
                             'not a full-training reproduction or causal performance ablation.')
    destination = ROOT / 'comparison.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
