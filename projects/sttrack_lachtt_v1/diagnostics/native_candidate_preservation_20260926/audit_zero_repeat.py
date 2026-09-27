"""Compare two completed weight-zero M89 runs before their first sequence ends."""
import hashlib
import json
from pathlib import Path


ROOT = Path('/root/autodl-tmp/sttrack_native_candidate_preservation_20260926')
OLD = Path('/root/autodl-tmp/sttrack_full152_paired_20260925/M82')
FIELDS = ('bbox', 'previous_bbox', 'best_score', 'resize_factor',
          'template_write', 'label', 'native_iou', 'preservation_kl',
          'competition_loss', 'competition_positive_index', 'competition_negatives')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(folder):
    result = json.loads((folder / 'result.json').read_text())
    trace = folder / 'sampled_state_trace.jsonl'
    assert sha(trace) == result['sampled_trace_sha256']
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    rows = {row['frame_index']: row for row in rows
            if row['sequence'] == 'cube04_indoor' and row['frame_index'] <= 750}
    return result, rows


def compare(left, right):
    frames = sorted(set(left) & set(right))
    differing = {field: [frame for frame in frames
                         if left[frame].get(field) != right[frame].get(field)]
                 for field in FIELDS}
    return dict(common_sampled_frames=len(frames),
                first_difference={key: values[0] for key, values in differing.items() if values},
                difference_counts={key: len(values) for key, values in differing.items()},
                selected={str(frame): {
                    'preflight_bbox': left[frame]['bbox'],
                    'formal_bbox': right[frame]['bbox'],
                    'preflight_score': left[frame]['best_score'],
                    'formal_score': right[frame]['best_score'],
                    'preflight_resize': left[frame]['resize_factor'],
                    'formal_resize': right[frame]['resize_factor'],
                    'preflight_template_write': left[frame]['template_write'],
                    'formal_template_write': right[frame]['template_write']}
                    for frame in (1, 50, 200, 350, 550)})


def main():
    preflight_path = ROOT / 'preflight/control'
    formal_path = ROOT / 'training/control'
    old_path = OLD / 'training/category'
    preflight, a = load_run(preflight_path)
    formal, b = load_run(formal_path)
    old, c = load_run(old_path)
    assert preflight['candidate_weight'] == formal['candidate_weight'] == 0
    for key in ('initial_adapter_state_sha256', 'base_state_before_sha256',
                'training_spec_sha256', 'preservation_loss_sha256',
                'window_loss_sha256'):
        assert preflight[key] == formal[key] == old[key], key
    assert preflight['total_track_calls'] == 768
    assert formal['total_track_calls'] == old['total_track_calls'] == 219802
    assert json.loads((ROOT / 'preflight_launch.json').read_text())['gpu_indices'][0] == 0
    assert json.loads((ROOT / 'training_launch.json').read_text())['gpu_indices'][0] == 0
    assert sorted(set(a) & set(b)) == sorted(set(a) & set(c))
    report = dict(status='completed_read_only_audit',
                  evidence='M89 preflight/control and training/control use the same sealed script, '
                           'weight zero, initial states, training spec and GPU0; both share the '
                           'first 768 tracking calls of cube04_indoor.',
                  trace_sha256=dict(preflight=sha(preflight_path / 'sampled_state_trace.jsonl'),
                                    formal=sha(formal_path / 'sampled_state_trace.jsonl'),
                                    old_m82=sha(old_path / 'sampled_state_trace.jsonl')),
                  preflight_vs_formal=compare(a, b),
                  preflight_vs_old_m82=compare(a, c),
                  limitation='Saved samples do not identify the first numerical divergence. '
                             'The preflight stops after frame 768 and does not prove that either '
                             'short trajectory would reproduce a full 152-sequence model. '
                             'The source of numerical differences is not established.')
    destination = ROOT / 'zero_repeat_audit.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(output=str(destination), sha256=sha(destination),
                          first_bbox=report['preflight_vs_formal']['first_difference'].get('bbox'),
                          first_template=report['preflight_vs_formal']['first_difference'].get('template_write'),
                          sampled_frames=report['preflight_vs_formal']['common_sampled_frames'])))


if __name__ == '__main__':
    main()
