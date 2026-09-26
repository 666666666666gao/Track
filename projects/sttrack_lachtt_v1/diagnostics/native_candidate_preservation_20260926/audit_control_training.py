"""Read-only comparison of old M82 and M89 weight-zero sampled training paths."""
import difflib
import hashlib
import json
from pathlib import Path

OLD = Path('/root/autodl-tmp/sttrack_full152_paired_20260925/M82')
NEW = Path('/root/autodl-tmp/sttrack_native_candidate_preservation_20260926')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    roots = [OLD / 'training/category', NEW / 'training/control']
    results = [read(root / 'result.json') for root in roots]
    paths = [root / 'sampled_state_trace.jsonl' for root in roots]
    for path, result in zip(paths, results):
        assert sha(path) == result['sampled_trace_sha256']
    for field in ['initial_adapter_state_sha256', 'base_state_before_sha256', 'training_spec_sha256',
                  'sequences', 'total_track_calls', 'optimizer_steps', 'window_loss_sha256',
                  'preservation_loss_sha256', 'preservation_weight']:
        assert results[0][field] == results[1][field], field
    rows = [[json.loads(line) for line in path.read_text().splitlines()] for path in paths]
    assert len(rows[0]) == len(rows[1])
    first = {}
    counts = {}
    examples = []
    original_fields = set().union(*(row.keys() for row in rows[0]))
    for old, new in zip(*rows):
        assert (old['sequence'], old['frame_index']) == (new['sequence'], new['frame_index'])
        # Conditional supervision fields genuinely disappear when labels change.
        before = {key: old.get(key, '<not recorded>') for key in original_fields}
        after = {key: new.get(key, '<not recorded>') for key in original_fields}
        differences = sorted(key for key in original_fields if before[key] != after[key])
        for key in differences:
            counts[key] = counts.get(key, 0) + 1
            if key not in first:
                first[key] = dict(sequence=old['sequence'], frame_index=old['frame_index'],
                                  old=before[key], new=after[key],
                                  old_updates=old['optimizer_steps_before_update'],
                                  new_updates=new['optimizer_steps_before_update'])
        if differences and len(examples) < 8:
            examples.append(dict(sequence=old['sequence'], frame_index=old['frame_index'],
                                 differences={key: [before[key], after[key]] for key in differences}))
    source_diff = ''.join(difflib.unified_diff((OLD / 'train_causal.py').read_text().splitlines(True),
                         (NEW / 'train_candidate.py').read_text().splitlines(True),
                         fromfile='old_m82_train', tofile='m89_train'))
    report = dict(status='sampled_training_comparison_complete', rows=len(rows[0]),
                  old_trace_sha256=sha(paths[0]), new_trace_sha256=sha(paths[1]),
                  equal_initial_adapter=True, equal_initial_base=True, equal_training_spec=True,
                  field_difference_counts=counts, first_sampled_difference_by_field=first,
                  early_differing_samples=examples,
                  limitation='Only saved sampled positions are compared; first sampled difference is not necessarily the first actual divergence. Source changes alone do not establish the cause of numerical or recursive differences.')
    with (NEW / 'control_training_audit.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    (NEW / 'control_training_source.diff').write_text(source_diff)
    print(json.dumps(dict(rows=report['rows'], first_sampled=examples[:2], first_bbox=first.get('bbox'),
                         first_resize=first.get('resize_factor'), first_template=first.get('template_write'))))


if __name__ == '__main__':
    main()
