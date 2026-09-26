"""Compare sealed sampled training states; never run a model or change training."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def crop(record):
    x, y, w, h = record['previous_bbox']
    size = math.ceil(math.sqrt(w * h) * 4.)
    assert record['resize_factor'] == 256 / size
    origin = [x + 0.5 * w - size * 0.5, y + 0.5 * h - size * 0.5]
    return [round(origin[0]), round(origin[1]), size], origin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    roots = [
        Path('/root/autodl-tmp/sttrack_m82_native_preservation_20260909/training/category'),
        Path('/root/autodl-tmp/sttrack_full152_paired_20260925/M82/training/category'),
    ]
    sources = {str(Path(__file__)): sha(Path(__file__))}
    records = []
    for root in roots:
        crop_source = root.parent.parent / 'code/lib/train/data/processing_utils.py'
        assert sha(crop_source) == '7aca916f5e5f62e1865fbd322ffb63dc08197c544a241938b6c822d4bfb89bf6'
        sources[str(crop_source)] = sha(crop_source)
        result_path = root / 'result.json'
        trace_path = root / 'sampled_state_trace.jsonl'
        result = json.loads(result_path.read_text())
        assert result['status'] == 'one_full_causal_fit_pass_complete'
        assert sha(trace_path) == result['sampled_trace_sha256']
        sources[str(result_path)] = sha(result_path)
        sources[str(trace_path)] = sha(trace_path)
        records.append([json.loads(line) for line in trace_path.read_text().splitlines()])
    old, new = records
    new_index = {(r['sequence'], r['frame_index']): r for r in new}
    assert len(new_index) == len(new)
    old_keys = [(r['sequence'], r['frame_index']) for r in old]
    assert len(set(old_keys)) == len(old_keys)
    assert old_keys == [(r['sequence'], r['frame_index']) for r in new[:len(old)]]
    assert len({r['sequence'] for r in old}) == 130
    different = Counter()
    first_events = {}
    rows = []
    for a in old:
        b = new_index[(a['sequence'], a['frame_index'])]
        assert a['optimizer_steps_before_update'] == b['optimizer_steps_before_update']
        # Crop-inside supervision has positive-index fields absent in crop-outside records.
        changes = [key for key in sorted(a.keys() | b.keys())
                   if key not in a or key not in b or a[key] != b[key]]
        different.update(changes)
        bbox_error = max(abs(x - y) for x, y in zip(a['bbox'], b['bbox']))
        previous_error = max(abs(x - y) for x, y in zip(a['previous_bbox'], b['previous_bbox']))
        old_crop, old_origin = crop(a)
        new_crop, new_origin = crop(b)
        row = dict(sequence=a['sequence'], frame_index=a['frame_index'],
                   optimizer_steps=a['optimizer_steps_before_update'],
                   bbox_max_abs_error=bbox_error, previous_bbox_max_abs_error=previous_error,
                   score_abs_error=abs(a['best_score'] - b['best_score']),
                   old_score=a['best_score'], new_score=b['best_score'],
                   old_label=a['label'], new_label=b['label'],
                   old_template_write=a['template_write'], new_template_write=b['template_write'],
                   old_crop_x=old_crop[0], old_crop_y=old_crop[1], old_crop_size=old_crop[2],
                   new_crop_x=new_crop[0], new_crop_y=new_crop[1], new_crop_size=new_crop[2],
                   old_unrounded_crop_y=old_origin[1], new_unrounded_crop_y=new_origin[1],
                   integer_crop_changed=old_crop != new_crop,
                   changed_fields='|'.join(changes))
        rows.append(row)
        events = {'any_field': bool(changes), 'score': a['best_score'] != b['best_score'],
                  'label': a['label'] != b['label'],
                  'template_write': a['template_write'] != b['template_write'],
                  'native_eligible': a['native_eligible'] != b['native_eligible'],
                  'integer_crop': old_crop != new_crop}
        for threshold in (0, 0.0001, 0.001, 0.1, 1, 10):
            events['bbox_error_gt_' + str(threshold)] = bbox_error > threshold
        for key, occurred in events.items():
            if occurred and key not in first_events:
                first_events[key] = dict(comparison=row, old=a, new=b)
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / 'M82_sampled_training_trace_audit.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(status='complete', source_sha256=sources, old_sampled_records=len(old),
                  new_sampled_records=len(new), paired_sequences=130,
                  first_record_exact=old[0] == new[0],
                  exact_records=sum(not r['changed_fields'] for r in rows),
                  integer_crop_changed_records=sum(r['integer_crop_changed'] for r in rows),
                  changed_record_counts=dict(different), first_observed_events=first_events,
                  comparison_csv_sha256=sha(csv_path),
                  limitations=[
                      'Samples are frame 1, every 50 frames, and sequence last; intervening states were not saved.',
                      'First observed difference is not necessarily the first true divergence.',
                      'Equal first predictions do not establish equal gradients or later floating-point execution.',
                      'This does not identify an environment, kernel, optimizer, or appended-data causal effect.',
                      'These are training-state differences, not evaluation gains or new training runs.',
                  ])
    path = args.output / 'M82_sampled_training_trace_audit.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(report_path=str(path), report_sha256=sha(path),
                         paired_records=len(old), first_record_exact=report['first_record_exact'],
                         changed_record_counts=dict(different),
                         first_observed_events={k: v['comparison'] for k, v in first_events.items()})))


if __name__ == '__main__':
    main()
