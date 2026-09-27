"""Compare completed M89 Control VOT failures with sealed full127 references."""

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DONE = HERE / 'completed'
OLD = HERE.parent / 'full152_paired_20260925/completed/full152_vot_per_sequence_comparison.csv'
vot = json.loads((DONE / 'control_vot_result.json').read_text())
with OLD.open(newline='') as stream:
    references = {row['sequence']: row for row in csv.DictReader(stream)}

assert vot['status'] == 'complete_full127'
assert len(vot['failure_outcomes']) == 1765
assert len(vot['per_sequence_failures']) == len(references) == 127
rows = []
for name, source in references.items():
    current = vot['per_sequence_failures'][name]
    assert int(source['anchors']) == current['anchors']
    failures = current['confirmed_failures']
    native = int(source['native'])
    old_full_m82 = int(source['full_M82'])
    rows.append(dict(sequence=name, anchors=current['anchors'], native=native,
                     old_full_m82=old_full_m82, m89_control=failures,
                     control_minus_native=failures - native,
                     control_minus_old_full_m82=failures - old_full_m82))

assert sum(row['anchors'] for row in rows) == 1765
assert sum(row['native'] for row in rows) == 183
assert sum(row['old_full_m82'] for row in rows) == 283
assert sum(row['m89_control'] for row in rows) == vot['confirmed_failures'] == 343

with (DONE / 'control_vot_failure_comparison.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
summary = dict(scope='Full127, 1765 anchors; descriptive confirmed failure counts, not an EAO decomposition.',
               native_failures=183, old_full_m82_failures=283, control_failures=343,
               control_minus_native=160, control_minus_old_full_m82=60,
               top_control_minus_native=sorted(rows, key=lambda row: row['control_minus_native'], reverse=True)[:10],
               top_control_minus_old_full_m82=sorted(rows, key=lambda row: row['control_minus_old_full_m82'], reverse=True)[:10],
               best_control_minus_native=sorted(rows, key=lambda row: row['control_minus_native'])[:10],
               best_control_minus_old_full_m82=sorted(rows, key=lambda row: row['control_minus_old_full_m82'])[:10])
(DONE / 'control_vot_failure_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
