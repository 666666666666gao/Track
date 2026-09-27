"""Compare completed M89 Candidate VOT failures with sealed full127 references."""

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DONE = HERE / 'completed'
OLD = HERE.parent / 'full152_paired_20260925/completed/full152_vot_per_sequence_comparison.csv'
candidate = json.loads((DONE / 'candidate_vot_result.json').read_text())
control = json.loads((DONE / 'control_vot_result.json').read_text())
with OLD.open(newline='') as stream:
    references = {row['sequence']: row for row in csv.DictReader(stream)}

assert candidate['status'] == 'complete_full127'
assert len(candidate['failure_outcomes']) == 1765
assert len(candidate['per_sequence_failures']) == len(references) == 127
rows = []
for name, source in references.items():
    current = candidate['per_sequence_failures'][name]
    previous = control['per_sequence_failures'][name]
    assert int(source['anchors']) == current['anchors'] == previous['anchors']
    failures = current['confirmed_failures']
    native = int(source['native'])
    old_full_m82 = int(source['full_M82'])
    control_failures = previous['confirmed_failures']
    rows.append(dict(sequence=name, anchors=current['anchors'], native=native,
                     old_full_m82=old_full_m82, m89_control=control_failures,
                     m89_candidate=failures, candidate_minus_native=failures - native,
                     candidate_minus_old_full_m82=failures - old_full_m82,
                     candidate_minus_control=failures - control_failures))

assert sum(row['anchors'] for row in rows) == 1765
assert sum(row['native'] for row in rows) == 183
assert sum(row['old_full_m82'] for row in rows) == 283
assert sum(row['m89_control'] for row in rows) == 343
assert sum(row['m89_candidate'] for row in rows) == candidate['confirmed_failures'] == 216

with (DONE / 'candidate_vot_failure_comparison.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

summary = dict(scope='Full127, 1765 anchors; descriptive confirmed failure counts, not an EAO decomposition.',
               native_failures=183, old_full_m82_failures=283, control_failures=343,
               candidate_failures=216, candidate_minus_native=33,
               candidate_minus_old_full_m82=-67, candidate_minus_control=-127)
for baseline in ('native', 'old_full_m82', 'control'):
    key = 'candidate_minus_' + baseline
    summary['worst_' + key] = sorted(rows, key=lambda row: row[key], reverse=True)[:10]
    summary['best_' + key] = sorted(rows, key=lambda row: row[key])[:10]
(DONE / 'candidate_vot_failure_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
