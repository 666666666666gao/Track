"""Compare sealed VOT outcomes at the same 1765 start anchors."""

import csv
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
DONE = HERE / 'completed'
sources = {
    'native': HERE.parent / 'native_vot_full127/result.json',
    'old_m82': HERE.parent / 'full152_paired_20260925/completed/M82_vot_result.json',
    'control': DONE / 'control_vot_result.json',
    'candidate': DONE / 'candidate_vot_result.json',
}
data = {name: json.loads(path.read_text()) for name, path in sources.items()}
keys = set(data['candidate']['failure_outcomes'])
assert len(keys) == 1765
assert all(set(item['failure_outcomes']) == keys for item in data.values())
rows = []
for key in sorted(keys):
    outcomes = {name: item['failure_outcomes'][key] for name, item in data.items()}
    assert len({(item['sequence'], item['anchor'], item['direction'], item['run_length'])
                for item in outcomes.values()}) == 1
    row = dict(anchor_key=key, sequence=outcomes['candidate']['sequence'],
               anchor=outcomes['candidate']['anchor'], direction=outcomes['candidate']['direction'],
               run_length=outcomes['candidate']['run_length'])
    for name, item in outcomes.items():
        row[name + '_failed'] = int(item['failed'])
        row[name + '_progress'] = item['progress']
    rows.append(row)

assert {name: sum(row[name + '_failed'] for row in rows) for name in sources} == {
    'native': 183, 'old_m82': 283, 'control': 343, 'candidate': 216}
with (DONE / 'candidate_vot_anchor_comparison.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

summary = {}
for baseline in ('native', 'old_m82', 'control'):
    new_failures = [row for row in rows if row['candidate_failed'] and not row[baseline + '_failed']]
    repaired = [row for row in rows if not row['candidate_failed'] and row[baseline + '_failed']]
    summary[baseline] = dict(new_failures=len(new_failures), repaired=len(repaired),
                             net_failures=len(new_failures) - len(repaired),
                             new_failure_sequences=Counter(row['sequence'] for row in new_failures).most_common(12),
                             repaired_sequences=Counter(row['sequence'] for row in repaired).most_common(12))
(DONE / 'candidate_vot_anchor_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
