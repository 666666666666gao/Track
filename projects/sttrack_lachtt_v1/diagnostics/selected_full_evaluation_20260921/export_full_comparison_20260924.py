"""Write a compact comparison table after all six official evaluations finish."""

import csv
import json
from pathlib import Path


ROOT = Path('/root/autodl-tmp/sttrack_selected_full_evaluation_20260921')
source = ROOT / 'all_results.json'
data = json.loads(source.read_text())
assert data['status'] == 'six_full_evaluations_complete'
assert [(r['model'], r['dataset']) for r in data['results']] == [
    ('M67', 'depthtrack'), ('M67', 'cdtb'), ('M67', 'vot'),
    ('M82', 'depthtrack'), ('M82', 'cdtb'), ('M82', 'vot'),
]

with (ROOT / 'full_comparison.csv').open('w', newline='') as out:
    writer = csv.writer(out)
    writer.writerow(['model', 'dataset', 'P_or_EAO_percent', 'R_or_ACC_percent',
                     'F_or_ROB_percent', 'checkpoint_sha256', 'result_sha256'])
    for row in data['results']:
        metrics = row['metrics']
        if row['dataset'] == 'vot':
            values = [metrics['EAO'], metrics['ACC'], metrics['ROB']]
        else:
            values = [metrics['precision_percent'], metrics['recall_percent'],
                      metrics['f_score_percent']]
        writer.writerow([row['model'], row['dataset'], *values,
                         row['checkpoint_sha256'], row['result_sha256']])
