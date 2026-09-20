"""Post-hoc tables from sealed M82 outputs; no inference or parameter selection."""
from pathlib import Path
import csv
import hashlib
import json
import math

R = Path(__file__).parent
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
result = read(R / 'recursive_result.json')
assert sha(R / 'recursive_result.json') == read(R / 'saved_development_audit.json')['result_sha256']
families = ['category', 'empty', 'category_empty', 'category_swapped']
metrics = ['mean_iou', 'macro_sequence_mean_iou', 'low_iou_frames', 'failure_episodes']
with (R / 'aggregate_metrics.csv').open('w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['condition'] + metrics)
    for name, a in result['aggregates'].items():
        writer.writerow([name] + [a[k] for k in metrics])
    for name, a in result['matched_M78_aggregates'].items():
        writer.writerow(['M78_' + name] + [a[k] for k in metrics])
with (R / 'sequence_metrics.csv').open('w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['condition', 'sequence', 'valid_frames', 'mean_iou', 'low_iou_frames', 'failure_episodes'])
    for name, rows in result['per_sequence'].items():
        for seq, a in rows.items():
            writer.writerow([name, seq] + [a[k] for k in ['valid_frames', 'mean_iou', 'low_iou_frames', 'failure_episodes']])
with (R / 'frozen_gates.csv').open('w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['group', 'condition', 'criterion', 'passed'])
    for k, v in result['gates'].items():
        writer.writerow(['primary', 'category', k, v])
    for group in ['content_gates', 'preservation_incremental_gates']:
        for arm, gates in result[group].items():
            for k, v in gates.items():
                writer.writerow([group, arm, k, v])

cat = result['aggregates']['category']
for condition, rows in result['content_leave_one_out'].items():
    other = result['aggregates'][condition]
    for seq, recorded in rows.items():
        a = result['per_sequence']['category'][seq]
        b = result['per_sequence'][condition][seq]
        value = (cat['iou_sum'] - a['iou_sum']) / (cat['valid_frames'] - a['valid_frames'])
        value -= (other['iou_sum'] - b['iou_sum']) / (other['valid_frames'] - b['valid_frames'])
        assert math.isclose(value, recorded, rel_tol=1e-10, abs_tol=1e-12)

updates = {}
for arm in families:
    receipt = read(R / (arm + '_recursive_receipt.json'))
    updates[arm] = {}
    for row in receipt['sequences']:
        path = R / 'recursive' / arm / (row['sequence'] + '.json')
        assert sha(path) == row['sha256']
        pred = read(path)['rows']
        updates[arm][row['sequence']] = [x['frame'] for x in pred if x['frame'] > 0 and x['frame'] % 50 == 0 and x['score'] > .75]

posthoc = {
    'status': 'completed_sealed_output_posthoc_descriptive_analysis',
    'source_sha256': sha(Path(__file__)),
    'result_sha256': sha(R / 'recursive_result.json'),
    'scope': 'DepthTrack Train development22, seed2027; no public evaluation, no same-state causal intervention',
    'leave_one_out_recomputed': True,
    'update_count_reconstructed_from_saved_score_and_verified_runtime_rule': {a: sum(map(len, v.values())) for a, v in updates.items()},
    'update_frames': updates,
    'note': 'Counts reconstruct the frozen frame%50==0 and Hann-score>.75 rule. Paths differ; counts do not establish update quality or causal utility.',
}
(R / 'posthoc_descriptive.json').write_text(json.dumps(posthoc, indent=2, allow_nan=False) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in posthoc.items() if k != 'update_frames'}, indent=2))
