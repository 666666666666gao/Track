"""Reaggregate published M56 event rows; not a fresh feature/GT evaluation."""
import hashlib
import json
from pathlib import Path


root = Path(__file__).parent / 'cpu_completed'
training = json.loads((root / 'training_result.json').read_text())
assert training['status'] == 'complete_train'
reports = {name: json.loads((root / 'training' / (name + '_result.json')).read_text()) for name in training['variants']}
for name, row in reports.items():
    path = root / 'training' / (name + '_result.json')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == training['variants'][name]['result_sha256']
    assert row['checkpoint_sha256'] == training['variants'][name]['checkpoint_sha256']
    assert row['optimizer_steps'] == 960 and row['epochs'] == 20 and row['parameters'] == 484387
    assert row['fit_events'] == 1511 and not row['development_targets_loaded']
for key in ['initial_state_sha256', 'sample_order_sha256', 'ordered_event_keys_sha256']:
    assert reports['attributes'][key] == reports['pooled'][key] == reports['empty'][key]
static = json.loads((root / 'static_result.json').read_text())
assert static['spec_sha256'] == hashlib.sha256((root / 'spec.json').read_bytes()).hexdigest()
assert static['training_result_sha256'] == hashlib.sha256((root / 'training_result.json').read_bytes()).hexdigest()
for name, report in static['variants'].items():
    rows, aggregate = report['rows'], report['aggregate']
    assert len(rows) == aggregate['events'] == len({row['key'] for row in rows})
    assert abs(sum(row['selected_iou'] for row in rows) / len(rows) - aggregate['mean_iou']) < 1e-6
    assert abs(sum(row['default_iou'] for row in rows) / len(rows) - aggregate['default_mean_iou']) < 1e-6
    assert sum(row['selected_iou'] >= .5 for row in rows) == aggregate['correct']
    assert sum(row['default_iou'] <= .1 and row['selected_iou'] >= .5 for row in rows) == aggregate['rescues']
    assert sum(row['default_iou'] >= .5 and row['selected_iou'] <= .1 for row in rows) == aggregate['breaks']
    assert sum(row['none'] for row in rows) == aggregate['none']
    assert sum(row['valid_gt'] for row in rows) == aggregate['valid_gt_events']
    assert sum(value['events'] for value in report['per_sequence'].values()) == len(rows)
controls = json.loads((root / 'text_controls.json').read_text())
development = [row for row in controls['rows'].values() if row['split'] == 'development']
assert len(development) == 22
assert sum(row['shuffled_category_changed'] for row in development) == 15
assert sum(row['shuffled_phrase_count_changed'] for row in development) == 10
assert len(controls['eligible_development_sequences']) == 9
result = dict(status='pass', equal_initialization_order_and_budget_verified=True, six_static_modes_reaggregated=True,
    private_checkpoint_bytes_checked_by_this_public_script=False, fresh_candidate_GT_overlap_recomputed=False,
    scope='Public report arithmetic, completion records and declared content-control coverage only')
(Path(__file__).parent / 'cpu_report_verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
