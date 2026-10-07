"""Read only the actual M118 final rows; no neural or policy execution."""
from pathlib import Path
from datetime import datetime
import json

root = Path(__file__).resolve().parents[4]
d = root / 'projects/sttrack_lachtt_v1/diagnostics/three_module_20260928'
completed = d / 'm118_completed'
analysis = {}
for arm in ['human_text', 'visual_query', 'generic']:
    report = json.loads((completed / ('train_' + arm) / 'result.json').read_text(encoding='utf-8'))
    analysis[arm] = dict(first_loss=report['history'][0]['mean_losses'],
                         final_loss=report['history'][-1]['mean_losses'], splits={})
    for split, filename in [('fit', 'fit_events.jsonl'), ('development', arm + '_development.jsonl')]:
        rows = [json.loads(line) for line in (completed / ('train_' + arm) / filename).read_text(encoding='utf-8').splitlines()]
        assert len(rows) == (2544 if split == 'fit' else 495)
        out = dict(states=len(rows), selected_correct=0, parent_correct=0, selected_mean_iou=0., parent_mean_iou=0.,
                   choice_changes=0, both_qualified_better=0, both_qualified_worse=0,
                   all_changes=[], healthy_changes=[])
        for row in rows:
            a, b = row['selected_iou'], row['parent_iou']
            assert row['candidate_iou'][row['selected']] == a
            assert row['candidate_iou'][row['parent_selected']] == b
            out['selected_correct'] += a >= .5
            out['parent_correct'] += b >= .5
            out['selected_mean_iou'] += a / len(rows)
            out['parent_mean_iou'] += b / len(rows)
            out['both_qualified_better'] += min(a, b) >= .5 and a > b
            out['both_qualified_worse'] += min(a, b) >= .5 and a < b
            if row['selected'] != row['parent_selected']:
                out['choice_changes'] += 1
                event = dict(key=row['key'], strata=row['strata'], selected=row['selected'], parent_selected=row['parent_selected'],
                             selected_iou=a, parent_iou=b, delta_iou=a-b)
                out['all_changes'].append(event)
                if 'healthy' in row['strata']:
                    out['healthy_changes'].append(event)
        analysis[arm]['splits'][split] = out
result = dict(status='complete_M118_saved_rows_failure_analysis', observed_at=datetime.now().astimezone().isoformat(),
              arms=analysis, no_model_execution=True, no_threshold_or_parameter_fitting=True,
              conclusion='Optimization works on fit but no human-content advantage passes on reused development. Current ROI/text/geometry ordering still lacks stable instance discrimination; this is not proof that confirmed input is wrong or that a specific loss caused the failure.')
target = d / 'M118_FAILURE_ANALYSIS.json'
assert not target.exists()
target.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
print(json.dumps({arm: dict(first_loss=value['first_loss'], final_loss=value['final_loss'],
    splits={name: {k:v for k,v in item.items() if k != 'all_changes'} for name,item in value['splits'].items()})
    for arm,value in analysis.items()}, indent=2))
