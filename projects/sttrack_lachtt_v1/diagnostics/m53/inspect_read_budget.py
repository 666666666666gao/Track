"""Exploratory fitting-set decomposition after the frozen M53 capacity test."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--raw-events', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    assert (root/'analysis.exit').read_text().strip() == '0'
    path = root/'capacity_result.json'
    capacity = json.loads(path.read_text())
    assert capacity['status'] == 'complete'
    receipt = json.loads((root/'collection_receipt.json').read_text())
    raw = {}
    for item in receipt['sequences']:
        data_path = args.raw_events/(item['sequence']+'.json')
        assert hashlib.sha256(data_path.read_bytes()).hexdigest() == item['sha256']
        for event in json.loads(data_path.read_text())['events']:
            raw[event['key']] = event
    rows = []
    for event in capacity['events']:
        if not event['valid_gt']:
            continue
        views = event['views']; baseline = views[0]; past = views[1:]
        options = dict(current=[baseline], initial_or_current=[baseline]+[v for v in past if v['template_frame'] == 0],
                       recent1_or_current=[baseline]+past[-1:],
                       initial_recent1_current=[baseline]+[v for v in past if v['template_frame'] == 0 or v in past[-1:]],
                       initial_recent3_current=[baseline]+[v for v in past if v['template_frame'] == 0 or v in past[-3:]],
                       all_past=views)
        chosen = {name: max(values, key=lambda v: v['top1']) for name, values in options.items()}
        original = raw[event['key']]
        scored = [dict(template_frame=original['active_template_frame'], **original['baseline'])]+original['alternatives']
        selected = max(scored, key=lambda v: v['score'])['template_frame']
        chosen['max_native_score'] = next(v for v in views if v['template_frame'] == selected)
        rows.append(dict(key=event['key'], sequence=event['sequence'], default_iou=baseline['top1'],
                         selected={name: dict(frame=value['template_frame'], iou=value['top1']) for name, value in chosen.items()}))
    summary = {}
    for name in rows[0]['selected']:
        correct = [r for r in rows if r['selected'][name]['iou'] >= .5]
        rescued = [r for r in correct if r['default_iou'] <= .1]
        harmed = [r for r in rows if r['default_iou'] >= .5 and r['selected'][name]['iou'] <= .1]
        summary[name] = dict(valid_events=len(rows), correct=len(correct), severe_rescued=len(rescued),
                             rescued_sequences=sorted({r['sequence'] for r in rescued}), severe_harms=len(harmed),
                             harmed_sequences=sorted({r['sequence'] for r in harmed}),
                             mean_iou=math.fsum(r['selected'][name]['iou'] for r in rows)/len(rows))
    assert summary['all_past']['severe_rescued'] == capacity['summary']['all_past']['severe_events_recovered']
    result = dict(scope='Post-hoc fitting-set anatomy; original capacity screen unchanged',
                  capacity_result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), summary=summary, rows=rows,
                  notes=['All modes except current and max_native_score use privileged best-view IoU selection.',
                         'max_native_score is an untrained saved-state comparator, without recursive deployment.',
                         'Recent memories are actual native writes; initialization may also be a dynamic input.',
                         'This fitting-set exploration informs reader design and adds no promotion condition.'])
    (root/'read_budget_exploration.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
