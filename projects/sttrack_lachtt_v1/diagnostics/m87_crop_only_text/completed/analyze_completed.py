"""Posthoc descriptive analysis of all sealed M87 development trajectories."""
from pathlib import Path
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parent
E = ROOT / 'evidence'
OUT = ROOT / 'analysis'
OUT.mkdir()
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
result = read(E / 'recursive_result.json')
spec = read(E / 'recursive_spec.json')
arms = ['category', 'category_empty', 'category_old', 'category_swapped']
register_path = ROOT.parent / 'semantic_screening/paired_register.csv'
register = {r['sequence']: r for r in csv.DictReader(register_path.open(encoding='utf-8-sig'))}

def iou(box, gt):
    if not all(math.isfinite(x) for x in gt) or gt[2] <= 0 or gt[3] <= 0:
        return None
    w = max(0, min(box[0]+box[2], gt[0]+gt[2])-max(box[0], gt[0]))
    h = max(0, min(box[1]+box[3], gt[1]+gt[3])-max(box[1], gt[1]))
    area = w*h
    return area/(box[2]*box[3]+gt[2]*gt[3]-area)

def intervals(mask):
    start = None
    out = []
    for index, value in enumerate(mask + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index-start >= 10:
                out.append([start, index])
            start = None
    return out

def summary(values):
    valid = [v for v in values if v is not None]
    return dict(valid_frames=len(valid), mean_iou=sum(valid)/len(valid),
                low_iou_frames=sum(v <= .1 for v in valid),
                failure_episodes=len(intervals([v is not None and v <= .1 for v in values])))

per = []
events = []
writes = []
for case in spec['cases']:
    name = case['sequence']
    gt_path = E / 'development_gt' / (name+'.txt')
    assert sha(gt_path) == case['gt_sha256']
    gt = [[float(x) for x in line.split(',')] for line in gt_path.read_text().splitlines()]
    streams = {a: read(E / 'recursive' / a / (name+'.json'))['rows'] for a in arms}
    assert all(len(rows) == len(gt) == case['frames'] for rows in streams.values())
    values = {a: [None] + [iou(row['bbox'], gt[i]) for i, row in enumerate(rows) if i > 0]
              for a, rows in streams.items()}
    row = dict(sequence=name, old_category=register[name]['old_category'],
               new_category=register[name]['new_category'],
               old_screening=register[name]['old_screening'], new_screening=register[name]['new_screening'])
    for arm in arms:
        measured = summary(values[arm])
        for key, val in measured.items():
            assert abs(val-result['per_sequence'][arm][name][key]) < 1e-10, (name, arm, key)
            row[arm+'_'+key] = val
        for i, pred in enumerate(streams[arm]):
            if i and i % 50 == 0 and pred['score'] > .75:
                val = values[arm][i]
                writes.append(dict(sequence=name, arm=arm, frame=i, score=pred['score'], iou=val,
                                   quality='invalid_gt' if val is None else 'correct' if val >= .5 else 'severe' if val <= .1 else 'intermediate'))
    for other in arms[1:]:
        row['delta_mean_vs_'+other] = row['category_mean_iou']-row[other+'_mean_iou']
        row['first_bbox_difference_vs_'+other] = next((i for i in range(1,len(gt))
            if streams['category'][i]['bbox'] != streams[other][i]['bbox']), None)
        for direction in ['damage', 'improvement']:
            bad, good = ('category', other) if direction == 'damage' else (other, 'category')
            mask = [b is not None and g is not None and b <= .1 and g >= .5
                    for b,g in zip(values[bad],values[good])]
            for start,end in intervals(mask):
                events.append(dict(sequence=name, reference=other, direction=direction,
                                   start=start,end_exclusive=end,frames=end-start))
    per.append(row)

def write_csv(path, rows):
    with path.open('w',encoding='utf-8-sig',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

write_csv(OUT/'per_sequence.csv',per)
write_csv(OUT/'strict_intervals.csv',events)
write_csv(OUT/'template_write_events.csv',writes)
groups = {}
for status in ['supported','conflicting','uncertain']:
    chosen = [row for row in per if row['new_screening'] == status]
    n = sum(row['category_valid_frames'] for row in chosen)
    groups[status] = dict(sequences=len(chosen), valid_frames=n,
        delta_category_old=sum(row['delta_mean_vs_category_old']*row['category_valid_frames'] for row in chosen)/n,
        delta_category_empty=sum(row['delta_mean_vs_category_empty']*row['category_valid_frames'] for row in chosen)/n)

out = dict(
    scope='Posthoc descriptive analysis, all22 repeated Train development sequences; not new training, official test or same-state intervention.',
    semantic_scope='Sealed pre-result assistant proxy labels, not independent human ground truth. Subgroups do not establish causal semantic accuracy.',
    update_scope='Reconstructed from saved scores and actual frame%50==0 and score>.75 code. Different recursive histories; counts do not establish update causality.',
    input_result_sha256=sha(E/'recursive_result.json'), input_register_sha256=sha(register_path),
    source_sha256=sha(Path(__file__)),
    aggregates=result['aggregates'], gates=result['gates'], native_success_damage=result['native_success_damage'],
    strict_intervals={other:{direction:dict(segments=sum(x['reference']==other and x['direction']==direction for x in events),
        frames=sum(x['frames'] for x in events if x['reference']==other and x['direction']==direction))
        for direction in ['damage','improvement']} for other in arms[1:]},
    template_writes={arm:{quality:sum(x['arm']==arm and x['quality']==quality for x in writes)
        for quality in ['correct','severe','intermediate','invalid_gt']} for arm in arms},
    semantic_subgroups=groups,
    leave_one_out_positive={arm:sum(v>0 for v in rows.values()) for arm,rows in result['content_leave_one_out'].items()},
    artifacts_sha256={name:sha(OUT/name) for name in ['per_sequence.csv','strict_intervals.csv','template_write_events.csv']},
)
(OUT/'analysis.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='aggregates'}))
for row in sorted(per,key=lambda r:r['delta_mean_vs_category_empty']):
    print(json.dumps({k:row[k] for k in ['sequence','old_category','new_category','new_screening',
        'category_mean_iou','category_empty_mean_iou','category_old_mean_iou','category_swapped_mean_iou',
        'category_failure_episodes','category_empty_failure_episodes']}))
