"""Posthoc saved-prediction comparison, not a causal replay or new model."""
import csv
import hashlib
import json
import math
from pathlib import Path

R = Path(__file__).parent
read = lambda p: json.loads(p.read_text())

def overlap(a, b):
    area = max(0., min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0])) * max(0., min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    return area / (a[2]*a[3]+b[2]*b[3]-area)

def intervals(mask):
    start = None
    for i, active in enumerate(mask + [False]):
        if active and start is None:
            start = i
        elif not active and start is not None:
            if i - start >= 10:
                yield [start, i]
            start = None

out = []
for case in read(R / 'recursive_spec.json')['cases']:
    seq = case['sequence']
    gt_path = R / 'dataset_gt' / (seq + '.txt')
    assert hashlib.sha256(gt_path.read_bytes()).hexdigest() == case['gt_sha256']
    gt = [[float(x) for x in row] for row in csv.reader(gt_path.read_text().splitlines()) if row]
    valid = [i > 0 and all(math.isfinite(v) for v in b) and b[2] > 0 and b[3] > 0 for i, b in enumerate(gt)]
    rows = {a: read(R/'recursive'/a/(seq+'.json'))['rows'] for a in ['category', 'category_empty', 'category_swapped']}
    scores = {a: [overlap(x['bbox'], b) if ok else None for x, b, ok in zip(v, gt, valid)] for a, v in rows.items()}
    damage = [ok and scores['category'][i] <= .1 and scores['category_empty'][i] >= .5 for i, ok in enumerate(valid)]
    rescued = [ok and scores['category_empty'][i] <= .1 and scores['category'][i] >= .5 for i, ok in enumerate(valid)]
    out.append(dict(sequence=seq,
        first_bbox_difference_gt1e_6={a: next((i for i, (c, o) in enumerate(zip(rows['category'], rows[a])) if max(abs(x-y) for x, y in zip(c['bbox'], o['bbox'])) > 1e-6), None) for a in ['category_empty', 'category_swapped']},
        H10={a:list(intervals([v is not None and v <= .1 for v in ss])) for a, ss in scores.items()},
        sustained_damage_vs_empty=list(intervals(damage)),
        sustained_rescue_vs_empty=list(intervals(rescued))))
result = dict(scope='Saved independent recursions; zero-based half-open intervals. Unknown GT breaks runs. This is posthoc localization of differences, not attribution of cause.',
    sequences=out,
    sustained_damage_runs=sum(len(x['sustained_damage_vs_empty']) for x in out),
    sustained_damage_frames=sum(b-a for x in out for a,b in x['sustained_damage_vs_empty']),
    sustained_rescue_runs=sum(len(x['sustained_rescue_vs_empty']) for x in out),
    sustained_rescue_frames=sum(b-a for x in out for a,b in x['sustained_rescue_vs_empty']))
(R/'saved_damage_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='sequences'}))
for item in out:
    if item['sequence'] in ['mobilephone02_indoor','car02_indoor']:
        print(json.dumps(item))
