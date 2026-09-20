"""Post-seal real-GT statistics on fixed Category visited states, not trajectories."""
import hashlib
import json
import math
from pathlib import Path

R = Path('/root/autodl-tmp/sttrack_m83_same_state_20260920')
P = Path('/root/autodl-tmp/sttrack_m82_native_preservation_20260909')
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
spec = read(R/'spec.json')
receipt = read(R/'predictions/receipt.json')
assert receipt['status'] == 'complete' and receipt['positions'] == 33108
assert receipt['spec_sha256'] == sha(R/'spec.json')
sealed = {}
for case, item in zip(spec['cases'], receipt['sequences']):
    assert case['sequence'] == item['sequence']
    path = R/'predictions'/(case['sequence']+'.json')
    assert sha(path) == item['sha256']
    rows = read(path)['rows']
    assert [x['frame'] for x in rows] == list(range(1, case['frames']))
    sealed[case['sequence']] = rows
assert len(sealed) == 22

def iou(a, b):
    dx = max(0., min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0]))
    dy = max(0., min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    inter = dx*dy
    return inter/(a[2]*a[3]+b[2]*b[3]-inter)

per = {}
for case in spec['cases']:
    seq = case['sequence']
    path = Path(read(P/'training_spec.json')['dataset_root'])/seq/'groundtruth.txt'
    assert sha(path) == case['gt_sha256']
    gt = [[float(x) for x in line.split(',')] for line in path.read_text().splitlines() if line.strip()]
    assert len(gt) == case['frames']
    groups = {}
    writes = {arm: 0 for arm in ['category', 'empty', 'swapped', 'native']}
    disagreements = {arm: 0 for arm in ['empty', 'swapped', 'native']}
    for row in sealed[seq]:
        v = row['variants']; f = row['frame']
        if f % 50 == 0:
            decisions = {a: x['hann_max'] > .75 for a, x in v.items()}
            for a in writes: writes[a] += int(decisions[a])
            for a in disagreements: disagreements[a] += int(decisions[a] != decisions['category'])
        g = gt[f]
        if not all(math.isfinite(x) for x in g) or min(g[2:]) <= 0: continue
        prev = row['previous_bbox']; side = row['search_side']
        x = round(prev[0]+.5*prev[2]-.5*side); y = round(prev[1]+.5*prev[3]-.5*side)
        inside = x <= g[0]+.5*g[2] < x+side and y <= g[1]+.5*g[3] < y+side
        overlaps = {a: iou(z['hann_bbox'], g) for a, z in v.items()}
        labels = ['all_valid', 'centre_inside' if inside else 'centre_outside',
                  'native_correct' if overlaps['native'] >= .5 else 'native_not_correct']
        for label in labels:
            group = groups.setdefault(label, {a: dict(n=0, iou_sum=0., low=0, correct=0, rescue_vs_category=0,
                harm_vs_category=0, raw_rescue_hann=0, raw_harm_hann=0, raw_hann_peak_changes=0,
                kl_native_sum=0., response_mass_sum=0.) for a in v})
            for a, z in v.items():
                out = group[a]; value = overlaps[a]; raw = iou(z['raw_bbox'], g)
                out['n'] += 1; out['iou_sum'] += value
                out['low'] += int(value <= .1); out['correct'] += int(value >= .5)
                out['rescue_vs_category'] += int(overlaps['category'] <= .1 and value >= .5)
                out['harm_vs_category'] += int(overlaps['category'] >= .5 and value <= .1)
                out['raw_rescue_hann'] += int(value <= .1 and raw >= .5)
                out['raw_harm_hann'] += int(value >= .5 and raw <= .1)
                out['raw_hann_peak_changes'] += int(z['raw_peak'] != z['hann_peak'])
                out['kl_native_sum'] += z['native_spatial_kl']; out['response_mass_sum'] += z['raw_mass']
    per[seq] = dict(groups=groups, update_qualification_count=writes, update_qualification_disagreements=disagreements)
aggregate = {}
for seq, data in per.items():
    for label, group in data['groups'].items():
        target = aggregate.setdefault(label, {})
        for arm, values in group.items():
            sums = target.setdefault(arm, {k: 0 for k in values})
            for k, v in values.items(): sums[k] += v
for group in aggregate.values():
    for sums in group.values(): sums['mean_one_step_iou'] = sums['iou_sum']/sums['n']
original = read(P/'recursive_result.json')['aggregates']['category']
current = aggregate['all_valid']['category']
assert current['n'] == original['valid_frames'] == 28897
assert math.isclose(current['iou_sum'], original['iou_sum'], abs_tol=1e-8)
assert current['low'] == original['low_iou_frames']
result = dict(status='complete_fixed_Category_state_diagnostic', source_sha256=sha(Path(__file__)),
    spec_sha256=sha(R/'spec.json'), receipt_sha256=sha(R/'predictions/receipt.json'),
    aggregates=aggregate, per_sequence=per, ground_truth_source='DepthTrack Train dataset files with frozen SHA',
    limitation='One-step alternative readouts on Category visited states. Native is not an independent native trajectory. No alternative H10 or long-term recovery claims; no new promotion.')
target = R/'diagnostic_result.json'; assert not target.exists()
target.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps(dict(status=result['status'], result_sha256=sha(target), aggregates=aggregate), indent=2))
