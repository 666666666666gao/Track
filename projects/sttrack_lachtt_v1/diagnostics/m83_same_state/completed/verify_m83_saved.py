"""Post-completion scalar checks of fixed-state readouts against original M82."""
from pathlib import Path
import hashlib
import json
import math
from datetime import datetime, timezone

R=Path('/root/autodl-tmp/sttrack_m83_same_state_20260920')
P=Path('/root/autodl-tmp/sttrack_m82_native_preservation_20260909')
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['replay','analysis','controller']:
    assert (R/(name+'.exit')).read_text().strip()=='0'
spec=read(R/'spec.json');launch=read(R/'launch.json');receipt=read(R/'predictions/receipt.json')
for name,digest in launch['sources'].items():assert sha(R/name)==digest
assert receipt['mode']=='full22' and receipt['positions']==33108 and len(receipt['sequences'])==22
assert receipt['spec_sha256']==sha(R/'spec.json')
result=read(R/'diagnostic_result.json')
assert result['receipt_sha256']==sha(R/'predictions/receipt.json')
assert result['source_sha256']==sha(R/'analyze_m83.py')
data={}
for case,item in zip(spec['cases'],receipt['sequences']):
    assert case['sequence']==item['sequence']
    path=R/'predictions'/(case['sequence']+'.json')
    assert sha(path)==item['sha256']
    old=P/'recursive/category'/(case['sequence']+'.json')
    assert sha(old)==case['sealed_category_sha256']
    rows=read(path)['rows'];native_rows=read(old)['rows']
    assert len(rows)==item['positions']==case['frames']-1
    for frame,row in enumerate(rows,1):
        assert row['frame']==frame and row['previous_bbox']==native_rows[frame-1]['bbox']
        assert row['variants']['category']['hann_bbox']==native_rows[frame]['bbox']
        assert row['variants']['category']['hann_max']==native_rows[frame]['score']
        assert set(row['variants'])=={'category','empty','swapped','native'}
        assert row['search_side']==math.ceil(math.sqrt(row['previous_bbox'][2]*row['previous_bbox'][3])*4.)
        for arm,out in row['variants'].items():
            for key in ['raw_bbox','hann_bbox']:
                assert len(out[key])==4 and all(math.isfinite(x) for x in out[key]) and min(out[key][2:])>0
            assert 0<=out['raw_peak']<256 and 0<=out['hann_peak']<256
            assert 0<out['hann_max']<=out['raw_max']<=1 and math.isfinite(out['native_spatial_kl'])
            if arm=='native':assert out['native_spatial_kl']==0.
    data[case['sequence']]=rows

def overlap(a,b):
    area=max(0.,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]))*max(0.,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))
    return area/(a[2]*a[3]+b[2]*b[3]-area)

overall={arm:dict(values=[],low=0,correct=0,rescue=0,harm=0,write=0) for arm in ['category','empty','swapped','native']}
for case in spec['cases']:
    path=Path(read(P/'training_spec.json')['dataset_root'])/case['sequence']/'groundtruth.txt'
    assert sha(path)==case['gt_sha256']
    gt=[[float(x) for x in s.split(',')] for s in path.read_text().splitlines() if s.strip()]
    assert len(gt)==case['frames']
    for row in data[case['sequence']]:
        i=row['frame'];g=gt[i]
        for arm,v in row['variants'].items():overall[arm]['write']+=int(i%50==0 and v['hann_max']>.75)
        if not all(math.isfinite(x) for x in g) or min(g[2:])<=0:continue
        values={arm:overlap(v['hann_bbox'],g) for arm,v in row['variants'].items()}
        for arm,value in values.items():
            x=overall[arm];x['values'].append(value);x['low']+=int(value<=.1);x['correct']+=int(value>=.5)
            x['rescue']+=int(values['category']<=.1 and value>=.5)
            x['harm']+=int(values['category']>=.5 and value<=.1)
for arm,x in overall.items():
    original=result['aggregates']['all_valid'][arm]
    assert len(x['values'])==original['n']==28897
    assert math.isclose(math.fsum(x['values']),original['iou_sum'],abs_tol=1e-8)
    for a,b in [('low','low'),('correct','correct'),('rescue','rescue_vs_category'),('harm','harm_vs_category')]:assert x[a]==original[b]
    assert x['write']==sum(v['update_qualification_count'][arm] for v in result['per_sequence'].values())
    x['mean_one_step_iou']=math.fsum(x.pop('values'))/28897
assert overall['category']['write']==260
report=dict(status='complete_scalar_fixed_state_verification',observed_utc=datetime.now(timezone.utc).isoformat(),
    positions=33108,valid_positions=28897,source_sha256=sha(Path(__file__)),result_sha256=sha(R/'diagnostic_result.json'),
    original_Category_previous_state_bbox_score_exact=True,scalar_means_low_rescue_harm_and_update_counts_verified=True,
    overall=overall,scope='Executor deterministic checks, not an independent model review; alternative outputs are not recursive trajectories')
target=R/'saved_diagnostic_verification.json';assert not target.exists()
target.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
