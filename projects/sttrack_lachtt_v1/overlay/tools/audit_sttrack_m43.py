"""Audit sealed recursive outputs and locate the first state-changing choice."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_overlap(boxes, gt):
    values=[];episodes=0;streak=0
    for frame,(a,b) in enumerate(zip(boxes,gt)):
        if frame==0 or not np.isfinite(b).all() or b[2]<=0 or b[3]<=0:
            value=float('nan')
        else:
            iw=max(0.,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]))
            ih=max(0.,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))
            intersection=iw*ih
            value=intersection/(a[2]*a[3]+b[2]*b[3]-intersection)
        values.append(value)
        if math.isfinite(value) and value<=.1:
            streak+=1
        else:
            episodes+=int(streak>=10);streak=0
    episodes+=int(streak>=10)
    valid=[x for x in values if math.isfinite(x)]
    metrics=dict(valid_frames=len(valid),iou_sum=math.fsum(valid),
                 mean_iou=math.fsum(valid)/len(valid),
                 low_iou_frames=sum(x<=.1 for x in valid),failure_episodes=episodes)
    return np.asarray(values),metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    root=p.parse_args().root
    assert (root/'controller.exit').read_text().strip()=='0'
    spec=json.loads((root/'spec.json').read_text());result=json.loads((root/'result.json').read_text())
    assert result['status']=='complete' and result['spec_sha256']==sha(root/'spec.json')
    assert result['primary']=='pooled_vs_default'
    for name,digest in spec['source_sha256'].items():assert sha(Path(spec['repository'])/name)==digest,name
    assert sha(spec['base_checkpoint'])==spec['base_checkpoint_sha256']
    for arm,path in spec['association_checkpoints'].items():assert sha(path)==spec['association_sha256'][arm]
    cases={c['name']:c for c in spec['sequences']};baseline=defaultdict(list)
    for path,digest in spec['baseline_trace_sha256'].items():
        assert sha(path)==digest
        for row in json.loads(Path(path).read_text())['rows']:
            if row['sequence'] in cases:baseline[row['sequence']].append(row)
    seals={}
    for arm in ['pooled','spatial']:
        receipt=json.loads((root/(arm+'_receipt.json')).read_text())
        assert receipt['status']=='complete' and receipt['source_unchanged']
        seals[arm]={r['sequence']:r for r in receipt['sequences']}
        assert set(seals[arm])==set(cases)
    details=[];csvrows=[];total=0
    for name,case in cases.items():
        base=sorted(baseline[name],key=lambda r:r['frame_index'])
        assert [r['frame_index'] for r in base]==list(range(case['frames']))
        boxes0=np.asarray([r['public_bbox'] for r in base],dtype=np.float64)
        gt=np.loadtxt(Path(spec['dataset_root'])/name/'groundtruth.txt',delimiter=',')[:len(base)]
        values0,metrics0=independent_overlap(boxes0,gt)
        sequence=dict(sequence=name,frames=len(base))
        for key,value in metrics0.items():
            assert math.isclose(value,result['per_sequence']['default'][name][key],rel_tol=1e-12,abs_tol=1e-10),(name,'default',key)
            sequence['default_'+key]=value
        for arm in ['pooled','spatial']:
            path=root/'trajectories'/(arm+'_'+name+'.json');assert sha(path)==seals[arm][name]['sha256']
            data=json.loads(path.read_text());rows=data['rows'];assert data['sequence']==name and data['arm']==arm
            assert len(rows)==case['frames'] and [r['frame'] for r in rows]==list(range(len(rows)))
            boxes=np.asarray([r['bbox'] for r in rows],dtype=np.float64)
            assert np.isfinite(boxes).all() and (boxes[:,2:]>0).all()
            assert all(math.isfinite(r['score']) and 0<=r['choice']<10 and (not r['none'] or r['choice']==0) for r in rows)
            assert np.array_equal(boxes[0],np.asarray(case['init_bbox']))
            values,metrics=independent_overlap(boxes,gt)
            for key,value in metrics.items():
                assert math.isclose(value,result['per_sequence'][arm][name][key],rel_tol=1e-12,abs_tol=1e-10),(name,arm,key)
                sequence[arm+'_'+key]=value
            changed=[i for i,r in enumerate(rows) if r['choice']!=0]
            assert len(changed)==seals[arm][name]['changes']
            first=changed[0] if changed else len(rows)
            prefix_error=float(np.abs(boxes[:first]-boxes0[:first]).max())
            assert prefix_error==0.,(name,arm,first,prefix_error)
            item=dict(sequence=name,arm=arm,changes=len(changed),first_override_frame_zero_based=first if changed else None,
                      prefix_max_bbox_error_px=prefix_error,trajectory_sha256=sha(path))
            if changed:
                start=first+1;end=min(first+11,len(rows));valid=np.isfinite(values0[start:end])
                item.update(first_choice=rows[first]['choice'],
                    first_default_iou=float(values0[first]) if np.isfinite(values0[first]) else None,
                    first_selected_iou=float(values[first]) if np.isfinite(values[first]) else None,
                    next10_valid_frames=int(valid.sum()),
                    next10_mean_gain=float((values[start:end][valid]-values0[start:end][valid]).mean()) if valid.any() else None)
            details.append(item);sequence[arm+'_changes']=len(changed);total+=len(rows)
        csvrows.append(sequence)
    audit=dict(status='complete',integrity_pass=True,sequences=len(cases),trajectory_files=len(details),frames_including_initialization=total,
        primary_pass=result['primary_pass'],source_and_weights_unchanged=True,independent_metrics_match=True,
        exact_default_prefix_before_first_override=True,first_overrides=details,
        result_sha256=sha(root/'result.json'),auditor_sha256=sha(__file__),
        scope='Train development only. First-override prefix equality supports matched-state diagnosis; later decisions occur on already diverged states. No new gate or model selection.')
    (root/'terminal_audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    with (root/'per_sequence.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    print(json.dumps(dict(status='complete',integrity_pass=True,frames=total,primary_pass=result['primary_pass'],result_sha256=audit['result_sha256'])),flush=True)


if __name__=='__main__':
    main()
