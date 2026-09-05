"""Posthoc Train development comparison; never a VOT or DepthTrack Test score."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def episodes(mask, length=10):
    starts = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)) == 1)
    ends = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)) == -1)
    return int(((ends-starts) >= length).sum())


def statistics(boxes, gt):
    boxes=np.asarray(boxes,dtype=np.float64); gt=np.asarray(gt[:len(boxes)],dtype=np.float64)
    valid=np.isfinite(gt).all(1)&(gt[:,2]>0)&(gt[:,3]>0)
    valid[0]=False
    a,b=boxes[valid],gt[valid]
    inter=np.maximum(0,np.minimum(a[:,:2]+a[:,2:],b[:,:2]+b[:,2:])-np.maximum(a[:,:2],b[:,:2])).prod(1)
    values=np.full(len(boxes),np.nan)
    values[valid]=inter/(a[:,2:].prod(1)+b[:,2:].prod(1)-inter)
    low=valid&(values<=.1)
    return dict(valid_frames=int(valid.sum()),iou_sum=float(values[valid].sum()),mean_iou=float(values[valid].mean()),
                low_iou_frames=int(low.sum()),failure_episodes=episodes(low),invalid_gt_frames=int((~valid[1:]).sum()))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); args=p.parse_args(); root=args.root
    spec=json.loads((root/'spec.json').read_text()); frozen=json.loads((root/'recursive_spec.json').read_text())
    assert sha(__file__)==frozen['source_sha256']['tools/analyze_sttrack_m42_recursive.py']
    training=json.loads((root/'training_result.json').read_text()); assert training['information_gate_pass']
    cases=[c for c in json.loads((root/'inference_inputs.json').read_text()) if c['split']=='development_holdout']
    sequences={c['sequence'] for c in cases}; assert len(sequences)==22
    baseline=defaultdict(list)
    trace_root=Path('/root/autodl-tmp/sttrack_innovation_v1/risk_recovery_full152_v1')
    for shard,digest in frozen['baseline_shard_sha256'].items():
        path=trace_root/shard; assert sha(path)==digest
        for row in json.loads(path.read_text())['rows']:
            if row['sequence'] in sequences: baseline[row['sequence']].append(row)
    by_arm=defaultdict(dict)
    for seq in sorted(sequences):
        rows=sorted(baseline[seq],key=lambda r:r['frame_index'])
        gt=np.loadtxt(Path(spec['dataset_root'])/seq/'groundtruth.txt',delimiter=',')
        by_arm['default'][seq]=statistics([r['public_bbox'] for r in rows],gt)
    for arm in ['spatial','pooled']:
        receipt=json.loads((root/(arm+'_recursive_receipt.json')).read_text())
        assert receipt['status']=='complete' and {r['sequence'] for r in receipt['sequences']}==sequences
        for item in receipt['sequences']:
            path=root/'recursive'/(arm+'_'+item['sequence']+'.json'); assert sha(path)==item['sha256']
            rows=json.loads(path.read_text())['rows']; seq=item['sequence']
            assert len(rows)==len(baseline[seq])
            assert [r['frame'] for r in rows]==list(range(len(rows)))
            gt=np.loadtxt(Path(spec['dataset_root'])/seq/'groundtruth.txt',delimiter=',')
            by_arm[arm][seq]=statistics([r['bbox'] for r in rows],gt)
            by_arm[arm][seq].update(changes=sum(r['choice']!=0 for r in rows),none=sum(r['none'] for r in rows))
    aggregates={}
    for arm, rows in by_arm.items():
        totals={k:sum(row[k] for row in rows.values()) for k in ['valid_frames','iou_sum','low_iou_frames','failure_episodes']}
        totals['mean_iou']=totals['iou_sum']/totals['valid_frames']
        totals['macro_sequence_mean_iou']=float(np.mean([r['mean_iou'] for r in rows.values()]))
        totals['failure_sequences']=sum(r['failure_episodes']>0 for r in rows.values())
        aggregates[arm]=totals
    a,b,c=aggregates['spatial'],aggregates['default'],aggregates['pooled']
    positives=sum(by_arm['spatial'][s]['mean_iou']>by_arm['default'][s]['mean_iou'] for s in sequences)
    broken=[s for s in sorted(sequences) if by_arm['default'][s]['failure_episodes']==0 and by_arm['spatial'][s]['failure_episodes']>0]
    gates=dict(mean_iou=a['mean_iou']>b['mean_iou'] and a['mean_iou']>c['mean_iou'],
               fewer_low_iou_frames=a['low_iou_frames']<b['low_iou_frames'],
               no_episode_increase=a['failure_episodes']<=b['failure_episodes'],
               sequence_coverage=positives>=3, successful_sequence_protection=len(broken)==0)
    result=dict(status='complete',dataset='DepthTrack Train development fold5 only',aggregates=aggregates,
                per_sequence=by_arm,positive_sequences=positives,new_failure_sequences=broken,
                recursive_gate=gates,recursive_gate_pass=all(gates.values()),
                source_sha256=frozen['source_sha256'],training_result_sha256=sha(root/'training_result.json'),
                metric='continuous xywh IoU; initialization and invalid GT excluded; episodes require10 consecutive valid IoU<=.1 frames',
                next='freeze low22 candidate evaluation' if all(gates.values()) else 'stop this frozen variant; no public evaluation')
    (root/'recursive_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(aggregates=aggregates,gates=gates,pass_gate=all(gates.values()))),flush=True)


if __name__=='__main__':
    main()
