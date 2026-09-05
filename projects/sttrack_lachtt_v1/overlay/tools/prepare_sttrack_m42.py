"""Freeze sequence-disjoint Train windows before spatial feature collection."""
import argparse
from collections import defaultdict, Counter
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def overlap(a, b):
    iw = max(0., min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0]))
    ih = max(0., min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2]*a[3]+b[2]*b[3]-inter)


def spaced(values, count):
    if not values:
        return []
    return [values[i] for i in sorted(set(np.linspace(0, len(values)-1, min(count, len(values))).round().astype(int).tolist()))]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    args = p.parse_args(); root = args.root; root.mkdir(parents=True, exist_ok=True)
    ledger_path = Path('/root/autodl-tmp/sttrack_lachtt_m18_0_causal_survival_target_closure_v1_20260901/split_ledger.json')
    ledger = json.loads(ledger_path.read_text())
    sequences = ledger['all_sequences']['training']
    assert len(sequences) == 85
    trace_root = Path('/root/autodl-tmp/sttrack_innovation_v1/risk_recovery_full152_v1')
    trace = defaultdict(list)
    for shard in [0, 1]:
        data = json.loads((trace_root / f'shard{shard}.json').read_text())
        assert data['complete']
        for row in data['rows']:
            if row['sequence'] in sequences:
                trace[row['sequence']].append(dict(frame_index=row['frame_index'], bbox=row['public_bbox'], score=row['public_score']))
    dataset = Path('/root/autodl-tmp/depthtrack/train/sequences')
    plans, targets, unused_annotations = [], {}, {}
    total_frames = [0, 0]
    for seq in sorted(sequences):
        rows = sorted(trace[seq], key=lambda r:r['frame_index'])
        gt = np.loadtxt(dataset / seq / 'groundtruth.txt', delimiter=',')
        assert len(rows) == len(list((dataset / seq / 'color').glob('*.jpg')))
        assert len(rows) == len(list((dataset / seq / 'depth').glob('*.png')))
        assert [r['frame_index'] for r in rows] == list(range(len(rows)))
        assert len(gt) >= len(rows)
        if len(gt) > len(rows):
            unused_annotations[seq] = len(gt) - len(rows)
        healthy, hard, unavailable = [], [], []
        # Sampling uses training GT. No GT appears in the separate inference input file except initialization.
        for f in range(10, len(rows)-4):
            g = gt[f]
            if not np.isfinite(g).all() or g[2] <= 0 or g[3] <= 0:
                unavailable.append(f)
                continue
            v = overlap(rows[f]['bbox'], g)
            if v >= .5:
                healthy.append(f)
            elif v <= .1:
                hard.append(f)
        selected = sorted(set(spaced(healthy, 8) + spaced(hard, 8) + spaced(unavailable, 2)))
        assert selected
        fold = int.from_bytes(hashlib.sha256(('sttrack-lachtt-outer-v1\0'+seq).encode()).digest()[:8], 'big') % 6
        assert fold in [2,3,4,5]
        shard = total_frames.index(min(total_frames)); total_frames[shard] += selected[-1] + 1
        plans.append(dict(sequence=seq, fold=fold, split='development_holdout' if fold == 5 else 'fit', shard=shard,
                          event_frames=selected, init_bbox=rows[0]['bbox'], expected_rows=rows[:selected[-1]+1]))
        for f in selected:
            key=f'{seq}@{f}'
            g=gt[f];valid=bool(np.isfinite(g).all() and g[2]>0 and g[3]>0)
            targets[key]=dict(sequence=seq,fold=fold,frame=f,visible=valid,gt_bbox=g.tolist() if valid else None)
    (root/'inference_inputs.json').write_text(json.dumps(plans)+'\n')
    (root/'training_labels.json').write_text(json.dumps(targets)+'\n')
    repo=Path('/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1')
    source_names=['lib/test/tracker/sttrack.py','lib/models/sttrack/sttrack.py',
                  'lib/test/tracker/sttrack_lachtt_observation.py','lib/train/data/processing_utils.py',
                  'lib/train/dataset/depth_utils.py','experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml',
                  'lib/test/tracker/sttrack_local_spatial_observation.py',
                  'lib/models/sttrack/lachtt_local_spatial_association.py',
                  'tools/prepare_sttrack_m42.py','tools/collect_sttrack_m42.py']
    spec=dict(schema='sttrack_m42_local_spatial_association_v1',repository=str(repo),dataset_root=str(dataset),
         checkpoint='/root/autodl-tmp/sttrack_checkpoints/STTrack_Vot22.pth.tar',
         checkpoint_sha256='cacbd799115be1aaeb049cee0db89270851e3b6dd68997553b4c2c31c1104f98',
         source_sha256={name:sha(repo/name) for name in source_names},ledger_sha256=sha(ledger_path),
         fit_folds=[2,3,4],development_holdout_fold=5,fold_history='All these development folds have prior experiment exposure; no fresh/unseen-test claim.',
         quarantine_folds_unopened=[0,1],public_datasets_used=False,
         sequence_count=len(plans),event_count=len(targets),shard_frames=total_frames,
         trailing_annotations_without_images=unused_annotations,
         split_counts=dict(Counter(x['split'] for x in plans)),
         inference_inputs_sha256=sha(root/'inference_inputs.json'),labels_sha256=sha(root/'training_labels.json'),
         candidates=10,nms_kernel=3,roi_cells=4,modalities=['rgb','depth'],references=['initial','dynamic','previous_prediction'],
         variants=['spatial','pooled'],feature_storage='FP16, converted to FP32 for fitting',
         optimization=dict(seed=2026,epochs=20,batch_size=32,lr=.001,weight_decay=.0001,optimizer='AdamW',grad_clip=5.,
             target='Highest-IoU candidate if max IoU>=.5, otherwise NONE; no class balancing or heldout epoch selection.',checkpoint='fixed final epoch only'),
         information_gate=dict(mean_iou='spatial > pooled and spatial > default',
             alternative_fixes='spatial fixes more default failures than it breaks default successes',
             sequence_coverage='positive selected-IoU gain over default on at least 3 heldout sequences',
             next='If passed, run paired prediction-crop recursive holdout evaluation; no public launch from static information gate.'),
         policy='Argmax among ten candidate logits and NONE; NONE preserves default top1. No threshold scan.',
         no_backbone_or_box_regression_training=True,language_enabled=False)
    (root/'spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    print(json.dumps({k:spec[k] for k in ['sequence_count','event_count','shard_frames','split_counts']},indent=2))


if __name__=='__main__':
    main()
