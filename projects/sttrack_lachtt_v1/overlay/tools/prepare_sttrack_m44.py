"""Freeze Train-only transition and healthy candidate-pair observations."""
import argparse
from collections import defaultdict,Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    root=args.root;root.mkdir(parents=True,exist_ok=True)
    repo=Path('/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1');sys.path.insert(0,str(repo))
    from tools.prepare_sttrack_m42 import overlap,spaced
    ledger_path=Path('/root/autodl-tmp/sttrack_lachtt_m18_0_causal_survival_target_closure_v1_20260901/split_ledger.json')
    sequences=json.loads(ledger_path.read_text())['all_sequences']['training'];assert len(sequences)==85
    traces={str(Path('/root/autodl-tmp/sttrack_innovation_v1/risk_recovery_full152_v1')/f'shard{i}.json'):None for i in [0,1]}
    rows=defaultdict(list)
    for path in traces:
        traces[path]=sha(path);data=json.loads(Path(path).read_text());assert data['complete']
        for x in data['rows']:
            if x['sequence'] in sequences:rows[x['sequence']].append(dict(frame_index=x['frame_index'],bbox=x['public_bbox'],score=x['public_score']))
    dataset=Path('/root/autodl-tmp/depthtrack/train/sequences');plans=[];labels={};load=[0,0];unused={};strata=Counter()
    for name in sorted(sequences):
        trace=sorted(rows[name],key=lambda x:x['frame_index']);n=len(trace);folder=dataset/name
        assert [x['frame_index'] for x in trace]==list(range(n))
        assert len(list((folder/'color').glob('*.jpg')))==n==len(list((folder/'depth').glob('*.png')))
        gt=np.loadtxt(folder/'groundtruth.txt',delimiter=',');assert len(gt)>=n
        if len(gt)>n:unused[name]=len(gt)-n
        valid=np.isfinite(gt[:n]).all(1)&(gt[:n,2]>0)&(gt[:n,3]>0)
        ious=np.full(n,np.nan)
        for f in range(1,n):
            if valid[f]:ious[f]=overlap(trace[f]['bbox'],gt[f])
        low=valid&(ious<=.1)
        starts=np.flatnonzero(np.diff(np.r_[False,low,False].astype(int))==1)
        ends=np.flatnonzero(np.diff(np.r_[False,low,False].astype(int))==-1)
        onset=[int(a) for a,b in zip(starts,ends) if b-a>=10 and 10<=a<n-4]
        groups=dict(healthy=spaced([f for f in range(10,n-4) if ious[f]>=.5],12),
            intermediate=spaced([f for f in range(10,n-4) if .1<ious[f]<.5],4),
            late_low=spaced([f for f in range(10,n-4) if low[f]],2),
            unavailable=spaced([f for f in range(10,n-4) if not valid[f]],2),
            transition=sorted({f+d for f in spaced(onset,6) for d in [-2,0,2] if 2<=f+d<n}))
        selected=sorted(set(f for group in groups.values() for f in group));assert selected
        fold=int.from_bytes(hashlib.sha256(('sttrack-lachtt-outer-v1\0'+name).encode()).digest()[:8],'big')%6
        assert fold in [2,3,4,5]
        shard=load.index(min(load));load[shard]+=selected[-1]
        plans.append(dict(sequence=name,fold=fold,split='development' if fold==5 else 'fit',shard=shard,
            event_frames=selected,frames=n,init_bbox=trace[0]['bbox'],expected_rows=trace[:selected[-1]+1]))
        for f in selected:
            tags=[key for key,value in groups.items() if f in value];strata.update(tags)
            labels[f'{name}@{f}']=dict(sequence=name,fold=fold,frame=f,strata=tags,
                current=gt[f].tolist() if valid[f] else None,previous=gt[f-1].tolist() if valid[f-1] else None)
    (root/'inference_inputs.json').write_text(json.dumps(plans)+'\n')
    (root/'training_labels.json').write_text(json.dumps(labels)+'\n')
    names=['lib/test/tracker/sttrack.py','lib/models/sttrack/sttrack.py',
        'lib/test/tracker/sttrack_lachtt_observation.py','lib/test/tracker/sttrack_local_spatial_observation.py',
        'lib/test/tracker/sttrack_candidate_set_observation.py','lib/models/sttrack/lachtt_candidate_set.py',
        'lib/train/data/processing_utils.py','lib/train/dataset/depth_utils.py',
        'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml','tools/prepare_sttrack_m42.py',
        'tools/prepare_sttrack_m44.py','tools/collect_sttrack_m44.py']
    spec=dict(schema='sttrack_m44_temporal_candidate_set_v1',repository=str(repo),dataset_root=str(dataset),
        checkpoint='/root/autodl-tmp/sttrack_checkpoints/STTrack_Vot22.pth.tar',
        checkpoint_sha256='cacbd799115be1aaeb049cee0db89270851e3b6dd68997553b4c2c31c1104f98',
        source_sha256={name:sha(repo/name) for name in names},baseline_trace_sha256=traces,
        ledger_sha256=sha(ledger_path),inference_inputs_sha256=sha(root/'inference_inputs.json'),labels_sha256=sha(root/'training_labels.json'),
        sequences=85,events=len(labels),split_sequences=dict(Counter(x['split'] for x in plans)),shard_frames=load,strata_counts=dict(strata),
        annotation_rows_without_images=unused,fit_folds=[2,3,4],development_fold=5,
        split_scope='Existing development folds; official pretrained backbone predates this head split. Mixed full152 prediction JSONs parsed and filtered; folds0/1 provide no new GT/native features/head fitting/evaluation.',
        architecture=dict(candidates_per_frame=10,previous_frames=1,roi_cells=4,template_references=2,
            descriptor_dim=128,transformer_layers=2,attention_heads=4,ffn_dim=256,dropout=0.,
            partial_matching_weight=.25,matching_temperature=.1,unmatched=True),
        variants=['geometry','appearance'],primary='geometry',ablation='Zero explicit candidate center/size coordinates, preserving all appearance, response, previous-selection and template inputs.',
        optimization=dict(seed=2026,epochs=20,batch_size=32,lr=.0003,weight_decay=.01,grad_clip=1.,optimizer='AdamW',checkpoint='fixed final epoch only'),
        feature_storage='FP16 raw RoIs; FP32 geometry/scores; conversion to FP32 for fitting and inference',
        estimated_feature_bytes=len(labels)*1081344,language_enabled=False,default_template_updates=True,
        backbone_and_box_head_frozen=True,policy='11-way argmax; NONE keeps default candidate0; first tracking frame collects preceding candidates using default.',
        training_target='Maximum-IoU candidate if IoU>=.5 else NONE; partial true-target pair correspondence, other identities unlabeled.',
        next='After collection integrity and fixed training, always run both complete22-sequence recursive development paths; static snapshots are diagnostic, not the promotion gate.',
        recursive_performance_gate=dict(mean_iou_gain_at_least=.01,fewer_low_frames=True,no_episode_increase=True,positive_sequences_at_least=3,no_new_failure_in_default_zero_episode_sequences=True),
        mechanism_claim='Geometry must exceed appearance control for an explicit-position incremental claim; failure of this attribution does not erase separately measured main-model performance.',
        public_gate='Frozen low22 only after recursive main performance pass; then EAO/ROB each >=M39default+1pp, ACC>=default-.10pp, fewer failures, all7 default-zero-failure sequences preserved before same-weight three-dataset validation.',
        upstream_reference='https://github.com/visionml/pytracking/tree/master/ltr/models/target_candidate_matching',
        no_public_gt_for_training=True)
    (root/'spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    print(json.dumps({k:spec[k] for k in ['sequences','events','split_sequences','shard_frames','strata_counts','estimated_feature_bytes']},indent=2))


if __name__=='__main__':main()
