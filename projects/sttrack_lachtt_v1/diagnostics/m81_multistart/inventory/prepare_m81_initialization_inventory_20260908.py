"""Train-only fixed multistart inventory; no model inference or training."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math
import numpy as np
from PIL import Image

B=Path('/root/autodl-tmp');P=B/'sttrack_m78_raw_competition_20260908';R=B/'sttrack_m81_multistart_20260908'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
assert not R.exists()
assert sha(P/'training_spec.json')=='57cdd314efd5359fa2e16be4f364c5568f1ca498b41df718517173610fe4d865'
t=read(P/'training_spec.json');e=read(P/'recursive_spec.json')
assert t['seed']==2027 and len(t['sequence_order'])==130
root=Path(t['dataset_root']);plans={};summaries={}
for split,rows in [('fit',t['sequence_order']),('development',e['cases'])]:
    episodes=[];sequences=[];updates=0;total_calls=0
    for row in rows:
        seq=row['sequence'];folder=root/seq;n=row['rgb_frames'] if split=='fit' else row['frames']
        gt_path=folder/'groundtruth.txt';gt_sha=row['groundtruth_sha256'] if split=='fit' else row['gt_sha256']
        assert sha(gt_path)==gt_sha
        gt=np.loadtxt(gt_path,delimiter=',').reshape(-1,4)
        if seq=='toy07_indoor_320':
            assert gt_sha=='683e8ae7ae401b71b8d10e9bb489c3956a150163606f5bac925a911f395444e2' and len(gt)==1406 and n==1367
            gt=gt[:n]
        assert len(gt)==n
        valid=np.isfinite(gt).all(1)&(gt[:,2:]>0).all(1)
        assert valid[0]
        first=row['first_box'] if split=='fit' else row['init_bbox']
        assert np.array_equal(gt[0],first)
        starts=[0];candidates=[];candidate=512
        while candidate<n-1:
            eligible=bool(valid[candidate]);reason='nonfinite_or_nonpositive_gt'
            if eligible:
                path=folder/'color'/('%08d.jpg'%(candidate+1))
                with Image.open(path) as image:w,h=image.size
                x,y,bw,bh=map(float,gt[candidate]);cx=x+bw/2;cy=y+bh/2
                eligible=0<=cx<w and 0<=cy<h
                reason='valid_box_center_in_image' if eligible else 'gt_center_outside_image'
            candidates.append(dict(frame=candidate,accepted=eligible,reason=reason))
            if eligible:
                starts.append(candidate);candidate+=512
            else:candidate+=32
        ends=starts[1:]+[n-1]
        seq_episodes=[]
        for index,(start,end) in enumerate(zip(starts,ends)):
            assert start<end and (index==0 or start%32==0)
            rgb=folder/'color'/('%08d.jpg'%(start+1));depth=folder/'depth'/('%08d.png'%(start+1))
            assert rgb.is_file() and depth.is_file()
            with Image.open(rgb) as image:width,height=image.size
            with Image.open(depth) as image:assert image.size==(width,height)
            identity=seq+':'+str(start)
            episode=dict(id=identity,sequence=seq,episode_index=index,start_frame=start,end_frame_inclusive=end,
                init_bbox=list(map(float,gt[start])),image=str(rgb),image_sha256=sha(rgb),depth=str(depth),depth_sha256=sha(depth),
                width=width,height=height,track_calls=end-start,groundtruth_sha256=gt_sha,
                text_source='reuse_frozen_t0_category_bank' if start==0 else 'pending_current_initialization_only_caption',
                groundtruth_reinit='fixed_episode_boundary_not_failure_triggered')
            seq_episodes.append(episode);episodes.append(episode)
        assert starts[0]==0 and ends[-1]==n-1
        assert sum(x['track_calls'] for x in seq_episodes)==n-1
        assert all(seq_episodes[i]['end_frame_inclusive']==seq_episodes[i+1]['start_frame'] for i in range(len(seq_episodes)-1))
        native_blocks=[(i,min(i+32,n-1)) for i in range(0,n-1,32)]
        split_blocks=[(i,min(i+32,x['end_frame_inclusive'])) for x in seq_episodes for i in range(x['start_frame'],x['end_frame_inclusive'],32)]
        assert native_blocks==split_blocks
        count=sum(bool(valid[a+1:b+1].any()) for a,b in native_blocks);updates+=count;total_calls+=n-1
        sequences.append(dict(sequence=seq,frames=n,track_calls=n-1,starts=starts,episodes=len(starts),valid_loss_update_windows=count,boundary_candidates=candidates))
    assert total_calls==(186694 if split=='fit' else 33108)
    plans[split]=dict(split=split,source='DepthTrack Train only',episodes=episodes,sequences=sequences)
    summaries[split]=dict(sequences=len(rows),episodes=len(episodes),additional_initializations=len(episodes)-len(rows),track_calls=total_calls,
        valid_loss_update_windows=updates,caption_requests_pending=sum(x['start_frame']>0 for x in episodes),
        maximum_episode_calls=max(x['track_calls'] for x in episodes),minimum_episode_calls=min(x['track_calls'] for x in episodes),
        every_original_transition_covered_exactly_once=True,gradient_accumulation_windows_identical=True)
assert summaries['fit']['valid_loss_update_windows']==5798
assert not set(x['sequence'] for x in plans['fit']['sequences'])&set(x['sequence'] for x in plans['development']['sequences'])
R.mkdir()
for split,data in plans.items():write(R/(split+'_initializations.json'),data)
spec=dict(status='initialization_inventory_prepared_no_training',observed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),seed=2027,additional_seeds=[],
    parent_training_spec_sha256=sha(P/'training_spec.json'),interval_track_frames=512,invalid_boundary_advance=32,
    eligibility='Finite positive GT box with center inside current image. Does not establish visibility, sharpness or caption correctness.',
    initialization_policy='Fixed boundaries from Train GT, never from tracking failures; all chronological transitions retained exactly once.',
    caption_policy='For each new boundary, current RGB image plus target box/crop only; no future frame, sequence name or category hint. Reuse t0 bank. Category/Empty share masks, slots and episode list.',
    learning_policy='M78 raw competition, optimizer and frozen architecture. No M80 dropout, reverse traversal, new memory, consistency loss, or new template rule.',
    compared_training_arms=['multistart_category','multistart_empty'],reference='Sealed M78 single-start Category/Empty with same tracking calls, accumulation windows and initial tensors.',
    evaluation_policy='Keep full t0 development22, additionally evaluate all compared models on this fixed development multistart manifest. Never compare reset-enhanced evaluation only against single-start native.',
    summaries=summaries,fit_manifest_sha256=sha(R/'fit_initializations.json'),development_manifest_sha256=sha(R/'development_initializations.json'),
    training_has_started=False,captions_generated=0,new_weights_created=False,public_evaluation_allowed=False,
    limitations=['Multiple initialization uses more GT initialization boxes and extra initialization forwards, despite identical tracking calls and optimizer windows.',
        'Initialization-specific captions differ with the new start; results test protocol coverage, not pure visual-reset causality.',
        'Later-start descriptions must be generated before training; no automatic t0 text reuse at later starts.',
        'Reused development sequences and historical controls are not independent heldout evidence.'])
write(R/'inventory_spec.json',spec);print(json.dumps(spec,indent=2))
