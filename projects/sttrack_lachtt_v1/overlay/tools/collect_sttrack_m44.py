"""Collect consecutive full candidate sets on exact default predicted crops."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--shard',type=int,required=True);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();root=args.root;spec=json.loads((root/'spec.json').read_text());repo=Path(spec['repository']);sys.path.insert(0,str(repo))
    for name,digest in spec['source_sha256'].items():assert sha(repo/name)==digest,name
    assert sha(spec['checkpoint'])==spec['checkpoint_sha256']
    assert sha(root/'inference_inputs.json')==spec['inference_inputs_sha256']
    from lib.config.sttrack.config import cfg,update_config_from_file
    from lib.test.tracker.sttrack import STTrack
    from lib.test.tracker.sttrack_candidate_set_observation import observe_candidate_set
    from lib.test.tracker.sttrack_local_spatial_observation import NativeReferenceBank
    from lib.train.dataset.depth_utils import get_rgbd_frame
    torch.set_num_threads(1);update_config_from_file(str(repo/'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    params=SimpleNamespace(cfg=cfg,checkpoint=spec['checkpoint'],template_factor=2.,template_size=128,search_factor=4.,search_size=256,save_all_boxes=False,debug=0)
    tracker=STTrack(params);forward=tracker.network.forward;capture={}
    def observed(*pos,**kw):
        kw['return_candidate_features']=True;out=forward(*pos,**kw);capture['output']=out[0];return out
    tracker.network.forward=observed
    plans=json.loads((root/'inference_inputs.json').read_text())
    if args.smoke:
        plans=[x for x in plans if x['split']=='fit'][:2]
        for x in plans:x['event_frames']=[10,50,60]
    else:plans=[x for x in plans if x['shard']==args.shard]
    outdir=root/('smoke_features' if args.smoke else 'features');outdir.mkdir(exist_ok=True);receipts=[];started=time.time()
    def image_at(name,frame):
        folder=Path(spec['dataset_root'])/name
        return get_rgbd_frame(str(folder/'color'/f'{frame+1:08d}.jpg'),str(folder/'depth'/f'{frame+1:08d}.png'),dtype='rgbcolormap',depth_clip=True)
    for index,case in enumerate(plans):
        tracker.initialize(image_at(case['sequence'],0),dict(init_bbox=list(case['init_bbox'])));bank=NativeReferenceBank(case['init_bbox'])
        events=set(case['event_frames']);needed=events|{f-1 for f in events};previous=None;previous_frame=None
        tensors={k:[] for k in ['current','previous','references','geometry','scores','current_boxes','previous_boxes','public_bbox','previous_public_bbox']}
        records=[];updates=0;maxbox=maxscore=0.
        for frame in range(1,max(events)+1):
            prior=list(tracker.state);dynamic=tracker.z_dict[1];im=image_at(case['sequence'],frame)
            out=tracker.track(im);expected=case['expected_rows'][frame]
            error=float(np.abs(np.asarray(out['target_bbox'])-expected['bbox']).max());score=abs(float(out['best_score'])-expected['score'])
            assert error<=1e-4 and score<=1e-6,(case['sequence'],frame,error,score)
            maxbox=max(maxbox,error);maxscore=max(maxscore,score)
            output=capture.pop('output');resize=256/math.ceil(math.sqrt(prior[2]*prior[3])*4.)
            with torch.no_grad():
                refs=bank.before_decision(output['candidate_features'],dynamic)
                if frame in needed:
                    current=observe_candidate_set(output,tracker.output_window,prior,resize,im.shape)
                    assert np.abs(current['boxes'][0].numpy()-np.asarray(out['target_bbox'])).max()<.001
                    if frame in events:
                        assert previous_frame==frame-1
                        tensors['current'].append(current['rois'].half().cpu());tensors['previous'].append(previous['rois'].half().cpu())
                        tensors['references'].append(refs[:2].half().cpu())
                        tensors['geometry'].append(torch.cat([current['geometry'],previous['geometry']]))
                        tensors['scores'].append(torch.cat([current['scores'],previous['scores']]))
                        tensors['current_boxes'].append(current['boxes']);tensors['previous_boxes'].append(previous['boxes'])
                        tensors['public_bbox'].append(torch.tensor(out['target_bbox']));tensors['previous_public_bbox'].append(torch.tensor(prior))
                        records.append(dict(key=f"{case['sequence']}@{frame}",frame=frame,previous_frame=frame-1,previous_choice=0,template_updates_before_frame=updates))
                    previous=current;previous_frame=frame
                bank.after_decision(output['candidate_features'],prior,resize,out['target_bbox'],tracker.z_dict[1])
            updates+=int(tracker.z_dict[1] is not dynamic)
        data={key:torch.stack(value) for key,value in tensors.items()};assert all(torch.isfinite(x).all() for x in data.values())
        data.update(records=records,sequence=case['sequence'],fold=case['fold'],split=case['split'],spec_sha256=sha(root/'spec.json'))
        path=outdir/(case['sequence']+'.pt');torch.save(data,path)
        receipt=dict(sequence=case['sequence'],events=len(records),frames=frame,max_bbox_error_px=maxbox,max_score_error=maxscore,
                     template_updates=updates,feature_sha256=sha(path),bytes=path.stat().st_size)
        receipts.append(receipt);print(json.dumps(dict(done=index+1,total=len(plans),elapsed=time.time()-started,**receipt)),flush=True)
    for name,digest in spec['source_sha256'].items():assert sha(repo/name)==digest,name
    receipt=dict(status='complete',elapsed_seconds=time.time()-started,sequences=receipts,frames=sum(x['frames'] for x in receipts),events=sum(x['events'] for x in receipts),
        checkpoint_sha256=spec['checkpoint_sha256'],source_unchanged=True,labels_opened=False,training_steps=0,spec_sha256=sha(root/'spec.json'))
    (root/('smoke_receipt.json' if args.smoke else f'shard{args.shard}_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
