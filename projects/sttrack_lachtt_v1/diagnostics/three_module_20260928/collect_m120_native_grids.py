"""Collect missing native grids on the original legal recursive Train states."""
import argparse,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from collect_train_states import sha
from train_fixed_visual_selector import load_initial_search
from region_patch_evidence import crop_geometry


def main():
    parser=argparse.ArgumentParser()
    for name in ['cache','origins','full152-spec','repository','checkpoint','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--shard',type=int,choices=[0,1],required=True)
    parser.add_argument('--mode',choices=['sanity','full'],required=True)
    args=parser.parse_args();assert not args.output.exists()
    args.output.mkdir(parents=True);(args.output/'features').mkdir()
    prep=json.loads((args.cache/'preparation.json').read_text())
    assert prep['source_spec_sha256']==sha(args.full152_spec)
    assert prep['inference_inputs_sha256']==sha(args.cache/'inference_inputs.json')
    spec=json.loads(args.full152_spec.read_text())
    plans=[r for r in json.loads((args.cache/'inference_inputs.json').read_text()) if r['shard']==args.shard]
    prior=json.loads((args.cache/('collect_shard'+str(args.shard)+'.json')).read_text())
    assert prior['status']=='complete' and not prior['smoke']
    assert prior['checkpoint_sha256']==sha(args.checkpoint)
    source={r['sequence']:r for r in prior['sequences']}
    initial_rois=load_initial_search(args.origins,args.cache)
    if args.mode=='sanity':plans=[next(r for r in plans if r['split']=='fit')]
    sys.path.insert(0,str(args.repository))
    from lib.config.sttrack.config import cfg,update_config_from_file
    from lib.test.tracker.sttrack import STTrack
    from lib.test.tracker.sttrack_candidate_set_observation import observe_candidate_set
    from lib.test.tracker.sttrack_local_spatial_observation import search_rois
    from lib.train.data.processing_utils import sample_target
    from lib.train.dataset.depth_utils import get_rgbd_frame
    torch.set_num_threads(1);torch.manual_seed(2027)
    config=args.repository/'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'
    update_config_from_file(str(config))
    params=SimpleNamespace(cfg=cfg,checkpoint=str(args.checkpoint),template_factor=2.,template_size=128,
        search_factor=4.,search_size=256,save_all_boxes=False,debug=0)
    tracker=STTrack(params);tracker.network.eval().requires_grad_(False)
    frozen={k:v.cpu().clone() for k,v in tracker.network.state_dict().items()}
    forward=tracker.network.forward;capture={}
    def observed(*pos,**kwargs):
        kwargs['return_candidate_features']=True
        outputs=forward(*pos,**kwargs);capture['output']=outputs[0]
        return outputs
    tracker.network.forward=observed
    branches={'rgb':'search_rgb_tokens','depth':'search_depth_tokens','fused':'search_fused_tokens'}
    channels=tracker.network.backbone.embed_dim;records=[];started=time.time()

    def grids(features):
        values={name:features[key].detach() for name,key in branches.items()}
        assert all(v.shape==(1,256,channels) and bool(torch.isfinite(v).all()) for v in values.values())
        return {k:v[0].half().cpu() for k,v in values.items()}

    def coverage(mask):
        observed=torch.from_numpy(~mask).float()[None,None]
        return F.avg_pool2d(observed,16,16).reshape(256)

    with torch.no_grad():
        for case in plans:
            sequence=case['sequence'];folder=Path(spec['dataset_root'])/sequence
            path=args.cache/'features'/(sequence+'.pt')
            assert sha(path)==source[sequence]['feature_sha256']
            cached=torch.load(path,map_location='cpu')
            assert not cached['labels_loaded'] and cached['split']==case['split']
            assert cached['event_frames']==case['event_frames']
            frames=case['event_frames'][:3] if args.mode=='sanity' else case['event_frames']
            events=set(frames)
            indices={frame:i for i,frame in enumerate(cached['event_frames'])}
            def image_at(frame):
                return get_rgbd_frame(str(folder/'color'/('%08d.jpg'%(frame+1))),
                    str(folder/'depth'/('%08d.png'%(frame+1))),dtype='rgbcolormap',depth_clip=True)
            init_box=list(case['init_bbox']);initial_image=image_at(0)
            tracker.initialize(initial_image,dict(init_bbox=init_box))
            patch,resize,mask=sample_target(initial_image,init_box,4.,output_sz=256)
            initial_output=tracker.network.forward(template=tracker.z_dict,search=[tracker.preprocessor.process(patch)],
                ce_template_mask=tracker.box_mask_z,track_query_before=None,keep_rate=tracker.keep_rate,
                return_candidate_features=True)[0]
            initial_features=initial_output['candidate_features'];initial=grids(initial_features)
            initial_roi=search_rois(initial_features,[dict(bbox=init_box)],init_box,resize)[0].half().cpu()
            assert torch.equal(initial_roi,initial_rois[sequence])
            assert tracker.frame_id==0 and tracker.track_query_before is None and list(tracker.state)==init_box
            initial_valid=coverage(mask);capture.clear();del initial_output,initial_features
            values={key:[] for key in list(branches)+['observed_fraction','crop_origin','prior_bbox','resize_factor']}
            maximum_box_error=maximum_score_error=0.
            for frame in range(1,max(frames)+1):
                before=list(tracker.state);image=image_at(frame);public=tracker.track(image)
                expected=case['expected_rows'][frame]
                box_error=float(np.abs(np.asarray(public['target_bbox'])-expected['bbox']).max())
                score_error=abs(float(public['best_score'])-expected['score'])
                assert box_error<=1e-4 and score_error<=1e-6,(sequence,frame,box_error,score_error)
                maximum_box_error=max(maximum_box_error,box_error);maximum_score_error=max(maximum_score_error,score_error)
                output=capture.pop('output')
                if frame in events:
                    at=indices[frame]
                    _,factor,mask=sample_target(image,before,4.,output_sz=256)
                    candidates=observe_candidate_set(output,tracker.output_window,before,factor,image.shape)
                    assert torch.equal(candidates['boxes'],cached['boxes'][at]),(sequence,frame,'boxes')
                    assert torch.equal(candidates['rois'].half().cpu(),cached['candidate_rois'][at]),(sequence,frame,'RoI')
                    assert torch.equal(torch.tensor(before,dtype=torch.float32),cached['prior_bbox'][at]),(sequence,frame,'prior')
                    assert float(torch.tensor(factor,dtype=torch.float32))==float(cached['resize_factor'][at])
                    search=grids(output['candidate_features'])
                    for key in branches:values[key].append(search[key])
                    values['observed_fraction'].append(coverage(mask))
                    values['crop_origin'].append(torch.tensor(crop_geometry(before),dtype=torch.float32))
                    values['prior_bbox'].append(torch.tensor(before,dtype=torch.float64))
                    values['resize_factor'].append(torch.tensor(factor,dtype=torch.float64))
            data={k:torch.stack(v) for k,v in values.items()}
            assert len(data['rgb'])==len(frames)
            assert all(bool(torch.isfinite(v).all()) for v in data.values())
            data.update(initial_grids=initial,initial_observed_fraction=initial_valid,
                initial_crop_origin=crop_geometry(init_box),initial_bbox=init_box,initial_resize_factor=float(resize),
                sequence=sequence,split=case['split'],event_frames=frames,channels=channels,search_positions=256,
                original_feature_sha256=source[sequence]['feature_sha256'],GT_loaded=False,text_loaded=False,
                native_tracker_history_replayed=True,auxiliary_state_committed=False,optimizer_steps=0)
            target=args.output/'features'/(sequence+'.pt');torch.save(data,target)
            row=dict(sequence=sequence,split=case['split'],events=len(frames),frames=frame,
                maximum_box_error_px=maximum_box_error,maximum_score_error=maximum_score_error,
                candidate_regions_exact=True,candidate_boxes_exact=True,initial_region_exact=True,
                original_feature_sha256=source[sequence]['feature_sha256'],feature_sha256=sha(target),bytes=target.stat().st_size,
                seconds=time.time()-started)
            records.append(row);print(json.dumps(dict(done=len(records),total=len(plans),**row)),flush=True)
    assert all(torch.equal(v.cpu(),frozen[k]) for k,v in tracker.network.state_dict().items())
    result=dict(status='complete_M120_'+args.mode+'_native_grid_shard',mode=args.mode,shard=args.shard,seed=2027,
        sequences=records,events=sum(r['events'] for r in records),frames=sum(r['frames'] for r in records),
        bytes=sum(r['bytes'] for r in records),channels=channels,optimizer_steps=0,GT_loaded=False,text_loaded=False,
        native_tracker_history_replayed=True,auxiliary_state_committed=False,frozen_state_exact=True,
        no_public_evaluation=True,checkpoint_sha256=sha(args.checkpoint),preparation_sha256=sha(args.cache/'preparation.json'),
        source_sha256=sha(__file__),native_model_source_sha256=sha(args.repository/'lib/models/sttrack/sttrack.py'),
        native_tracker_source_sha256=sha(args.repository/'lib/test/tracker/sttrack.py'),config_sha256=sha(config),
        elapsed_seconds=time.time()-started,gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],events=result['events'],seconds=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':
    main()
