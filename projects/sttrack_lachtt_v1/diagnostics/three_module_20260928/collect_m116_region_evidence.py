"""Frozen CLIP patch-token cache plus human phrase diagnostics, DepthTrack Train only."""
import argparse,importlib.util,json,time
from pathlib import Path
import clip,cv2,numpy as np,torch
from PIL import Image
from torch.nn import functional as F
from collect_train_states import sha
from region_patch_evidence import crop_geometry,phrase_scores,coordinate_sanity


def load_utility(path,name):
    spec=importlib.util.spec_from_file_location(name,str(path))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def patch_features(model,images):
    captured=[]
    hook=model.visual.transformer.register_forward_hook(lambda module,args,out:captured.append(out.detach()))
    cls=model.encode_image(images)
    hook.remove()
    hidden=captured[0].permute(1,0,2)
    manual=model.visual.ln_post(hidden[:,0])@model.visual.proj
    assert torch.equal(cls,manual),'Same input CLS extraction must reproduce the official image encoder.'
    patches=model.visual.ln_post(hidden[:,1:])@model.visual.proj
    assert patches.shape==(len(images),256,768)
    assert bool(torch.isfinite(patches).all()) and bool(torch.isfinite(cls).all())
    return patches,cls


def main():
    p=argparse.ArgumentParser()
    for key in ['cache','full152-spec','repository','bank','labels','clip-weight','output']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--shard',type=int,choices=[0,1],required=True)
    p.add_argument('--mode',choices=['sanity','full'],required=True)
    a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
    torch.set_num_threads(1);torch.manual_seed(2027)
    prep=json.loads((a.cache/'preparation.json').read_text())
    assert prep['source_spec_sha256']==sha(a.full152_spec)
    assert prep['inference_inputs_sha256']==sha(a.cache/'inference_inputs.json')
    spec=json.loads(a.full152_spec.read_text())
    plans=[r for r in json.loads((a.cache/'inference_inputs.json').read_text()) if r['shard']==a.shard]
    if a.mode=='sanity':plans=[next(r for r in plans if r['split']=='fit')]
    bank=torch.load(a.bank,map_location='cpu');labels=json.loads(a.labels.read_text())
    assert bank['human_confirmed'] and labels['human_confirmed']
    assert bank['dataset']==labels['dataset']=='depthtrack'
    assert bank['labels_sha256']==sha(a.labels) and bank['encoder_sha256']==sha(a.clip_weight)
    assert bank['sequences']==[r['sequence'] for r in labels['initial']]
    model,preprocess=clip.load(str(a.clip_weight),device='cuda',jit=False)
    model=model.float().eval().requires_grad_(False)
    assert model.visual.input_resolution==224 and model.visual.conv1.kernel_size==(14,14)
    frozen={k:v.cpu().clone() for k,v in model.state_dict().items()}
    utility=load_utility(a.repository/'lib/train/data/processing_utils.py','native_m116_crop')
    depth=load_utility(a.repository/'lib/train/dataset/depth_utils.py','native_m116_rgbd')
    coordinates=coordinate_sanity();started=time.time();records=[];response_rows=[]
    (a.output/'features').mkdir()
    source_rows={r['sequence']:r for r in json.loads((a.cache/('collect_shard'+str(a.shard)+'.json')).read_text())['sequences']}
    rgb_verified=False
    for case in plans:
        sequence=case['sequence'];source=a.cache/'features'/(sequence+'.pt')
        assert sha(source)==source_rows[sequence]['feature_sha256']
        cached=torch.load(source,map_location='cpu')
        assert not cached['labels_loaded'] and cached['event_frames']==case['event_frames']
        assert cached['split']==case['split']
        index=bank['sequences'].index(sequence);assert bank['splits'][index]==case['split']
        text=torch.cat((bank['tokens'][index],bank['generic'][None],bank['empty'][None])).cuda()
        frames=cached['event_frames'][:3] if a.mode=='sanity' else cached['event_frames']
        boxes=[torch.tensor(case['init_bbox'])[None]]+[cached['boxes'][i] for i in range(len(frames))]
        priors=[case['init_bbox']]+[case['expected_rows'][frame-1]['bbox'] for frame in frames]
        tensors=[];masks=[];origins=[]
        folder=Path(spec['dataset_root'])/sequence
        for i,(frame,prior) in enumerate(zip([0]+frames,priors)):
            image=depth.get_rgbd_frame(str(folder/'color'/('%08d.jpg'%(frame+1))),str(folder/'depth'/('%08d.png'%(frame+1))),dtype='rgbcolormap',depth_clip=True)
            if i:
                assert list(image.shape[:2])==cached['image_shape'][i-1].tolist()
                assert torch.equal(torch.tensor(prior,dtype=torch.float32),cached['prior_bbox'][i-1])
            native_patch,resize,mask=utility.sample_target(image,prior,4.,output_sz=256)
            patch=native_patch[:,:,:3]
            if i:
                assert float(torch.tensor(resize,dtype=torch.float32))==float(cached['resize_factor'][i-1])
            if not rgb_verified:
                rgb=cv2.cvtColor(cv2.imread(str(folder/'color'/'00000001.jpg')),cv2.COLOR_BGR2RGB)
                assert np.array_equal(image[:,:,:3],rgb)
                rgb_verified=True
            tensors.append(preprocess(Image.fromarray(patch)))
            valid=torch.from_numpy(~mask).float()[None,None]
            valid=F.interpolate(valid,size=(224,224),mode='area')
            masks.append(F.avg_pool2d(valid,14,14).reshape(256))
            origins.append(crop_geometry(prior))
        features=[];cls_features=[]
        with torch.no_grad():
            for batch in torch.stack(tensors).split(16):
                patch,cls=patch_features(model,batch.cuda())
                features.append(patch.half().cpu());cls_features.append(cls.half().cpu())
        features=torch.cat(features);cls_features=torch.cat(cls_features);valid=torch.stack(masks)
        # Diagnostics use the same persisted half features that a later prototype will consume.
        for i,frame in enumerate([0]+frames):
            tokens=features[i].float().cuda();weight=valid[i].cuda();bb=boxes[i].float().cuda()
            region,coverage=phrase_scores(tokens,weight,bb,origins[i],text)
            context,context_coverage=phrase_scores(tokens,weight,bb,origins[i],text,True)
            global_cosine=F.normalize(cls_features[i].float().cuda(),dim=-1)@F.normalize(text,dim=-1).T
            response_rows.append(dict(key=sequence+'@'+str(frame),sequence=sequence,split=case['split'],
                is_initialization=i==0,phrase_mask=bank['mask'][index].tolist(),
                region_cosine=region.cpu().tolist(),ring_cosine=context.cpu().tolist(),
                region_minus_ring=(region-context).cpu().tolist(),global_cosine=global_cosine.cpu().tolist(),
                observed_fraction=coverage.cpu().tolist(),ring_observed_fraction=context_coverage.cpu().tolist()))
        target=a.output/'features'/(sequence+'.pt')
        payload=dict(sequence=sequence,split=case['split'],event_frames=frames,
            initial_search_tokens=features[0],search_tokens=features[1:],initial_cls=cls_features[0],search_cls=cls_features[1:],
            initial_observed_fraction=valid[0],observed_fraction=valid[1:],crop_origins=origins,
            original_feature_sha256=sha(source),GT_loaded=False,human_confirmed=True,
            tracker_state_committed=False,model_updates=0)
        torch.save(payload,target)
        record=dict(sequence=sequence,split=case['split'],events=len(frames),feature_sha256=sha(target),bytes=target.stat().st_size,
            original_feature_sha256=sha(source),seconds=time.time()-started)
        records.append(record);print(json.dumps(dict(done=len(records),total=len(plans),**record)),flush=True)
    assert all(torch.equal(v.cpu(),frozen[k]) for k,v in model.state_dict().items())
    responses=a.output/'phrase_responses.jsonl'
    responses.write_text(''.join(json.dumps(r)+'\n' for r in response_rows))
    report=dict(status='complete_M116_'+a.mode+'_shard',shard=a.shard,mode=a.mode,seed=2027,sequences=records,
        events=sum(r['events'] for r in records),source_sha256=sha(__file__),interface_sha256=sha(Path(__file__).with_name('region_patch_evidence.py')),
        native_crop_source_sha256=sha(a.repository/'lib/train/data/processing_utils.py'),
        native_rgbd_source_sha256=sha(a.repository/'lib/train/dataset/depth_utils.py'),
        bank_sha256=sha(a.bank),labels_sha256=sha(a.labels),encoder_sha256=sha(a.clip_weight),
        response_sha256=sha(responses),coordinates=coordinates,rgb_same_as_native_verified=rgb_verified,
        cls_same_input_exact=True,frozen_state_exact=True,human_confirmed=True,GT_loaded=False,
        model_updates=0,checkpoints_created=False,tracker_state_committed=False,
        patch_projection_is_grounding_hypothesis=True,elapsed_seconds=time.time()-started,
        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],events=report['events'],seconds=report['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
