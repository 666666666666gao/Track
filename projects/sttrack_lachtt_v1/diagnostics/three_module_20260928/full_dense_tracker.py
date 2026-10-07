"""Own-history dense RGB-D/text tracking; no current GT or cached event states."""
from pathlib import Path
import sys
from types import SimpleNamespace
import clip
from PIL import Image
import torch
from torch.nn import functional as F
from dense_target_decoder import DenseTargetDecoder,clip_boxes
from dense_region_encoding import dense_patch_features
from region_patch_evidence import crop_geometry


def state_digest(model):
    import hashlib
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class FullDenseTracker:
    def __init__(self,repository,checkpoint,clip_weight,bank,decoder):
        sys.path.insert(0,str(repository))
        from lib.config.sttrack.config import cfg,update_config_from_file
        from lib.test.tracker.sttrack import STTrack
        from lib.train.data.processing_utils import sample_target
        update_config_from_file(str(repository/'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
        params=SimpleNamespace(cfg=cfg,checkpoint=str(checkpoint),template_factor=2.,template_size=128,
            search_factor=4.,search_size=256,save_all_boxes=False,debug=0)
        self.native=STTrack(params);self.native.network.eval().requires_grad_(False)
        self.clip,self.clip_preprocess=clip.load(str(clip_weight),device='cuda',jit=False)
        self.clip=self.clip.float().eval().requires_grad_(False)
        self.decoder=decoder;self.bank=bank;self.sample_target=sample_target
        self.cell_response=torch.eye(256,device='cuda').reshape(256,1,16,16)
        assert self.native.feat_sz==16 and self.native.num_template==2
        assert self.clip.visual.input_resolution==224 and len(self.clip.visual.transformer.resblocks)==24

    def frozen_digest(self):
        return dict(native=state_digest(self.native.network),clip=state_digest(self.clip))

    def pixels(self,output,prior,resize,image_shape):
        # Preserve native cal_bbox -> *256 -> /resize ordering before Python-double mapping.
        normalized=self.native.network.box_head.cal_bbox(self.cell_response,
            output['size_map'].expand(256,-1,-1,-1),output['offset_map'].expand(256,-1,-1,-1))
        pixels=(normalized*256/resize).double()
        cx=pixels[:,0]+(prior[0]+.5*prior[2]-.5*256/resize)
        cy=pixels[:,1]+(prior[1]+.5*prior[3]-.5*256/resize)
        raw=torch.stack((cx-.5*pixels[:,2],cy-.5*pixels[:,3],pixels[:,2],pixels[:,3]),dim=-1)[None]
        shape=torch.tensor([image_shape[:2]],device='cuda',dtype=torch.float64)
        return clip_boxes(raw,shape)

    def observe(self,image,prior,query):
        patch,resize,mask=self.sample_target(image,prior,4.,output_sz=256)
        with torch.no_grad():
            outputs=self.native.network.forward(template=self.native.z_dict,
                search=[self.native.preprocessor.process(patch)],ce_template_mask=self.native.box_mask_z,
                track_query_before=query,keep_rate=self.native.keep_rate,return_candidate_features=True)
            native=outputs[0];features=native['candidate_features']
            tensor=self.clip_preprocess(Image.fromarray(patch[:,:,:3]))[None].cuda()
            dense,_=dense_patch_features(self.clip,tensor)
            valid=torch.from_numpy(~mask).float()[None,None].cuda()
            boxes=self.pixels(native,prior,resize,image.shape)
            data={name:features[key].detach().half().float() for name,key in dict(
                rgb='search_rgb_tokens',depth='search_depth_tokens',fused='search_fused_tokens').items()}
            data.update(clip=dense.detach().half().float(),native_valid=F.avg_pool2d(valid,16,16).reshape(1,256),
                clip_valid=F.avg_pool2d(F.interpolate(valid,size=(224,224),mode='area'),14,14).reshape(1,256),
                boxes=boxes.float(),native_score=native['score_map'].reshape(1,256),
                native_response=(self.native.output_window*native['score_map']).reshape(1,256),
                origin=torch.tensor([crop_geometry(prior)],device='cuda',dtype=torch.float32),
                image_shape=torch.tensor([image.shape[:2]],device='cuda',dtype=torch.float32),condition='human_text')
            assert all(bool(torch.isfinite(v).all()) for v in data.values() if torch.is_tensor(v))
        return data,boxes,native,resize

    def initialize(self,image,box,bank_index):
        self.native.initialize(image,dict(init_bbox=list(box)));self.initial={}
        data,_,_,_=self.observe(image,list(box),None)
        for name in ['rgb','depth','fused','clip','native_valid','clip_valid','origin']:
            self.initial['initial_'+name]=data[name].detach()
        self.initial['initial_box']=torch.tensor([box],device='cuda',dtype=torch.float32)
        self.words=self.bank['tokens'][bank_index:bank_index+1].cuda().float()
        self.word_mask=self.bank['mask'][bank_index:bank_index+1].cuda()
        self.empty=self.bank['empty'].cuda().float()
        assert bool(self.word_mask[:,0].all())
        assert self.native.frame_id==0 and self.native.track_query_before is None
        assert list(self.native.state)==list(box)

    def step(self,image):
        prior=list(self.native.state);self.native.frame_id+=1
        data,base,native,resize=self.observe(image,prior,self.native.track_query_before)
        data.update(self.initial);data.update(query=self.words,query_mask=self.word_mask,empty_query=self.empty)
        out=self.decoder(data)
        # Submit the learned delta on the native pixel box without a zero-only fallback.
        boxes=clip_boxes(base+(out['raw_boxes']-data['boxes']).double(),data['image_shape'].double())
        selected=out['selected_index'];row=torch.arange(1,device=selected.device)
        out.update(boxes=boxes,selected_box=boxes[row,selected])
        assert all(bool(torch.isfinite(x).all()) for x in out.values())
        assert torch.equal(out['selected_score'],out['response'][row,selected])
        assert torch.equal(out['selected_feature'],out['spatial_features'][row,selected])
        assert torch.equal(out['selected_quality'],out['quality_logits'].sigmoid()[row,selected])
        self.native.state=out['selected_box'][0].detach().cpu().tolist()
        self.native.track_query_before=[x.detach() for x in native['track_query_before']]
        self.selected_feature=out['selected_feature'].detach()
        native_confidence=float(data['native_response'][0,selected[0]])
        write=self.native.frame_id%self.native.update_intervals==0 and native_confidence>self.native.update_threshold
        if write:
            patch,_,_=self.sample_target(image,self.native.state,2.,output_sz=128)
            self.native.z_patch_arr=patch;template=self.native.preprocessor.process(patch)
            self.native.z_dict.append(template);self.native.z_dict.pop(1)
        record=dict(frame=self.native.frame_id,previous_bbox=prior,bbox=list(self.native.state),
            selected=int(selected[0]),best_score=float(out['selected_score'][0].detach()),
            selected_quality=float(out['selected_quality'][0].detach()),native_same_position_response=native_confidence,
            observation_intersection_probability=float(out['observation_logits'][0].detach().sigmoid()),
            template_write=bool(write),resize_factor=float(resize),crop_origin=data['origin'][0].tolist())
        return out,data,record

    def track(self,image,info=None):
        # Official tracking needs no current annotation; training calls step outside no_grad.
        with torch.no_grad():out,data,record=self.step(image)
        return dict(target_bbox=record['bbox'],best_score=record['best_score'])
