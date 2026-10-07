"""Dense first-reference support; language never changes the geometry branch."""
import torch
from torch import nn
from torch.nn import functional as F
from dense_region_decoder import sample_regions, weighted_mean
from instance_ab_prototype import roi_positions


def clip_boxes(boxes, image_shape):
    height=image_shape[:,0,None];width=image_shape[:,1,None]
    x1=boxes[...,0].clamp_min(0).minimum(width-10)
    y1=boxes[...,1].clamp_min(0).minimum(height-10)
    x2=(boxes[...,0]+boxes[...,2]).clamp_min(10).minimum(width)
    y2=(boxes[...,1]+boxes[...,3]).clamp_min(10).minimum(height)
    return torch.stack((x1,y1,(x2-x1).clamp_min(10),(y2-y1).clamp_min(10)),dim=-1)


def odds_residual(probability,delta):
    gate=delta.sigmoid()
    positive=probability*gate
    negative=(1-probability)*(1-gate)
    return positive/(positive+negative)


class DenseTargetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.native=nn.ModuleDict({k:nn.Sequential(nn.Linear(768,64),nn.LayerNorm(64)) for k in ['rgb','depth','fused']})
        self.semantic_projection=nn.Sequential(nn.Linear(768,64,bias=False),nn.LayerNorm(64))
        self.position=nn.Linear(2,64,bias=False)
        self.search_fusion=nn.Sequential(nn.Linear(256,64),nn.LayerNorm(64))
        self.reference_read=nn.MultiheadAttention(64,4,dropout=0.,batch_first=True)
        self.visual_fusion=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.LayerNorm(64))
        self.phrase_bind=nn.MultiheadAttention(64,4,dropout=0.,batch_first=True)
        self.phrase_calibrate=nn.MultiheadAttention(64,4,dropout=0.,batch_first=True)
        self.phrase_read=nn.MultiheadAttention(64,4,dropout=0.,batch_first=True)
        self.visual_support=nn.Linear(64,1)
        self.semantic_support=nn.Sequential(nn.Linear(134,64),nn.GELU(),nn.Linear(64,1,bias=False))
        self.geometry=nn.Sequential(nn.Conv2d(128,64,3,padding=1),nn.GELU())
        self.box_delta=nn.Conv2d(64,4,1)
        self.quality=nn.Conv2d(64,1,1)
        self.observation=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
        for module in [self.visual_support,self.box_delta,self.quality]:
            nn.init.zeros_(module.weight);nn.init.zeros_(module.bias)
        nn.init.zeros_(self.semantic_support[-1].weight)
        self.register_buffer('roi_xy',roi_positions())
        axis=(torch.arange(16)+.5)/8.-1.
        y,x=torch.meshgrid(axis,axis,indexing='ij')
        self.register_buffer('search_xy',torch.stack((x,y),dim=-1).reshape(256,2))

    def observe(self,data):
        references=[];current=[];initial_weights=None;initial_clip=None
        for kind in ['rgb','depth','fused','clip']:
            grid=data['initial_'+kind];valid=data['initial_clip_valid' if kind=='clip' else 'initial_native_valid']
            roi,weight=sample_regions(grid,valid,data['initial_box'][:,None],data['initial_origin'],(self.roi_xy+1.)*.5)
            roi=F.normalize(roi[:,0],dim=-1);weight=weight[:,0]
            projection=self.semantic_projection if kind=='clip' else self.native[kind]
            references.append(projection(roi)+self.position(self.roi_xy)[None])
            current.append(projection(F.normalize(data[kind],dim=-1)))
            if kind=='clip':initial_clip=weighted_mean(roi,weight)
            if initial_weights is None:initial_weights=weight
            else:assert bool((weight.sum(-1)>0).all())
        reference=torch.cat(references,dim=1)
        # Native and CLIP observe the same crop, with their own sampled padding.
        _,clip_weight=sample_regions(data['initial_clip'],data['initial_clip_valid'],data['initial_box'][:,None],
            data['initial_origin'],(self.roi_xy+1.)*.5)
        mask=torch.cat([initial_weights]*3+[clip_weight[:,0]],dim=-1)>0
        assert bool(mask.any(-1).all())
        search=self.search_fusion(torch.cat(current,dim=-1))+self.position(self.search_xy)[None]
        context=self.reference_read(search,reference,reference,key_padding_mask=~mask,need_weights=False)[0]
        visual=self.visual_fusion(torch.cat((search,context),dim=-1))
        return dict(reference=reference,reference_mask=mask,visual=visual,fused=current[2],initial_query=initial_clip)

    def branch(self,data,observed,query):
        query=F.normalize(query,dim=-1)
        words=self.semantic_projection(query)
        bound=words+self.phrase_bind(words,observed['reference'],observed['reference'],
            key_padding_mask=~observed['reference_mask'],need_weights=False)[0]
        valid=data['native_valid']>0
        assert bool(valid.any(-1).all())
        calibrated=bound+self.phrase_calibrate(bound,observed['visual'],observed['visual'],
            key_padding_mask=~valid,need_weights=False)[0]
        context=self.phrase_read(observed['visual'],calibrated,calibrated,
            key_padding_mask=~data['query_mask'],need_weights=False)[0]
        cosine=F.normalize(data['clip'],dim=-1)@query.transpose(-2,-1)
        cosine=cosine*data['query_mask'][:,None]
        features=torch.cat((observed['visual'],context,cosine,data['clip_valid'][...,None]),dim=-1)
        assert features.shape[-1]==134
        return self.semantic_support(features).squeeze(-1),context,cosine

    def forward(self,data):
        observed=self.observe(data);batch=len(data['boxes'])
        query=observed['initial_query'][:,None].expand(-1,5,-1) if data['condition']=='visual_query' else data['query']
        empty=data['empty_query'][None,None].expand(batch,5,-1)
        full,semantic_features,cosine=self.branch(data,observed,query)
        null,_,_=self.branch(data,observed,empty);semantic_delta=full-null
        native=data['native_score'];assert bool(((native>0)&(native<1)).all())
        base_logits=torch.log(native)-torch.log1p(-native)
        visual_delta=self.visual_support(observed['visual']).squeeze(-1)
        visual_logits=base_logits+visual_delta
        support_logits=visual_logits+semantic_delta
        geometry_input=torch.cat((observed['visual'],observed['fused']),dim=-1).transpose(1,2).reshape(batch,128,16,16)
        local=self.geometry(geometry_input)
        delta=self.box_delta(local).flatten(2).transpose(1,2)
        base=data['boxes'];scale=base[...,2:]
        raw_boxes=torch.cat((base[...,:2]+delta[...,:2]*scale-.5*scale*torch.expm1(delta[...,2:]),
            scale*torch.exp(delta[...,2:])),dim=-1)
        boxes=clip_boxes(raw_boxes,data['image_shape'])
        quality_logits=self.quality(local).flatten(1)
        observation=self.observation(torch.cat((observed['visual'].mean(1),observed['reference'].mean(1)),dim=-1)).squeeze(-1)
        probability=odds_residual(native,visual_delta+semantic_delta)
        visual_probability=odds_residual(native,visual_delta)
        # At zero residual, keep the measured native Hann response, not an inverse-logit reconstruction.
        response=data['native_response']*(probability/native)*quality_logits.sigmoid()
        visual_response=data['native_response']*(visual_probability/native)*quality_logits.sigmoid()
        selected=response.argmax(-1);row=torch.arange(batch,device=selected.device)
        features=torch.cat((observed['visual'],semantic_features),dim=-1)
        return dict(support_logits=support_logits,visual_support_logits=visual_logits,semantic_delta=semantic_delta,
            response=response,visual_response=visual_response,boxes=boxes,raw_boxes=raw_boxes,
            quality_logits=quality_logits,observation_logits=observation,phrase_cosine=cosine,
            selected_index=selected,selected_box=boxes[row,selected],selected_score=response[row,selected],
            selected_quality=quality_logits.sigmoid()[row,selected],selected_feature=features[row,selected],
            spatial_features=features)
