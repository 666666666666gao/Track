"""Spatial CLIP observations on the same native search crop; no tracking action."""
import math
import torch
from torch.nn import functional as F
from instance_ab_prototype import roi_positions, RING


def crop_geometry(prior):
    x,y,w,h=map(float,prior)
    side=math.ceil(math.sqrt(w*h)*4.)
    return round(x+.5*w-.5*side),round(y+.5*h-.5*side),side


def region_samples(tokens,valid,boxes,origin,ring=False):
    """4x4 points in global xywh boxes, sampling only the observed 16x16 grid."""
    left,top,side=origin
    xy=(roi_positions()+1.)*.5
    if ring:
        xy=xy[RING]
        boxes=torch.cat((boxes[:,:2]-.5*boxes[:,2:],2.*boxes[:,2:]),-1)
    points=boxes[:,None,:2]+xy.to(boxes.device)[None]*boxes[:,None,2:]
    grid=2.*(points-points.new_tensor([left,top]))/side-1.
    count=len(boxes)
    image=tokens.reshape(16,16,-1).permute(2,0,1)[None].expand(count,-1,-1,-1)
    weights=valid.reshape(1,1,16,16).expand(count,-1,-1,-1)
    values=F.grid_sample(image,grid[:,None],align_corners=False,padding_mode='zeros')
    sampled=F.grid_sample(weights,grid[:,None],align_corners=False,padding_mode='zeros')
    return values[:,:,0].transpose(1,2),sampled[:,0,0]


def phrase_scores(tokens,valid,boxes,origin,text,ring=False):
    values,weights=region_samples(tokens,valid,boxes,origin,ring)
    values=F.normalize(values,dim=-1)
    cosine=values@F.normalize(text,dim=-1).T
    # Zero observations remain zero evidence, never an invented positive label.
    return (cosine*weights[...,None]).sum(1)/weights.sum(1,keepdim=True).clamp_min(1.),weights.mean(1)


def coordinate_sanity():
    axis=(torch.arange(16)+.5)/16
    y,x=torch.meshgrid(axis,axis,indexing='ij')
    values=torch.stack((x,y),-1).reshape(256,2)
    boxes=torch.tensor([[20.,20.,40.,40.],[100.,100.,10.,10.]])
    samples,weights=region_samples(values,torch.ones(256),boxes,(0,0,80))
    expected=(torch.arange(4)+.5)/8+.25
    ey,ex=torch.meshgrid(expected,expected,indexing='ij')
    assert torch.equal(samples[0],torch.stack((ex,ey),-1).reshape(16,2))
    assert torch.equal(weights[0],torch.ones(16))
    assert not bool(samples[1].any()) and not bool(weights[1].any())
    return dict(interior_coordinate_ramp_exact=True,outside_search_zero=True)
