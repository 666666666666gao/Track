"""Box-level support supervision; no inference state or caption labels changed."""

def support_supervision(out, groundtruth, previous, resize, search_size):
    import torch
    from lib.train.data.processing_utils import transform_image_to_crop
    evidence=out['semantic_features']['attribute_scores']
    assert evidence.shape[:2]==(1,256)
    gt=torch.as_tensor(groundtruth,dtype=torch.float32,device=evidence.device)
    # Caller has already applied the original finite/positive GT validity check.
    target=transform_image_to_crop(gt,gt.new_tensor(previous),resize,gt.new_tensor([search_size,search_size]),normalize=True)
    centre=target[:2]+.5*target[2:]
    positive=torch.zeros((16,16),dtype=torch.bool,device=evidence.device)
    if bool(((centre>=0)&(centre<1)).all()):
        cell=(centre*16).round().clamp(0,15).long()
        positive[cell[1],cell[0]]=True
    grid=(torch.arange(16,device=evidence.device,dtype=gt.dtype)+.5)/16
    yy,xx=torch.meshgrid(grid,grid,indexing='ij')
    # A one-cell ignored border avoids claiming foreground segmentation from boxes.
    negative=(xx<target[0]-1/16)|(xx>target[0]+target[2]+1/16)|(yy<target[1]-1/16)|(yy>target[1]+target[3]+1/16)
    negative=negative & ~positive
    word_logsum=torch.logsumexp(evidence,dim=-1).reshape(16,16)
    normalizer=torch.logaddexp(word_logsum,torch.zeros_like(word_logsum))
    positive_n=positive.sum();negative_n=negative.sum()
    fg=((normalizer-word_logsum)*positive).sum()/positive_n.clamp_min(1)
    bg=(normalizer*negative).sum()/negative_n.clamp_min(1)
    groups=(positive_n>0).to(gt.dtype)+(negative_n>0).to(gt.dtype)
    loss=(fg+bg)/groups.clamp_min(1)
    null_mass=torch.exp(-normalizer)
    diagnostics=dict(support_loss=float(loss.detach()),support_positive_cells=int(positive_n),support_negative_cells=int(negative_n),
        positive_null_mass=float((null_mass*positive).sum().detach()/positive_n.clamp_min(1)),
        negative_null_mass=float((null_mass*negative).sum().detach()/negative_n.clamp_min(1)))
    return loss,diagnostics
