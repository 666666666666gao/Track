"""CLIP Surgery mechanism control using the existing frozen OpenAI CLIP weights.

Li et al., https://github.com/xmed-lab/CLIP_Surgery/blob/master/clip/clip_surgery_model.py
The public ViT implementation starts its FFN-free path at the last six blocks.
Keep the native original path and its parameters intact; this is not a new method.
"""
import torch
from torch.nn import functional as F


def dense_patch_features(model,images):
    assert len(model.visual.transformer.resblocks)==24
    dense=[None];hooks=[]

    def observe_original(block,args):
        original=args[0]
        normalized=block.ln_1(original)
        width=original.shape[-1];heads=block.attn.num_heads
        value=F.linear(normalized,block.attn.in_proj_weight[2*width:],block.attn.in_proj_bias[2*width:])
        count,batch,_=value.shape
        value=value.permute(1,0,2).reshape(batch,count,heads,width//heads).permute(0,2,1,3)
        attention=((value@value.transpose(-2,-1))*(width//heads)**-.5).softmax(-1)
        update=(attention@value).transpose(1,2).reshape(batch,count,width)
        update=F.linear(update,block.attn.out_proj.weight,block.attn.out_proj.bias).transpose(0,1)
        dense[0]=(original if dense[0] is None else dense[0])+update
        # Returning None leaves the original block's input and forward untouched.

    for block in list(model.visual.transformer.resblocks)[-6:]:
        hooks.append(block.register_forward_pre_hook(observe_original))
    cls=model.encode_image(images)
    for hook in hooks:hook.remove()
    patches=model.visual.ln_post(dense[0].permute(1,0,2)[:,1:])@model.visual.proj
    assert patches.shape==(len(images),256,768)
    assert bool(torch.isfinite(patches).all()) and bool(torch.isfinite(cls).all())
    return patches,cls


def verify_author_reference(model,images,source):
    """One sanity batch: compare normalized dense features with the actual author ViT."""
    import importlib.util
    spec=importlib.util.spec_from_file_location('author_clip_surgery_reference',str(source))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    visual=model.visual
    author=module.VisionTransformer(224,14,1024,24,16,768).float().cuda().eval().requires_grad_(False)
    author.load_state_dict(visual.state_dict(),strict=True)
    ordinary=model.encode_image(images)
    actual,cls=dense_patch_features(model,images)
    assert torch.equal(cls,ordinary),'Extra dense observations must not change the original CLS path.'
    expected=author(images)[:,1:]
    # Different GEMM/QKV implementations can round differently (M115 witnessed this).
    # Compare the direction used by the cosine reader with PyTorch's default tolerances.
    torch.testing.assert_close(F.normalize(actual,dim=-1),F.normalize(expected,dim=-1))
    return dict(original_cls_exact=True,author_normalized_patch_default_assert_close=True,
        raw_max_abs_difference=float((actual-expected).abs().max()),
        normalized_max_abs_difference=float((F.normalize(actual,dim=-1)-F.normalize(expected,dim=-1)).abs().max()),
        tokens_compared=actual.numel()//768,changed_parameters=0)
