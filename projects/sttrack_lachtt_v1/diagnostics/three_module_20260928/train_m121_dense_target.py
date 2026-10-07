"""Complete-cache dense target training; GT stays in losses and reporting."""
import argparse,hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from analyze_train_states import sha
from dense_target_decoder import DenseTargetDecoder
from m121_dense_inputs import load_panel,inputs,observation_targets
from train_ab_visual_control import summarize
from train_m113_human_initialization import paired_vs_empty


PARAMETER_GROUPS=['native','semantic_projection','position','search_fusion','reference_read','visual_fusion',
    'phrase_bind','phrase_calibrate','phrase_read','visual_support','semantic_support',
    'geometry','box_delta','quality','observation']


def box_overlap(boxes,target):
    target=target[:,None]
    left=torch.maximum(boxes[...,:2],target[...,:2])
    right=torch.minimum(boxes[...,:2]+boxes[...,2:],target[...,:2]+target[...,2:])
    intersection=(right-left).clamp_min(0).prod(-1)
    union=boxes[...,2:].prod(-1)+target[...,2:].prod(-1)-intersection
    iou=intersection/union
    enclosing=(torch.maximum(boxes[...,:2]+boxes[...,2:],target[...,:2]+target[...,2:])-
        torch.minimum(boxes[...,:2],target[...,:2])).prod(-1)
    return iou,iou-(enclosing-union)/enclosing


def focal(logits,target):
    probability=logits.sigmoid().clamp(1e-4,1-1e-4)
    positive=target.eq(1);negative=~positive
    loss=-torch.log(probability)*(1-probability).pow(2)*positive
    loss-=torch.log1p(-probability)*probability.pow(2)*(1-target).pow(4)*negative
    return loss.sum()/positive.sum().clamp_min(1)


def objective(out,data,target):
    heat,cell,observed=observation_targets(target,data['origin'],data['image_shape'])
    actual,giou=box_overlap(out['boxes'],target)
    original,_=box_overlap(data['boxes'],target)
    native_index=data['native_response'].argmax(-1);row=torch.arange(len(target),device=target.device)
    reliable=original[row,native_index]>=.5
    regression=torch.zeros_like(actual,dtype=torch.bool)
    regression[row,cell]=observed;regression[row,native_index]|=reliable
    count=regression.sum().clamp_min(1)
    geometry=((1-giou)*regression).sum()/count
    l1=((out['boxes']-target[:,None]).abs()/data['origin'][:,None,2:]).mean(-1)
    l1=(l1*regression).sum()/count
    quality=F.binary_cross_entropy_with_logits(out['quality_logits'],actual.detach())
    observation=F.binary_cross_entropy_with_logits(out['observation_logits'],observed.float())
    eligible=reliable & (actual.detach()[row,native_index]>=.5)
    bad=actual.detach()<=.1
    negatives=out['response'].detach().masked_fill(~bad,-1).topk(9,dim=-1).indices
    mask=bad.gather(-1,negatives)&eligible[:,None]
    native_positive=data['native_response'][row,native_index,None]
    native_negative=data['native_response'].gather(-1,negatives)
    teacher_gap=(native_positive-native_negative)/(native_positive+native_negative)
    positive=out['response'][row,native_index,None];negative=out['response'].gather(-1,negatives)
    gap=(positive-negative)/(positive+negative).clamp_min(1e-12)
    preserve=(F.relu(teacher_gap-gap)*mask).sum()/mask.sum().clamp_min(1)
    center=focal(out['support_logits'],heat)
    loss=center+2*geometry+5*l1+quality+observation+preserve
    return loss,torch.stack([center,geometry,l1,quality,observation,preserve]),dict(
        observation_intersection=int(observed.sum()),regression_positions=int(regression.sum()),preservation_pairs=int(mask.sum()))


def digest(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def evaluate(model,part,initial,bank,condition,device):
    rows=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(part['key']),64):
            ids=torch.arange(start,min(start+64,len(part['key'])))
            data=inputs(part,ids,initial,bank,condition,device);out=model(data)
            assert all(bool(torch.isfinite(x).all()) for x in out.values())
            if condition=='empty':
                assert not bool(out['semantic_delta'].any())
                assert torch.equal(out['response'],out['visual_response'])
            target=part['target'][ids].to(device)
            actual,_=box_overlap(out['boxes'],target);native,_=box_overlap(data['boxes'],target)
            selected=out['selected_index'];original=data['native_response'].argmax(-1)
            index=torch.arange(len(ids),device=device)
            assert torch.equal(out['selected_box'],out['boxes'][index,selected])
            assert torch.equal(out['selected_score'],out['response'][index,selected])
            assert torch.equal(out['selected_feature'],out['spatial_features'][index,selected])
            assert torch.equal(out['selected_quality'],out['quality_logits'].sigmoid()[index,selected])
            _,_,intersects=observation_targets(target,data['origin'],data['image_shape'])
            for j,i in enumerate(ids.tolist()):
                pick=int(selected[j]);base=int(original[j])
                rows.append(dict(key=part['key'][i],strata=part['strata'][i],native_iou=float(native[j,base]),
                    selected_iou=float(actual[j,pick]),oracle_iou=float(actual[j].max()),
                    native_full256_oracle_iou=float(native[j].max()),selected=pick,native_selected=base,
                    selected_box=out['selected_box'][j].cpu().tolist(),selected_score=float(out['selected_score'][j]),
                    selected_quality=float(out['selected_quality'][j]),
                    observation_probability=float(out['observation_logits'][j].sigmoid()),
                    GT_box_intersects_observed_window=bool(intersects[j]),
                    native_position_refined_iou=float(actual[j,base]),semantic_delta=out['semantic_delta'][j].cpu().tolist()))
    return summarize(rows),rows


def main():
    parser=argparse.ArgumentParser()
    for name in ['cache','native','dense','bank','labels','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--arm',choices=['human_text','visual_query','generic'],required=True)
    parser.add_argument('--mode',choices=['sanity','train'],required=True)
    args=parser.parse_args();assert not args.output.exists()
    torch.set_num_threads(1);random.seed(2027);np.random.seed(2027)
    torch.manual_seed(2027);torch.cuda.manual_seed_all(2027);device=torch.device('cuda')
    bank=torch.load(args.bank,map_location='cpu');labels=json.loads(args.labels.read_text())
    assert bank['human_confirmed'] and labels['human_confirmed'] and bank['dataset']==labels['dataset']=='depthtrack'
    assert bank['labels_sha256']==sha(args.labels)=='6ffb6e9907fee6a520e31fa78b4d3ff044a46a7daf8ad1aaa42c4d638b8c50c2'
    assert sha(args.bank)=='a599e063b79b9458aab9e63ec21b9f4ac9735e420bceeec95275bfd1289603b2'
    assert bank['sequences']==[r['sequence'] for r in labels['initial']]
    assert bank['splits']==[r['split'] for r in labels['initial']]
    started=time.time();panel,initial,source=load_panel(args,bank)
    torch.manual_seed(2027);torch.cuda.manual_seed_all(2027)
    model=DenseTargetDecoder().to(device).eval();initial_sha=digest(model)
    initial_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
    fit=panel['fit'];ids=torch.arange(3);data=inputs(fit,ids,initial,bank,'human_text',device)
    with torch.no_grad():
        first=model(data)
        assert torch.equal(first['response'],data['native_response']*.5)
        assert torch.equal(first['selected_index'],data['native_response'].argmax(-1))
        torch.testing.assert_close(first['boxes'],data['boxes'],atol=1e-4,rtol=0.)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    history=[];gradients=[];updates=0
    for epoch in range(16 if args.mode=='train' else 1):
        permutation=torch.randperm(len(fit['key']),generator=torch.Generator().manual_seed(2027+epoch))
        batches=permutation.split(64) if args.mode=='train' else [permutation[:64]]*3
        losses=[];counts=dict(observation_intersection=0,regression_positions=0,preservation_pairs=0)
        model.train()
        for ids in batches:
            optimizer.zero_grad(set_to_none=True)
            data=inputs(fit,ids,initial,bank,args.arm,device);out=model(data)
            assert all(bool(torch.isfinite(v).all()) for v in out.values())
            loss,terms,stats=objective(out,data,fit['target'][ids].to(device))
            assert bool(torch.isfinite(loss));loss.backward()
            assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
            if args.mode=='sanity':
                gradients.append({name:sum(float(p.grad.norm()) for k,p in model.named_parameters() if k.startswith(name+'.') and p.grad is not None)
                    for name in PARAMETER_GROUPS})
            optimizer.step();updates+=1;losses.append(terms.detach().cpu().tolist())
            for k,v in stats.items():counts[k]+=v
        history.append(dict(epoch=epoch+1,updates=updates,mean_losses=np.mean(losses,axis=0).tolist(),seconds=time.time()-started,**counts))
        print(json.dumps(history[-1]),flush=True)
    assert updates==(3 if args.mode=='sanity' else 640)
    assert all(torch.equal(v.cpu(),initial_state[k]) for k,v in model.named_buffers())
    args.output.mkdir();summaries={};rows_by={};model.eval()
    for condition in ['empty',args.arm]:
        ids=torch.arange(3);out=model(inputs(fit,ids,initial,bank,condition,device))
        assert all(bool(torch.isfinite(x).all()) for x in out.values())
        selected=out['selected_index'];index=torch.arange(len(ids),device=device)
        assert torch.equal(out['selected_box'],out['boxes'][index,selected])
        assert torch.equal(out['selected_score'],out['response'][index,selected])
        assert torch.equal(out['selected_feature'],out['spatial_features'][index,selected])
        assert torch.equal(out['selected_quality'],out['quality_logits'].sigmoid()[index,selected])
        if condition=='empty':assert not bool(out['semantic_delta'].any())
    result=dict(status='complete_M121_sanity' if args.mode=='sanity' else 'complete_M121_training',
        arm=args.arm,seed=2027,optimizer_steps=updates,optimized_parameters=sum(p.numel() for p in model.parameters()),
        initial_state_sha256=initial_sha,source=source,source_sha256=sha(__file__),
        model_source_sha256=sha(Path(__file__).with_name('dense_target_decoder.py')),
        loader_source_sha256=sha(Path(__file__).with_name('m121_dense_inputs.py')),
        bank_sha256=sha(args.bank),labels_sha256=sha(args.labels),history=history,
        native_response_zero_initialization_exact=True,initial_geometry_default_close_atol1e4=True,
        Empty_semantic_delta_same_forward_exact=True,geometry_has_no_current_text_input=True,
        no_phrase_visibility_or_physical_identity_truth=True,no_recursive_or_public_evaluation=True,
        elapsed_seconds=time.time()-started,gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    if args.mode=='sanity':
        assert all(v>0 for v in gradients[-1].values()),gradients
        changes={}
        for name in PARAMETER_GROUPS:
            differences=[p.detach().cpu()-initial_state[k] for k,p in model.named_parameters() if k.startswith(name+'.')]
            changes[name]=dict(changed_values=sum(int(d.ne(0).sum()) for d in differences),
                absolute_delta=sum(float(d.abs().sum()) for d in differences),
                maximum_delta=max(float(d.abs().max()) for d in differences))
        assert all(r['changed_values']>0 and r['absolute_delta']>0 for r in changes.values()),changes
        result.update(gradients=gradients,parameter_changes=changes,all_architectural_groups_updated=True,checkpoint_saved=False)
    else:
        for condition in ['human_text','empty','generic','visual_query']:
            summaries[condition],rows_by[condition]=evaluate(model,panel['development'],initial,bank,condition,device)
            (args.output/(condition+'_development.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows_by[condition]))
        result['fit'],fit_rows=evaluate(model,fit,initial,bank,args.arm,device)
        (args.output/'fit_events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in fit_rows))
        result.update(content_conditions=summaries,development=summaries[args.arm],
            paired_vs_own_empty={k:paired_vs_empty(v,rows_by['empty']) for k,v in rows_by.items() if k!='empty'})
        torch.save(model.state_dict(),args.output/'final.pt')
        reloaded=torch.load(args.output/'final.pt',map_location='cpu')
        assert all(torch.equal(v.cpu(),reloaded[k]) for k,v in model.state_dict().items())
        result.update(final_sha256=sha(args.output/'final.pt'),final_state_roundtrip_exact=True)
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],arm=args.arm,updates=updates)),flush=True)


if __name__=='__main__':main()
