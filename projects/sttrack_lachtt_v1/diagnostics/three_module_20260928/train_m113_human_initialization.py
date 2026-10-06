"""M113 human initialization versus generic text, with the M101 visual parent fixed."""
import argparse,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from analyze_train_states import sha
from instance_ab_prototype import InstanceCandidatePrototype
from train_ab_visual_control import load_inputs,native_candidate_preservation
from train_m110_no_weak_rank import TRAIN_PREFIXES,inputs,snapshot,evaluate


def paired_vs_empty(rows,empty_rows):
    assert [r['key'] for r in rows]==[r['key'] for r in empty_rows]
    result={}
    for group in ['all','healthy','transition']:
        pairs=[(r,e) for r,e in zip(rows,empty_rows) if group=='all' or group in r['strata']]
        assert pairs,group
        result[group]=dict(events=len(pairs),
            empty_iou50=sum(e['selected_iou']>=0.5 for r,e in pairs),
            condition_iou50=sum(r['selected_iou']>=0.5 for r,e in pairs),
            mean_iou_delta=sum(r['selected_iou']-e['selected_iou'] for r,e in pairs)/len(pairs),
            rescue=sum(r['selected_iou']>=0.5 and e['selected_iou']<=0.1 for r,e in pairs),
            harm=sum(r['selected_iou']<=0.1 and e['selected_iou']>=0.5 for r,e in pairs))
    return result


def main():
    p=argparse.ArgumentParser()
    for k in ['cache','contexts','origins','bank','labels','parent','parent-result','output']:
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--arm',choices=['generic','human_text'],required=True)
    p.add_argument('--mode',choices=['sanity','train'],required=True)
    a=p.parse_args();assert not a.output.exists()
    torch.set_num_threads(1);random.seed(2027);np.random.seed(2027)
    torch.manual_seed(2027);torch.cuda.manual_seed_all(2027);device=torch.device('cuda')
    panel=load_inputs(a.cache,a.contexts,a.origins,False)
    bank=torch.load(a.bank,map_location='cpu');labels=json.loads(a.labels.read_text())
    assert bank['human_confirmed'] and labels['human_confirmed']
    assert bank['labels_sha256']==sha(a.labels) and bank['dataset']==labels['dataset']=='depthtrack'
    assert bank['sequences']==[r['sequence'] for r in labels['initial']]
    assert bank['splits']==[r['split'] for r in labels['initial']]
    assert len(bank['sequences'])==152
    for split in ['fit','development']:
        assert all(bank['splits'][bank['sequences'].index(k.rsplit('@',1)[0])]==split for k in panel[split]['key'])
    old=json.loads(a.parent_result.read_text())
    assert sha(a.parent)==old['final_weights_sha256']=='1d187d78c1dd76ebf59e41fd31af16fa69cecc37b4cb6c74fbdfe3ed5c237293'
    model=InstanceCandidatePrototype().to(device)
    model.load_state_dict(torch.load(a.parent,map_location='cpu'),strict=True);model.requires_grad_(False)
    for name,param in model.named_parameters():
        if name.startswith(TRAIN_PREFIXES):param.requires_grad_(True)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==95683
    torch.nn.init.zeros_(model.semantic.weight);torch.nn.init.zeros_(model.phrase.weight)
    frozen={k:v.detach().cpu().clone() for k,v in model.named_parameters() if not v.requires_grad}
    buffers={k:v.detach().cpu().clone() for k,v in model.named_buffers()}
    initial={s:snapshot(model,panel[s],bank,device) for s in ['fit','development']}
    parent_indices={s:x[0].argmax(-1) for s,x in initial.items()}
    assert parent_indices['development'].tolist()==[r['selected'] for r in old['development_rows']]
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4)
    fit=panel['fit'];started=time.time();history=[];gradients=[]
    epochs=12 if a.mode=='train' else 1
    for epoch in range(epochs):
        batches=torch.randperm(len(fit['key']),generator=torch.Generator().manual_seed(2027+epoch)).split(64) if a.mode=='train' else [torch.arange(64)]*3
        terms=[];model.train()
        for ids in batches:
            optimizer.zero_grad(set_to_none=True)
            data=inputs(fit,ids,bank,a.arm,device);out=model(data);target=fit['iou'][ids].to(device)
            localization=F.binary_cross_entropy_with_logits(out['selection_logits'],target)
            preservation,_=native_candidate_preservation(out['selection_logits'],data['base_scores'],target)
            loss=localization+preservation
            assert bool(torch.isfinite(loss));loss.backward()
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
            if a.mode=='sanity':
                gradients.append({n:sum(float(p.grad.norm()) for k,p in model.named_parameters() if k.startswith(n+'.') and p.grad is not None) for n in ['evidence.text','semantic','phrase']})
            optimizer.step();terms.append([float(x.detach()) for x in [loss,localization,preservation]])
        history.append(dict(epoch=epoch+1,optimizer_steps=len(terms),mean_losses=np.mean(terms,axis=0).tolist(),seconds=time.time()-started))
        print(json.dumps(history[-1]),flush=True)
    for k,v in model.named_parameters():
        if k in frozen:assert torch.equal(v.detach().cpu(),frozen[k]),k
    for k,v in model.named_buffers():assert torch.equal(v.cpu(),buffers[k]),k
    for split in initial:
        final=snapshot(model,panel[split],bank,device)
        assert all(torch.equal(x,y) for x,y in zip(final,initial[split])),split
    result=dict(status='complete_M113_human_initialization_arm' if a.mode=='train' else 'complete_M113_gpu_sanity',
        arm=a.arm,mode=a.mode,seed=2027,epochs=epochs,batch_size=64,learning_rate=3e-4,
        optimizer_steps=sum(r['optimizer_steps'] for r in history),optimized_parameters=95683,
        human_confirmed=True,review_subject='initialization_only',current_candidate_semantic_labels=0,
        model_candidate_weak_labels_used=False,loss_components=['localization','preservation'],
        source_sha256=sha(__file__),parent_sha256=sha(a.parent),bank_sha256=sha(a.bank),labels_sha256=sha(a.labels),
        history=history,empty_all3039_scores_quality_exact=True,frozen_parameters_buffers_exact=True,
        no_recursive_or_public_evaluation=True,elapsed_seconds=time.time()-started)
    a.output.mkdir()
    if a.mode=='sanity':
        assert gradients[-1]['evidence.text']>0 and gradients[-1]['semantic']>0
        result.update(gradient_norms=gradients,checkpoint_saved=False,development_metrics_reported=False)
    else:
        result['fit'],rows=evaluate(model,fit,bank,a.arm,device,parent_indices['fit'])
        (a.output/'fit_events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        conditions={}
        result['content_conditions']={}
        for mode in ['empty','generic','human_text']:
            summary,rows=evaluate(model,panel['development'],bank,mode,device,parent_indices['development'])
            result['content_conditions'][mode]=summary;conditions[mode]=rows
            (a.output/(mode+'_development.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        result['development']=result['content_conditions'][a.arm]
        result['paired_vs_own_empty']={mode:paired_vs_empty(conditions[mode],conditions['empty']) for mode in ['generic','human_text']}
        dev=result['development']
        result['fixed_state_checks']=dict(correct_exceeds_native=dev['all']['selected_iou50']>dev['all']['native_iou50'],
            mean_iou_exceeds_native=dev['all']['selected_mean_iou']>dev['all']['native_mean_iou'],
            healthy_breaks_zero=dev['healthy']['breaks']==0,
            transition_correct_at_least_native=dev['transition']['selected_iou50']>=dev['transition']['native_iou50'])
        torch.save(model.state_dict(),a.output/'final.pt')
        reloaded=torch.load(a.output/'final.pt',map_location='cpu')
        assert all(torch.equal(v.cpu(),reloaded[k]) for k,v in model.state_dict().items())
        result.update(final_state_roundtrip_exact=True,final_sha256=sha(a.output/'final.pt'))
    result['gpu_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],arm=a.arm,optimizer_steps=result['optimizer_steps'])),flush=True)


if __name__=='__main__':main()
