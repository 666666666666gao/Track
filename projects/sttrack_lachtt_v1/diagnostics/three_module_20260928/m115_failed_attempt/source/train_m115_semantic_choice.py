"""M115 matched direct semantic head: human BCE, human choice, generic choice."""
import argparse,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from analyze_train_states import sha
from semantic_choice_prototype import SemanticChoicePrototype
from train_ab_visual_control import load_inputs
from train_m110_no_weak_rank import inputs,snapshot,evaluate
from train_m113_human_initialization import paired_vs_empty


def relative_choice(logits, target):
    mass = target * (target >= .5)
    total = mass.sum(-1, keepdim=True)
    distribution = mass / total.clamp_min(.5)
    eligible = total.squeeze(-1) > 0
    per_state = -(distribution * F.log_softmax(logits, dim=-1)).sum(-1)
    return (per_state * eligible).sum() / eligible.sum().clamp_min(1), eligible.sum()


def parent_preservation(logits, teacher, target):
    teacher = teacher.detach()
    index = teacher.argmax(-1, keepdim=True)
    reliable = target.gather(-1, index) >= .5
    eligible = reliable & (target <= .1)
    parent_gap = teacher.gather(-1, index) - teacher
    student_gap = logits.gather(-1, index) - logits
    return (F.relu(parent_gap - student_gap) * eligible).sum() / eligible.sum().clamp_min(1), eligible.sum()


def main():
    p=argparse.ArgumentParser()
    for k in ['cache','contexts','origins','bank','labels','parent','parent-result','output']:
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--arm',choices=['human_bce','human_choice','generic_choice'],required=True)
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
    model=SemanticChoicePrototype().to(device)
    model.initialize_parent(torch.load(a.parent,map_location='cpu'))
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==99907
    frozen={k:v.detach().cpu().clone() for k,v in model.named_parameters() if not v.requires_grad}
    buffers={k:v.detach().cpu().clone() for k,v in model.named_buffers()}
    initial={s:snapshot(model,panel[s],bank,device) for s in ['fit','development']}
    parent_indices={s:x[0].argmax(-1) for s,x in initial.items()}
    assert parent_indices['development'].tolist()==[r['selected'] for r in old['development_rows']]
    text_condition='generic' if a.arm=='generic_choice' else 'human_text'
    fit=panel['fit'];ids=torch.arange(64)
    with torch.no_grad():
        start=model(inputs(fit,ids,bank,text_condition,device))
        assert torch.equal(start['selection_logits'].cpu(),initial['fit'][0][ids])
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4)
    started=time.time();history=[];gradients=[]
    epochs=12 if a.mode=='train' else 1
    for epoch in range(epochs):
        batches=torch.randperm(len(fit['key']),generator=torch.Generator().manual_seed(2027+epoch)).split(64) if a.mode=='train' else [torch.arange(64)]*3
        # Keep the frozen parent's inference attention path; autograd stays enabled.
        terms=[];eligible_states=0;eligible_pairs=0;model.eval()
        for ids in batches:
            optimizer.zero_grad(set_to_none=True)
            data=inputs(fit,ids,bank,text_condition,device);out=model(data);target=fit['iou'][ids].to(device)
            assert torch.equal(out['visual_selection_logits'].detach().cpu(),initial['fit'][0][ids])
            if a.arm=='human_bce':
                localization=F.binary_cross_entropy_with_logits(out['selection_logits'],target)
                count=target.shape[0]
            else:
                localization,count=relative_choice(out['selection_logits'],target)
            preservation,pairs=parent_preservation(out['selection_logits'],out['visual_selection_logits'],target)
            loss=localization+preservation
            assert bool(torch.isfinite(loss));loss.backward()
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
            if a.mode=='sanity':
                gradients.append({n:sum(float(p.grad.norm()) for k,p in model.named_parameters() if k.startswith(n+'.') and p.grad is not None) for n in ['parent.evidence.text','parent.evidence.phrase_read','parent.evidence.text_read','parent.evidence.evidence','semantic_choice']})
            optimizer.step();terms.append([float(x.detach()) for x in [loss,localization,preservation]])
            eligible_states+=int(count);eligible_pairs+=int(pairs)
        history.append(dict(epoch=epoch+1,optimizer_steps=len(terms),mean_losses=np.mean(terms,axis=0).tolist(),localization_states=eligible_states,preservation_pairs=eligible_pairs,seconds=time.time()-started))
        print(json.dumps(history[-1]),flush=True)
    for k,v in model.named_parameters():
        if k in frozen:assert torch.equal(v.detach().cpu(),frozen[k]),k
    for k,v in model.named_buffers():assert torch.equal(v.cpu(),buffers[k]),k
    for split in initial:
        final=snapshot(model,panel[split],bank,device)
        assert all(torch.equal(x,y) for x,y in zip(final,initial[split])),split
    for split in ['fit','development']:
        model.eval()
        with torch.no_grad():
            for start in range(0,len(panel[split]['key']),64):
                ids=torch.arange(start,min(start+64,len(panel[split]['key'])))
                data=inputs(panel[split],ids,bank,text_condition,device);out=model(data)
                selected=out['selected_index'];row=torch.arange(len(ids),device=device)
                assert all(bool(torch.isfinite(v).all()) for v in out.values())
                assert torch.equal(out['quality_logits'].cpu(),initial[split][1][ids])
                assert torch.equal(out['selected_box'],data['boxes'][row,selected])
                assert torch.equal(out['selected_score'],out['selection_logits'][row,selected])
                assert torch.equal(out['selected_quality'],out['quality_logits'][row,selected])
                assert torch.equal(out['selected_feature'],out['candidate_features'][row,selected])
    result=dict(status='complete_M115_semantic_choice_arm' if a.mode=='train' else 'complete_M115_gpu_sanity',
        arm=a.arm,text_condition=text_condition,mode=a.mode,seed=2027,epochs=epochs,batch_size=64,learning_rate=3e-4,
        optimizer_steps=sum(r['optimizer_steps'] for r in history),optimized_parameters=99907,
        human_confirmed=True,review_subject='initialization_only',current_candidate_semantic_labels=0,
        model_candidate_weak_labels_used=False,loss_components=['bce' if a.arm=='human_bce' else 'relative_choice','reliable_parent_preservation'],
        source_sha256=sha(__file__),model_sha256=sha(Path(__file__).with_name('semantic_choice_prototype.py')),
        parent_sha256=sha(a.parent),bank_sha256=sha(a.bank),labels_sha256=sha(a.labels),
        history=history,empty_all3039_scores_quality_exact=True,frozen_parameters_buffers_exact=True,
        nonempty_quality_exact=True,selected_fields_consistent=True,
        no_recursive_or_public_evaluation=True,elapsed_seconds=time.time()-started)
    a.output.mkdir()
    if a.mode=='sanity':
        assert gradients[-1]['parent.evidence.text']>0 and gradients[-1]['semantic_choice']>0
        result.update(gradient_norms=gradients,checkpoint_saved=False,development_metrics_reported=False)
    else:
        result['fit'],rows=evaluate(model,fit,bank,text_condition,device,parent_indices['fit'])
        (a.output/'fit_events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        conditions={};result['content_conditions']={}
        for mode in ['empty','generic','human_text']:
            summary,rows=evaluate(model,panel['development'],bank,mode,device,parent_indices['development'])
            result['content_conditions'][mode]=summary;conditions[mode]=rows
            (a.output/(mode+'_development.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        result['development']=result['content_conditions'][text_condition]
        result['paired_vs_own_empty']={mode:paired_vs_empty(conditions[mode],conditions['empty']) for mode in ['generic','human_text']}
        torch.save(model.state_dict(),a.output/'final.pt')
        reloaded=torch.load(a.output/'final.pt',map_location='cpu')
        assert all(torch.equal(v.cpu(),reloaded[k]) for k,v in model.state_dict().items())
        result.update(final_state_roundtrip_exact=True,final_sha256=sha(a.output/'final.pt'))
    result['gpu_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],arm=a.arm,optimizer_steps=result['optimizer_steps'])),flush=True)


if __name__=='__main__':main()
