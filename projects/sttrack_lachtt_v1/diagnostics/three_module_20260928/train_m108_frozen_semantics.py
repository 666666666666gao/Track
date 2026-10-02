"""M108 generic/reviewed text pair; the M101 visual parent is immutable."""
import argparse,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from analyze_train_states import sha
from instance_ab_prototype import InstanceCandidatePrototype
from train_ab_visual_control import load_inputs,summarize,native_candidate_preservation
from train_m107_weak_semantics import inputs as m107_inputs

TRAIN_PREFIXES=('evidence.text.','evidence.phrase_slots','evidence.phrase_read.',
                'evidence.text_read.','evidence.evidence.','semantic.','phrase.')


def condition_bank(bank,mode):
    if mode=='generic':
        return dict(bank,tokens=bank['generic'][None,None].expand_as(bank['tokens'])*bank['mask'][...,None])
    return bank


def inputs(panel,indices,bank,mode,device):
    return m107_inputs(panel,indices,condition_bank(bank,mode),
                      'empty' if mode=='empty' else 'weak_text',device)


def snapshot(model,panel,bank,device):
    model.eval();scores=[];qualities=[]
    with torch.no_grad():
        for start in range(0,len(panel['key']),64):
            ids=torch.arange(start,min(start+64,len(panel['key'])))
            out=model(inputs(panel,ids,bank,'empty',device))
            assert torch.equal(out['selection_logits'],out['visual_selection_logits'])
            scores.append(out['selection_logits'].cpu());qualities.append(out['quality_logits'].cpu())
    return torch.cat(scores),torch.cat(qualities)


def evaluate(model,panel,bank,mode,device,parent_indices):
    model.eval();rows=[]
    with torch.no_grad():
        for start in range(0,len(panel['key']),64):
            ids=torch.arange(start,min(start+64,len(panel['key'])))
            out=model(inputs(panel,ids,bank,mode,device))
            for j,i in enumerate(ids.tolist()):
                iou=panel['iou'][i];selected=int(out['selected_index'][j]);parent=int(parent_indices[i])
                rows.append(dict(key=panel['key'][i],strata=panel['strata'][i],
                                 native_iou=float(iou[0]),selected_iou=float(iou[selected]),
                                 oracle_iou=float(iou.max()),selected=selected,
                                 parent_selected=parent,parent_iou=float(iou[parent])))
    return summarize(rows),rows


def main():
    p=argparse.ArgumentParser()
    for k in ['cache','contexts','origins','bank','labels','parent','parent-result','output']:
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--arm',choices=['generic','weak_text'],required=True)
    p.add_argument('--mode',choices=['sanity','train'],required=True)
    a=p.parse_args();assert not a.output.exists()
    torch.set_num_threads(1);random.seed(2027);np.random.seed(2027)
    torch.manual_seed(2027);torch.cuda.manual_seed_all(2027);device=torch.device('cuda')
    panel=load_inputs(a.cache,a.contexts,a.origins,False)
    bank=torch.load(a.bank,map_location='cpu');labels=json.loads(a.labels.read_text())
    assert bank['labels_sha256']==sha(a.labels) and not bank['human_confirmed']
    assert bank['m107_bank_sha256']=='7acb2f5509b23bc0a900f6c252017672f04961394054a85deb9b01e697ef388c'
    old=json.loads(a.parent_result.read_text())
    assert sha(a.parent)==old['final_weights_sha256']=='1d187d78c1dd76ebf59e41fd31af16fa69cecc37b4cb6c74fbdfe3ed5c237293'
    model=InstanceCandidatePrototype().to(device)
    model.load_state_dict(torch.load(a.parent,map_location='cpu'),strict=True)
    model.requires_grad_(False)
    for name,param in model.named_parameters():
        if name.startswith(TRAIN_PREFIXES):param.requires_grad_(True)
    torch.nn.init.zeros_(model.semantic.weight);torch.nn.init.zeros_(model.phrase.weight)
    frozen={k:v.detach().cpu().clone() for k,v in model.named_parameters() if not v.requires_grad}
    buffers={k:v.detach().cpu().clone() for k,v in model.named_buffers()}
    initial={s:snapshot(model,panel[s],bank,device) for s in ['fit','development']}
    parent_indices={s:x[0].argmax(-1) for s,x in initial.items()}
    assert parent_indices['development'].tolist()==[r['selected'] for r in old['development_rows']]
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4)
    fit=panel['fit'];key_index={k:i for i,k in enumerate(fit['key'])};pairs={}
    for r in labels['candidates']:
        if r['choice'] in ['A','B']:
            assert r['key'] in key_index and bank['splits'][bank['sequences'].index(r['sequence'])]=='fit'
            good=r['a_index'] if r['choice']=='A' else r['b_index']
            other=r['b_index'] if r['choice']=='A' else r['a_index']
            pairs[key_index[r['key']]]=(good,other)
    assert len(pairs)==11
    started=time.time();history=[];gradients=[];epochs=12 if a.mode=='train' else 1
    for epoch in range(epochs):
        batches=torch.randperm(len(fit['key']),generator=torch.Generator().manual_seed(2027+epoch)).split(64) if a.mode=='train' else [torch.arange(64)]*3
        terms=[];pair_calls=0;model.train()
        for ids in batches:
            optimizer.zero_grad(set_to_none=True)
            data=inputs(fit,ids,bank,a.arm,device);out=model(data);target=fit['iou'][ids].to(device)
            localization=F.binary_cross_entropy_with_logits(out['selection_logits'],target)
            preservation,_=native_candidate_preservation(out['selection_logits'],data['base_scores'],target)
            positions=[(j,pairs[i]) for j,i in enumerate(ids.tolist()) if i in pairs]
            weak=out['selection_logits'].sum()*0
            if positions:
                weak=torch.stack([F.softplus(out['selection_logits'][j,b]-out['selection_logits'][j,g]) for j,(g,b) in positions]).mean()
                pair_calls+=len(positions)
            loss=localization+preservation+weak
            assert bool(torch.isfinite(loss));loss.backward()
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
            if a.mode=='sanity':
                norms={n:sum(float(p.grad.norm()) for k,p in model.named_parameters() if k.startswith(n+'.') and p.grad is not None)
                       for n in ['evidence.text','evidence.phrase_read','evidence.text_read','semantic','phrase']}
                gradients.append(norms)
            optimizer.step();terms.append([float(x.detach()) for x in [loss,localization,preservation,weak]])
        history.append(dict(epoch=epoch+1,optimizer_steps=len(terms),mean_losses=np.mean(terms,axis=0).tolist(),weak_pair_calls=pair_calls,seconds=time.time()-started))
        print(json.dumps(history[-1]),flush=True)
    for k,v in model.named_parameters():
        if k in frozen:assert torch.equal(v.detach().cpu(),frozen[k]),k
    for k,v in model.named_buffers():assert torch.equal(v.cpu(),buffers[k]),k
    for split in initial:
        final=snapshot(model,panel[split],bank,device)
        assert all(torch.equal(x,y) for x,y in zip(final,initial[split])),split
    result=dict(status='complete_M108_frozen_semantic_arm' if a.mode=='train' else 'complete_M108_gpu_sanity',
                arm=a.arm,mode=a.mode,seed=2027,epochs=epochs,batch_size=64,learning_rate=3e-4,
                optimizer_steps=sum(r['optimizer_steps'] for r in history),trainable_prefixes=TRAIN_PREFIXES,
                optimized_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                source_sha256=sha(__file__),prototype_sha256=sha(Path(__file__).with_name('instance_ab_prototype.py')),
                input_helper_sha256=sha(Path(__file__).with_name('train_m107_weak_semantics.py')),
                parent_sha256=sha(a.parent),bank_sha256=sha(a.bank),labels_sha256=sha(a.labels),
                human_confirmed=False,weak_identity_pairs=11,history=history,
                empty_all3039_scores_quality_exact=True,frozen_parameters_buffers_exact=True,
                no_recursive_or_public_evaluation=True,elapsed_seconds=time.time()-started)
    a.output.mkdir()
    if a.mode=='sanity':
        assert gradients[-1]['evidence.text']>0 and gradients[-1]['semantic']>0
        result.update(gradient_norms=gradients,checkpoint_saved=False,development_metrics_reported=False)
    else:
        result['fit'],rows=evaluate(model,fit,bank,a.arm,device,parent_indices['fit'])
        (a.output/'fit_events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        result['content_conditions']={}
        for mode in ['empty','generic','weak_text']:
            summary,rows=evaluate(model,panel['development'],bank,mode,device,parent_indices['development'])
            result['content_conditions'][mode]=summary
            (a.output/(mode+'_development.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        result['development']=result['content_conditions'][a.arm]
        dev=result['development']
        result['fixed_state_checks']=dict(correct_exceeds_native=dev['all']['selected_iou50']>dev['all']['native_iou50'],
            mean_iou_exceeds_native=dev['all']['selected_mean_iou']>dev['all']['native_mean_iou'],
            healthy_breaks_zero=dev['healthy']['breaks']==0,
            transition_correct_at_least_native=dev['transition']['selected_iou50']>=dev['transition']['native_iou50'])
        torch.save(model.state_dict(),a.output/'final.pt')
        reloaded=torch.load(a.output/'final.pt',map_location='cpu')
        assert all(torch.equal(v.cpu(),reloaded[k]) for k,v in model.state_dict().items())
        result['final_state_roundtrip_exact']=True
        result['final_sha256']=sha(a.output/'final.pt')
    result['gpu_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['history','content_conditions','fit','development']}),flush=True)


if __name__=='__main__':main()
