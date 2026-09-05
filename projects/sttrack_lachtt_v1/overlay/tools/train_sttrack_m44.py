"""Fit one frozen M44 arm on Train folds2-4 and evaluate fold5 once."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(value):return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def check_binding(root,spec):
    binding=json.loads((root/'training_binding.json').read_text());repo=Path(spec['repository'])
    assert binding['spec_sha256']==sha(root/'spec.json')
    for name,digest in {**spec['source_sha256'],**binding['source_sha256']}.items():assert sha(repo/name)==digest,name
    assert sha(spec['checkpoint'])==spec['checkpoint_sha256']
    return binding


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--variant',choices=['geometry','appearance'],required=True)
    args=p.parse_args();root=args.root;variant=args.variant;spec=json.loads((root/'spec.json').read_text());repo=Path(spec['repository']);sys.path.insert(0,str(repo))
    binding=check_binding(root,spec)
    from lib.models.sttrack.lachtt_candidate_set import CandidateSetAssociation,select_candidate,supervised_loss
    from tools.train_sttrack_m42 import overlaps
    assert sha(root/'training_labels.json')==spec['labels_sha256'];labels=json.loads((root/'training_labels.json').read_text())
    receipts=[]
    for shard in [0,1]:
        x=json.loads((root/f'shard{shard}_receipt.json').read_text());assert x['status']=='complete' and x['spec_sha256']==sha(root/'spec.json');receipts.extend(x['sequences'])
    assert len(receipts)==spec['sequences'] and len({x['sequence'] for x in receipts})==spec['sequences']
    collected={key:[] for key in ['current','previous','references','geometry','scores']};keys=[];folds=[];ious=[];previous_ious=[]
    for receipt in sorted(receipts,key=lambda x:x['sequence']):
        path=root/'features'/(receipt['sequence']+'.pt');assert sha(path)==receipt['feature_sha256'];data=torch.load(path,map_location='cpu')
        assert data['spec_sha256']==sha(root/'spec.json')
        for key in collected:collected[key].append(data[key])
        for i,row in enumerate(data['records']):
            label=labels[row['key']];assert label['fold']==data['fold'] and label['sequence']==data['sequence']
            assert row['previous_choice']==0 and row['previous_frame']==row['frame']-1
            values=[]
            for side,boxkey,publickey in [('current','current_boxes','public_bbox'),('previous','previous_boxes','previous_public_bbox')]:
                if label[side] is None:v=torch.zeros(10)
                else:
                    gt=torch.tensor(label[side]);v=overlaps(data[boxkey][i],gt);v[0]=overlaps(data[publickey][i],gt)
                values.append(v)
            keys.append(row['key']);folds.append(data['fold']);ious.append(values[0]);previous_ious.append(values[1])
    assert len(keys)==spec['events'] and set(keys)==set(labels)
    tensors={key:torch.cat(value) for key,value in collected.items()};del collected,data
    ious=torch.stack(ious);previous_ious=torch.stack(previous_ious)
    current_target=ious.argmax(1);current_target[ious.max(1).values<.5]=10
    previous_target=previous_ious.argmax(1);previous_target[previous_ious.max(1).values<.5]=10
    fit=torch.tensor([i for i,x in enumerate(folds) if x in spec['fit_folds']]);dev=torch.tensor([i for i,x in enumerate(folds) if x==spec['development_fold']])
    assert len(fit)+len(dev)==len(keys)
    assert {keys[i].split('@')[0] for i in fit}.isdisjoint({keys[i].split('@')[0] for i in dev})
    opt=spec['optimization'];torch.set_num_threads(1);torch.manual_seed(opt['seed']);torch.cuda.manual_seed_all(opt['seed']);np.random.seed(opt['seed']);random.seed(opt['seed'])
    model=CandidateSetAssociation(variant=='geometry').cuda();initial={name:tensor_sha(v) for name,v in model.state_dict().items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=opt['lr'],weight_decay=opt['weight_decay']);order=torch.Generator().manual_seed(opt['seed']);digest=hashlib.sha256()
    steps=0;losses=[];started=time.time()
    def inputs(index):
        return [tensors[key][index].cuda().float() for key in ['current','previous','references','geometry','scores']]+[torch.zeros(len(index),dtype=torch.long,device='cuda')]
    for epoch in range(opt['epochs']):
        model.train();total=main_total=pair_total=0.;shuffled=fit[torch.randperm(len(fit),generator=order)];digest.update(shuffled.numpy().tobytes())
        for index in shuffled.split(opt['batch_size']):
            logits,affinity=model(*inputs(index));loss,main,pair=supervised_loss(logits,affinity,current_target[index].cuda(),previous_target[index].cuda())
            assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),opt['grad_clip']);assert torch.isfinite(norm)
            optimizer.step();steps+=1;total+=float(loss.detach())*len(index);main_total+=float(main.detach())*len(index);pair_total+=float(pair.detach())*len(index)
        row=dict(epoch=epoch+1,loss=total/len(fit),identity_loss=main_total/len(fit),matching_loss=pair_total/len(fit));losses.append(row)
        print(json.dumps(dict(variant=variant,elapsed=time.time()-started,optimizer_steps=steps,**row)),flush=True)
    checkpoint=root/(variant+'_final.pth')
    torch.save(dict(model=model.state_dict(),variant=variant,spec_sha256=sha(root/'spec.json'),binding_sha256=sha(root/'training_binding.json'),
        base_checkpoint_sha256=spec['checkpoint_sha256'],epochs=opt['epochs'],optimizer_steps=steps),checkpoint)
    def evaluate(net,index):
        net.eval();logits_all=[];matching_correct=matching_count=0
        with torch.no_grad():
            for b in index.split(opt['batch_size']):
                logits,affinity=net(*inputs(b));logits_all.append(logits.cpu());valid=current_target[b]<10
                predicted=affinity[torch.arange(len(b),device='cuda'),current_target[b].cuda()].argmax(1).cpu()
                matching_correct+=int(((predicted==previous_target[b])&valid).sum());matching_count+=int(valid.sum())
        logits=torch.cat(logits_all);selected=select_candidate(logits);value=ious[index].gather(1,selected[:,None]).flatten();default=ious[index,0]
        seqstats={}
        for name in sorted({keys[i].split('@')[0] for i in index}):
            mask=torch.tensor([keys[i].split('@')[0]==name for i in index]);seqstats[name]=dict(events=int(mask.sum()),mean_iou=float(value[mask].mean()),default_mean_iou=float(default[mask].mean()),gain=float((value-default)[mask].mean()))
        stats=dict(events=len(index),mean_iou=float(value.mean()),default_mean_iou=float(default.mean()),correct=int((value>=.5).sum()),default_correct=int((default>=.5).sum()),
            changes=int((selected!=0).sum()),none=int((logits.argmax(1)==10).sum()),rescues=int(((default<=.1)&(value>=.5)).sum()),breaks=int(((default>=.5)&(value<=.1)).sum()),
            positive_sequences=sum(x['gain']>0 for x in seqstats.values()),target_match_correct=matching_correct,target_match_count=matching_count,sequence_stats=seqstats,
            rows=[dict(key=keys[i],chosen=int(k),action_none=bool(n==10),default_iou=float(d),selected_iou=float(v),oracle_iou=float(ious[i].max()),
                target=int(current_target[i]),previous_target=int(previous_target[i])) for i,k,n,d,v in zip(index,selected,logits.argmax(1),default,value)])
        return stats,logits
    fitstats,_=evaluate(model,fit);devstats,devlogits=evaluate(model,dev)
    loaded=CandidateSetAssociation(variant=='geometry').cuda();loaded.load_state_dict(torch.load(checkpoint,map_location='cpu')['model'],strict=True)
    _,restored=evaluate(loaded,dev);assert torch.equal(restored,devlogits)
    check_binding(root,spec)
    result=dict(status='complete',variant=variant,parameters=sum(x.numel() for x in model.parameters()),optimizer_steps=steps,epochs=opt['epochs'],losses=losses,
        checkpoint_sha256=sha(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,reload_logits_exact=True,initial_state_sha256=initial,
        changed_tensors=[name for name,v in model.state_dict().items() if tensor_sha(v)!=initial[name]],sample_order_sha256=digest.hexdigest(),
        fit=fitstats,development=devstats,current_target_counts_fit=dict(Counter(current_target[fit].tolist())),current_target_counts_development=dict(Counter(current_target[dev].tolist())),
        source_binding_sha256=sha(root/'training_binding.json'),spec_sha256=sha(root/'spec.json'),elapsed_seconds=time.time()-started,
        claim='Static Train diagnostics; primary performance requires complete recursive validation. No public metric.')
    (root/(variant+'_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='complete',variant=variant,development={key:value for key,value in devstats.items() if key not in ['rows','sequence_stats']})),flush=True)


if __name__=='__main__':main()
