"""Exercise set permutation, explicit geometry ablation and real optimization."""
import argparse
import json
from pathlib import Path
import sys
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--repository',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();sys.path.insert(0,str(args.repository))
    from lib.models.sttrack.lachtt_candidate_set import CandidateSetAssociation,select_candidate,supervised_loss
    torch.set_num_threads(1);torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
    current=torch.randn(3,10,2,16,768,device='cuda');previous=torch.randn_like(current)
    refs=torch.randn(3,2,2,16,768,device='cuda');coords=torch.rand(3,20,4,device='cuda')
    scores=-torch.arange(1,21,device='cuda',dtype=torch.float32).expand(3,-1)/10
    previous_choice=torch.tensor([0,3,8],device='cuda');inputs=[current,previous,refs,coords,scores,previous_choice]
    model=CandidateSetAssociation().cuda().eval()
    with torch.no_grad():
        initial,_=model(*inputs);assert select_candidate(initial).tolist()==[0,0,0]
        torch.nn.init.normal_(model.identity.weight,std=.1)
        logits,affinity=model(*inputs)
        cp=torch.tensor([4,0,9,1,3,2,8,6,5,7],device='cuda');pp=torch.tensor([8,3,2,9,0,7,1,5,4,6],device='cuda')
        permutation=torch.cat([cp,pp+10]);pc=torch.argsort(pp)[previous_choice]
        changed,matched=model(current[:,cp],previous[:,pp],refs,coords[:,permutation],scores[:,permutation],pc)
        outperm=torch.cat([cp,torch.tensor([10],device='cuda')]);prevperm=torch.cat([pp,torch.tensor([10],device='cuda')])
        logerr=float((changed-logits[:,outperm]).abs().max());matcherr=float((matched-affinity[:,outperm][:,:,prevperm]).abs().max())
        assert logerr<1e-5 and matcherr<1e-5,(logerr,matcherr)
        control=CandidateSetAssociation(False).cuda().eval();control.load_state_dict(model.state_dict(),strict=True)
        zeros,_=control(*inputs);different=coords.clone();different[:,:10,:2]+=.2
        zeros2,_=control(current,previous,refs,different,scores,previous_choice)
        actual,_=model(current,previous,refs,different,scores,previous_choice)
        assert torch.equal(zeros,zeros2)
        geometry_delta=float((actual-logits).abs().max());assert geometry_delta>1e-6
    model=CandidateSetAssociation().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
    ct=torch.tensor([2,10,5],device='cuda');pt=torch.tensor([1,3,10],device='cuda');losses=[]
    for step in range(24):
        optimizer.zero_grad(set_to_none=True);a,b=model(*inputs);loss,main,pair=supervised_loss(a,b,ct,pt)
        assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
        optimizer.step();losses.append(float(loss))
    assert losses[-1]<losses[0]
    result=dict(status='PASS',parameters=sum(p.numel() for p in model.parameters()),initial_default_exact=True,
        candidate_permutation_logit_error=logerr,candidate_permutation_match_error=matcherr,
        geometry_ablation_exact=True,geometry_delta=geometry_delta,optimizer_steps=24,
        initial_synthetic_loss=losses[0],final_synthetic_loss=losses[-1],scope='Contract test only; no real development or public metric.')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
