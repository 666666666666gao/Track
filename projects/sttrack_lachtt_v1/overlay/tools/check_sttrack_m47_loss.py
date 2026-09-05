"""Check M47 loss semantics before any fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo))
    from lib.models.sttrack.lachtt_candidate_set import supervised_loss as old_loss
    from lib.models.sttrack.lachtt_candidate_multipositive_loss import supervised_loss as new_loss
    torch.manual_seed(2026);torch.set_num_threads(1)
    current=torch.tensor([2,10,4,10]);previous=torch.tensor([1,3,10,10])
    ciou=torch.zeros(4,10,dtype=torch.float64);piou=ciou.clone()
    for i in range(4):
        if current[i]<10:ciou[i,current[i]]=.8
        if previous[i]<10:piou[i,previous[i]]=.8
    logits=torch.randn(4,11,dtype=torch.float64,requires_grad=True)
    affinity=torch.randn(4,11,11,dtype=torch.float64,requires_grad=True)
    old=old_loss(logits,affinity,current,previous);new=new_loss(logits,affinity,current,previous,ciou,piou)
    loss_error=max(float((a-b).abs()) for a,b in zip(old,new));assert loss_error<1e-12
    grad_old=torch.autograd.grad(old[0],(logits,affinity),retain_graph=True)
    grad_new=torch.autograd.grad(new[0],(logits,affinity),retain_graph=True)
    gradient_error=max(float((a-b).abs().max()) for a,b in zip(grad_old,grad_new));assert gradient_error<1e-12
    assert float(grad_new[1][0,0,0])==0.,'Unknown distractor-to-distractor entry gained supervision'
    empty=torch.full((4,),10);zeros=torch.zeros(4,10,dtype=torch.float64)
    both_empty=new_loss(logits,affinity,empty,empty,zeros,zeros)
    assert float(both_empty[2])==0.
    empty_gradient=torch.autograd.grad(both_empty[2],affinity)[0];assert not empty_gradient.any()
    action=torch.zeros(1,dtype=torch.long);l=torch.zeros(1,11,dtype=torch.float64)
    a=torch.zeros(1,11,11,dtype=torch.float64);c=torch.zeros(1,10,dtype=torch.float64);p=c.clone();c[0,0]=.8;p[0,:2]=.8
    boosted=a.clone();boosted[0,0,1]=2.
    old_before=float(old_loss(l,a,action,action)[2]);old_after=float(old_loss(l,boosted,action,action)[2])
    new_before=float(new_loss(l,a,action,action,c,p)[2]);new_after=float(new_loss(l,boosted,action,action,c,p)[2])
    assert old_after>old_before and new_after<new_before
    assert float(new_loss(l,a,action,action,c,p)[1])==float(old_loss(l,a,action,action)[1])
    source=repo/'lib/models/sttrack/lachtt_candidate_multipositive_loss.py'
    result=dict(status='pass',single_positive_loss_max_error=loss_error,single_positive_gradient_max_error=gradient_error,
        both_empty_matching_zero=True,both_empty_matching_gradient_zero=True,unknown_distractor_pair_unsupervised=True,
        action_loss_unchanged=True,valid_alternative_boost=dict(old_before=old_before,old_after=old_after,new_before=new_before,new_after=new_after),
        loss_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),test_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        training_steps=0,scope='CPU loss/gradient contract checks, not a model performance result.')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
