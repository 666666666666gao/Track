"""Partial target correspondence with multiple IoU-valid destination boxes."""
import torch
import torch.nn.functional as F


def _target_positions(ious):
    valid=ious>=.5
    return torch.cat([valid,~valid.any(dim=1,keepdim=True)],dim=1)


def _set_cross_entropy(logits,positive):
    return -torch.logsumexp(F.log_softmax(logits,dim=-1).masked_fill(~positive,-torch.inf),dim=-1)


def supervised_loss(logits,affinity,current_target,previous_target,current_ious,previous_ious):
    """Keep the single action loss and the original two selected target queries.

    Only the opposite-frame positive destination set changes. Real distractor
    identities remain unknown. An empty destination set supervises unmatched;
    both-empty input pairs contribute no correspondence loss, as in M45.
    """
    main=F.cross_entropy(logits,current_target)
    index=torch.arange(len(logits),device=logits.device)
    current_valid=current_target<10;previous_valid=previous_target<10
    forward=_set_cross_entropy(affinity[index,current_target],_target_positions(previous_ious))
    backward=_set_cross_entropy(affinity.transpose(1,2)[index,previous_target],_target_positions(current_ious))
    count=(current_valid.sum()+previous_valid.sum()).clamp_min(1)
    matching=(forward[current_valid].sum()+backward[previous_valid].sum())/count
    return main+.25*matching,main,matching
