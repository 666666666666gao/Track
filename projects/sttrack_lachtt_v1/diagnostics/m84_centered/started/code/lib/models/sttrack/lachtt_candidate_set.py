"""Learn target identity over two complete candidate sets and causal templates.

Inspired by KeepTrack's candidate-set and partial matching formulation; this
implementation uses native STTrack RGB/depth RoIs and PyTorch attention.
"""
import torch
from torch import nn
import torch.nn.functional as F


class CandidateSetAssociation(nn.Module):
    def __init__(self, use_geometry=True):
        super().__init__()
        self.use_geometry=use_geometry
        self.cell=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,32),nn.GELU())
        self.object=nn.Sequential(nn.Linear(2*16*32,128),nn.LayerNorm(128))
        self.position=nn.Sequential(nn.Linear(4,64),nn.GELU(),nn.Linear(64,128))
        self.response=nn.Linear(1,128)
        self.role=nn.Embedding(4,128)
        self.previous_target=nn.Parameter(torch.zeros(128))
        layer=nn.TransformerEncoderLayer(128,4,256,dropout=0.,batch_first=True,norm_first=True)
        self.context=nn.TransformerEncoder(layer,2)
        self.identity=nn.Linear(128,1)
        self.none=nn.Linear(128,1)
        self.match=nn.Linear(128,128,bias=False)
        self.unmatched=nn.Parameter(torch.tensor(0.))
        nn.init.zeros_(self.identity.weight);nn.init.zeros_(self.identity.bias)
        nn.init.zeros_(self.none.weight);nn.init.constant_(self.none.bias,-10.)

    def forward(self,current,previous,references,geometry,scores,previous_choice):
        # Current/previous: B,10,2,16,768; templates: B,2,2,16,768.
        batch=current.shape[0];device=current.device
        x=torch.cat([current,previous,references],dim=1)
        x=self.object(self.cell(x).flatten(-3))
        coords=geometry if self.use_geometry else torch.zeros_like(geometry)
        x=torch.cat([x[:,:20]+self.position(coords)+self.response(scores[...,None]),x[:,20:]],dim=1)
        roles=torch.tensor([0]*10+[1]*10+[2,3],device=device)
        x=x+self.role(roles)[None]
        marker=F.one_hot(previous_choice,10).to(x.dtype)[...,None]*self.previous_target
        x=torch.cat([x[:,:10],x[:,10:20]+marker,x[:,20:]],dim=1)
        x=self.context(x)
        logits=torch.cat([self.identity(x[:,:10]).squeeze(-1)+scores[:,:10],self.none(x[:,20])],dim=1)
        c=F.normalize(self.match(x[:,:10]),dim=-1)
        p=F.normalize(self.match(x[:,10:20]),dim=-1)
        affinity=torch.matmul(c,p.transpose(1,2))/.1
        affinity=torch.cat([affinity,self.unmatched.expand(batch,10,1)],dim=2)
        affinity=torch.cat([affinity,self.unmatched.expand(batch,1,11)],dim=1)
        return logits,affinity


def select_candidate(logits):
    choices=logits.argmax(1)
    return torch.where(choices==10,torch.zeros_like(choices),choices)


def supervised_loss(logits,affinity,current_target,previous_target):
    """Supervise only the known target correspondence; other instances unlabeled."""
    main=F.cross_entropy(logits,current_target)
    index=torch.arange(len(logits),device=logits.device)
    cvalid=current_target<10;pvalid=previous_target<10
    forward=F.cross_entropy(affinity[index,current_target],previous_target,reduction='none')
    backward=F.cross_entropy(affinity.transpose(1,2)[index,previous_target],current_target,reduction='none')
    # Both-empty pairs have no correspondence label and contribute zero.
    count=(cvalid.sum()+pvalid.sum()).clamp_min(1)
    matching=(forward[cvalid].sum()+backward[pvalid].sum())/count
    return main+.25*matching,main,matching
