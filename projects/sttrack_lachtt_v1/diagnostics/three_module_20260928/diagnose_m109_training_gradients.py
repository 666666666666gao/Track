"""M109: fit-only M108 objective diagnostics, no optimizer or state mutation."""
import argparse,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from analyze_train_states import sha
from instance_ab_prototype import InstanceCandidatePrototype
from train_ab_visual_control import load_inputs,native_candidate_preservation
from train_m108_frozen_semantics import inputs,TRAIN_PREFIXES


def cosine(a,b):
    denominator=a.norm()*b.norm()
    return float(torch.dot(a,b)/denominator) if denominator>0 else None


def main():
    p=argparse.ArgumentParser()
    for name in ['cache','contexts','origins','bank','labels','parent','completed','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arm',choices=['generic','weak_text'],required=True)
    a=p.parse_args();assert not a.output.exists()
    torch.set_num_threads(1);torch.manual_seed(2027);torch.cuda.manual_seed_all(2027)
    device=torch.device('cuda');started=time.time()
    driver=json.loads((a.completed/'result.json').read_text())
    assert driver['status']=='complete_M108_frozen_semantic_pair'
    assert sha(a.bank)==driver['bank_sha256'] and sha(a.labels)==driver['labels_sha256']
    assert sha(a.parent)=='1d187d78c1dd76ebf59e41fd31af16fa69cecc37b4cb6c74fbdfe3ed5c237293'
    final=a.completed/('train_'+a.arm)/'final.pt'
    assert sha(final)==driver['arms'][a.arm]['final_sha256']
    fit=load_inputs(a.cache,a.contexts,a.origins,False)['fit']
    bank=torch.load(a.bank,map_location='cpu');labels=json.loads(a.labels.read_text())
    keys={k:i for i,k in enumerate(fit['key'])};pairs={};relations=[]
    for row in labels['candidates']:
        if row['choice'] in ['A','B']:
            assert row['key'] in keys
            i=keys[row['key']]
            assert bank['splits'][bank['sequences'].index(row['sequence'])]=='fit'
            g=row['a_index'] if row['choice']=='A' else row['b_index']
            b=row['b_index'] if row['choice']=='A' else row['a_index']
            pairs[i]=(g,b)
            gi=float(fit['iou'][i,g]);bi=float(fit['iou'][i,b])
            relations.append(dict(key=row['key'],preferred_iou=gi,other_iou=bi,
                                  preferred_iou50=gi>=.5,other_iou50=bi>=.5))
    assert len(pairs)==11
    batches=torch.randperm(len(fit['key']),generator=torch.Generator().manual_seed(2038)).split(64)
    records={}
    for snapshot,weights in [('initial',a.parent),('final',final)]:
        model=InstanceCandidatePrototype().to(device)
        model.load_state_dict(torch.load(weights,map_location='cpu'),strict=True)
        model.requires_grad_(False)
        for name,param in model.named_parameters():
            if name.startswith(TRAIN_PREFIXES):param.requires_grad_(True)
        if snapshot=='initial':
            torch.nn.init.zeros_(model.semantic.weight);torch.nn.init.zeros_(model.phrase.weight)
        before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        parameters=[p for p in model.parameters() if p.requires_grad]
        assert sum(p.numel() for p in parameters)==95683
        model.eval();rows=[]
        for batch_id,ids in enumerate(batches):
            data=inputs(fit,ids,bank,a.arm,device);out=model(data)
            if snapshot=='initial':assert torch.equal(out['selection_logits'],out['visual_selection_logits'])
            target=fit['iou'][ids].to(device)
            localization=F.binary_cross_entropy_with_logits(out['selection_logits'],target)
            preservation,_=native_candidate_preservation(out['selection_logits'],data['base_scores'],target)
            positions=[(j,pairs[i]) for j,i in enumerate(ids.tolist()) if i in pairs]
            weak=out['selection_logits'].sum()*0
            if positions:
                weak=torch.stack([F.softplus(out['selection_logits'][j,b]-out['selection_logits'][j,g]) for j,(g,b) in positions]).mean()
            losses=[localization,preservation,weak];gradients=[]
            for index,loss in enumerate(losses):
                assert bool(torch.isfinite(loss))
                grad=torch.autograd.grad(loss,parameters,retain_graph=index<2)
                vector=torch.cat([g.reshape(-1) for g in grad])
                assert bool(torch.isfinite(vector).all());gradients.append(vector.detach())
            loc,keep,pair=gradients;baseline=loc+keep
            norms=[float(g.norm()) for g in gradients]
            rows.append(dict(batch=batch_id,count=len(ids),weak_pairs=len(positions),
                losses=[float(x.detach()) for x in losses],gradient_norms=norms,
                weak_vs_localization_cosine=cosine(pair,loc),weak_vs_preservation_cosine=cosine(pair,keep),
                localization_vs_preservation_cosine=cosine(loc,keep),weak_vs_baseline_cosine=cosine(pair,baseline),
                weak_to_baseline_norm=norms[2]/float(baseline.norm()) if baseline.norm()>0 else None,
                total_vs_baseline_cosine=cosine(baseline+pair,baseline)))
        assert sum(r['weak_pairs'] for r in rows)==11 and sum(r['count'] for r in rows)==2544
        assert all(p.grad is None for p in model.parameters())
        assert all(torch.equal(before[k],v.detach().cpu()) for k,v in model.state_dict().items())
        records[snapshot]=rows
    a.output.mkdir()
    # This per-label relation is private and is not included in the public result.
    (a.output/'private_pair_relations.json').write_text(json.dumps(relations,indent=2)+'\n')
    result=dict(status='complete_M109_readonly_fit_gradient_diagnostic',arm=a.arm,
        source_sha256=sha(__file__),trainer_sha256=sha(Path(__file__).with_name('train_m108_frozen_semantics.py')),
        parent_sha256=sha(a.parent),final_sha256=sha(final),bank_sha256=sha(a.bank),labels_sha256=sha(a.labels),
        fit_states=2544,weak_pairs=11,selected_epoch_order=12,seed=2027,order_seed=2038,
        no_optimizer_or_training=True,no_development_model_evaluation=True,no_public_evaluation=True,model_state_exact=True,
        pairing=dict(preferred_higher_iou=sum(r['preferred_iou']>r['other_iou'] for r in relations),
                     preferred_lower_iou=sum(r['preferred_iou']<r['other_iou'] for r in relations),
                     equal_iou=sum(r['preferred_iou']==r['other_iou'] for r in relations),
                     preferred_iou50=sum(r['preferred_iou50'] for r in relations),
                     other_iou50=sum(r['other_iou50'] for r in relations)),
        batches=records,elapsed_seconds=time.time()-started,
        scope='Replay exact fit loss formulas/order at initial/final weights; no historical gradient replay. IoU is not physical identity.')
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='batches'}),flush=True)


if __name__=='__main__':main()
