"""M107 model-weak A+B versus Empty, fixed-state localization/weak-pair learning."""
import argparse,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from analyze_train_states import sha
from instance_ab_prototype import InstanceCandidatePrototype
from train_ab_visual_control import load_inputs,batch,summarize,native_candidate_preservation

def inputs(panel,indices,bank,mode,device):
 data=batch(panel,indices,bank['empty'].to(device),device)
 ids=torch.tensor([bank['sequences'].index(panel['key'][i].rsplit('@',1)[0]) for i in indices.tolist()])
 mask=bank['mask'][ids].to(device)
 text=bank['tokens'][ids].to(device).float()
 if mode=='empty':
  text=bank['empty'].to(device)[None,None].expand_as(text)*mask[...,None]
 data['text']=text;data['text_mask']=mask
 return data

def evaluate(model,panel,bank,mode,device,parent_rows):
 model.eval();rows=[]
 with torch.no_grad():
  for start in range(0,len(panel['key']),64):
   ids=torch.arange(start,min(start+64,len(panel['key'])))
   out=model(inputs(panel,ids,bank,mode,device))
   if mode=='empty':
    assert torch.equal(out['selection_logits'],out['visual_selection_logits'])
    assert not bool(out['semantic_delta'].any()) and not bool(out['phrase_delta'].any())
   for j,i in enumerate(ids.tolist()):
    iou=panel['iou'][i];selected=int(out['selected_index'][j]);key=panel['key'][i]
    rows.append(dict(key=key,strata=panel['strata'][i],native_iou=float(iou[0]),selected_iou=float(iou[selected]),oracle_iou=float(iou.max()),selected=selected,parent_selected=parent_rows[key]['selected'],parent_iou=parent_rows[key]['selected_iou']))
 return summarize(rows),rows

def main():
 p=argparse.ArgumentParser()
 for k in ['cache','contexts','origins','bank','labels','parent','parent-result','output']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--arm',choices=['empty','weak_text'],required=True)
 p.add_argument('--mode',choices=['sanity','train'],required=True)
 a=p.parse_args();assert not a.output.exists()
 torch.set_num_threads(1)
 random.seed(2027);np.random.seed(2027);torch.manual_seed(2027);torch.cuda.manual_seed_all(2027)
 device=torch.device('cuda')
 panel=load_inputs(a.cache,a.contexts,a.origins,False)
 bank=torch.load(a.bank,map_location='cpu');labels=json.loads(a.labels.read_text())
 assert bank['labels_sha256']==sha(a.labels) and not bank['human_confirmed']
 assert len(bank['sequences'])==152
 old=json.loads(a.parent_result.read_text());assert sha(a.parent)==old['final_weights_sha256']
 model=InstanceCandidatePrototype().to(device);model.load_state_dict(torch.load(a.parent,map_location='cpu'),strict=True)
 # The previously unused semantic readout is zero-initialized in both arms.
 torch.nn.init.zeros_(model.semantic.weight);torch.nn.init.zeros_(model.phrase.weight)
 for param in model.observation.parameters():param.requires_grad_(False)
 frozen={k:v.detach().cpu().clone() for k,v in model.observation.state_dict().items()}
 optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4)
 fit=panel['fit'];key_index={k:i for i,k in enumerate(fit['key'])}
 pairs={}
 for r in labels['candidates']:
  if r['choice'] in ['A','B']:
   assert r['key'] in key_index and bank['splits'][bank['sequences'].index(r['sequence'])]=='fit'
   good=r['a_index'] if r['choice']=='A' else r['b_index']
   other=r['b_index'] if r['choice']=='A' else r['a_index']
   pairs[key_index[r['key']]]=(good,other,r['audit_id'])
 assert len(pairs)==11
 ids=torch.arange(64)
 with torch.no_grad():
  initial=model(inputs(fit,ids,bank,a.arm,device))
  assert torch.equal(initial['selection_logits'],initial['visual_selection_logits'])
  dev=panel['development'];dev_ids=torch.arange(64)
  start=model(inputs(dev,dev_ids,bank,a.arm,device))
  assert [int(s) for s in start['selected_index'].cpu()]==[old['development_rows'][i]['selected'] for i in dev_ids.tolist()]
 started=time.time();history=[];sanity_gradients=[]
 epochs=12 if a.mode=='train' else 1
 for epoch in range(epochs):
  indices=torch.randperm(len(fit['key']),generator=torch.Generator().manual_seed(2027+epoch)).split(64) if a.mode=='train' else [ids,ids,ids]
  terms=[];pair_calls=0
  model.train()
  for ids in indices:
   optimizer.zero_grad(set_to_none=True)
   data=inputs(fit,ids,bank,a.arm,device);out=model(data);target=fit['iou'][ids].to(device)
   if a.arm=='empty':assert torch.equal(out['selection_logits'],out['visual_selection_logits'])
   localization=F.binary_cross_entropy_with_logits(out['selection_logits'],target)+F.binary_cross_entropy_with_logits(out['quality_logits'],target)
   preservation,_=native_candidate_preservation(out['selection_logits'],data['base_scores'],target)
   positions=[(j,pairs[i]) for j,i in enumerate(ids.tolist()) if i in pairs]
   weak=out['selection_logits'].sum()*0
   if positions:
    weak=torch.stack([F.softplus(out['selection_logits'][j,b]-out['selection_logits'][j,g]) for j,(g,b,_) in positions]).mean()
    pair_calls+=len(positions)
   loss=localization+preservation+weak
   assert bool(torch.isfinite(loss));loss.backward()
   assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
   if a.mode=='sanity':
    names=['evidence.text','evidence.phrase_read','evidence.text_read','semantic','selection']
    norms={n:sum(float(p.grad.norm()) for k,p in model.named_parameters() if k.startswith(n+'.') and p.grad is not None) for n in names}
    sanity_gradients.append(norms)
   optimizer.step();terms.append([float(x.detach()) for x in [loss,localization,preservation,weak]])
  history.append(dict(epoch=epoch+1,optimizer_steps=len(terms),mean_losses=np.mean(terms,axis=0).tolist(),weak_pair_calls=pair_calls,seconds=time.time()-started))
  print(json.dumps(history[-1]),flush=True)
 for k,v in model.observation.state_dict().items():assert torch.equal(v.cpu(),frozen[k])
 result=dict(status='complete_M107_weak_semantic_pair_arm' if a.mode=='train' else 'complete_M107_gpu_sanity',arm=a.arm,mode=a.mode,source_sha256=sha(__file__),prototype_sha256=sha(Path(__file__).with_name('instance_ab_prototype.py')),seed=2027,epochs=epochs,batch_size=64,learning_rate=3e-4,optimizer_steps=sum(r['optimizer_steps'] for r in history),parent_sha256=sha(a.parent),bank_sha256=sha(a.bank),labels_sha256=sha(a.labels),total_parameters=sum(p.numel() for p in model.parameters()),optimized_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),weak_identity_pairs=11,model_weak_labels=True,human_confirmed=False,initial_parent_selection_checked=True,history=history,no_recursive_or_public_evaluation=True,elapsed_seconds=time.time()-started,gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
 a.output.mkdir()
 if a.mode=='sanity':
  result['gradient_norms']=sanity_gradients
  if a.arm=='weak_text':
   assert sanity_gradients[-1]['evidence.text']>0 and sanity_gradients[-1]['semantic']>0
  result['development_evaluated']=False;result['checkpoint_saved']=False
 else:
  parent_dev={r['key']:r for r in old['development_rows']}
  # Fitting parent choices are replayed on the same frozen cache.
  parent=InstanceCandidatePrototype().to(device);parent.load_state_dict(torch.load(a.parent,map_location='cpu'),strict=True);parent.eval()
  parent_fit={}
  with torch.no_grad():
   for start in range(0,len(fit['key']),64):
    ids=torch.arange(start,min(start+64,len(fit['key'])))
    pred=parent(inputs(fit,ids,bank,'empty',device))
    for j,i in enumerate(ids.tolist()):
     si=int(pred['selected_index'][j]);parent_fit[fit['key'][i]]=dict(selected=si,selected_iou=float(fit['iou'][i,si]))
  result['fit'],fit_rows=evaluate(model,fit,bank,a.arm,device,parent_fit)
  result['development'],dev_rows=evaluate(model,panel['development'],bank,a.arm,device,parent_dev)
  # Same final weight content interventions, never checkpoint selection.
  result['content_conditions']={}
  for mode in ['empty','weak_text']:
   summary,rows=evaluate(model,panel['development'],bank,mode,device,parent_dev)
   result['content_conditions'][mode]=summary
   (a.output/(mode+'_development.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
  for name,rows in [('fit',fit_rows),('development',dev_rows)]:
   (a.output/(name+'_events.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
  dev=result['development']
  result['fixed_state_checks']=dict(correct_exceeds_native=dev['all']['selected_iou50']>dev['all']['native_iou50'],mean_iou_exceeds_native=dev['all']['selected_mean_iou']>dev['all']['native_mean_iou'],healthy_breaks_zero=dev['healthy']['breaks']==0,transition_correct_at_least_native=dev['transition']['selected_iou50']>=dev['transition']['native_iou50'])
  torch.save(model.state_dict(),a.output/'final.pt');result['final_sha256']=sha(a.output/'final.pt')
 (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ['history','content_conditions','fit','development']}),flush=True)
if __name__=='__main__':main()
