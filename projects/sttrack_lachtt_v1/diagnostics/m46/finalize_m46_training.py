"""Finalize the saved M46 checkpoint after its report-only variable-name error."""
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


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root
    plan=json.loads((root/'spec.json').read_text());parent=Path(plan['source_root']);early=root/'early'
    spec=json.loads((parent/'spec.json').read_text());repo=Path(spec['repository']);sys.path.insert(0,str(repo))
    from tools.train_sttrack_m44 import sha,tensor_sha,check_binding
    from tools.train_sttrack_m42 import overlaps
    from lib.models.sttrack.lachtt_candidate_set import CandidateSetAssociation,select_candidate,supervised_loss
    check_binding(parent,spec)
    for name,source_digest in plan['source_sha256'].items():assert sha(repo/name)==source_digest,name
    assert sha(parent/'spec.json')==plan['source_spec_sha256']
    assert sha(early/'spec.json')==plan['early_spec_sha256']
    control_root=Path(plan['control_root']);assert sha(control_root/'geometry_result.json')==plan['control_training_result_sha256']
    assert sha(control_root/'recursive_result.json')==plan['control_recursive_result_sha256']
    control=json.loads((control_root/'geometry_result.json').read_text())
    assert not json.loads((control_root/'recursive_result.json').read_text())['primary_pass']
    assert (root/'training.exit').read_text().strip()=='1'
    assert "AttributeError: 'str' object has no attribute 'hexdigest'" in (root/'training.log').read_text()
    checkpoint=root/'geometry_final.pth';checkpoint_before=sha(checkpoint)
    collected={k:[] for k in ['current','previous','references','geometry','scores']};keys=[];folds=[];ious=[];previous_ious=[];early_keys=[]
    for source in [parent,early]:
        source_spec=json.loads((source/'spec.json').read_text())
        assert sha(source/'training_labels.json')==source_spec['labels_sha256']
        labels=json.loads((source/'training_labels.json').read_text());receipts=[];source_keys=[]
        for shard in [0,1]:
            receipt=json.loads((source/f'shard{shard}_receipt.json').read_text())
            assert receipt['status']=='complete' and receipt['spec_sha256']==sha(source/'spec.json') and not receipt['labels_opened']
            receipts.extend(receipt['sequences'])
        assert len(receipts)==len({r['sequence'] for r in receipts})==source_spec['sequences']
        for receipt in sorted(receipts,key=lambda x:x['sequence']):
            path=source/'features'/(receipt['sequence']+'.pt');assert sha(path)==receipt['feature_sha256'];data=torch.load(path,map_location='cpu')
            assert data['spec_sha256']==sha(source/'spec.json')
            for key in collected:collected[key].append(data[key])
            for i,row in enumerate(data['records']):
                label=labels[row['key']];assert label['fold']==data['fold'] and label['sequence']==data['sequence']
                assert row['previous_choice']==0 and row['previous_frame']==row['frame']-1
                if source==early:
                    assert data['fold'] in spec['fit_folds'] and 2<=row['frame']<=9
                    early_keys.append(row['key'])
                values=[]
                for side,boxkey,publickey in [('current','current_boxes','public_bbox'),('previous','previous_boxes','previous_public_bbox')]:
                    if label[side] is None:v=torch.zeros(10)
                    else:
                        gt=torch.tensor(label[side]);v=overlaps(data[boxkey][i],gt);v[0]=overlaps(data[publickey][i],gt)
                    values.append(v)
                keys.append(row['key']);source_keys.append(row['key']);folds.append(data['fold']);ious.append(values[0]);previous_ious.append(values[1])
        assert len(source_keys)==source_spec['events'] and set(source_keys)==set(labels)
    assert len(keys)==len(set(keys))==2605 and len(early_keys)==504
    tensors={k:torch.cat(v) for k,v in collected.items()};del collected,data
    ious=torch.stack(ious);previous_ious=torch.stack(previous_ious)
    current_target=ious.argmax(1);current_target[ious.max(1).values<.5]=10;current_target[ious[:,0]>=.5]=0
    previous_target=previous_ious.argmax(1);previous_target[previous_ious.max(1).values<.5]=10;previous_target[previous_ious[:,0]>=.5]=0
    fit=torch.tensor([i for i,f in enumerate(folds) if f in spec['fit_folds']]);dev=torch.tensor([i for i,f in enumerate(folds) if f==spec['development_fold']])
    assert len(fit)==2015 and len(dev)==590
    assert {keys[i].split('@')[0] for i in fit}.isdisjoint({keys[i].split('@')[0] for i in dev})
    opt=plan['optimization'];torch.set_num_threads(1);torch.manual_seed(opt['seed']);torch.cuda.manual_seed_all(opt['seed']);np.random.seed(opt['seed']);random.seed(opt['seed'])
    model=CandidateSetAssociation(True).cuda();initial={k:tensor_sha(v) for k,v in model.state_dict().items()}
    assert initial==control['initial_state_sha256']
    order=torch.Generator().manual_seed(opt['seed'])
    digest=hashlib.sha256();seen=set();steps=0;losses=[];started=time.time()
    def inputs(index):
        return [tensors[k][index].cuda().float() for k in ['current','previous','references','geometry','scores']]+[torch.zeros(len(index),dtype=torch.long,device='cuda')]
    logged=[json.loads(line) for line in (root/'training.log').read_text().splitlines() if line.startswith('{"elapsed"')]
    assert len(logged)==20 and [r['round'] for r in logged]==list(range(1,21))
    assert [r['optimizer_steps'] for r in logged]==list(range(48,961,48))
    for round_id in range(opt['rounds']):
        shuffled=fit[torch.randperm(len(fit),generator=order)[:opt['examples_per_round']]]
        digest.update(shuffled.numpy().tobytes());seen.update(shuffled.tolist())
        row=logged[round_id];losses.append({k:row[k] for k in ['round','loss','identity_loss','matching_loss']})
    steps=logged[-1]['optimizer_steps'];assert steps==control['optimizer_steps']==960
    saved=torch.load(checkpoint,map_location='cpu')
    assert saved['m46_spec_sha256']==sha(root/'spec.json') and saved['optimizer_steps']==960 and saved['optimization_rounds']==20
    assert saved['base_checkpoint_sha256']==spec['checkpoint_sha256'] and saved['spec_sha256']==sha(parent/'spec.json')
    assert saved['binding_sha256']==sha(parent/'training_binding.json') and saved['target_rule']=='default_if_iou_at_least_half'
    model.load_state_dict(saved['model'],strict=True)
    assert all(torch.isfinite(v).all() for v in saved['model'].values())
    def evaluate(net,index):
        net.eval();outputs=[]
        with torch.no_grad():
            for batch in index.split(opt['batch_size']):outputs.append(net(*inputs(batch))[0].cpu())
        logits=torch.cat(outputs);selected=select_candidate(logits);value=ious[index].gather(1,selected[:,None]).flatten();default=ious[index,0];sequence_stats={}
        for name in sorted({keys[i].split('@')[0] for i in index}):
            mask=torch.tensor([keys[i].split('@')[0]==name for i in index]);sequence_stats[name]=dict(events=int(mask.sum()),mean_iou=float(value[mask].mean()),default_mean_iou=float(default[mask].mean()),gain=float((value-default)[mask].mean()))
        return dict(events=len(index),mean_iou=float(value.mean()),default_mean_iou=float(default.mean()),correct=int((value>=.5).sum()),default_correct=int((default>=.5).sum()),
            changes=int((selected!=0).sum()),none=int((logits.argmax(1)==10).sum()),rescues=int(((default<=.1)&(value>=.5)).sum()),breaks=int(((default>=.5)&(value<=.1)).sum()),
            positive_sequences=sum(v['gain']>0 for v in sequence_stats.values()),sequence_stats=sequence_stats,
            rows=[dict(key=keys[i],chosen=int(k),action_none=bool(n==10),default_iou=float(d),selected_iou=float(v),oracle_iou=float(ious[i].max()),target=int(current_target[i]))
                  for i,k,n,d,v in zip(index,selected,logits.argmax(1),default,value)]),logits
    fitstats,_=evaluate(model,fit);devstats,logits=evaluate(model,dev)
    loaded=CandidateSetAssociation(True).cuda();loaded.load_state_dict(torch.load(checkpoint,map_location='cpu')['model'],strict=True)
    _,restored=evaluate(loaded,dev);assert torch.equal(restored,logits)
    check_binding(parent,spec)
    for name,source_digest in plan['source_sha256'].items():assert sha(repo/name)==source_digest,name
    result=dict(status='complete',variant='geometry',m46_spec_sha256=sha(root/'spec.json'),trainer_sha256=sha(repo/'tools/train_sttrack_m46.py'),completion_source_sha256=sha(__file__),metadata_recovered_after_training=True,optimization_steps_during_finalize=0,
        parameters=sum(v.numel() for v in model.parameters()),optimization_rounds=opt['rounds'],optimizer_steps=steps,losses=losses,
        original_fit_pairs=1511,early_fit_pairs=504,fit_sequences=63,examples_per_round=1511,distinct_fit_inputs_seen=len(seen),
        early_inputs_seen=sum(keys[i] in set(early_keys) for i in seen),target_rule='default_if_iou_at_least_half',
        matched_control_initialization_and_optimizer_budget=True,initial_state_sha256=initial,sample_order_sha256=digest.hexdigest(),
        changed_tensors=[k for k,v in model.state_dict().items() if tensor_sha(v)!=initial[k]],reload_logits_exact=True,
        checkpoint_sha256=sha(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,source_binding_sha256=sha(parent/'training_binding.json'),
        spec_sha256=sha(parent/'spec.json'),control_training_result_sha256=sha(control_root/'geometry_result.json'),
        fit=fitstats,development=devstats,current_target_counts_fit=dict(Counter(current_target[fit].tolist())),elapsed_seconds=logged[-1]['elapsed'],static_finalize_elapsed_seconds=time.time()-started,
        claim='Only the fitting-input pool adds frames2-9. Twenty sampled rounds of1511 examples retain960updates; these are not20complete epochs over2015inputs. Static Train diagnostics, no public result.')
    assert sha(checkpoint)==checkpoint_before
    (root/'geometry_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='complete',development={k:v for k,v in devstats.items() if k not in ['rows','sequence_stats']})),flush=True)


if __name__=='__main__':main()
