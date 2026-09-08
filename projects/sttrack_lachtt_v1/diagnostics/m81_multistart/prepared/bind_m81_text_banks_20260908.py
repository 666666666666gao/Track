"""Bind sealed Train multistart observations to exact paired category/empty banks."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import torch
B=Path('/root/autodl-tmp');R=B/'sttrack_m81_multistart_20260908';P=B/'sttrack_m78_raw_competition_20260908'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
torch.set_num_threads(1)
s=read(R/'inventory_spec.json');prep=read(R/'caption_preparation.json');plan=read(R/'captions/plan.json')
assert not (R/'banks').exists()
assert sha(R/'inventory_spec.json')==prep['inventory_spec_sha256']
assert sha(R/'captions/plan.json')==prep['caption_plan_sha256']
for n in ['caption_generate.exit','caption_encode.exit','caption_controller.exit']:assert (R/n).read_text().strip()=='0'
g=read(R/'captions/generation_result.json');e=read(R/'captions/encoding_result.json')
assert g['status']=='all_initialization_captions_generated' and g['cases']==348 and g['unique_observations']==348
assert g['plan_sha256']==e['plan_sha256']==sha(R/'captions/plan.json')
assert e['status']=='initialization_text_bank_complete' and e['encoding_device']=='cpu'
assert e['generation_result_sha256']==sha(R/'captions/generation_result.json')
assert g['records_sha256']==sha(R/'captions/records.jsonl')
assert e['bank_sha256']==sha(R/'captions/text_bank.pt')
raw=torch.load(R/'captions/text_bank.pt',map_location='cpu')
records=[json.loads(x) for x in (R/'captions/records.jsonl').read_text().splitlines()]
assert raw['keys']==[x['key'] for x in records]==[x['key'] for x in plan['rows']]
for x,y in zip(plan['rows'],records):
    assert x['image_sha256']==y['image_sha256']==sha(x['image'])
    assert all(k in y for k in ['raw','category','attributes'])
case_to_key={x['id']:x['key'] for x in plan['cases']};raw_index={k:i for i,k in enumerate(raw['keys'])}
assert len(case_to_key)==348
t=read(P/'training_spec.json');assert sha(P/'training_spec.json')==s['parent_training_spec_sha256']
out=R/'banks';out.mkdir();results={};binding=[]
for split in ['fit','development']:
    mpath=R/(split+'_initializations.json');assert sha(mpath)==s[split+'_manifest_sha256'];manifest=read(mpath)
    oldspec=t['banks'][split]['category'];empty_spec=t['banks'][split]['empty']
    assert sha(oldspec['path'])==oldspec['sha256'] and sha(empty_spec['path'])==empty_spec['sha256']
    old=torch.load(oldspec['path'],map_location='cpu');oldempty=torch.load(empty_spec['path'],map_location='cpu')
    assert torch.equal(old['empty'],oldempty['empty'])
    tokens=[];masks=[];ids=[];sequences=[];rows=[]
    for ep in manifest['episodes']:
        assert sha(ep['image'])==ep['image_sha256'] and sha(ep['depth'])==ep['depth_sha256']
        seq=ep['sequence'];index=old['sequences'].index(seq)
        if ep['start_frame']==0:
            vector=old['tokens'][index].clone();mask=old['mask'][index].clone();source='sealed_t0_bank'
        else:
            key=case_to_key[ep['id']];j=raw_index[key];observation=plan['rows'][j]
            assert observation['init_bbox']==ep['init_bbox'] and observation['image_sha256']==ep['image_sha256']
            vector=raw['tokens'][j].clone();mask=raw['mask'][j].clone();source=key
            assert mask[0]
            vector[1:][mask[1:]]=old['empty']
        assert vector.shape==(5,768) and mask.shape==(5,) and torch.isfinite(vector).all()
        assert torch.equal(vector[1:][mask[1:]],old['empty'].expand_as(vector[1:][mask[1:]]))
        tokens.append(vector);masks.append(mask);ids.append(ep['id']);sequences.append(seq)
        rows.append(dict(id=ep['id'],source=source,image_sha256=ep['image_sha256'],init_bbox=ep['init_bbox']))
    category=dict(format='multistart_observation_v1',ids=ids,sequences=sequences,tokens=torch.stack(tokens),mask=torch.stack(masks),empty=old['empty'].clone(),manifest_sha256=sha(mpath),caption_plan_sha256=sha(R/'captions/plan.json'),protocol_sha256=raw['protocol_sha256'])
    blank={**category,'tokens':category['tokens'].clone()};blank['tokens'][blank['mask']]=blank['empty']
    assert torch.equal(blank['mask'],category['mask'])
    assert torch.equal(blank['tokens'][~blank['mask']],category['tokens'][~category['mask']])
    assert torch.equal(blank['tokens'][:,1:],category['tokens'][:,1:])
    for ep in manifest['episodes']:
        if ep['start_frame']!=0:continue
        i=ids.index(ep['id']);j=old['sequences'].index(ep['sequence']);k=oldempty['sequences'].index(ep['sequence'])
        assert torch.equal(category['tokens'][i],old['tokens'][j]) and torch.equal(category['mask'][i],old['mask'][j])
        assert torch.equal(blank['tokens'][i],oldempty['tokens'][k]) and torch.equal(blank['mask'][i],oldempty['mask'][k])
    for condition,bank in [('category',category),('empty',blank)]:
        path=out/(split+'_'+condition+'.pt');torch.save(bank,path)
        check=torch.load(path,map_location='cpu');assert check['ids']==ids and torch.equal(check['tokens'],bank['tokens'])
        results[split+'_'+condition]=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,episodes=len(ids))
    binding.extend(dict(split=split,**row) for row in rows)
write(out/'bindings.json',binding)
report=dict(status='complete_paired_multistart_text_binding',observed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),inventory_sha256=sha(R/'inventory_spec.json'),generation_result_sha256=sha(R/'captions/generation_result.json'),encoding_result_sha256=sha(R/'captions/encoding_result.json'),banks=results,total_episode_bindings=len(binding),new_initialization_captions=348,t0_bindings_preserved_exactly=152,fit_development_disjoint=True,padding_and_masks_paired_exactly=True,attributes_blank=True,canonical_empty='Exact existing per-split M78 CLIP empty vector used for every blanked valid slot.',new_vs_existing_empty_max_abs=float((raw['empty']-old['empty']).abs().max()),bindings_sha256=sha(out/'bindings.json'),tracking_training_started=False,semantic_correctness_verified=False)
write(out/'binding_result.json',report);print(json.dumps(report,indent=2))
