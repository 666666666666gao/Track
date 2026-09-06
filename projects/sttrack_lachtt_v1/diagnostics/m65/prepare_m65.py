"""Isolate a paired category-trained null-support experiment from sealed M58."""
import argparse
from datetime import datetime, timezone
import hashlib,json,shutil
from pathlib import Path

PARENT=Path('/root/autodl-tmp/sttrack_m58_semantic_spatial_v2_20260906')
ROOT=Path('/root/autodl-tmp/sttrack_m65_category_null_support_20260907')
BASE=Path('/root/autodl-tmp')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def change(s,a,b):
    assert s.count(a)==1,a
    return s.replace(a,b)

def prepare():
    import torch
    expected={'integration.json':'f6720047fdaec934efe9610d448fbdab8e2f6125318d0c917ee809d3e8458a96',
      'training_spec.json':'c592109d10579efac4dce5d3dd6f4881900c2c264acc3db3b63a6021855ea3b7',
      'train_causal.py':'acca756f0b7158c16a4d0ae68d141dcacd4955b72e6399f7c3744a9ad7c5767c',
      'causal_training.py':'672b7575437f9d5000377311b949bb46f7908f0dce2f7cfe5891390b8f1bfb95',
      'run_recursive.py':'63406ce954e62f03f170b09b9796484caa79abdf20880d8b9b97535a69aed254',
      'text_fit.pt':'d7ce0833b8e0e96c881a31ced85be43b6eeccb777adc577144fad0fb2420e3d0',
      'text_development.pt':'81879fa1ace5da373c77b898a76dd5fea23c5911c94184322fee4f2755d56e89',
      'native_parity/text_zero.pth':'ea90ec858551d5edbd1cd48f7c11408c367b57cf172be795b42e9fbd03e4e0d1'}
    for n,h in expected.items():assert sha(PARENT/n)==h,n
    assert not ROOT.exists();ROOT.mkdir()
    i=json.loads((PARENT/'integration.json').read_text());assert len(i['source_sha256'])==160
    for n,h in i['source_sha256'].items():
        assert sha(PARENT/'code'/n)==h
        target=ROOT/'code'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PARENT/'code'/n,target)
    adapter=ROOT/'code/lib/models/sttrack/semantic_spatial_adapter.py'
    s=adapter.read_text();s=change(s,'slots=5):','slots=5, null_support=False):')
    s=change(s,'        self.scale = 1.0 / math.sqrt(hidden)','        self.scale = 1.0 / math.sqrt(hidden)\n        assert isinstance(null_support, bool)\n        self.null_support = null_support')
    s=change(s,'        context = evidence.softmax(dim=-1) @ grounded',
      "        if self.null_support:\n            # Fixed zero-logit / zero-value alternative; no new learned parameter.\n            scores = torch.cat((evidence, torch.zeros_like(evidence[..., :1])), dim=-1)\n            context = scores.softmax(dim=-1)[..., :-1] @ grounded\n        else:\n            context = evidence.softmax(dim=-1) @ grounded")
    adapter.write_text(s)
    tracker=ROOT/'code/lib/test/tracker/sttrack_semantic.py'
    s=tracker.read_text();s=change(s,"checkpoint['architecture'] == 'semantic_spatial_v1'","checkpoint['architecture'] == 'semantic_spatial_support_v1'")
    s=change(s,'adapter = SemanticSpatialAdapter()',"adapter = SemanticSpatialAdapter(null_support=checkpoint['null_support'])")
    tracker.write_text(s)
    changed=[n for n,h in i['source_sha256'].items() if sha(ROOT/'code'/n)!=h]
    assert sorted(changed)==sorted(['lib/models/sttrack/semantic_spatial_adapter.py','lib/test/tracker/sttrack_semantic.py'])
    write(ROOT/'integration.json',dict(status='isolated_paired_category_null_support',
      parent_integration_sha256=sha(PARENT/'integration.json'),native_checkpoint_sha256=i['native_checkpoint_sha256'],
      changed_or_added_paths=changed,source_sha256={n:sha(ROOT/'code'/n) for n in i['source_sha256']}))
    for n in ['causal_training.py','recursive_metric.py','data_inventory.json']:shutil.copyfile(PARENT/n,ROOT/n)
    bank_receipts={}
    for split,count in [('fit',130),('development',22)]:
        p=PARENT/('text_'+split+'.pt');b=torch.load(p,map_location='cpu');assert len(b['sequences'])==count
        original=b['tokens'].clone();original_mask=b['mask'].clone();valid=b['mask'].clone();valid[:,0]=False
        b['tokens']=b['tokens'].clone();b['tokens'][valid]=b['empty'].to(b['tokens'].dtype)
        assert torch.equal(b['tokens'][:,0],original[:,0]) and torch.equal(b['mask'],original_mask)
        assert torch.equal(b['tokens'][~b['mask']],original[~b['mask']])
        assert torch.equal(b['tokens'][valid],b['empty'].to(b['tokens'].dtype).expand(int(valid.sum()),-1))
        b['lexical_policy']='Keep original category slot0; valid attrs1..4 become frozen CLIP empty; retain original 5-slot mask and padding. Both training and deployment.'
        b['original_initialization_phrases']=b.pop('initialization_phrases')
        b['source_bank_sha256']=sha(p)
        torch.save(b,ROOT/('text_'+split+'.pt'))
        bank_receipts[split]=dict(sequences=count,source_sha256=sha(p),output_sha256=sha(ROOT/('text_'+split+'.pt')),
          valid_attribute_vectors_replaced=int(valid.sum()),category_exact=True,mask_exact=True,padding_exact=True,
          regenerated_captions=0,regenerated_embeddings=0)
    (ROOT/'native_parity').mkdir()
    initial=torch.load(PARENT/'native_parity/text_zero.pth',map_location='cpu')
    assert initial['actual_dataset_optimizer_steps']==0 and initial['use_text']
    h=hashlib.sha256()
    for n,t in initial['model'].items():h.update(n.encode());h.update(t.numpy().tobytes())
    initial_paths={}
    for arm in ['control','null']:
        c=dict(initial,architecture='semantic_spatial_support_v1',null_support=arm=='null',arm=arm)
        path=ROOT/'native_parity'/(arm+'_zero.pth');torch.save(c,path);initial_paths[arm]=sha(path)
    s=(PARENT/'train_causal.py').read_text().replace(str(PARENT),str(ROOT))
    s=change(s,"choices=['text', 'visual']","choices=['control', 'null']")
    s=change(s,"sha(root / 'native_parity/text_zero.pth') == spec['initial_checkpoint_sha256']","sha(root / 'native_parity' / (args.arm + '_zero.pth')) == spec['initial_checkpoint_sha256'][args.arm]")
    s=change(s,"str(root / 'native_parity/text_zero.pth')","str(root / 'native_parity' / (args.arm + '_zero.pth'))")
    s=change(s,"tracker.use_text = args.arm == 'text'","assert tracker.use_text and tracker.network.semantic_adapter.null_support == (args.arm == 'null')")
    s=change(s,"return dict(architecture='semantic_spatial_v1', model=tracker.network.semantic_adapter.state_dict(),","return dict(architecture='semantic_spatial_support_v1', null_support=args.arm == 'null', arm=args.arm, model=tracker.network.semantic_adapter.state_dict(),")
    s=change(s,"status='one_full_causal_fit_pass_complete', arm=args.arm, sequences=len(receipts),","status='one_full_causal_fit_pass_complete', arm=args.arm, null_support=args.arm == 'null', sequences=len(receipts),")
    s=change(s,'One frozen-budget full-sequence training pass per text/visual arm.','Matched category-only full-sequence training; control versus fixed null-support normalization.')
    (ROOT/'train_causal.py').write_text(s)
    s=(PARENT/'run_recursive.py').read_text().replace(str(PARENT),str(ROOT))
    s=s.replace("'text'","'null'").replace("'visual'","'control'")
    s=change(s,"assert saved['use_text'] == (arm == 'null')","assert saved['use_text'] and saved['null_support'] == (arm == 'null')")
    start=s.index('    gates=dict(mean_vs_native=');end=s.index('\n    result=dict(',start)
    s=s[:start]+'''    control_broken=[n for n in per['control'] if per['control'][n]['failure_episodes']==0 and per['null'][n]['failure_episodes']>0]
    gates=dict(mean_vs_native=primary['mean_iou']>=baseline['mean_iou']+rule['null_pooled_mean_vs_native_minimum'],
        mean_vs_control=primary['mean_iou']>=control['mean_iou']+rule['null_pooled_mean_vs_control_minimum'],
        macro_vs_native=primary['macro_sequence_mean_iou']>=baseline['macro_sequence_mean_iou'],
        macro_vs_control=primary['macro_sequence_mean_iou']>=control['macro_sequence_mean_iou'],
        low_frames_vs_native=primary['low_iou_frames']<=baseline['low_iou_frames'],
        low_frames_vs_control=primary['low_iou_frames']<=control['low_iou_frames'],
        H10_vs_native=primary['failure_episodes']<=baseline['failure_episodes'],
        H10_vs_control=primary['failure_episodes']<=control['failure_episodes'],
        native_success_protection=not broken, control_success_protection=not control_broken)'''+s[end:]
    s=change(s,'new_failure_sequences=broken, receipts=receipts,','new_failure_sequences=broken, broken_control_success_sequences=control_broken, receipts=receipts,')
    s=change(s,"next='Fixed-weight content counterfactuals before low22' if all(gates.values()) else 'Stop this frozen revision; diagnose complete trajectories'", "next='Fixed-weight category versus empty and swapped category controls before low22' if all(gates.values()) else 'Stop this frozen revision; diagnose complete trajectories'")
    (ROOT/'run_recursive.py').write_text(s)
    for name in ['check_m65.py','freeze_m65.py','run_m65.sh']:
        shutil.copyfile(BASE/name,ROOT/name)
    write(ROOT/'preparation.json',dict(status='ready_for_real_fit_smoke',observed_utc=datetime.now(timezone.utc).isoformat(),
      prepare_script_sha256=sha(__file__),parent_inputs=expected,source_files=160,changed_source_paths=changed,
      bank_receipts=bank_receipts,initial_checkpoint_sha256=initial_paths,initial_parameter_tensor_sha256=h.hexdigest(),
      learned_parameters_each=289154,formal_training_started=False,new_caption_calls=0,
      architecture='semantic_spatial_support_v1',primary='null',control='control'))
    for n in i['source_sha256']:assert sha(PARENT/'code'/n)==i['source_sha256'][n]
    print(json.dumps(dict(root=str(ROOT),status='ready_for_real_fit_smoke',preparation_sha256=sha(ROOT/'preparation.json'),bank_receipts=bank_receipts),indent=2))

if __name__=='__main__':prepare()
