"""Predeclared same-final-head category controls, conditional on M67's frozen gates."""
import argparse
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import sys,time
from types import SimpleNamespace

BASE=Path('/root/autodl-tmp');ROOT=BASE/'sttrack_m67_supervised_semantic_support_20260907'
OUT=ROOT/'content_counterfactuals';M60=BASE/'sttrack_m60_category_isolation_20260906'
TRAIN_SHA='2027b1893343e5e1520c447bd82a5c45468542cecaab0f59b3ec72a345e48b2e'
RECURSIVE_SHA='d4dd7ca74b8f4b318f8e32f961bc21a10b95aa3802137dd293b49948d83411c5'
AUDITOR=BASE/'audit_m67_completed_20260907.py'
AUDITOR_SHA='1773887deb3be27aa11a4409aa77bd7581f32b8d891ecab2a03454d277e769bf'
M60_SPEC_SHA='567944c9dbe2e79340bfabc737a0975563fa7e5be17d16a8b2fc3d75fd1990c5'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def parents():
    assert sha(ROOT/'training_spec.json')==TRAIN_SHA and sha(ROOT/'recursive_spec.json')==RECURSIVE_SHA
    training=read(ROOT/'training_spec.json');recursive=read(ROOT/'recursive_spec.json')
    assert sha(ROOT/'run_recursive.py')==recursive['runner_sha256']
    assert sha(ROOT/'integration.json')==training['integration_sha256']
    for p,h in read(ROOT/'integration.json')['source_sha256'].items():assert sha(ROOT/'code'/p)==h
    assert sha(training['native_checkpoint'])==training['native_checkpoint_sha256']
    assert sha(ROOT/'text_development.pt')==training['text_development_sha256']
    return training,recursive

def prepare():
    import torch
    training,recursive=parents()
    assert not (ROOT/'support_recursive_receipt.json').exists() and not (ROOT/'recursive_result.json').exists()
    assert not OUT.exists();OUT.mkdir()
    assert sha(M60/'spec.json')==M60_SPEC_SHA
    previous=read(M60/'spec.json');assert sha(M60/'swapped_category.pt')==previous['bank_sha256']
    original=torch.load(ROOT/'text_development.pt',map_location='cpu')
    donor=torch.load(M60/'swapped_category.pt',map_location='cpu')
    assert set(original['sequences'])==set(donor['sequences'])=={c['sequence'] for c in recursive['cases']}
    order=[donor['sequences'].index(n) for n in original['sequences']]
    donor_tokens=donor['tokens'][order];donor_mask=donor['mask'][order]
    assert original['tokens'].shape==(22,5,768) and torch.equal(original['mask'],donor_mask)
    assert torch.equal(original['empty'],donor['empty'])
    assert torch.equal(original['tokens'][:,1:],donor_tokens[:,1:])
    assert torch.equal(original['tokens'][~original['mask']],donor_tokens[~donor_mask])
    changed=(original['tokens'][:,0]!=donor_tokens[:,0]).any(1);assert bool(changed.all())
    banks={'category':dict(path=str(ROOT/'text_development.pt'),sha256=sha(ROOT/'text_development.pt'))}
    for name in ['empty','swapped']:
        b=dict(original);b['tokens']=original['tokens'].clone()
        if name=='empty':b['tokens'][b['mask']]=b['empty']
        else:b['tokens'][:,0]=donor_tokens[:,0]
        b['input_variant']=name
        b['lexical_policy']='All valid slots contain the frozen CLIP empty vector.' if name=='empty' else 'Only category slot0 replaced by the frozen M60 donor category; attrs remain empty.'
        assert torch.equal(b['mask'],original['mask']) and torch.equal(b['tokens'][~b['mask']],original['tokens'][~b['mask']])
        path=OUT/(name+'.pt');torch.save(b,path);banks[name]=dict(path=str(path),sha256=sha(path))
    queue=OUT/'run_controls.sh'
    queue.write_text('''#!/bin/bash
set -u
root=/root/autodl-tmp/sttrack_m67_supervised_semantic_support_20260907/content_counterfactuals
python=/root/autodl-tmp/envs/sttrack/bin/python
script=/root/autodl-tmp/m67_content_counterfactuals_20260907.py
cd "$root" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0 "$python" -u "$script" prefix > prefix.log 2>&1
status=$?
printf '%s\\n' "$status" > prefix.exit
if [ "$status" -ne 0 ]; then printf '1\\n' > controller.exit; exit 1; fi
run_arm() {
    CUDA_VISIBLE_DEVICES="$2" "$python" -u "$script" "$1" > "$1.log" 2>&1
    status=$?
    printf '%s\\n' "$status" > "$1.exit"
    return "$status"
}
run_arm empty 0 & first=$!
run_arm swapped 1 & second=$!
wait "$first"; first_status=$?
wait "$second"; second_status=$?
if [ "$first_status" -ne 0 ] || [ "$second_status" -ne 0 ]; then printf '1\\n' > controller.exit; exit 1; fi
CUDA_VISIBLE_DEVICES='' "$python" -u "$script" analyze > analysis.log 2>&1
status=$?
printf '%s\\n' "$status" > analysis.exit
printf '%s\\n' "$status" > controller.exit
exit "$status"
''')
    spec=dict(status='frozen_before_M67_development_outputs',observed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__),queue_sha256=sha(queue),training_spec_sha256=TRAIN_SHA,recursive_spec_sha256=RECURSIVE_SHA,
        completion_auditor_sha256=AUDITOR_SHA,banks=banks,cases=recursive['cases'],prefix_sequences=previous['prefix_sequences'],
        prefix_frames=previous['prefix_frames'],M60_spec_sha256=M60_SPEC_SHA,donor_bank_sha256=previous['bank_sha256'],
        donor_mapping_sha256=previous['donor_mapping_sha256'],category_vectors_changed=22,all_masks_padding_and_noncategory_slots_exact=True,
        head_selection='Only M67 support final after completed evidence audit and every original development gate passes.',
        inference='Same final head, base, 5 slots, mask, initialization, crop, query and native template code. No online text or optimizer.',
        primary='category',controls=['empty','swapped'],reuse_complete_category_predictions=True,new_full_track_calls=66216,
        metric='DepthTrack Train reused development22; continuous IoU with initialization/invalid GT excluded; H10 same convention as M67.',
        gate=dict(category_pooled_margin_vs_each_control=.001,category_macro_no_less_than_each_control=True,
            category_low_frames_no_more_than_each_control=True,category_H10_no_more_than_each_control=True),
        gate_scope='Necessary lexical-content evidence, not proof of correct captions or fine-grained same-class identity. Requires all original M67 gates too.',
        fresh_captions=0,new_embeddings=0,new_learned_parameters=0,new_optimizer_steps=0,
        public_full_evaluation_allowed=False,independent_model_review_pass=False,
        limitations=['Original categories are noisy automatic descriptions, not semantic ground truth.',
            'Frozen replacement category need not be a verified contradiction to the target.',
            'One seed and repeatedly used development sequences; lexical sensitivity alone is insufficient for a language-gain claim.'])
    write(OUT/'spec.json',spec)
    report=dict(status='conditional_content_inputs_and_source_prepared',source_sha256=sha(__file__),spec_sha256=sha(OUT/'spec.json'),
        banks=banks,category_vectors_changed=22,masks_padding_noncategory_exact=True,learned_head_loaded=False,
        new_tracking_calls=0,new_GT_files_opened=False,new_captions=0,new_embeddings=0,
        M67_training_or_evaluation_modified=False,conditional_execution_started=False)
    write(OUT/'preparation_result.json',report);print(json.dumps(report,indent=2))

def checked():
    training,recursive=parents();s=read(OUT/'spec.json')
    assert sha(__file__)==s['source_sha256'] and sha(OUT/'run_controls.sh')==s['queue_sha256']
    assert s['training_spec_sha256']==TRAIN_SHA and s['recursive_spec_sha256']==RECURSIVE_SHA
    for b in s['banks'].values():assert sha(b['path'])==b['sha256']
    return s,training,recursive

def eligible():
    s,training,recursive=checked()
    assert sha(AUDITOR)==AUDITOR_SHA==s['completion_auditor_sha256']
    audit=read(ROOT/'completed_evidence_audit.json')
    assert audit['status']=='completed_M67_artifacts_and_development_audited' and audit['auditor_sha256']==AUDITOR_SHA
    assert audit['training_spec_sha256']==TRAIN_SHA and audit['recursive_spec_sha256']==RECURSIVE_SHA
    assert audit['paired_development_gate_pass'] and audit['content_counterfactuals_allowed']
    assert all(audit['recomputed_frozen_gates'].values())
    assert sha(ROOT/'recursive_result.json')==audit['result_sha256']
    assert sha(ROOT/'training/support/final.pth')==audit['training']['support']['final_checkpoint_sha256']
    assert sha(ROOT/'support_recursive_receipt.json')==audit['families']['support']['receipt_sha256']
    return s,training,audit

def track(name,prefix=False):
    s,training,audit=eligible()
    if not prefix:
        assert (OUT/'prefix.exit').read_text().strip()=='0'
        receipt=read(OUT/'prefix/receipt.json');assert receipt['exact_original_prefix_parity']
        assert receipt['head_sha256']==sha(ROOT/'training/support/final.pth') and receipt['spec_sha256']==sha(OUT/'spec.json')
    import torch
    sys.path.insert(0,str(ROOT/'code'))
    from lib.config.sttrack.config import cfg,update_config_from_file
    from lib.test.tracker.sttrack_semantic import STTrackSemantic
    from lib.train.dataset.depth_utils import get_rgbd_frame
    torch.set_num_threads(1);torch.manual_seed(training['seed']);torch.cuda.manual_seed_all(training['seed'])
    update_config_from_file(str(ROOT/'code/experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    params=SimpleNamespace(cfg=cfg,checkpoint=training['native_checkpoint'],base_checkpoint_sha256=training['native_checkpoint_sha256'],
        template_factor=2.,template_size=128,search_factor=4.,search_size=256,save_all_boxes=False,debug=0)
    head=ROOT/'training/support/final.pth';tracker=STTrackSemantic(params,str(head))
    assert tracker.use_text and tracker.network.semantic_adapter.null_support
    bank=torch.load(s['banks'][name]['path'],map_location='cpu');folder=OUT/('prefix' if prefix else name);folder.mkdir()
    refs={r['sequence']:r for r in read(ROOT/'support_recursive_receipt.json')['sequences']}
    started=time.time();records=[]
    for case in s['cases']:
        seq=case['sequence']
        if prefix and seq not in s['prefix_sequences']:continue
        base=Path(training['dataset_root'])/seq
        def frame(n):return get_rgbd_frame(str(base/'color'/('%08d.jpg'%(n+1))),str(base/'depth'/('%08d.png'%(n+1))),dtype='rgbcolormap',depth_clip=True)
        index=bank['sequences'].index(seq)
        tracker.initialize(frame(0),dict(init_bbox=case['init_bbox'],text_tokens=bank['tokens'][index],text_mask=bank['mask'][index],empty_text=bank['empty']))
        rows=[dict(frame=0,bbox=list(tracker.state),score=None)]
        count=s['prefix_frames'] if prefix else case['frames']
        for n in range(1,count):
            prediction=tracker.track(frame(n));rows.append(dict(frame=n,bbox=list(prediction['target_bbox']),score=float(prediction['best_score'])))
        if prefix:
            reference=ROOT/'recursive/support'/(seq+'.json');assert sha(reference)==refs[seq]['sha256']
            assert rows==read(reference)['rows'][:count],seq
        path=folder/(seq+'.json');write(path,dict(sequence=seq,arm=name,rows=rows))
        item=dict(sequence=seq,frames=len(rows),sha256=sha(path),elapsed_seconds=time.time()-started);records.append(item);print(json.dumps(item),flush=True)
    checked()
    receipt=dict(status='complete',variant=name,prefix_only=prefix,source_sha256=sha(__file__),spec_sha256=sha(OUT/'spec.json'),
        head_sha256=sha(head),bank_sha256=s['banks'][name]['sha256'],completed_audit_sha256=sha(ROOT/'completed_evidence_audit.json'),
        sequences=records,total_frames=sum(x['frames'] for x in records),subsequent_GT_opened=False,optimizer_steps=0,
        exact_original_prefix_parity=prefix,elapsed_seconds=time.time()-started)
    write(folder/'receipt.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='sequences'},indent=2),flush=True)

def analyze():
    import numpy as np
    s,training,audit=eligible();data={};receipts={}
    original=read(ROOT/'support_recursive_receipt.json')
    for name in ['category','empty','swapped']:
        if name=='category':receipt=original;directory=ROOT/'recursive/support';expected_arm='support'
        else:
            assert (OUT/(name+'.exit')).read_text().strip()=='0'
            directory=OUT/name;receipt=read(directory/'receipt.json');expected_arm=name
            assert receipt['spec_sha256']==sha(OUT/'spec.json') and receipt['variant']==name and not receipt['prefix_only']
            assert receipt['bank_sha256']==s['banks'][name]['sha256']
            assert receipt['completed_audit_sha256']==sha(ROOT/'completed_evidence_audit.json')
        assert receipt['status']=='complete' and receipt['head_sha256']==sha(ROOT/'training/support/final.pth')
        assert receipt['total_frames']==33130 and [x['sequence'] for x in receipt['sequences']]==[x['sequence'] for x in s['cases']]
        data[name]={}
        for item,case in zip(receipt['sequences'],s['cases']):
            p=directory/(case['sequence']+'.json');assert sha(p)==item['sha256'];x=read(p);rows=x['rows']
            assert x['sequence']==case['sequence'] and x['arm']==expected_arm
            assert len(rows)==case['frames'] and [r['frame'] for r in rows]==list(range(case['frames']))
            assert rows[0]==dict(frame=0,bbox=case['init_bbox'],score=None)
            boxes=np.asarray([r['bbox'] for r in rows]);assert np.isfinite(boxes).all() and (boxes[:,2:]>0).all()
            data[name][case['sequence']]=rows
        receipts[name]=sha(ROOT/'support_recursive_receipt.json') if name=='category' else sha(directory/'receipt.json')
    assert sha(ROOT/'recursive_metric.py')==read(ROOT/'recursive_spec.json')['metric_sha256']
    sys.path.insert(0,str(ROOT));from recursive_metric import statistics
    per={name:{} for name in data};writes={name:0 for name in data}
    for case in s['cases']:
        seq=case['sequence'];p=Path(training['dataset_root'])/seq/'groundtruth.txt';assert sha(p)==case['gt_sha256']
        gt=np.loadtxt(p,delimiter=',').reshape(-1,4);assert len(gt)==case['frames']
        for name in data:
            rows=data[name][seq];per[name][seq]=statistics([x['bbox'] for x in rows],gt)
            writes[name]+=sum(x['frame']%50==0 and x['score']>.75 for x in rows[1:])
    aggregates={}
    for name,values in per.items():
        x={k:sum(r[k] for r in values.values()) for k in ['valid_frames','iou_sum','low_iou_frames','failure_episodes']}
        x.update(mean_iou=x['iou_sum']/x['valid_frames'],macro_sequence_mean_iou=float(np.mean([r['mean_iou'] for r in values.values()])))
        assert x['valid_frames']==28897;aggregates[name]=x
    for k,v in aggregates['category'].items():assert abs(v-audit['recomputed_aggregates']['support'][k])<1e-8
    primary=aggregates['category'];gates={}
    for name in ['empty','swapped']:
        other=aggregates[name]
        gates[name]=dict(pooled_margin=primary['mean_iou']>=other['mean_iou']+s['gate']['category_pooled_margin_vs_each_control'],
            macro=primary['macro_sequence_mean_iou']>=other['macro_sequence_mean_iou'],low_frames=primary['low_iou_frames']<=other['low_iou_frames'],
            H10=primary['failure_episodes']<=other['failure_episodes'])
    passed=all(value for conditions in gates.values() for value in conditions.values())
    r=dict(status='completed_fixed_head_category_content_controls',observed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__),spec_sha256=sha(OUT/'spec.json'),head_sha256=sha(ROOT/'training/support/final.pth'),
        M67_completed_audit_sha256=sha(ROOT/'completed_evidence_audit.json'),receipts=receipts,aggregates=aggregates,per_sequence=per,
        reconstructed_template_writes=writes,gates=gates,content_gate_pass=passed,new_full_track_calls=66216,
        scope='Reused Train development22. Original automatic categories are not ground-truth semantic labels.',
        low22_candidate_preparation_allowed=passed,full_three_dataset_evaluation_allowed=False,independent_model_review_pass=False,
        next='Freeze a new exact-bundle low22 candidate after result audit' if passed else 'Stop this fixed candidate; retain negative content evidence without threshold changes')
    write(OUT/'result.json',r);print(json.dumps({k:v for k,v in r.items() if k!='per_sequence'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','check','prefix','empty','swapped','analyze']);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='check':checked();print('CONTENT_SOURCE_AND_INPUTS_VERIFIED_NO_EXECUTION')
    elif a.action=='prefix':track('category',prefix=True)
    elif a.action in ['empty','swapped']:track(a.action)
    else:analyze()
