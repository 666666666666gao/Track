"""Isolate automatic category content at a fixed M58 weight, with all attribute inputs held fixed."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path('/root/autodl-tmp/sttrack_m60_category_isolation_20260906')
M59=Path('/root/autodl-tmp/sttrack_m59_fixed_head_content_diagnostic_20260906')
PARENT=Path('/root/autodl-tmp/sttrack_m58_semantic_spatial_v2_20260906')
INTERFACE=Path('/root/autodl-tmp/sttrack_m58_evaluation_preparation_20260906')
M59_RESULT_SHA='6e9373d1d9252eae104421d664703490a13541353988ac8851fe2bc5c4d0756e'
sys.path.insert(0,str(M59))
import run_controls as previous
sys.path.insert(0,str(INTERFACE))
from semantic_runtime import checked_plan, text_bank


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def completed_parent():
    for name in ['worker0.exit','worker1.exit','analysis.exit','finalization.exit']:
        assert (M59/name).read_text().strip()=='0'
    assert sha(M59/'result.json')==M59_RESULT_SHA
    spec,training,_,_=previous.parent_ready()
    return spec,training,json.loads((M59/'result.json').read_text())


def prepare():
    old,training,result=completed_parent()
    ROOT.mkdir()
    category=torch.load(M59/'category.pt',map_location='cpu')
    mismatch=torch.load(M59/'mismatch.pt',map_location='cpu')
    assert category['keys']==mismatch['keys'] and category['sequences']==mismatch['sequences']
    assert torch.equal(category['mask'],mismatch['mask'])
    assert torch.equal(category['empty'],mismatch['empty'])
    assert category['mask'][:,0].all()
    active_attributes=category['tokens'][:,1:][category['mask'][:,1:]]
    assert torch.equal(active_attributes,category['empty'].expand_as(active_attributes))
    bank=dict(category)
    bank['tokens']=category['tokens'].clone()
    bank['tokens'][:,0]=mismatch['tokens'][:,0]
    bank['control']='swapped_category_attributes_empty'
    assert torch.equal(bank['tokens'][:,1:],category['tokens'][:,1:])
    changed=(bank['tokens'][:,0]!=category['tokens'][:,0]).any(1)
    assert int(changed.sum())==22
    torch.save(bank,ROOT/'swapped_category.pt')
    old_plan=json.loads((M59/'category_plan.json').read_text())
    plan=dict(old_plan,text_bank_path=str(ROOT/'swapped_category.pt'),text_bank_sha256=sha(ROOT/'swapped_category.pt'))
    write(ROOT/'plan.json',plan)
    checked,bundle=checked_plan(ROOT/'plan.json')
    loaded=text_bank(checked,bundle)
    sys.path.insert(0,str(PARENT/'code'))
    for i,case in enumerate(old['cases']):
        rgb,_=previous.load_frame(Path(training['dataset_root'])/case['sequence'],0)
        info=loaded.info(rgb,case['init_bbox'])
        assert torch.equal(info['text_tokens'],bank['tokens'][i])
        assert torch.equal(info['text_mask'],category['mask'][i])
        assert info['init_bbox']==case['init_bbox']
    spec=dict(status='frozen_before_category_isolation_gpu_inference',observed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__),m59_source_sha256=sha(M59/'run_controls.py'),
        m59_spec_sha256=sha(M59/'spec.json'),m59_result_sha256=M59_RESULT_SHA,
        bundle_sha256=sha(M59/'bundle.json'),head_sha256=result['head_sha256'],
        reference_bank_sha256=sha(M59/'category.pt'),donor_bank_sha256=sha(M59/'mismatch.pt'),
        bank_sha256=sha(ROOT/'swapped_category.pt'),plan_sha256=sha(ROOT/'plan.json'),
        attribute_tokens_and_all_masks_exactly_unchanged=True,category_slots_changed=22,
        cases=old['cases'],new_image_frames=33130,new_track_calls=33108,
        reference='M59 category with all valid attribute slots replaced by CLIP empty; original masks retained.',
        control='Only slot0 replaced using the previously frozen cross-auto-category donors; all remaining tokens/masks unchanged.',
        donor_mapping_sha256=old['mappings_sha256'],
        hypothesis='The original automatic category word improves full recursion relative to a different automatic category under exactly the same empty attribute inputs.',
        primary_descriptive_margin=.001,secondary_reference='M59 fixed-head empty; not the separately trained visual head.',
        selected_after_m59_results=True,development_set_reused=True,
        new_optimizer_steps=0,new_network_parameters=0,new_captions=0,
        prefix_sequences=old['prefix_parity_sequences'],prefix_frames=102,
        public_evaluation_allowed=False,independent_review_pass=False,
        limitations=['Original automatic categories include documented errors; do not call them semantic ground truth.',
            'The masks preserve original caption slot counts, so this is not a one-token interface.',
            'The category-retention protocol was selected after M59 development results; no unseen-set claim.',
            'Do not promote M58 full-attribute input, or infer language utility merely from sensitivity to a replacement.'],
        after_result='Compare original-category versus swapped-category and fixed-head empty, then freeze a subsequent training or evaluation plan based on all evidence.')
    write(ROOT/'spec.json',spec)
    write(ROOT/'preparation_result.json',dict(status='single_category_slot_intervention_sealed',
        spec_sha256=sha(ROOT/'spec.json'),source_sha256=sha(__file__),bank_sha256=spec['bank_sha256'],
        changed_category_slots=22,unchanged_attribute_tokens_and_mask=True,actual_initialization_routes_checked=22,
        new_weight_files=0,new_model_calls=0,new_optimizer_steps=0,public_evaluation_allowed=False))
    print(json.dumps(json.loads((ROOT/'preparation_result.json').read_text()),indent=2))


def plans():
    old,training,result=completed_parent()
    spec=json.loads((ROOT/'spec.json').read_text())
    assert sha(__file__)==spec['source_sha256']
    assert sha(M59/'run_controls.py')==spec['m59_source_sha256']
    assert sha(M59/'spec.json')==spec['m59_spec_sha256']
    assert sha(ROOT/'plan.json')==spec['plan_sha256']
    assert sha(ROOT/'swapped_category.pt')==spec['bank_sha256']
    assert sha(M59/'category.pt')==spec['reference_bank_sha256']
    assert sha(M59/'mismatch.pt')==spec['donor_bank_sha256']
    return spec,training,result


def parity():
    spec,training,_=plans()
    _,_,tracker,bank=previous.runtime('category')
    receipt=json.loads((M59/'category_receipt.json').read_text())
    sealed={r['sequence']:r for r in receipt['sequences']}
    reports=[];started=time.time()
    for name in spec['prefix_sequences']:
        case=next(c for c in spec['cases'] if c['sequence']==name)
        path=M59/'predictions/category'/(name+'.json')
        assert sha(path)==sealed[name]['sha256']
        reference=json.loads(path.read_text())['rows']
        folder=Path(training['dataset_root'])/name
        rgb,frame=previous.load_frame(folder,0)
        tracker.initialize(frame,bank.info(rgb,case['init_bbox']))
        box_error=score_error=0.;updates=0
        for i in range(1,spec['prefix_frames']):
            old_template=tracker.z_dict[1]
            _,frame=previous.load_frame(folder,i)
            output=tracker.track(frame)
            box_error=max(box_error,float(np.abs(np.asarray(output['target_bbox'])-reference[i]['bbox']).max()))
            score_error=max(score_error,abs(float(output['best_score'])-reference[i]['score']))
            updates+=tracker.z_dict[1] is not old_template
        assert box_error<=1e-4 and score_error<=1e-6
        reports.append(dict(sequence=name,frames=spec['prefix_frames'],max_bbox_error=box_error,max_score_error=score_error,template_writes=updates))
    write(ROOT/'parity_result.json',dict(status='reference_category_prefix_reproduced',spec_sha256=sha(ROOT/'spec.json'),
        rows=reports,elapsed_seconds=time.time()-started,subsequent_gt_opened=False,
        entire_sequence_and_all_internal_states_verified=False,trax_verified=False))


def run():
    spec,training,_=plans()
    assert (ROOT/'parity.exit').read_text().strip()=='0'
    parity_result=json.loads((ROOT/'parity_result.json').read_text())
    assert parity_result['spec_sha256']==sha(ROOT/'spec.json')
    _,_,tracker,_=previous.runtime('category')
    plan,bundle=checked_plan(ROOT/'plan.json')
    bank=text_bank(plan,bundle)
    output=ROOT/'predictions';output.mkdir()
    receipts=[];started=time.time()
    for case in spec['cases']:
        folder=Path(training['dataset_root'])/case['sequence']
        rgb,frame=previous.load_frame(folder,0)
        tracker.initialize(frame,bank.info(rgb,case['init_bbox']))
        rows=[dict(frame=0,bbox=list(tracker.state),score=None)]
        for i in range(1,case['frames']):
            _,frame=previous.load_frame(folder,i)
            prediction=tracker.track(frame)
            rows.append(dict(frame=i,bbox=list(prediction['target_bbox']),score=float(prediction['best_score'])))
        path=output/(case['sequence']+'.json')
        write(path,dict(sequence=case['sequence'],control='swapped_category',rows=rows))
        item=dict(sequence=case['sequence'],frames=len(rows),sha256=sha(path),elapsed_seconds=time.time()-started)
        receipts.append(item);print(json.dumps(item),flush=True)
    plans()
    write(ROOT/'receipt.json',dict(status='complete',spec_sha256=sha(ROOT/'spec.json'),plan_sha256=sha(ROOT/'plan.json'),
        head_sha256=spec['head_sha256'],bank_sha256=spec['bank_sha256'],sequences=receipts,
        frames=sum(r['frames'] for r in receipts),subsequent_gt_opened=False,new_optimizer_steps=0,elapsed_seconds=time.time()-started))


def analyze():
    spec,training,m59=plans()
    assert (ROOT/'tracking.exit').read_text().strip()=='0'
    receipt=json.loads((ROOT/'receipt.json').read_text())
    assert receipt['status']=='complete' and receipt['spec_sha256']==sha(ROOT/'spec.json')
    assert receipt['frames']==33130 and len(receipt['sequences'])==22
    boxes={};scores={}
    assert [r['sequence'] for r in receipt['sequences']]==[c['sequence'] for c in spec['cases']]
    for item in receipt['sequences']:
        path=ROOT/'predictions'/(item['sequence']+'.json');assert sha(path)==item['sha256']
        data=json.loads(path.read_text());rows=data['rows']
        case=next(c for c in spec['cases'] if c['sequence']==item['sequence'])
        assert data['control']=='swapped_category' and data['sequence']==case['sequence']
        assert len(rows)==case['frames'] and rows[0]['bbox']==case['init_bbox']
        assert [r['frame'] for r in rows]==list(range(case['frames']))
        values=np.asarray([r['bbox'] for r in rows]);assert np.isfinite(values).all() and (values[:,2:]>0).all()
        boxes[case['sequence']]=values;scores[case['sequence']]=[r['score'] for r in rows]
    sys.path.insert(0,str(PARENT))
    from recursive_metric import statistics
    per={};writes={}
    # New control is fully sealed; reference families were sealed in M59.
    for case in spec['cases']:
        path=Path(training['dataset_root'])/case['sequence']/'groundtruth.txt'
        assert sha(path)==case['gt_sha256']
        gt=np.loadtxt(path,delimiter=',').reshape(-1,4)
        assert len(gt)==case['frames']
        per[case['sequence']]=statistics(boxes[case['sequence']],gt)
        writes[case['sequence']]=sum(i>0 and i%50==0 and s>.75 for i,s in enumerate(scores[case['sequence']]))
    aggregate=previous.aggregate(per)
    reference=m59['aggregates']['category'];empty=m59['aggregates']['empty']
    delta=reference['mean_iou']-aggregate['mean_iou']
    result=dict(status='completed_fixed_weight_category_content_isolation',observed_utc=datetime.now(timezone.utc).isoformat(),
        spec_sha256=sha(ROOT/'spec.json'),receipt_sha256=sha(ROOT/'receipt.json'),m59_result_sha256=M59_RESULT_SHA,
        head_sha256=spec['head_sha256'],aggregates=dict(original_category=reference,swapped_category=aggregate,empty=empty),
        per_sequence=dict(original_category=m59['per_sequence']['category'],swapped_category=per,empty=m59['per_sequence']['empty']),
        original_minus_swapped_mean_iou=delta,original_minus_swapped_H10=reference['failure_episodes']-aggregate['failure_episodes'],
        original_minus_swapped_low_frames=reference['low_iou_frames']-aggregate['low_iou_frames'],
        original_minus_empty_mean_iou=reference['mean_iou']-empty['mean_iou'],
        descriptive_category_margin_pass=delta>=spec['primary_descriptive_margin'],
        swapped_category_reconstructed_template_writes=sum(writes.values()),per_sequence_swapped_writes=writes,
        public_evaluation_allowed=False,independent_review_pass=False,
        scope='Same trained weight, masks and empty attribute tokens. Only the original automatic category is replaced; original category is not semantic ground truth. Reused development22, chosen after M59.')
    write(ROOT/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['per_sequence','per_sequence_swapped_writes']},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','parity','run','analyze'])
    action=parser.parse_args().mode
    {'prepare':prepare,'parity':parity,'run':run,'analyze':analyze}[action]()
