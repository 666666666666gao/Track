"""M80 fixed-final-head development/content evaluation; no external auto-promotion."""
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace
import argparse, hashlib, json, sys, time

R=Path('/root/autodl-tmp/sttrack_m80_block_text_dropout_20260908')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def plans():
    frozen=read(R/'frozen.json');t=read(R/'training_spec.json');e=read(R/'evaluation_spec.json')
    assert sha(R/'training_spec.json')==frozen['training_spec_sha256']
    assert sha(R/'evaluation_spec.json')==frozen['evaluation_spec_sha256']
    assert sha(__file__)==e['evaluator_sha256']
    assert sha(R/'recursive_metric.py')==e['metric_sha256']
    assert sha(R/'integration.json')==t['integration_sha256']
    for n,h in read(R/'integration.json')['source_sha256'].items(): assert sha(R/'code'/n)==h
    assert (R/'training_category.exit').read_text().strip()=='0'
    result=read(R/'training/category/result.json')
    assert result['status']=='one_full_causal_fit_pass_complete' and result['sequences']==130
    assert result['total_track_calls']==186694 and result['optimizer_steps']==5798
    assert result['training_spec_sha256']==sha(R/'training_spec.json')
    assert result['base_parameters_and_buffers_unchanged'] and result['learned_parameters']==289154
    assert result['initial_adapter_state_sha256']=='56d03acff2972871788dd51f0eff40d0b6d888aad79fe74db4c44cdfb276ca42'
    assert result['base_state_before_sha256']=='c07022117c6efa8399b6df57e275ec81c6f3efce44eca5e4bc0282b52dd0dae0'
    assert sha(R/'training/category/final.pth')==result['final_checkpoint_sha256']
    schedule=read(R/'text_schedule.json')
    assert sha(R/'text_schedule.json')==t['text_schedule_sha256']==result['text_schedule_sha256']
    assert result['actual_text_condition_calls']['empty']==schedule['dropout_track_calls']
    assert sum(result['actual_text_condition_calls'].values())==186694
    for condition in ['category','empty']:
        b=t['banks']['development'][condition];assert sha(b['path'])==b['sha256']
    assert sha(e['swapped_bank']['path'])==e['swapped_bank']['sha256']
    return t,e,result

def run(condition):
    t,e,result=plans()
    import torch
    sys.path.insert(0,str(R/'code'))
    from lib.config.sttrack.config import cfg,update_config_from_file
    from lib.test.tracker.sttrack_semantic import STTrackSemantic
    from lib.train.dataset.depth_utils import get_rgbd_frame
    torch.set_num_threads(1);torch.manual_seed(2027);torch.cuda.manual_seed_all(2027)
    update_config_from_file(str(R/'code/experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    params=SimpleNamespace(cfg=cfg,checkpoint=t['native_checkpoint'],base_checkpoint_sha256=t['native_checkpoint_sha256'],template_factor=2.,template_size=128,search_factor=4.,search_size=256,save_all_boxes=False,debug=0)
    checkpoint=R/'training/category/final.pth'
    saved=torch.load(checkpoint,map_location='cpu')
    assert saved['status']=='complete' and saved['seed']==2027 and saved['use_text'] and saved['null_support']
    assert saved['training_spec_sha256']==sha(R/'training_spec.json')
    tracker=STTrackSemantic(params,str(checkpoint))
    b=e['swapped_bank'] if condition=='swapped' else t['banks']['development'][condition]
    bank=torch.load(b['path'],map_location='cpu')
    category=torch.load(t['banks']['development']['category']['path'],map_location='cpu')
    assert set(bank['sequences'])==set(t['development_sequences'])
    for seq in bank['sequences']:
        i=bank['sequences'].index(seq);j=category['sequences'].index(seq)
        assert torch.equal(bank['mask'][i],category['mask'][j])
        valid=bank['mask'][i].bool()
        assert torch.equal(bank['tokens'][i][~valid],category['tokens'][j][~valid])
        if condition=='empty': assert torch.equal(bank['tokens'][i][valid],bank['empty'].expand_as(bank['tokens'][i][valid]))
        if condition=='swapped': assert torch.equal(bank['tokens'][i][1:],category['tokens'][j][1:])
    output=R/'recursive'/condition;output.mkdir(parents=True)
    receipts=[];started=time.time()
    for case in e['cases']:
        folder=Path(t['dataset_root'])/case['sequence']
        def frame(i):return get_rgbd_frame(str(folder/'color'/('%08d.jpg'%(i+1))),str(folder/'depth'/('%08d.png'%(i+1))),dtype='rgbcolormap',depth_clip=True)
        i=bank['sequences'].index(case['sequence'])
        tracker.initialize(frame(0),dict(init_bbox=case['init_bbox'],text_tokens=bank['tokens'][i],text_mask=bank['mask'][i],empty_text=bank['empty']))
        rows=[dict(frame=0,bbox=list(tracker.state),score=None)]
        for index in range(1,case['frames']):
            prediction=tracker.track(frame(index))
            rows.append(dict(frame=index,bbox=list(prediction['target_bbox']),score=float(prediction['best_score'])))
        path=output/(case['sequence']+'.json');write(path,dict(sequence=case['sequence'],condition=condition,rows=rows))
        receipt=dict(sequence=case['sequence'],frames=len(rows),sha256=sha(path),elapsed_seconds=time.time()-started)
        receipts.append(receipt);print(json.dumps(receipt),flush=True)
    plans()
    write(R/('receipt_'+condition+'.json'),dict(status='complete',condition=condition,head_sha256=sha(checkpoint),text_bank_sha256=sha(b['path']),evaluation_spec_sha256=sha(R/'evaluation_spec.json'),sequences=receipts,total_frames=sum(x['frames'] for x in receipts),subsequent_gt_opened=False,text_updated_online=False))

def analyze():
    t,e,training=plans()
    import numpy as np
    sys.path.insert(0,str(R));from recursive_metric import statistics
    assert sha(e['parent_result_path'])==e['parent_result_sha256']
    parent=read(e['parent_result_path'])
    assert sha(e['native_result_path'])==e['native_result_sha256']
    native=read(e['native_result_path'])
    predictions={};receipt_hashes={}
    for condition in ['category','empty','swapped']:
        assert (R/('eval_'+condition+'.exit')).read_text().strip()=='0'
        receipt=read(R/('receipt_'+condition+'.json'))
        assert receipt['status']=='complete' and receipt['condition']==condition
        assert receipt['head_sha256']==training['final_checkpoint_sha256']
        assert receipt['evaluation_spec_sha256']==sha(R/'evaluation_spec.json')
        assert receipt['total_frames']==33130 and len(receipt['sequences'])==22
        predictions[condition]={};receipt_hashes[condition]=sha(R/('receipt_'+condition+'.json'))
        for row in receipt['sequences']:
            path=R/'recursive'/condition/(row['sequence']+'.json');assert sha(path)==row['sha256']
            data=read(path);case=next(c for c in e['cases'] if c['sequence']==row['sequence'])
            assert data['condition']==condition and data['sequence']==case['sequence']
            assert [x['frame'] for x in data['rows']]==list(range(case['frames']))
            assert data['rows'][0]['bbox']==case['init_bbox']
            boxes=np.asarray([x['bbox'] for x in data['rows']]);assert np.isfinite(boxes).all() and (boxes[:,2:]>0).all()
            predictions[condition][case['sequence']]=boxes
        assert set(predictions[condition])==set(t['development_sequences'])
    # All three complete prediction families have been sealed before reading GT.
    per={'native':native['per_sequence']['native'],'M78_category':parent['per_sequence']['category'],'M78_empty_trained':parent['per_sequence']['empty']}
    per.update({c:{} for c in predictions})
    for case in e['cases']:
        gt_path=Path(t['dataset_root'])/case['sequence']/'groundtruth.txt';assert sha(gt_path)==case['gt_sha256']
        gt=np.loadtxt(gt_path,delimiter=',').reshape(-1,4);assert len(gt)==case['frames']
        for c in predictions:per[c][case['sequence']]=statistics(predictions[c][case['sequence']],gt)
    aggregates={}
    for c,seqs in per.items():
        a={k:sum(x[k] for x in seqs.values()) for k in ['valid_frames','iou_sum','low_iou_frames','failure_episodes']}
        assert a['valid_frames']==28897
        a.update(mean_iou=a['iou_sum']/a['valid_frames'],macro_sequence_mean_iou=float(np.mean([x['mean_iou'] for x in seqs.values()])))
        aggregates[c]=a
    primary=aggregates['category'];gates={};broken={}
    for ref,margin in [('native',.002),('M78_empty_trained',.001)]:
        baseline=aggregates[ref]
        broken[ref]=[n for n in per[ref] if per[ref][n]['failure_episodes']==0 and per['category'][n]['failure_episodes']>0]
        gates[ref+'_pooled']=primary['mean_iou']>=baseline['mean_iou']+margin
        gates[ref+'_macro']=primary['macro_sequence_mean_iou']>=baseline['macro_sequence_mean_iou']
        gates[ref+'_low']=primary['low_iou_frames']<=baseline['low_iou_frames']
        gates[ref+'_H10']=primary['failure_episodes']<=baseline['failure_episodes']
        gates[ref+'_protect']=not broken[ref]
    content={};mechanism={}
    for ref in ['empty','swapped','M78_category']:
        b=aggregates[ref]
        target=mechanism if ref=='M78_category' else content
        target[ref+'_pooled']=primary['mean_iou']>b['mean_iou'] if ref!='M78_category' else primary['mean_iou']>=b['mean_iou']
        target[ref+'_macro']=primary['macro_sequence_mean_iou']>b['macro_sequence_mean_iou'] if ref!='M78_category' else primary['macro_sequence_mean_iou']>=b['macro_sequence_mean_iou']
        target[ref+'_low']=primary['low_iou_frames']<=b['low_iou_frames']
        target[ref+'_H10']=primary['failure_episodes']<=b['failure_episodes']
    result=dict(status='completed_M80_development_and_content',observed_utc=datetime.now(timezone.utc).isoformat(),aggregates=aggregates,per_sequence=per,primary_gates=gates,content_gates=content,dropout_vs_no_dropout_gates=mechanism,broken_success_sequences=broken,primary_pass=all(gates.values()),content_pass=all(content.values()),dropout_mechanism_pass=all(mechanism.values()),all_development_checks_pass=all(list(gates.values())+list(content.values())+list(mechanism.values())),receipts=receipt_hashes,training_spec_sha256=sha(R/'training_spec.json'),evaluation_spec_sha256=sha(R/'evaluation_spec.json'),head_sha256=training['final_checkpoint_sha256'],parent_result_sha256=e['parent_result_sha256'],public_evaluation_allowed=False,independent_model_review_pass=False,scope='Fixed seed2027; reused DepthTrack Train development22. No external dataset score or automatic promotion.')
    write(R/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='per_sequence'},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();a=p.add_mutually_exclusive_group(required=True);a.add_argument('--condition',choices=['category','empty','swapped']);a.add_argument('--analyze',action='store_true');args=p.parse_args()
    analyze() if args.analyze else run(args.condition)
