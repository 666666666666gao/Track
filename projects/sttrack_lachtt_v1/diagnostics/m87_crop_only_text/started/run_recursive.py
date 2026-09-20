"""Four fixed-content full recursions for one centered Category final checkpoint."""
import argparse,hashlib,json,math,sys,time
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).parent
FAMILIES=['category','category_empty','category_old','category_swapped']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def plans():
    frozen=read(ROOT/'frozen.json');spec=read(ROOT/'recursive_spec.json');training=read(ROOT/'training_spec.json')
    assert sha(ROOT/'recursive_spec.json')==frozen['recursive_spec_sha256']
    assert sha(ROOT/'training_spec.json')==frozen['training_spec_sha256']==spec['training_spec_sha256']
    assert sha(Path(__file__))==spec['runner_sha256'] and sha(ROOT/'recursive_metric.py')==spec['metric_sha256']
    assert sha(ROOT/'integration.json')==training['integration_sha256']
    for n,h in read(ROOT/'integration.json')['source_sha256'].items():assert sha(ROOT/'code'/n)==h
    for a in ['category','empty','old','swapped']:
        b=training['banks']['development'][a];assert sha(b['path'])==b['sha256']
    assert (ROOT/'training_category.exit').read_text().strip()=='0'
    result=read(ROOT/'training/category/result.json')
    assert result['status']=='one_full_causal_fit_pass_complete' and result['sequences']==130
    assert result['total_track_calls']==186694 and result['optimizer_steps']==5798
    assert result['training_spec_sha256']==spec['training_spec_sha256'] and result['base_parameters_and_buffers_unchanged']
    assert sha(ROOT/'training/category/final.pth')==result['final_checkpoint_sha256']
    return spec,training,result

def run(arm):
    spec,training,result=plans()
    import torch
    sys.path.insert(0,str(ROOT/'code'))
    from lib.config.sttrack.config import cfg,update_config_from_file
    from lib.test.tracker.sttrack_semantic import STTrackSemantic
    from lib.train.dataset.depth_utils import get_rgbd_frame
    torch.set_num_threads(1);torch.manual_seed(2027);torch.cuda.manual_seed_all(2027)
    update_config_from_file(str(ROOT/'code/experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    params=SimpleNamespace(cfg=cfg,checkpoint=training['native_checkpoint'],
        base_checkpoint_sha256=training['native_checkpoint_sha256'],template_factor=2.,template_size=128,
        search_factor=4.,search_size=256,save_all_boxes=False,debug=0)
    checkpoint=ROOT/'training/category/final.pth'
    saved=torch.load(checkpoint,map_location='cpu')
    assert saved['architecture']=='semantic_spatial_centered_v1' and saved['status']=='complete'
    assert saved['training_spec_sha256']==spec['training_spec_sha256']
    tracker=STTrackSemantic(params,str(checkpoint))
    content={'category':'category','category_empty':'empty','category_old':'old','category_swapped':'swapped'}[arm]
    bank=torch.load(training['banks']['development'][content]['path'],map_location='cpu')
    assert set(bank['sequences'])==set(training['development_sequences'])
    assert torch.equal(bank['empty'].float(),tracker.network.semantic_adapter.empty_text.cpu())
    output=ROOT/'recursive'/arm;output.mkdir(parents=True)
    receipts=[];started=time.time()
    for case in spec['cases']:
        folder=Path(training['dataset_root'])/case['sequence']
        def frame(i):
            return get_rgbd_frame(str(folder/'color'/('%08d.jpg'%(i+1))),
                str(folder/'depth'/('%08d.png'%(i+1))),dtype='rgbcolormap',depth_clip=True)
        idx=bank['sequences'].index(case['sequence'])
        tracker.initialize(frame(0),dict(init_bbox=case['init_bbox'],text_tokens=bank['tokens'][idx],
            text_mask=bank['mask'][idx],empty_text=bank['empty']))
        rows=[dict(frame=0,bbox=list(tracker.state),score=None)]
        for i in range(1,case['frames']):
            prediction=tracker.track(frame(i))
            rows.append(dict(frame=i,bbox=list(prediction['target_bbox']),score=float(prediction['best_score'])))
        path=output/(case['sequence']+'.json');save(path,dict(sequence=case['sequence'],arm=arm,rows=rows))
        row=dict(sequence=case['sequence'],frames=len(rows),sha256=sha(path),elapsed_seconds=time.time()-started)
        receipts.append(row);print(json.dumps(row),flush=True)
    plans()
    receipt=dict(status='complete',arm=arm,recursive_spec_sha256=sha(ROOT/'recursive_spec.json'),
        head_sha256=sha(checkpoint),training_result_sha256=sha(ROOT/'training/category/result.json'),
        total_frames=sum(r['frames'] for r in receipts),sequences=receipts,elapsed_seconds=time.time()-started,
        subsequent_gt_opened=False,text_updated_online=False)
    save(ROOT/(arm+'_recursive_receipt.json'),receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='sequences'}),flush=True)

def analyze():
    spec,training,trained=plans()
    import numpy as np
    sys.path.insert(0,str(ROOT))
    from recursive_metric import statistics
    predictions={a:{} for a in FAMILIES};receipts={}
    for arm in FAMILIES:
        assert (ROOT/(arm+'_recursive.exit')).read_text().strip()=='0'
        receipt=read(ROOT/(arm+'_recursive_receipt.json'))
        assert receipt['status']=='complete' and receipt['total_frames']==33130 and len(receipt['sequences'])==22
        assert receipt['recursive_spec_sha256']==sha(ROOT/'recursive_spec.json')
        assert receipt['head_sha256']==trained['final_checkpoint_sha256']
        for case,row in zip(spec['cases'],receipt['sequences']):
            assert case['sequence']==row['sequence'] and case['frames']==row['frames']
            path=ROOT/'recursive'/arm/(case['sequence']+'.json');assert sha(path)==row['sha256']
            data=read(path);assert data['arm']==arm and data['sequence']==case['sequence']
            assert [r['frame'] for r in data['rows']]==list(range(case['frames']))
            assert data['rows'][0]['bbox']==case['init_bbox']
            boxes=np.asarray([r['bbox'] for r in data['rows']]);assert np.isfinite(boxes).all() and (boxes[:,2:]>0).all()
            predictions[arm][case['sequence']]=boxes
        receipts[arm]=sha(ROOT/(arm+'_recursive_receipt.json'))
    assert sha(spec['native_result_path'])==training['native_result_sha256']
    assert sha(training['parent_result_path'])==training['parent_result_sha256']
    native=read(spec['native_result_path']);parent=read(training['parent_result_path'])
    per={'native':native['per_sequence']['native'],'M84_protocol_control':parent['per_sequence']['category'],'M82_historical':parent['per_sequence']['M82_mask_control'],**{a:{} for a in FAMILIES}}
    # Every family is sealed before loading any development GT.
    for case in spec['cases']:
        gtpath=Path(training['dataset_root'])/case['sequence']/'groundtruth.txt';assert sha(gtpath)==case['gt_sha256']
        gt=np.loadtxt(gtpath,delimiter=',').reshape(-1,4);assert len(gt)==case['frames']
        for arm in FAMILIES:per[arm][case['sequence']]=statistics(predictions[arm][case['sequence']],gt)
    aggregates={}
    for arm,values in per.items():
        sums={k:sum(v[k] for v in values.values()) for k in ['valid_frames','iou_sum','low_iou_frames','failure_episodes']}
        assert sums['valid_frames']==28897
        sums['mean_iou']=sums['iou_sum']/sums['valid_frames']
        sums['macro_sequence_mean_iou']=sum(v['mean_iou'] for v in values.values())/22
        aggregates[arm]=sums
    current=aggregates['category']
    def compare(other):
        return dict(mean=current['mean_iou']>other['mean_iou'],macro=current['macro_sequence_mean_iou']>=other['macro_sequence_mean_iou'],
            low=current['low_iou_frames']<=other['low_iou_frames'],H10=current['failure_episodes']<=other['failure_episodes'])
    native_gates=compare(aggregates['native'])
    broken=[s for s,v in per['native'].items() if v['failure_episodes']==0 and per['category'][s]['failure_episodes']>0]
    native_gates['native_zero_H10_protection']=not broken
    parity={s:all(per['category_empty'][s][k]==v[k] for k in ['valid_frames','low_iou_frames','failure_episodes']) and
        abs(per['category_empty'][s]['iou_sum']-v['iou_sum'])<=1e-8 for s,v in per['native'].items()}
    gates=dict(M84_increment=compare(aggregates['M84_protocol_control']),native=native_gates,
        old_content=compare(aggregates['category_old']),swapped_content=compare(aggregates['category_swapped']),empty_native_parity={'all_sequences':all(parity.values())})
    loo={}
    for arm in ['category_empty','category_old','category_swapped']:
        other=aggregates[arm];loo[arm]={}
        for s,v in per['category'].items():
            o=per[arm][s]
            loo[arm][s]=(current['iou_sum']-v['iou_sum'])/(current['valid_frames']-v['valid_frames'])-(other['iou_sum']-o['iou_sum'])/(other['valid_frames']-o['valid_frames'])
    result=dict(status='complete_recursive_development',observed_utc=datetime.now(timezone.utc).isoformat(),
        training_spec_sha256=sha(ROOT/'training_spec.json'),recursive_spec_sha256=sha(ROOT/'recursive_spec.json'),
        aggregates=aggregates,per_sequence=per,gates=gates,all_gates_pass=all(all(g.values()) for g in gates.values()),
        gate_count=sum(len(g) for g in gates.values()),native_success_damage=broken,empty_native_parity=parity,
        content_leave_one_out=loo,receipts=receipts,head_sha256=trained['final_checkpoint_sha256'],
        scope='Repeated DepthTrack Train development22; one trained Category head and four content recursions, not four trained models.',
        text_scope='Frozen automatic captions, not semantically verified ground truth.',public_evaluation_started=False)
    assert result['gate_count']==18
    save(ROOT/'recursive_result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='per_sequence'},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--arm',choices=FAMILIES);g.add_argument('--analyze',action='store_true');a=p.parse_args()
    analyze() if a.analyze else run(a.arm)
