"""Fixed-final-head development recursion; all prediction families sealed before GT."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace


ROOT = Path('/root/autodl-tmp/sttrack_m73_paired_lexical_replication_20260907/seed2028')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def plans():
    spec = json.loads((ROOT/'recursive_spec.json').read_text())
    assert sha(ROOT/'recursive_spec.json') == json.loads((ROOT/'frozen.json').read_text())['recursive_spec_sha256']
    assert sha(__file__) == spec['runner_sha256']
    training = json.loads((ROOT/'training_spec.json').read_text())
    assert sha(ROOT/'training_spec.json') == spec['training_spec_sha256']
    assert sha(ROOT/'data_inventory.json') == training['inventory_sha256']
    for arm in ['category', 'empty']:
        assert sha(training['banks']['development'][arm]['path']) == training['banks']['development'][arm]['sha256']
    assert sha(ROOT/'integration.json') == training['integration_sha256']
    integration = json.loads((ROOT/'integration.json').read_text())
    for name, digest in integration['source_sha256'].items():
        assert sha(ROOT/'code'/name) == digest
    return spec, training


def trained():
    spec, training = plans()
    results = {}
    for arm in ['category', 'empty']:
        assert (ROOT/('training_'+arm+'.exit')).read_text().strip() == '0'
        r = json.loads((ROOT/'training'/arm/'result.json').read_text())
        assert r['status'] == 'one_full_causal_fit_pass_complete' and r['sequences'] == 130
        assert r['training_spec_sha256'] == sha(ROOT/'training_spec.json')
        assert r['total_track_calls'] == training['total_training_track_calls']
        assert r['base_parameters_and_buffers_unchanged']
        assert sha(ROOT/'training'/arm/'final.pth') == r['final_checkpoint_sha256']
        results[arm] = r
    for key in ['initial_adapter_state_sha256', 'base_state_before_sha256', 'optimizer_steps', 'total_track_calls']:
        assert results['category'][key] == results['empty'][key], key
    return spec, training, results


def run(arm):
    spec, training, results = trained()
    import torch
    sys.path.insert(0, str(ROOT/'code'))
    from lib.config.sttrack.config import cfg, update_config_from_file
    from lib.test.tracker.sttrack_semantic import STTrackSemantic
    from lib.train.dataset.depth_utils import get_rgbd_frame
    torch.set_num_threads(1)
    torch.manual_seed(training['seed']); torch.cuda.manual_seed_all(training['seed'])
    update_config_from_file(str(ROOT/'code/experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    params = SimpleNamespace(cfg=cfg, checkpoint=training['native_checkpoint'],
        base_checkpoint_sha256=training['native_checkpoint_sha256'], template_factor=2., template_size=128,
        search_factor=4., search_size=256, save_all_boxes=False, debug=0)
    checkpoint = ROOT/'training'/arm/'final.pth'
    saved = torch.load(checkpoint, map_location='cpu')
    assert saved['status'] == 'complete' and saved['completed_sequences'] == 130
    assert saved['training_spec_sha256'] == spec['training_spec_sha256']
    assert saved['use_text'] and saved['null_support']
    tracker = STTrackSemantic(params, str(checkpoint))
    bank = torch.load(training['banks']['development'][arm]['path'], map_location='cpu')
    assert set(bank['sequences']) == set(training['development_sequences'])
    output = ROOT/'recursive'/arm
    output.mkdir(parents=True)
    started = time.time()
    receipts = []
    for case in spec['cases']:
        folder = Path(training['dataset_root'])/case['sequence']
        def frame(i):
            return get_rgbd_frame(str(folder/'color'/('%08d.jpg'%(i+1))),
                str(folder/'depth'/('%08d.png'%(i+1))), dtype='rgbcolormap', depth_clip=True)
        index = bank['sequences'].index(case['sequence'])
        tracker.initialize(frame(0), dict(init_bbox=case['init_bbox'], text_tokens=bank['tokens'][index],
            text_mask=bank['mask'][index], empty_text=bank['empty']))
        rows = [dict(frame=0, bbox=list(tracker.state), score=None)]
        for i in range(1, case['frames']):
            prediction = tracker.track(frame(i))
            rows.append(dict(frame=i, bbox=list(prediction['target_bbox']), score=float(prediction['best_score'])))
        path = output/(case['sequence']+'.json')
        write(path, dict(sequence=case['sequence'], arm=arm, rows=rows))
        row = dict(sequence=case['sequence'], frames=len(rows), sha256=sha(path), elapsed_seconds=time.time()-started)
        receipts.append(row)
        print(json.dumps(row), flush=True)
    plans()
    receipt = dict(status='complete', arm=arm, recursive_spec_sha256=sha(ROOT/'recursive_spec.json'),
        training_result_sha256=sha(ROOT/'training'/arm/'result.json'), head_sha256=sha(checkpoint),
        sequences=receipts, total_frames=sum(r['frames'] for r in receipts),
        elapsed_seconds=time.time()-started, subsequent_gt_opened=False, text_updated_online=False)
    write(ROOT/(arm+'_recursive_receipt.json'), receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='sequences'}), flush=True)


def analyze():
    spec, training, results = trained()
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from recursive_metric import statistics
    assert sha(ROOT/'recursive_metric.py') == spec['metric_sha256']
    native_path = Path(spec['native_result_path'])
    assert sha(native_path) == training['native_result_sha256']
    native = json.loads(native_path.read_text())
    predicted = {a:{} for a in ['category','empty']}
    receipts = {}
    for arm in predicted:
        assert (ROOT/(arm+'_recursive.exit')).read_text().strip() == '0'
        receipt = json.loads((ROOT/(arm+'_recursive_receipt.json')).read_text())
        assert receipt['status']=='complete' and receipt['recursive_spec_sha256']==sha(ROOT/'recursive_spec.json')
        assert receipt['head_sha256']==results[arm]['final_checkpoint_sha256']
        assert receipt['total_frames']==33130 and len(receipt['sequences'])==22
        for row in receipt['sequences']:
            path=ROOT/'recursive'/arm/(row['sequence']+'.json')
            assert sha(path)==row['sha256']
            data=json.loads(path.read_text())
            case=next(c for c in spec['cases'] if c['sequence']==row['sequence'])
            assert data['sequence']==case['sequence'] and data['arm']==arm
            assert [r['frame'] for r in data['rows']]==list(range(case['frames']))
            assert data['rows'][0]['bbox']==case['init_bbox']
            boxes=np.asarray([r['bbox'] for r in data['rows']])
            assert np.isfinite(boxes).all() and (boxes[:,2:]>0).all()
            predicted[arm][case['sequence']]=boxes
        receipts[arm]=sha(ROOT/(arm+'_recursive_receipt.json'))
    assert all(set(v)==set(training['development_sequences']) for v in predicted.values())
    # Both complete families are now sealed; GT is first opened for metric analysis.
    per = {'native':native['per_sequence']['native'], 'category':{}, 'empty':{}}
    for case in spec['cases']:
        path=Path(training['dataset_root'])/case['sequence']/'groundtruth.txt'
        assert sha(path)==case['gt_sha256']
        gt=np.loadtxt(path,delimiter=',').reshape(-1,4)
        assert len(gt)==case['frames']
        for arm in predicted:
            per[arm][case['sequence']]=statistics(predicted[arm][case['sequence']],gt)
    aggregates={}
    for arm,values in per.items():
        counts={k:sum(r[k] for r in values.values()) for k in ['valid_frames','iou_sum','low_iou_frames','failure_episodes']}
        counts['mean_iou']=counts['iou_sum']/counts['valid_frames']
        counts['macro_sequence_mean_iou']=float(np.mean([r['mean_iou'] for r in values.values()]))
        aggregates[arm]=counts
        assert counts['valid_frames']==28897
    primary=aggregates['category']; baseline=aggregates['native']; control=aggregates['empty']
    broken=[n for n in per['native'] if per['native'][n]['failure_episodes']==0 and per['category'][n]['failure_episodes']>0]
    rule=training['promotion_gates']
    control_broken=[n for n in per['empty'] if per['empty'][n]['failure_episodes']==0 and per['category'][n]['failure_episodes']>0]
    legacy_broken=[n for n in training['protected_prior_control_sequences'] if per['category'][n]['failure_episodes']>0]
    gates=dict(prior_control_success_protection=not legacy_broken, mean_vs_native=primary['mean_iou']>=baseline['mean_iou']+rule['category_pooled_mean_vs_native_minimum'],
        mean_vs_control=primary['mean_iou']>=control['mean_iou']+rule['category_pooled_mean_vs_control_minimum'],
        macro_vs_native=primary['macro_sequence_mean_iou']>=baseline['macro_sequence_mean_iou'],
        macro_vs_control=primary['macro_sequence_mean_iou']>=control['macro_sequence_mean_iou'],
        low_frames_vs_native=primary['low_iou_frames']<=baseline['low_iou_frames'],
        low_frames_vs_control=primary['low_iou_frames']<=control['low_iou_frames'],
        H10_vs_native=primary['failure_episodes']<=baseline['failure_episodes'],
        H10_vs_control=primary['failure_episodes']<=control['failure_episodes'],
        native_success_protection=not broken, control_success_protection=not control_broken)
    result=dict(status='complete_recursive_development', observed_utc=datetime.now(timezone.utc).isoformat(),
        recursive_spec_sha256=sha(ROOT/'recursive_spec.json'), training_spec_sha256=sha(ROOT/'training_spec.json'),
        aggregates=aggregates, per_sequence=per, gates=gates, primary_pass=all(gates.values()),
        new_failure_sequences=broken, broken_control_success_sequences=control_broken, broken_prior_control_success_sequences=legacy_broken, receipts=receipts, frozen_native_result_sha256=sha(native_path),
        scope='Reused DepthTrack Train development22; no public dataset results.',
        caption_scope='Automatic initialization-only captions with documented errors; no semantic ground-truth guarantee.',
        independent_review_pass=False, next='Fixed-weight category versus empty and swapped category controls before low22' if all(gates.values()) else 'Stop this frozen revision; diagnose complete trajectories')
    write(ROOT/'recursive_result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='per_sequence'},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--arm',choices=['category','empty'])
    action.add_argument('--analyze',action='store_true')
    action.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    if args.preflight:
        plans(); print('FROZEN_RECURSIVE_PLAN_VALID')
    elif args.analyze:
        analyze()
    else:
        run(args.arm)
