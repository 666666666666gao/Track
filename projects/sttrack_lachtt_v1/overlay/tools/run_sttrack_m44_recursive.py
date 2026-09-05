"""Run a frozen M44 arm on complete development sequences and compare both."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--variant',choices=['geometry','appearance']);p.add_argument('--analyze',action='store_true')
    args=p.parse_args();root=args.root;spec=json.loads((root/'spec.json').read_text());repo=Path(spec['repository']);sys.path.insert(0,str(repo))
    from tools.train_sttrack_m44 import sha,check_binding
    check_binding(root,spec);training=json.loads((root/'training_result.json').read_text());assert training['status']=='complete'
    for arm in spec['variants']:assert sha(root/(arm+'_final.pth'))==training['variants'][arm]['checkpoint_sha256']
    cases=[x for x in json.loads((root/'inference_inputs.json').read_text()) if x['split']=='development'];assert len(cases)==22
    if args.analyze:
        from tools.analyze_sttrack_m42_recursive import statistics
        default=defaultdict(list);names={x['sequence'] for x in cases}
        for path,digest in spec['baseline_trace_sha256'].items():
            assert sha(path)==digest
            for x in json.loads(Path(path).read_text())['rows']:
                if x['sequence'] in names:default[x['sequence']].append(x)
        per={arm:{} for arm in ['default']+spec['variants']}
        for case in cases:
            name=case['sequence'];rows=sorted(default[name],key=lambda x:x['frame_index']);assert len(rows)==case['frames']
            gt=np.loadtxt(Path(spec['dataset_root'])/name/'groundtruth.txt',delimiter=',');per['default'][name]=statistics([x['public_bbox'] for x in rows],gt)
        for arm in spec['variants']:
            receipt=json.loads((root/(arm+'_recursive_receipt.json')).read_text());assert receipt['status']=='complete' and {x['sequence'] for x in receipt['sequences']}==names
            for item in receipt['sequences']:
                name=item['sequence'];path=root/'recursive'/(arm+'_'+name+'.json');assert sha(path)==item['sha256']
                rows=json.loads(path.read_text())['rows'];assert len(rows)==len(default[name]);assert [x['frame'] for x in rows]==list(range(len(rows)))
                gt=np.loadtxt(Path(spec['dataset_root'])/name/'groundtruth.txt',delimiter=',');per[arm][name]=statistics([x['bbox'] for x in rows],gt);per[arm][name]['changes']=item['changes']
        aggregates={}
        for arm,rows in per.items():
            totals={key:sum(x[key] for x in rows.values()) for key in ['valid_frames','iou_sum','low_iou_frames','failure_episodes']}
            totals['mean_iou']=totals['iou_sum']/totals['valid_frames'];totals['macro_sequence_mean_iou']=float(np.mean([x['mean_iou'] for x in rows.values()]));aggregates[arm]=totals
        base=aggregates['default'];comparisons={};rule=spec['recursive_performance_gate']
        for arm in spec['variants']:
            x=aggregates[arm];positive=sum(per[arm][n]['mean_iou']>per['default'][n]['mean_iou'] for n in names)
            broken=[n for n in sorted(names) if per['default'][n]['failure_episodes']==0 and per[arm][n]['failure_episodes']>0]
            gates=dict(mean_iou=x['mean_iou']>=base['mean_iou']+rule['mean_iou_gain_at_least'],fewer_low_frames=x['low_iou_frames']<base['low_iou_frames'],
                no_episode_increase=x['failure_episodes']<=base['failure_episodes'],sequence_coverage=positive>=rule['positive_sequences_at_least'],successful_sequence_protection=len(broken)==0)
            comparisons[arm]=dict(gates=gates,pass_gate=all(gates.values()),positive_sequences=positive,new_failure_sequences=broken)
        result=dict(status='complete',primary='geometry',primary_pass=comparisons['geometry']['pass_gate'],aggregates=aggregates,per_sequence=per,comparisons=comparisons,
            geometry_incremental_mean_gain=aggregates['geometry']['mean_iou']-aggregates['appearance']['mean_iou'],spec_sha256=sha(root/'spec.json'),training_result_sha256=sha(root/'training_result.json'),
            claim='Train development recursive proxies only; primary and positional attribution remain separate.',
            next='freeze low22 geometry comparison' if comparisons['geometry']['pass_gate'] else 'stop this frozen variant; no public evaluation')
        (root/'recursive_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(aggregates=aggregates,comparisons=comparisons)),flush=True);return
    arm=args.variant;assert arm in spec['variants']
    from lib.config.sttrack.config import cfg,update_config_from_file
    from lib.test.tracker.sttrack_candidate_set import STTrackCandidateSet
    from lib.train.dataset.depth_utils import get_rgbd_frame
    torch.set_num_threads(1);update_config_from_file(str(repo/'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    params=SimpleNamespace(cfg=cfg,checkpoint=spec['checkpoint'],template_factor=2.,template_size=128,search_factor=4.,search_size=256,save_all_boxes=False,debug=0)
    tracker=STTrackCandidateSet(params,root/(arm+'_final.pth'));outdir=root/'recursive';outdir.mkdir(exist_ok=True);receipts=[];started=time.time()
    for case in cases:
        name=case['sequence'];folder=Path(spec['dataset_root'])/name
        def image_at(frame):
            return get_rgbd_frame(str(folder/'color'/f'{frame+1:08d}.jpg'),str(folder/'depth'/f'{frame+1:08d}.png'),dtype='rgbcolormap',depth_clip=True)
        tracker.initialize(image_at(0),dict(init_bbox=list(case['init_bbox'])));rows=[dict(frame=0,bbox=case['init_bbox'],score=1.,choice=0,none=False)]
        for frame in range(1,case['frames']):
            out=tracker.track(image_at(frame));rows.append(dict(frame=frame,bbox=out['target_bbox'],score=float(out['best_score']),choice=out['association_candidate'],none=out['association_none']))
        path=outdir/(arm+'_'+name+'.json');path.write_text(json.dumps(dict(sequence=name,arm=arm,rows=rows))+'\n')
        item=dict(sequence=name,frames=len(rows),changes=sum(x['choice']!=0 for x in rows),sha256=sha(path),elapsed_seconds=time.time()-started);receipts.append(item);print(json.dumps(item),flush=True)
    check_binding(root,spec)
    (root/(arm+'_recursive_receipt.json')).write_text(json.dumps(dict(status='complete',sequences=receipts,elapsed_seconds=time.time()-started,
        checkpoint_sha256=sha(root/(arm+'_final.pth')),source_unchanged=True,ground_truth_files_opened=False),indent=2)+'\n')


if __name__=='__main__':main()
