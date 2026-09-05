"""Run the frozen M45 weight in two sequence shards, then audit both outputs."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--shard',type=int);p.add_argument('--analyze',action='store_true')
    args=p.parse_args();root=args.root;plan=json.loads((root/'spec.json').read_text());parent=Path(plan['source_root'])
    spec=json.loads((parent/'spec.json').read_text());repo=Path(spec['repository']);sys.path.insert(0,str(repo))
    from tools.train_sttrack_m44 import sha,check_binding
    from tools.audit_sttrack_m43 import independent_overlap
    check_binding(parent,spec)
    assert sha(parent/'inference_inputs.json')==spec['inference_inputs_sha256']
    for name,digest in plan['source_sha256'].items():assert sha(repo/name)==digest,name
    training=json.loads((root/'geometry_result.json').read_text());assert training['status']=='complete'
    assert training['m45_spec_sha256']==sha(root/'spec.json') and training['matched_control_initialization_and_sample_order']
    checkpoint=root/'geometry_final.pth';assert sha(checkpoint)==training['checkpoint_sha256']
    cases={c['sequence']:c for c in json.loads((parent/'inference_inputs.json').read_text()) if c['split']=='development'}
    assert len(cases)==22
    if args.analyze:
        assert [(root/f'recursive_s{i}.exit').read_text().strip() for i in [0,1]]==['0','0']
        control=json.loads((parent/'recursive_result.json').read_text());assert sha(parent/'recursive_result.json')==plan['control_recursive_result_sha256']
        receipts=[]
        for shard in [0,1]:
            receipt=json.loads((root/f'shard{shard}_receipt.json').read_text());assert receipt['status']=='complete' and receipt['checkpoint_sha256']==sha(checkpoint)
            assert receipt['spec_sha256']==sha(root/'spec.json');receipts.extend(receipt['sequences'])
        assert len(receipts)==22 and {r['sequence'] for r in receipts}==set(cases)
        baseline=defaultdict(list)
        for path,digest in spec['baseline_trace_sha256'].items():
            assert sha(path)==digest
            for row in json.loads(Path(path).read_text())['rows']:
                if row['sequence'] in cases:baseline[row['sequence']].append(row)
        per={};details=[];table=[]
        for item in receipts:
            name=item['sequence'];case=cases[name];path=root/'recursive'/(name+'.json');assert sha(path)==item['sha256']
            data=json.loads(path.read_text());rows=data['rows'];assert data['sequence']==name
            assert len(rows)==case['frames'] and [r['frame'] for r in rows]==list(range(len(rows)))
            gt=np.loadtxt(Path(spec['dataset_root'])/name/'groundtruth.txt',delimiter=',')[:len(rows)];assert len(gt)==len(rows)
            boxes=np.asarray([r['bbox'] for r in rows],dtype=np.float64);assert np.isfinite(boxes).all() and (boxes[:,2:]>0).all()
            assert all(math.isfinite(r['score']) and 0<=r['choice']<10 and (not r['none'] or r['choice']==0) for r in rows)
            assert np.array_equal(boxes[0],np.asarray(case['init_bbox']))
            values,metrics=independent_overlap(boxes,gt);metrics['low_iou_frames']=int(metrics['low_iou_frames']);per[name]=metrics
            base=sorted(baseline[name],key=lambda x:x['frame_index']);assert [r['frame_index'] for r in base]==list(range(len(rows)))
            baseboxes=np.asarray([r['public_bbox'] for r in base]);basevalues,basemetrics=independent_overlap(baseboxes,gt);basemetrics['low_iou_frames']=int(basemetrics['low_iou_frames'])
            for key,value in basemetrics.items():assert math.isclose(value,control['per_sequence']['default'][name][key],rel_tol=1e-12,abs_tol=1e-10)
            changed=[i for i,r in enumerate(rows) if r['choice']!=0];assert len(changed)==item['changes']
            first=changed[0] if changed else len(rows);error=float(np.abs(boxes[:first]-baseboxes[:first]).max());assert error==0.,(name,first,error)
            details.append(dict(sequence=name,changes=len(changed),first_override=first if changed else None,prefix_error_px=error,
                first_default_iou=float(basevalues[first]) if changed and np.isfinite(basevalues[first]) else None,
                first_selected_iou=float(values[first]) if changed and np.isfinite(values[first]) else None,trajectory_sha256=sha(path)))
            table.append(dict(sequence=name,**metrics,changes=len(changed),mean_iou_gain=metrics['mean_iou']-basemetrics['mean_iou'],
                              low_frame_delta=metrics['low_iou_frames']-basemetrics['low_iou_frames'],episode_delta=metrics['failure_episodes']-basemetrics['failure_episodes']))
        totals={key:sum(x[key] for x in per.values()) for key in ['valid_frames','iou_sum','low_iou_frames','failure_episodes']}
        totals['mean_iou']=totals['iou_sum']/totals['valid_frames'];totals['macro_sequence_mean_iou']=float(np.mean([x['mean_iou'] for x in per.values()]))
        base=control['aggregates']['default'];positive=sum(x['mean_iou']>control['per_sequence']['default'][n]['mean_iou'] for n,x in per.items())
        broken=sorted(n for n,x in per.items() if control['per_sequence']['default'][n]['failure_episodes']==0 and x['failure_episodes']>0)
        rule=plan['performance_gate'];gates=dict(mean_iou=totals['mean_iou']>=base['mean_iou']+rule['mean_iou_gain_at_least'],
            fewer_low_frames=totals['low_iou_frames']<base['low_iou_frames'],no_episode_increase=totals['failure_episodes']<=base['failure_episodes'],
            sequence_coverage=positive>=rule['positive_sequences_at_least'],successful_sequence_protection=not broken)
        saved=torch.load(checkpoint,map_location='cpu');assert saved['variant']=='geometry' and saved['m45_spec_sha256']==sha(root/'spec.json')
        assert saved['optimizer_steps']==960 and saved['base_checkpoint_sha256']==spec['checkpoint_sha256']
        result=dict(status='complete',integrity_pass=True,primary='default_priority',primary_pass=all(gates.values()),gates=gates,
            aggregates=dict(default=base,m44_geometry=control['aggregates']['geometry'],m45=totals),per_sequence=per,
            positive_sequences=positive,new_failure_sequences=broken,first_overrides=details,frames=33130,
            exact_default_prefix=True,baseline_scalar_recomputation_matches=True,checkpoint_sha256=sha(checkpoint),spec_sha256=sha(root/'spec.json'),
            training_result_sha256=sha(root/'geometry_result.json'),control_result_sha256=sha(parent/'recursive_result.json'),
            label_effect_mean_gain=totals['mean_iou']-control['aggregates']['geometry']['mean_iou'],
            next='freeze low22' if all(gates.values()) else 'stop this label variant; no public evaluation',
            scope='Existing Train development sequences. Historical M44 control has matched initialization, sample order, architecture and optimizer; only training target rule changes. No public metric.')
        for name,digest in plan['source_sha256'].items():assert sha(repo/name)==digest
        (root/'recursive_result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        with (root/'per_sequence.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
        print(json.dumps({k:v for k,v in result.items() if k not in ['per_sequence','first_overrides']},indent=2));return
    assert args.shard in [0,1]
    from lib.config.sttrack.config import cfg,update_config_from_file
    from lib.test.tracker.sttrack_candidate_set import STTrackCandidateSet
    from lib.train.dataset.depth_utils import get_rgbd_frame
    torch.set_num_threads(1);update_config_from_file(str(repo/'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    params=SimpleNamespace(cfg=cfg,checkpoint=spec['checkpoint'],template_factor=2.,template_size=128,search_factor=4.,search_size=256,save_all_boxes=False,debug=0)
    tracker=STTrackCandidateSet(params,checkpoint);outdir=root/'recursive';outdir.mkdir(exist_ok=True);receipts=[];started=time.time()
    for name in plan['shards'][args.shard]:
        case=cases[name];folder=Path(spec['dataset_root'])/name
        def image_at(frame):return get_rgbd_frame(str(folder/'color'/f'{frame+1:08d}.jpg'),str(folder/'depth'/f'{frame+1:08d}.png'),dtype='rgbcolormap',depth_clip=True)
        tracker.initialize(image_at(0),dict(init_bbox=case['init_bbox']));rows=[dict(frame=0,bbox=case['init_bbox'],score=1.,choice=0,none=False)]
        for frame in range(1,case['frames']):
            out=tracker.track(image_at(frame));rows.append(dict(frame=frame,bbox=out['target_bbox'],score=float(out['best_score']),choice=out['association_candidate'],none=out['association_none']))
        path=outdir/(name+'.json');path.write_text(json.dumps(dict(sequence=name,rows=rows))+'\n')
        item=dict(sequence=name,frames=len(rows),changes=sum(r['choice']!=0 for r in rows),sha256=sha(path),elapsed_seconds=time.time()-started);receipts.append(item);print(json.dumps(item),flush=True)
    check_binding(parent,spec)
    for name,digest in plan['source_sha256'].items():assert sha(repo/name)==digest
    (root/f'shard{args.shard}_receipt.json').write_text(json.dumps(dict(status='complete',sequences=receipts,checkpoint_sha256=sha(checkpoint),spec_sha256=sha(root/'spec.json'),ground_truth_files_opened=False),indent=2)+'\n')


if __name__=='__main__':main()
