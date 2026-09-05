"""Prospective recursive test of M42's trained pooled control versus default."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_sources(spec):
    repo=Path(spec['repository'])
    for name,digest in spec['source_sha256'].items():assert sha(repo/name)==digest,name
    assert sha(spec['base_checkpoint'])==spec['base_checkpoint_sha256']
    for arm,path in spec['association_checkpoints'].items():assert sha(path)==spec['association_sha256'][arm]


def run_arm(root,spec,arm):
    import torch
    sys.path.insert(0,spec['repository'])
    from lib.config.sttrack.config import cfg,update_config_from_file
    from lib.test.tracker.sttrack_local_spatial import STTrackLocalSpatial
    from lib.train.dataset.depth_utils import get_rgbd_frame
    torch.set_num_threads(1)
    update_config_from_file(str(Path(spec['repository'])/'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    params=SimpleNamespace(cfg=cfg,checkpoint=spec['base_checkpoint'],template_factor=2.,template_size=128,
                            search_factor=4.,search_size=256,save_all_boxes=False,debug=0)
    tracker=STTrackLocalSpatial(params,spec['association_checkpoints'][arm])
    outdir=root/'trajectories';outdir.mkdir(exist_ok=True)
    receipts=[];start=time.time()
    for case in spec['sequences']:
        folder=Path(spec['dataset_root'])/case['name']
        def image_at(frame):
            return get_rgbd_frame(str(folder/'color'/f'{frame+1:08d}.jpg'),str(folder/'depth'/f'{frame+1:08d}.png'),
                                  dtype='rgbcolormap',depth_clip=True)
        tracker.initialize(image_at(0),dict(init_bbox=list(case['init_bbox'])))
        rows=[dict(frame=0,bbox=case['init_bbox'],score=1.,choice=0,none=False)]
        for frame in range(1,case['frames']):
            out=tracker.track(image_at(frame))
            rows.append(dict(frame=frame,bbox=out['target_bbox'],score=float(out['best_score']),
                             choice=out['association_candidate'],none=out['association_none']))
        path=outdir/(arm+'_'+case['name']+'.json')
        path.write_text(json.dumps(dict(sequence=case['name'],arm=arm,rows=rows))+'\n')
        item=dict(sequence=case['name'],frames=len(rows),changes=sum(r['choice']!=0 for r in rows),
                  sha256=sha(path),elapsed_seconds=time.time()-start)
        receipts.append(item);print(json.dumps(item),flush=True)
    check_sources(spec)
    (root/(arm+'_receipt.json')).write_text(json.dumps(dict(status='complete',sequences=receipts,
        elapsed_seconds=time.time()-start,association_sha256=spec['association_sha256'][arm],
        source_unchanged=True,ground_truth_files_opened=False),indent=2)+'\n')


def analyze(root,spec):
    import numpy as np
    sys.path.insert(0,spec['repository'])
    from tools.analyze_sttrack_m42_recursive import statistics
    names={c['name'] for c in spec['sequences']};baseline=defaultdict(list)
    for path,digest in spec['baseline_trace_sha256'].items():
        assert sha(path)==digest
        for row in json.loads(Path(path).read_text())['rows']:
            if row['sequence'] in names:baseline[row['sequence']].append(row)
    by_arm={arm:{} for arm in ['default','pooled','spatial']}
    for case in spec['sequences']:
        name=case['name'];gt=np.loadtxt(Path(spec['dataset_root'])/name/'groundtruth.txt',delimiter=',')
        rows=sorted(baseline[name],key=lambda r:r['frame_index']);assert len(rows)==case['frames']
        by_arm['default'][name]=statistics([r['public_bbox'] for r in rows],gt)
    for arm in ['pooled','spatial']:
        receipt=json.loads((root/(arm+'_receipt.json')).read_text());assert receipt['status']=='complete'
        assert {r['sequence'] for r in receipt['sequences']}==names
        for item in receipt['sequences']:
            name=item['sequence'];path=root/'trajectories'/(arm+'_'+name+'.json');assert sha(path)==item['sha256']
            rows=json.loads(path.read_text())['rows'];assert len(rows)==len(baseline[name])
            gt=np.loadtxt(Path(spec['dataset_root'])/name/'groundtruth.txt',delimiter=',')
            by_arm[arm][name]=statistics([r['bbox'] for r in rows],gt)
            by_arm[arm][name]['changes']=item['changes']
    aggregates={}
    for arm,rows in by_arm.items():
        totals={k:sum(r[k] for r in rows.values()) for k in ['valid_frames','iou_sum','low_iou_frames','failure_episodes']}
        totals['mean_iou']=totals['iou_sum']/totals['valid_frames']
        totals['macro_sequence_mean_iou']=float(np.mean([r['mean_iou'] for r in rows.values()]))
        totals['failure_sequences']=sum(r['failure_episodes']>0 for r in rows.values())
        aggregates[arm]=totals
    comparisons={}
    for arm in ['pooled','spatial']:
        candidate,base=aggregates[arm],aggregates['default']
        positives=sum(by_arm[arm][n]['mean_iou']>by_arm['default'][n]['mean_iou'] for n in names)
        broken=[n for n in sorted(names) if by_arm['default'][n]['failure_episodes']==0 and by_arm[arm][n]['failure_episodes']>0]
        gates=dict(mean_iou=candidate['mean_iou']>base['mean_iou'],
            fewer_low_iou_frames=candidate['low_iou_frames']<base['low_iou_frames'],
            no_episode_increase=candidate['failure_episodes']<=base['failure_episodes'],
            sequence_coverage=positives>=3,successful_sequence_protection=len(broken)==0)
        comparisons[arm]=dict(gates=gates,pass_gate=all(gates.values()),positive_sequences=positives,new_failure_sequences=broken)
    result=dict(status='complete',primary='pooled_vs_default',secondary='spatial_vs_default_no_posthoc_model_switch',
        aggregates=aggregates,per_sequence=by_arm,comparisons=comparisons,
        primary_pass=comparisons['pooled']['pass_gate'],spec_sha256=sha(root/'spec.json'),
        claim='Exploratory follow-up to the useful trained pooled control. M42 local-information superiority remains failed. Train development proxies only.',
        next='freeze low22 pooled comparison' if comparisons['pooled']['pass_gate'] else 'no public launch; inspect actual recursive failure evidence')
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(aggregates=aggregates,comparisons=comparisons)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',choices=['pooled','spatial'])
    args=p.parse_args();root=args.root;spec=json.loads((root/'spec.json').read_text());check_sources(spec)
    if args.arm:
        run_arm(root,spec,args.arm);return
    audit=json.loads(Path(spec['m42_audit']).read_text());assert audit['integrity_pass']
    assert sha(spec['m42_audit'])==spec['m42_audit_sha256']
    assert not audit['information_gate_pass'],'M42 outcome differs from the declared M43 rationale'
    usage=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    assert all(int(x.strip())<500 for x in usage.splitlines()),usage
    jobs=[];logs=[]
    for gpu,arm in enumerate(['pooled','spatial']):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        log=(root/(arm+'.log')).open('w');logs.append(log)
        job=subprocess.Popen([sys.executable,__file__,'--root',str(root),'--arm',arm],cwd=spec['repository'],env=env,stdout=log,stderr=subprocess.STDOUT)
        jobs.append(job)
    (root/'launch.json').write_text(json.dumps(dict(pids=[j.pid for j in jobs],started_unix=time.time(),spec_sha256=sha(root/'spec.json')),indent=2)+'\n')
    codes=[job.wait() for job in jobs]
    for log in logs:log.close()
    assert codes==[0,0],codes
    analyze(root,spec)


if __name__=='__main__':
    main()
