"""Register a new performance question without changing M42's failed gate."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root
    root.mkdir(parents=True,exist_ok=False)
    previous=Path('/root/autodl-tmp/sttrack_m42_local_spatial_v1_20260905')
    old=json.loads((previous/'spec.json').read_text());training=json.loads((previous/'training_result.json').read_text())
    audit=json.loads((previous/'terminal_audit.json').read_text());assert audit['integrity_pass'] and not training['information_gate_pass']
    stopped=json.loads((previous/'recursive_gate_receipt.json').read_text());assert stopped['GPU_jobs_launched']==0
    a=training['variants']['pooled']['development_holdout'];b=training['variants']['spatial']['development_holdout']
    assert a['mean_iou']>a['default_mean_iou'] and a['rescues']==10 and a['breaks']==0 and a['positive_sequences']==6
    assert [r['chosen'] for r in a['rows']]==[r['chosen'] for r in b['rows']]
    repo=Path(old['repository']);sources=dict(old['source_sha256'])
    for f in ['lib/test/tracker/sttrack_local_spatial.py','tools/analyze_sttrack_m42_recursive.py',
              'tools/run_sttrack_m43.py','tools/prepare_sttrack_m43.py']:
        sources[f]=sha(repo/f)
    cases=[]
    for c in json.loads((previous/'inference_inputs.json').read_text()):
        if c['split']=='development_holdout':
            n=len(list((Path(old['dataset_root'])/c['sequence']/'color').glob('*.jpg')))
            cases.append(dict(name=c['sequence'],frames=n,init_bbox=c['init_bbox']))
    assert len(cases)==22
    base_trace=Path('/root/autodl-tmp/sttrack_innovation_v1/risk_recovery_full152_v1')
    spec=dict(schema='sttrack_m43_pooled_recursive_performance_v1',created_unix=time.time(),repository=str(repo),
        dataset_root=old['dataset_root'],sequences=cases,frames_per_arm=sum(c['frames'] for c in cases),
        source_sha256=sources,base_checkpoint=old['checkpoint'],base_checkpoint_sha256=old['checkpoint_sha256'],
        association_checkpoints={arm:str(previous/(arm+'_final.pth')) for arm in ['pooled','spatial']},
        association_sha256={arm:training['variants'][arm]['checkpoint_sha256'] for arm in ['pooled','spatial']},
        baseline_trace_sha256={str(base_trace/f'shard{i}.json'):sha(base_trace/f'shard{i}.json') for i in [0,1]},
        m42_audit=str(previous/'terminal_audit.json'),m42_audit_sha256=sha(previous/'terminal_audit.json'),
        m42_training_result_sha256=sha(previous/'training_result.json'),
        rationale='M42 local-versus-pooled superiority failed exactly. Both trained heads nevertheless made the same12 choices:10 rescued windows, one joint failure, one slight still-correct regression. Test pooled-control performance in recursion as a new exploratory question; do not relabel M42 as passed.',
        primary='pooled_vs_default',secondary='spatial tracked for diagnosis, without selecting the better arm posthoc',
        optimization_steps=0,threshold_changes=False,new_architecture_changes=False,
        development_scope='same previously used22 Train fold5 sequences; all frames; no fresh-test claim',
        metrics='same fixed M42 recursive metrics: initialization/invalid GT excluded, continuous xywh IoU, severe frames<=.1, episodes>=10 consecutive valid low-IoU frames',
        primary_gate=['mean IoU > default','fewer severe-low-IoU frames','no increase in persistent failure episodes',
                      'positive mean IoU gain on at least3 sequences','no new failure on default-zero-episode sequences'],
        next='Only a pooled primary pass permits freezing low22 against M39 default. No automatic public or full benchmark launch.')
    (root/'spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    print(json.dumps(dict(status='frozen',frames_per_arm=spec['frames_per_arm'],sequences=len(cases),spec_sha256=sha(root/'spec.json'))))


if __name__=='__main__':
    main()
