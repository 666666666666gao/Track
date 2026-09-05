"""Freeze fitting-only frames2-9 and a fixed-budget M46 experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root
    assert not root.exists()
    parent=Path('/root/autodl-tmp/sttrack_m44_candidate_set_v1_20260905')
    control=Path('/root/autodl-tmp/sttrack_m45_default_priority_v1_20260905')
    source=json.loads((parent/'spec.json').read_text());repo=Path(source['repository']);sys.path.insert(0,str(repo))
    from tools.train_sttrack_m44 import check_binding
    check_binding(parent,source)
    result=json.loads((control/'recursive_result.json').read_text());assert result['integrity_pass'] and not result['primary_pass']
    assert (control/'analysis_fixed.exit').read_text().strip()=='0'
    assert json.loads((control/'weight_audit.json').read_text())['status']=='pass'
    control_plan=json.loads((control/'spec.json').read_text())
    for name,digest in control_plan['source_sha256'].items():assert sha(repo/name)==digest,name
    training=json.loads((control/'geometry_result.json').read_text());assert sha(control/'geometry_final.pth')==training['checkpoint_sha256']
    plans=[];labels={};loads=[0,0]
    for original in json.loads((parent/'inference_inputs.json').read_text()):
        if original['split']!='fit':continue
        case=dict(original);assert case['fold'] in source['fit_folds'] and len(case['expected_rows'])>=10
        case['event_frames']=list(range(2,10));case['shard']=loads.index(min(loads));loads[case['shard']]+=9
        gt=np.loadtxt(Path(source['dataset_root'])/case['sequence']/'groundtruth.txt',delimiter=',')
        for frame in case['event_frames']:
            entry=dict(sequence=case['sequence'],fold=case['fold'])
            for key,index in [('current',frame),('previous',frame-1)]:
                value=gt[index];entry[key]=value.tolist() if np.isfinite(value).all() and value[2]>0 and value[3]>0 else None
            labels[f"{case['sequence']}@{frame}"]=entry
        plans.append(case)
    assert len(plans)==63 and len(labels)==504 and sum(loads)==567
    gpu=[int(v) for v in subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()]
    assert len(gpu)==2 and max(gpu)<500,gpu
    free=shutil.disk_usage(root.parent).free;assert free>564400000
    source_hashes={name:sha(repo/name) for name in ['tools/prepare_sttrack_m46.py','tools/train_sttrack_m46.py','tools/run_sttrack_m46.py']}
    root.mkdir();early=root/'early';early.mkdir()
    write(early/'inference_inputs.json',plans);write(early/'training_labels.json',labels)
    early_spec=dict(schema='m46_initial_frame_collection_v1',repository=str(repo),dataset_root=source['dataset_root'],checkpoint=source['checkpoint'],
        checkpoint_sha256=source['checkpoint_sha256'],source_sha256=source['source_sha256'],inference_inputs_sha256=sha(early/'inference_inputs.json'),
        labels_sha256=sha(early/'training_labels.json'),sequences=63,events=504,shard_frames=loads,
        scope='Fitting sequences only; frames2-9, collected on exact native default predicted crops. Collector does not open label file.')
    write(early/'spec.json',early_spec)
    plan=dict(schema='sttrack_m46_initial_frame_coverage_v1',source_root=str(parent),control_root=str(control),
        source_spec_sha256=sha(parent/'spec.json'),source_recursive_result_sha256=sha(parent/'recursive_result.json'),
        control_training_result_sha256=sha(control/'geometry_result.json'),control_recursive_result_sha256=sha(control/'recursive_result.json'),
        early_spec_sha256=sha(early/'spec.json'),source_sha256=source_hashes,primary='initial_frame_coverage',
        single_change='Add fitting-only frame2-9 inputs to the M45 default-priority training pool.',
        inference='Unchanged native candidate-set geometry runtime, NONE/default behavior, query, templates and search. No warmup exclusion or text.',
        fit_sequences=63,old_fit_pairs=1511,early_fit_pairs=504,fit_pairs=2015,development_sequences=22,development_pairs=590,
        optimization=dict(seed=2026,rounds=20,examples_per_round=1511,batch_size=32,lr=.0003,weight_decay=.01,grad_clip=1.,optimizer='AdamW',checkpoint='fixed final only',
            sampler='Fresh seeded permutation of2015fitting pairs per round; take first1511 without replacement. Total960updates, not20full epochs.'),
        initialization='Fresh seed2026 initialization exactly paired with M45; no warm start.',
        control='Reuse sealed M45: same initial weights, architecture, constant optimizer and960updates. Sampling order changes with the input-pool intervention.',
        parameters=448739,performance_gate=control_plan['performance_gate'],public_gate=control_plan['public_gate'],
        static_is_diagnostic_only=True,shards=control_plan['shards'],shard_frames=control_plan['shard_frames'],public_automatic_launch=False,
        rationale='Runtime begins association at2, but all original fitting frames>=10. M45 egg first error at6; this does not establish the cause of later errors.',
        next='Fit fixed model and complete all22recursive sequences before deciding advancement.',initial_free_bytes=free)
    write(root/'spec.json',plan)
    python='/root/autodl-tmp/envs/sttrack/bin/python'
    script=f'''#!/bin/bash
set -u
cd {repo}
echo $$ > {root}/controller.pid
CUDA_VISIBLE_DEVICES=0 {python} tools/collect_sttrack_m44.py --root {early} --shard 0 > {root}/collect_s0.log 2>&1 &
collection0=$!
echo $collection0 > {root}/collect_s0.pid
CUDA_VISIBLE_DEVICES=1 {python} tools/collect_sttrack_m44.py --root {early} --shard 1 > {root}/collect_s1.log 2>&1 &
collection1=$!
echo $collection1 > {root}/collect_s1.pid
wait $collection0
status0=$?
echo $status0 > {early}/collect_s0.exit
wait $collection1
status1=$?
echo $status1 > {early}/collect_s1.exit
if [ $status0 -ne 0 ] || [ $status1 -ne 0 ]; then echo 1 > {root}/controller.exit; exit 1; fi
CUDA_VISIBLE_DEVICES=0 {python} tools/train_sttrack_m46.py --root {root} > {root}/training.log 2>&1 &
training_pid=$!
echo $training_pid > {root}/training.pid
wait $training_pid
training_status=$?
echo $training_status > {root}/training.exit
if [ $training_status -ne 0 ]; then echo 1 > {root}/controller.exit; exit 1; fi
CUDA_VISIBLE_DEVICES=0 {python} tools/run_sttrack_m46.py --root {root} --shard 0 > {root}/recursive_s0.log 2>&1 &
recursive0=$!
echo $recursive0 > {root}/recursive_s0.pid
CUDA_VISIBLE_DEVICES=1 {python} tools/run_sttrack_m46.py --root {root} --shard 1 > {root}/recursive_s1.log 2>&1 &
recursive1=$!
echo $recursive1 > {root}/recursive_s1.pid
wait $recursive0
status0=$?
echo $status0 > {root}/recursive_s0.exit
wait $recursive1
status1=$?
echo $status1 > {root}/recursive_s1.exit
if [ $status0 -ne 0 ] || [ $status1 -ne 0 ]; then echo 1 > {root}/controller.exit; exit 1; fi
CUDA_VISIBLE_DEVICES='' {python} tools/run_sttrack_m46.py --root {root} --analyze > {root}/analysis.log 2>&1
analysis_status=$?
echo $analysis_status > {root}/controller.exit
exit $analysis_status
'''
    (root/'pipeline.sh').write_text(script)
    session='sttrack_m46_initial_frames_20260905'
    subprocess.run(['screen','-dmS',session,'bash',str(root/'pipeline.sh')],check=True)
    launch=dict(status='launched',screen=session,root=str(root),spec_sha256=sha(root/'spec.json'),early_spec_sha256=sha(early/'spec.json'),source_sha256=source_hashes,early_shard_frames=loads,recursive_shard_frames=plan['shard_frames'])
    write(root/'launch.json',launch);print(json.dumps(launch,indent=2))


if __name__=='__main__':main()
