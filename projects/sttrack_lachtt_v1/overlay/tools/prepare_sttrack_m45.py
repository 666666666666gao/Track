"""Freeze and launch the single M45 training-target intervention."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True);args=p.parse_args();root=args.root;parent=args.source_root
    assert not root.exists();root.mkdir()
    source=json.loads((parent/'spec.json').read_text());repo=Path(source['repository']);audit=json.loads((parent/'terminal_audit.json').read_text())
    assert audit['integrity_pass'] and not audit['primary_pass'] and (parent/'controller.exit').read_text().strip()=='0'
    cases=[c for c in json.loads((parent/'inference_inputs.json').read_text()) if c['split']=='development'];assert len(cases)==22
    shards=[[],[]];loads=[0,0]
    for case in sorted(cases,key=lambda c:(-c['frames'],c['sequence'])):
        index=loads.index(min(loads));shards[index].append(case['sequence']);loads[index]+=case['frames']
    for names in shards:names.sort()
    names=['tools/train_sttrack_m45.py','tools/run_sttrack_m45.py','tools/prepare_sttrack_m45.py']
    plan=dict(schema='sttrack_m45_default_priority_targets_v1',source_root=str(parent),source_spec_sha256=sha(parent/'spec.json'),
        control_training_result_sha256=sha(parent/'training_result.json'),control_recursive_result_sha256=sha(parent/'recursive_result.json'),
        control_terminal_audit_sha256=sha(parent/'terminal_audit.json'),trainer_sha256=sha(repo/'tools/train_sttrack_m45.py'),source_sha256={name:sha(repo/name) for name in names},
        primary='default_priority',single_change='Current and previous identity supervision keep candidate0 when its IoU>=.5; otherwise max-IoU if>=.5, else NONE.',
        inference='Unchanged M44 geometry runtime; no GT, new threshold, text, search, query or template change.',
        fit_sequences=63,fit_pairs=1511,development_sequences=22,development_pairs=590,current_fit_labels_changed=121,
        optimization=source['optimization'],initialization='Fresh seed2026 initialization, not additional epochs on M44 weights.',
        control='Reuse sealed M44 geometry with exact matched initial-state and sample-order hashes; same architecture/data/20epochs960steps.',
        parameters=448739,performance_gate=source['recursive_performance_gate'],public_gate=source['public_gate'],
        static_is_diagnostic_only=True,shards=shards,shard_frames=loads,public_automatic_launch=False,
        rationale='M44 has121 fitting windows where a correct default is supervised away for higher instantaneous IoU; no causal proof that this is the sole failure source.',
        next='Complete fixed fitting and all22 recursive sequences. Strong advancement baseline remains STTrack default.')
    (root/'spec.json').write_text(json.dumps(plan,indent=2)+'\n')
    usage=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True);assert all(int(x)<500 for x in usage.splitlines()),usage
    py='/root/autodl-tmp/envs/sttrack/bin/python'
    # Each shell receipt records its actual child exit; the controller never relaunches a failed stage.
    for shard in [0,1]:
        (root/f'recursive_s{shard}.sh').write_text(f'''#!/bin/bash
cd {repo}
CUDA_VISIBLE_DEVICES={shard} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 {py} {repo}/tools/run_sttrack_m45.py --root {root} --shard {shard} > {root}/recursive_s{shard}.log 2>&1 &
job=$!
printf '%s\\n' "$job" > {root}/recursive_s{shard}.pid
wait "$job"
status=$?
printf '%s\\n' "$status" > {root}/recursive_s{shard}.exit
exit "$status"
''')
    pipeline=f'''#!/bin/bash
cd {repo}
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 {py} {repo}/tools/train_sttrack_m45.py --root {root} > {root}/training.log 2>&1 &
trainer=$!
printf '%s\\n' "$trainer" > {root}/training.pid
wait "$trainer"
status=$?
printf '%s\\n' "$status" > {root}/training.exit
if [ "$status" -ne 0 ]; then exit "$status"; fi
bash {root}/recursive_s0.sh &
first=$!
bash {root}/recursive_s1.sh &
second=$!
wait "$first"
first_status=$?
wait "$second"
second_status=$?
if [ "$first_status" -ne 0 ] || [ "$second_status" -ne 0 ]; then exit 1; fi
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' {py} {repo}/tools/run_sttrack_m45.py --root {root} --analyze > {root}/analysis.log 2>&1
'''
    (root/'pipeline.sh').write_text(pipeline)
    (root/'controller.sh').write_text(f'''#!/bin/bash
bash {root}/pipeline.sh &
job=$!
printf '%s\\n' "$job" > {root}/controller.pid
wait "$job"
status=$?
printf '%s\\n' "$status" > {root}/controller.exit
exit "$status"
''')
    screen='sttrack_m45_default_priority_20260905';subprocess.run(['screen','-dmS',screen,'bash',str(root/'controller.sh')],check=True);time.sleep(2)
    pid=int((root/'controller.pid').read_text());os.kill(pid,0)
    (root/'launch.json').write_text(json.dumps(dict(status='started',pid=pid,screen=screen,started_unix=time.time(),spec_sha256=sha(root/'spec.json')),indent=2)+'\n')
    print(json.dumps(dict(status='started',pid=pid,shard_frames=loads,spec_sha256=sha(root/'spec.json')),indent=2))


if __name__=='__main__':main()
