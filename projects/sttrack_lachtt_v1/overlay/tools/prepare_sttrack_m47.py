"""Freeze and launch the M47 auxiliary correspondence destination-set test."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root
    assert not root.exists()
    parent=Path('/root/autodl-tmp/sttrack_m44_candidate_set_v1_20260905')
    control=Path('/root/autodl-tmp/sttrack_m45_default_priority_v1_20260905')
    audit_root=Path('/root/autodl-tmp/sttrack_m47_correspondence_audit_v1_20260905')
    source=json.loads((parent/'spec.json').read_text());repo=Path(source['repository']);sys.path.insert(0,str(repo))
    from tools.train_sttrack_m44 import check_binding
    check_binding(parent,source)
    previous=json.loads((control/'recursive_result.json').read_text());assert previous['integrity_pass'] and not previous['primary_pass']
    assert (control/'analysis_fixed.exit').read_text().strip()=='0'
    control_plan=json.loads((control/'spec.json').read_text())
    for name,source_digest in control_plan['source_sha256'].items():assert sha(repo/name)==source_digest,name
    training=json.loads((control/'geometry_result.json').read_text());assert sha(control/'geometry_final.pth')==training['checkpoint_sha256']
    contract=json.loads((audit_root/'loss_contract.json').read_text());assert contract['status']=='pass' and contract['training_steps']==0
    assert contract['loss_source_sha256']==sha(repo/'lib/models/sttrack/lachtt_candidate_multipositive_loss.py')
    assert contract['test_source_sha256']==sha(repo/'tools/check_sttrack_m47_loss.py')
    assert sha(audit_root/'valid_candidate_multiplicity.json')=='969846496607a1681a75e0c1acae2000801b55846bb973f5d60f5a17ba65ff0e'
    usage=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    assert len(usage.splitlines())==2 and all(int(v)<500 for v in usage.splitlines()),usage
    names=['lib/models/sttrack/lachtt_candidate_multipositive_loss.py','tools/check_sttrack_m47_loss.py',
           'tools/train_sttrack_m47.py','tools/run_sttrack_m47.py','tools/prepare_sttrack_m47.py']
    plan=dict(schema='sttrack_m47_multipositive_correspondence_v1',source_root=str(parent),control_root=str(control),
        source_spec_sha256=sha(parent/'spec.json'),source_training_result_sha256=sha(parent/'training_result.json'),
        source_recursive_result_sha256=sha(parent/'recursive_result.json'),control_training_result_sha256=sha(control/'geometry_result.json'),
        control_recursive_result_sha256=sha(control/'recursive_result.json'),trainer_sha256=sha(repo/'tools/train_sttrack_m47.py'),
        source_sha256={name:sha(repo/name) for name in names},loss_contract_sha256=sha(audit_root/'loss_contract.json'),
        primary='multipositive_correspondence',single_change='Auxiliary partial matching uses probability mass of all opposite-frame IoU>=.5 target boxes; empty set uses unmatched.',
        queries='The same single current and preceding M45 action-target queries are supervised; other real distractor identities remain unknown.',
        action_supervision='Exactly the M45 default-priority labels and action cross-entropy. No new action threshold.',
        inference='Unchanged STTrackCandidateSet geometry runtime, native confidence, NONE/default behavior, query, templates and search. Language OFF.',
        fit_sequences=63,fit_pairs=1511,development_sequences=22,development_pairs=590,
        optimization=source['optimization'],initialization='Fresh seed2026 weights, exactly paired with M45; no warm start.',
        control='Reuse sealed M45: identical action labels, initial state, sample order, data, architecture and20epochs960updates.',
        parameters=448739,matching_loss_weight=.25,positive_iou_threshold=.5,
        performance_gate=control_plan['performance_gate'],public_gate=control_plan['public_gate'],static_is_diagnostic_only=True,
        shards=control_plan['shards'],shard_frames=control_plan['shard_frames'],public_automatic_launch=False,
        matching_metric='Membership in valid destination set; do not compare this to old exact-slot match accuracy.',
        next='Complete fixed fitting and all22 recursive sequences before applying the unchanged STTrack-default advancement gate.')
    root.mkdir();(root/'spec.json').write_text(json.dumps(plan,indent=2)+'\n')
    (root/'loss_contract.json').write_bytes((audit_root/'loss_contract.json').read_bytes())
    py='/root/autodl-tmp/envs/sttrack/bin/python'
    for shard in [0,1]:
        (root/f'recursive_s{shard}.sh').write_text(f'''#!/bin/bash
cd {repo}
CUDA_VISIBLE_DEVICES={shard} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 {py} {repo}/tools/run_sttrack_m47.py --root {root} --shard {shard} > {root}/recursive_s{shard}.log 2>&1 &
job=$!
printf '%s\\n' "$job" > {root}/recursive_s{shard}.pid
wait "$job"
status=$?
printf '%s\\n' "$status" > {root}/recursive_s{shard}.exit
exit "$status"
''')
    (root/'pipeline.sh').write_text(f'''#!/bin/bash
cd {repo}
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 {py} {repo}/tools/train_sttrack_m47.py --root {root} > {root}/training.log 2>&1 &
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
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' {py} {repo}/tools/run_sttrack_m47.py --root {root} --analyze > {root}/analysis.log 2>&1
''')
    (root/'controller.sh').write_text(f'''#!/bin/bash
bash {root}/pipeline.sh &
job=$!
printf '%s\\n' "$job" > {root}/controller.pid
wait "$job"
status=$?
printf '%s\\n' "$status" > {root}/controller.exit
exit "$status"
''')
    screen='sttrack_m47_multipositive_20260905';subprocess.run(['screen','-dmS',screen,'bash',str(root/'controller.sh')],check=True)
    time.sleep(2);pid=int((root/'controller.pid').read_text());os.kill(pid,0)
    launch=dict(status='started',pid=pid,screen=screen,started_unix=time.time(),spec_sha256=sha(root/'spec.json'),shard_frames=plan['shard_frames'])
    (root/'launch.json').write_text(json.dumps(launch,indent=2)+'\n');print(json.dumps(launch,indent=2))


if __name__=='__main__':main()
