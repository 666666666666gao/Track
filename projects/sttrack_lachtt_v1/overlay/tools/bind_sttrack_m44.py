"""Bind the completed training/runtime implementation before optimization."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root
    spec=json.loads((root/'spec.json').read_text());repo=Path(spec['repository'])
    for name,digest in spec['source_sha256'].items():assert sha(repo/name)==digest,name
    assert not any((root/(arm+'_final.pth')).exists() for arm in spec['variants'])
    names=['lib/test/tracker/sttrack_candidate_set.py','tools/train_sttrack_m44.py','tools/smoke_sttrack_m44_runtime.py',
        'tools/run_sttrack_m44_recursive.py','tools/run_sttrack_m44_pipeline.py','tools/bind_sttrack_m44.py',
        'tools/train_sttrack_m42.py','tools/analyze_sttrack_m42_recursive.py']
    for name in names:ast.parse((repo/name).read_text())
    cases=json.loads((root/'inference_inputs.json').read_text());fit=sum(len(x['event_frames']) for x in cases if x['split']=='fit');dev=sum(len(x['event_frames']) for x in cases if x['split']=='development')
    binding=dict(status='frozen_before_first_optimizer',bound_unix=time.time(),spec_sha256=sha(root/'spec.json'),source_sha256={name:sha(repo/name) for name in names},
        fit_events=fit,development_events=dev,optimizer_steps_per_arm=math.ceil(fit/spec['optimization']['batch_size'])*spec['optimization']['epochs'],
        recursive_sequences=22,recursive_frames_per_arm=sum(x['frames'] for x in cases if x['split']=='development'),
        primary='geometry',static_is_diagnostic_only=True,public_automatic_launch=False)
    (root/'training_binding.json').write_text(json.dumps(binding,indent=2)+'\n')
    script=root/'pipeline.sh';py='/root/autodl-tmp/envs/sttrack/bin/python'
    script.write_text(f'''#!/bin/bash
cd {repo}
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 {py} {repo}/tools/run_sttrack_m44_pipeline.py --root {root} > {root}/pipeline.log 2>&1 &
job=$!
printf '%s\\n' "$job" > {root}/pipeline.pid
wait "$job"
status=$?
printf '%s\\n' "$status" > {root}/controller.exit
exit "$status"
''')
    screen='sttrack_m44_pipeline_20260905';subprocess.run(['screen','-dmS',screen,'bash',str(script)],check=True);time.sleep(2)
    pid=int((root/'pipeline.pid').read_text());os.kill(pid,0)
    launch=dict(status='started',pid=pid,screen=screen,binding_sha256=sha(root/'training_binding.json'),started_unix=time.time(),
        collector_pids=json.loads((root/'launch.json').read_text())['pids'])
    (root/'pipeline_launch.json').write_text(json.dumps(launch,indent=2)+'\n');print(json.dumps(dict(binding=binding,launch=launch)),flush=True)


if __name__=='__main__':main()
