"""Wait for both fixed training arms, then run only the frozen development pair."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

root=Path('/root/autodl-tmp/sttrack_m58_semantic_spatial_v1_20260906')
spec=json.loads((root/'recursive_spec.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(__file__)==spec['queue_sha256']
assert sha(root/'run_recursive.py')==spec['runner_sha256']
print('WAIT_FOR_BOTH_FIXED_TRAINING_ARMS',datetime.now(timezone.utc).isoformat(),flush=True)
while not all((root/('training_'+arm+'.exit')).exists() for arm in ['text','visual']):
    for arm in ['text','visual']:
        path=root/('training_'+arm+'.exit')
        if path.exists(): assert path.read_text().strip()=='0',arm
    time.sleep(240)
assert all((root/('training_'+arm+'.exit')).read_text().strip()=='0' for arm in ['text','visual'])
from run_recursive import trained
trained()
memory=[int(x) for x in subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits']).decode().splitlines()]
assert len(memory)==2 and all(v<500 for v in memory),memory
launch=[]
for gpu,arm in [(0,'text'),(1,'visual')]:
    script=root/('run_recursive_'+arm+'.sh')
    script.write_text('#!/bin/bash\ncd '+str(root)+'\nCUDA_VISIBLE_DEVICES='+str(gpu)+
        ' /root/autodl-tmp/envs/sttrack/bin/python -u run_recursive.py --arm '+arm+
        ' > '+arm+'_recursive.log 2>&1\nstatus=$?\nprintf "%s\\n" "$status" > '+arm+'_recursive.exit\nexit "$status"\n')
    screen='m58_rec_'+arm+'_20260906'
    subprocess.run(['screen','-dmS',screen,'bash',str(script)],check=True)
    launch.append(dict(arm=arm,gpu=gpu,screen=screen,gpu_memory_mib_before=memory[gpu]))
(root/'recursive_launch.json').write_text(json.dumps(dict(started_utc=datetime.now(timezone.utc).isoformat(),
    recursive_spec_sha256=sha(root/'recursive_spec.json'),runs=launch),indent=2)+'\n')
print('DEVELOPMENT_PAIR_LAUNCHED',json.dumps(launch),flush=True)
while not all((root/(arm+'_recursive.exit')).exists() for arm in ['text','visual']):
    for arm in ['text','visual']:
        path=root/(arm+'_recursive.exit')
        if path.exists():assert path.read_text().strip()=='0',arm
    time.sleep(240)
assert all((root/(arm+'_recursive.exit')).read_text().strip()=='0' for arm in ['text','visual'])
with (root/'recursive_analysis.log').open('x') as log:
    completed=subprocess.run(['/root/autodl-tmp/envs/sttrack/bin/python',str(root/'run_recursive.py'),'--analyze'],
        cwd=str(root),stdout=log,stderr=subprocess.STDOUT)
(root/'recursive_analysis.exit').write_text(str(completed.returncode)+'\n')
assert completed.returncode==0
print('DEVELOPMENT_COMPLETE_NO_PUBLIC_EVALUATION_LAUNCHED',flush=True)
