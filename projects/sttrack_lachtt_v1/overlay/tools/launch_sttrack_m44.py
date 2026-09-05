"""Launch the two frozen observation shards after the measured preflight."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root
    spec=json.loads((root/'spec.json').read_text());contracts=json.loads((root/'contracts.json').read_text());smoke=json.loads((root/'smoke_receipt.json').read_text())
    assert contracts['status']=='PASS' and smoke['status']=='complete' and smoke['frames']==120
    assert all(x['max_bbox_error_px']==0 and x['max_score_error']==0 for x in smoke['sequences'])
    usage=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    assert all(int(x.strip())<500 for x in usage.splitlines()),usage
    assert shutil.disk_usage(root).free>spec['estimated_feature_bytes']+1_000_000_000
    py='/root/autodl-tmp/envs/sttrack/bin/python';repo=spec['repository'];screens=[]
    for shard in [0,1]:
        script=root/f'collect_s{shard}.sh';screen=f'sttrack_m44_collect_s{shard}_20260905';screens.append(screen)
        script.write_text(f'''#!/bin/bash
cd {repo}
CUDA_VISIBLE_DEVICES={shard} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 {py} {repo}/tools/collect_sttrack_m44.py --root {root} --shard {shard} > {root}/collect_s{shard}.log 2>&1 &
job=$!
printf '%s\\n' "$job" > {root}/collect_s{shard}.pid
wait "$job"
status=$?
printf '%s\\n' "$status" > {root}/collect_s{shard}.exit
exit "$status"
''')
        subprocess.run(['screen','-dmS',screen,'bash',str(script)],check=True)
    time.sleep(2)
    pids=[int((root/f'collect_s{shard}.pid').read_text()) for shard in [0,1]]
    for pid in pids:os.kill(pid,0)
    launch=dict(status='running',pids=pids,screens=screens,started_unix=time.time(),spec_sha256=hashlib.sha256((root/'spec.json').read_bytes()).hexdigest(),estimated_minutes=60,
                next='Finish and bind training/runtime source while collection runs; no optimizer or public metrics yet.')
    (root/'launch.json').write_text(json.dumps(launch,indent=2)+'\n');print(json.dumps(launch),flush=True)


if __name__=='__main__':main()
