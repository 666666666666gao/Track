"""Wait for the authorized M42 training; run frozen recursive tests only on pass."""
import argparse
import hashlib
import json
from pathlib import Path
import os
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); args=p.parse_args(); root=args.root
    plan=json.loads((root/'recursive_spec.json').read_text()); spec=json.loads((root/'spec.json').read_text())
    repo=Path(spec['repository']); python='/root/autodl-tmp/envs/sttrack/bin/python'
    for name,digest in plan['source_sha256'].items(): assert sha(repo/name)==digest,name
    while not (root/'controller.exit').exists(): time.sleep(240)
    assert (root/'controller.exit').read_text().strip()=='0','M42 collection/fitting failed'
    training=json.loads((root/'training_result.json').read_text())
    assert training['status']=='complete'
    launch=json.loads((root/'launch.json').read_text())
    assert training['trainer_sha256']==launch['source_sha256']['tools/train_sttrack_m42.py']
    if not training['information_gate_pass']:
        (root/'recursive_gate_receipt.json').write_text(json.dumps(dict(status='not_launched_information_gate_failed',
            training_result_sha256=sha(root/'training_result.json'),GPU_jobs_launched=0),indent=2)+'\n')
        print('M42 static information gate failed; recursive tracking and public benchmarks not launched.',flush=True)
        return
    usage=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    assert all(int(x.strip())<500 for x in usage.splitlines()),usage
    jobs=[]; logs=[]
    for gpu,arm in enumerate(['spatial','pooled']):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        log=(root/(arm+'_recursive.log')).open('w'); logs.append(log)
        job=subprocess.Popen([python,'tools/run_sttrack_m42_recursive.py','--root',str(root),'--variant',arm],
                             cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
        jobs.append(job)
    (root/'recursive_launch.json').write_text(json.dumps(dict(pids=[j.pid for j in jobs],started_unix=time.time(),
        recursive_spec_sha256=sha(root/'recursive_spec.json'),training_result_sha256=sha(root/'training_result.json')),indent=2)+'\n')
    codes=[j.wait() for j in jobs]
    for log in logs: log.close()
    assert codes==[0,0],codes
    subprocess.run([python,'tools/analyze_sttrack_m42_recursive.py','--root',str(root)],cwd=repo,check=True)


if __name__=='__main__':
    main()
