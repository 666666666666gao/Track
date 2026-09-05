"""Wait for the original collectors, then verify, fit and recursively evaluate."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root
    spec=json.loads((root/'spec.json').read_text());repo=Path(spec['repository']);sys.path.insert(0,str(repo))
    from tools.train_sttrack_m44 import sha,check_binding
    check_binding(root,spec)
    print(json.dumps(dict(stage='waiting_for_original_collectors',poll_seconds=240)),flush=True)
    while not all((root/f'collect_s{i}.exit').exists() for i in [0,1]):time.sleep(240)
    assert [(root/f'collect_s{i}.exit').read_text().strip() for i in [0,1]]==['0','0']
    receipts=[json.loads((root/f'shard{i}_receipt.json').read_text()) for i in [0,1]]
    assert all(x['status']=='complete' and x['spec_sha256']==sha(root/'spec.json') for x in receipts)
    assert sum(x['events'] for x in receipts)==spec['events']
    print(json.dumps(dict(stage='collection_complete',events=spec['events'])),flush=True)
    usage=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    assert all(int(x.strip())<500 for x in usage.splitlines()),usage
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (root/'runtime_smoke.log').open('w') as log:
        subprocess.run([sys.executable,str(repo/'tools/smoke_sttrack_m44_runtime.py'),'--root',str(root)],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert json.loads((root/'runtime_smoke.json').read_text())['status']=='PASS'
    print(json.dumps(dict(stage='runtime_smoke_pass')),flush=True)
    def pair(script,suffix):
        jobs=[];logs=[]
        for gpu,arm in enumerate(spec['variants']):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');log=(root/(arm+'_'+suffix+'.log')).open('w');logs.append(log)
            job=subprocess.Popen([sys.executable,str(repo/'tools'/script),'--root',str(root),'--variant',arm],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT);jobs.append(job)
        (root/(suffix+'_launch.json')).write_text(json.dumps(dict(pids=[x.pid for x in jobs],started_unix=time.time(),source_binding_sha256=sha(root/'training_binding.json')),indent=2)+'\n')
        codes=[x.wait() for x in jobs]
        for log in logs:log.close()
        assert codes==[0,0],(suffix,codes)
    pair('train_sttrack_m44.py','training')
    results={arm:json.loads((root/(arm+'_result.json')).read_text()) for arm in spec['variants']};a=results['geometry'];b=results['appearance']
    assert all(x['status']=='complete' and x['reload_logits_exact'] for x in results.values())
    assert a['initial_state_sha256']==b['initial_state_sha256'] and a['sample_order_sha256']==b['sample_order_sha256']
    assert a['parameters']==b['parameters'] and a['optimizer_steps']==b['optimizer_steps']
    check_binding(root,spec)
    summary=dict(status='complete',variants=results,spec_sha256=sha(root/'spec.json'),source_binding_sha256=sha(root/'training_binding.json'),
        primary='geometry',matched_initialization_and_sample_order=True,geometry_static_gain_over_appearance=a['development']['mean_iou']-b['development']['mean_iou'],
        next='complete paired recursive development regardless of static ranking; no public automatic launch')
    (root/'training_result.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(stage='training_complete',optimizer_steps_per_arm=a['optimizer_steps'],geometry_static_gain_over_appearance=summary['geometry_static_gain_over_appearance'])),flush=True)
    pair('run_sttrack_m44_recursive.py','recursive')
    with (root/'recursive_analysis.log').open('w') as log:
        subprocess.run([sys.executable,str(repo/'tools/run_sttrack_m44_recursive.py'),'--root',str(root),'--analyze'],cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads((root/'recursive_result.json').read_text());print(json.dumps(dict(status='complete',primary_pass=result['primary_pass'],result_sha256=sha(root/'recursive_result.json'))),flush=True)


if __name__=='__main__':main()
