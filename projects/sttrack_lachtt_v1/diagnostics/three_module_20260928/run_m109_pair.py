"""M109 two-GPU read-only diagnostics; one execution, no retry or training."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from analyze_train_states import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True);a=p.parse_args()
    root=a.output;root.mkdir(exist_ok=False);started=time.time();children=[]
    for gpu,arm in enumerate(['generic','weak_text']):
        args=[sys.executable,'-u',str(Path(__file__).with_name('diagnose_m109_training_gradients.py')),
              '--cache','/root/autodl-tmp/sttrack_m90_train_states_20260928',
              '--contexts','/root/autodl-tmp/sttrack_m98_train_contexts_20260928',
              '--origins','/root/autodl-tmp/sttrack_m95_initial_origins_20260928',
              '--bank','/root/autodl-tmp/sttrack_m108_frozen_semantic_pair_20261002/text.pt',
              '--labels',str(a.labels),
              '--parent','/root/autodl-tmp/sttrack_m101_ab_native_pair_20260928/train_weight1/final.pt',
              '--completed','/root/autodl-tmp/sttrack_m108_frozen_semantic_pair_20261002',
              '--output',str(root/arm),'--arm',arm]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        with (root/(arm+'.log')).open('w') as f:
            child=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,env=env)
        receipt=dict(pid=child.pid,gpu=gpu,arm=arm,args=args,started=time.time())
        children.append((child,receipt))
    (root/'launch.json').write_text(json.dumps([r for _,r in children],indent=2)+'\n')
    for child,r in children:
        r.update(exit=child.wait(),finished=time.time())
        (root/(r['arm']+'.exit')).write_text(str(r['exit'])+'\n')
    assert all(r['exit']==0 for _,r in children),'Inspect preserved logs; no retry.'
    result=dict(status='complete_M109_two_gpu_readonly_diagnostic',source_sha256=sha(__file__),
                arms={r['arm']:json.loads((root/r['arm']/'result.json').read_text()) for _,r in children},
                launches=[r for _,r in children],elapsed_seconds=time.time()-started)
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'elapsed_seconds':result['elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
