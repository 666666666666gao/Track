"""M108 fixed generic/reviewed semantic pair, both GPUs, no automatic promotion."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from analyze_train_states import sha


def phase(root,mode,bank,labels):
    children=[]
    for gpu,arm in enumerate(['generic','weak_text']):
        args=[sys.executable,'-u',str(Path(__file__).with_name('train_m108_frozen_semantics.py')),
              '--cache','/root/autodl-tmp/sttrack_m90_train_states_20260928',
              '--contexts','/root/autodl-tmp/sttrack_m98_train_contexts_20260928',
              '--origins','/root/autodl-tmp/sttrack_m95_initial_origins_20260928',
              '--bank',str(bank),'--labels',str(labels),
              '--parent','/root/autodl-tmp/sttrack_m101_ab_native_pair_20260928/train_weight1/final.pt',
              '--parent-result','/root/autodl-tmp/sttrack_m101_ab_native_pair_20260928/train_weight1/result.json',
              '--output',str(root/(mode+'_'+arm)),'--arm',arm,'--mode',mode]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        with (root/(mode+'_'+arm+'.log')).open('w') as f:
            child=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,env=env)
        receipt=dict(pid=child.pid,gpu=gpu,arm=arm,mode=mode,args=args,started=time.time())
        children.append((child,receipt));print(json.dumps(receipt),flush=True)
    (root/(mode+'_launch.json')).write_text(json.dumps([r for _,r in children],indent=2)+'\n')
    for child,receipt in children:
        receipt.update(exit=child.wait(),finished=time.time())
        (root/(mode+'_'+receipt['arm']+'.exit')).write_text(str(receipt['exit'])+'\n')
    assert all(r['exit']==0 for _,r in children),'Inspect preserved logs; no retry.'
    return [r for _,r in children]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True);a=p.parse_args()
    root=a.output;root.mkdir(exist_ok=False);started=time.time();bank=root/'text.pt'
    prep=[sys.executable,'-u',str(Path(__file__).with_name('prepare_m108_generic.py')),
          '--bank','/root/autodl-tmp/sttrack_m107_model_weak_pair_20260928/weak_text.pt','--output',str(bank)]
    with (root/'prepare.log').open('w') as f:
        code=subprocess.call(prep,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1'))
    (root/'prepare.exit').write_text(str(code)+'\n');assert code==0
    sanity=phase(root,'sanity',bank,a.labels)
    for r in sanity:
        x=json.loads((root/('sanity_'+r['arm'])/'result.json').read_text())
        assert x['status']=='complete_M108_gpu_sanity' and x['optimizer_steps']==3 and not x['checkpoint_saved']
    trained=phase(root,'train',bank,a.labels);results={}
    for r in trained:
        x=json.loads((root/('train_'+r['arm'])/'result.json').read_text())
        assert x['status']=='complete_M108_frozen_semantic_arm' and x['optimizer_steps']==480
        results[r['arm']]={k:x[k] for k in ['fit','development','content_conditions','fixed_state_checks','final_sha256','empty_all3039_scores_quality_exact']}
    receipt=dict(status='complete_M108_frozen_semantic_pair',source_sha256=sha(__file__),
                 labels_sha256=sha(a.labels),bank_sha256=sha(bank),human_confirmed=False,
                 started=started,finished=time.time(),elapsed_seconds=time.time()-started,
                 sanity_launches=sanity,train_launches=trained,arms=results,no_automatic_promotion=True,
                 no_recursive_or_public_evaluation=True)
    (root/'result.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (root/'driver.exit').write_text('0\n')
    print(json.dumps({'status':receipt['status'],'elapsed_seconds':receipt['elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
