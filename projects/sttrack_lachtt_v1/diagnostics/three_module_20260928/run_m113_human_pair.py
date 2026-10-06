"""Encode the human Train bank, accept two-card sanity, then train the fixed pair."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path


def phase(root,mode,bank,labels):
    children=[]
    for gpu,arm in enumerate(['human_text','generic']):
        args=[sys.executable,'-u',str(Path(__file__).with_name('train_m113_human_initialization.py')),
            '--cache','/root/autodl-tmp/sttrack_m90_train_states_20260928',
            '--contexts','/root/autodl-tmp/sttrack_m98_train_contexts_20260928',
            '--origins','/root/autodl-tmp/sttrack_m95_initial_origins_20260928',
            '--bank',str(bank),'--labels',str(labels),
            '--parent','/root/autodl-tmp/sttrack_m101_ab_native_pair_20260928/train_weight1/final.pt',
            '--parent-result','/root/autodl-tmp/sttrack_m101_ab_native_pair_20260928/train_weight1/result.json',
            '--output',str(root/(mode+'_'+arm)),'--arm',arm,'--mode',mode]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        with (root/(mode+'_'+arm+'.log')).open('w') as log:
            child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env)
        receipt=dict(pid=child.pid,gpu=gpu,arm=arm,mode=mode,args=args,started=time.time())
        children.append((child,receipt));print(json.dumps(receipt),flush=True)
    (root/(mode+'_launch.json')).write_text(json.dumps([r for _,r in children],indent=2)+'\n')
    for child,receipt in children:
        receipt.update(exit=child.wait(),finished=time.time())
        (root/(mode+'_'+receipt['arm']+'.exit')).write_text(str(receipt['exit'])+'\n')
    (root/(mode+'_launch.json')).write_text(json.dumps([r for _,r in children],indent=2)+'\n')
    assert all(r['exit']==0 for _,r in children),'Inspect preserved logs; no retry.'
    return [r for _,r in children]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True);p.add_argument('--labels',type=Path,required=True)
    a=p.parse_args();root=a.output;root.mkdir(exist_ok=False);started=time.time();bank=root/'human_text.pt'
    args=[sys.executable,'-u',str(Path(__file__).with_name('prepare_m113_human_text.py')),'--labels',str(a.labels),'--output',str(bank)]
    with (root/'prepare.log').open('w') as log:
        code=subprocess.call(args,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1'))
    (root/'prepare.exit').write_text(str(code)+'\n');assert code==0,'Inspect prepare.log; no training or retry.'
    sanity=phase(root,'sanity',bank,a.labels)
    for r in sanity:
        result=json.loads((root/('sanity_'+r['arm'])/'result.json').read_text())
        assert result['status']=='complete_M113_gpu_sanity' and result['optimizer_steps']==3
        assert result['empty_all3039_scores_quality_exact'] and result['frozen_parameters_buffers_exact']
        assert not result['checkpoint_saved']
    trained=phase(root,'train',bank,a.labels);results={}
    for r in trained:
        result=json.loads((root/('train_'+r['arm'])/'result.json').read_text())
        assert result['status']=='complete_M113_human_initialization_arm' and result['optimizer_steps']==480
        assert result['empty_all3039_scores_quality_exact'] and result['frozen_parameters_buffers_exact']
        results[r['arm']]={k:result[k] for k in ['fit','development','content_conditions','paired_vs_own_empty','fixed_state_checks','final_sha256']}
    receipt=dict(status='complete_M113_human_initialization_pair',human_confirmed=True,
        started=started,finished=time.time(),elapsed_seconds=time.time()-started,
        sanity_launches=sanity,train_launches=trained,arms=results,no_automatic_promotion=True,
        no_recursive_or_public_evaluation=True)
    (root/'result.json').write_text(json.dumps(receipt,indent=2)+'\n');(root/'driver.exit').write_text('0\n')
    print(json.dumps(dict(status=receipt['status'],elapsed_seconds=receipt['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
