"""Two-card human objective pair, followed by the matched generic control."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path


def phase(root,mode,arms):
    children=[]
    for gpu,arm in enumerate(arms):
        args=[sys.executable,'-u',str(Path(__file__).with_name('train_m115_semantic_choice.py')),
            '--cache','/root/autodl-tmp/sttrack_m90_train_states_20260928',
            '--contexts','/root/autodl-tmp/sttrack_m98_train_contexts_20260928',
            '--origins','/root/autodl-tmp/sttrack_m95_initial_origins_20260928',
            '--bank','/root/autodl-tmp/sttrack_m113_human_initialization_20261006/human_text.pt',
            '--labels','/home/SUTrack_RGBD_L/.aris/m113_human_initialization_20261006/human_train_labels.json',
            '--parent','/root/autodl-tmp/sttrack_m101_ab_native_pair_20260928/train_weight1/final.pt',
            '--parent-result','/root/autodl-tmp/sttrack_m101_ab_native_pair_20260928/train_weight1/result.json',
            '--output',str(root/(mode+'_'+arm)),'--arm',arm,'--mode',mode]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        with (root/(mode+'_'+arm+'.log')).open('w') as log:
            child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env)
        receipt=dict(pid=child.pid,gpu=gpu,arm=arm,mode=mode,args=args,started=time.time())
        children.append((child,receipt));print(json.dumps(receipt),flush=True)
    group='human_pair' if len(arms)==2 else 'generic_control'
    launch=root/(mode+'_'+group+'_launch.json')
    launch.write_text(json.dumps([r for _,r in children],indent=2)+'\n')
    for child,receipt in children:
        receipt.update(exit=child.wait(),finished=time.time())
        (root/(mode+'_'+receipt['arm']+'.exit')).write_text(str(receipt['exit'])+'\n')
    launch.write_text(json.dumps([r for _,r in children],indent=2)+'\n')
    assert all(r['exit']==0 for _,r in children),'Inspect preserved logs; no retry.'
    for _,r in children:
        result=json.loads((root/(mode+'_'+r['arm'])/'result.json').read_text())
        assert result['status']==('complete_M115_gpu_sanity' if mode=='sanity' else 'complete_M115_semantic_choice_arm')
        assert result['optimizer_steps']==(3 if mode=='sanity' else 480)
        assert all(result[k] for k in ['empty_all3039_scores_quality_exact','frozen_parameters_buffers_exact','nonempty_quality_exact','selected_fields_consistent'])
    return [r for _,r in children]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.output;root.mkdir(exist_ok=False);started=time.time();launches=[]
    for arms in [['human_bce','human_choice'],['generic_choice']]:
        launches+=phase(root,'sanity',arms)
        launches+=phase(root,'train',arms)
    results={}
    for arm in ['human_bce','human_choice','generic_choice']:
        result=json.loads((root/('train_'+arm)/'result.json').read_text())
        results[arm]={k:result[k] for k in ['fit','development','content_conditions','paired_vs_own_empty','final_sha256']}
    receipt=dict(status='complete_M115_semantic_choice_suite',human_confirmed=True,
        started=started,finished=time.time(),elapsed_seconds=time.time()-started,
        launches=launches,arms=results,no_automatic_promotion=True,no_recursive_or_public_evaluation=True)
    (root/'result.json').write_text(json.dumps(receipt,indent=2)+'\n');(root/'driver.exit').write_text('0\n')
    print(json.dumps(dict(status=receipt['status'],elapsed_seconds=receipt['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
