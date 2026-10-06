"""Accept a bounded forward replay before reading both completed M113 finals."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path


def phase(root,mode):
    children=[];base=Path('/root/autodl-tmp/sttrack_m113_human_initialization_20261006')
    for gpu,arm in enumerate(['human_text','generic']):
        reference=base/('train_'+arm)
        args=[sys.executable,'-u',str(Path(__file__).with_name('diagnose_m114_text_response.py')),
            '--cache','/root/autodl-tmp/sttrack_m90_train_states_20260928',
            '--contexts','/root/autodl-tmp/sttrack_m98_train_contexts_20260928',
            '--origins','/root/autodl-tmp/sttrack_m95_initial_origins_20260928',
            '--bank',str(base/'human_text.pt'),
            '--labels','/home/SUTrack_RGBD_L/.aris/m113_human_initialization_20261006/human_train_labels.json',
            '--checkpoint',str(reference/'final.pt'),'--reference',str(reference),
            '--output',str(root/(mode+'_'+arm)),'--arm',arm,'--mode',mode]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        with (root/(mode+'_'+arm+'.log')).open('w') as log:
            child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env)
        receipt=dict(pid=child.pid,gpu=gpu,arm=arm,mode=mode,args=args,started=time.time())
        children.append((child,receipt));print(json.dumps(receipt),flush=True)
    for child,r in children:
        r.update(exit=child.wait(),finished=time.time())
        (root/(mode+'_'+r['arm']+'.exit')).write_text(str(r['exit'])+'\n')
    (root/(mode+'_launch.json')).write_text(json.dumps([r for _,r in children],indent=2)+'\n')
    assert all(r['exit']==0 for _,r in children),'Read preserved logs; no retry.'
    for _,r in children:
        result=json.loads((root/(mode+'_'+r['arm'])/'result.json').read_text())
        assert result['status']=='complete_M114_response_'+mode and result['optimizer_steps']==0
        assert result['stored_selection_replay_matches'] and result['weights_buffers_unchanged']
        assert result['replay_scope']==('first_batch_per_split' if mode=='sanity' else 'all_cached_states')
        assert result['empty_exact_visual'] and result['geometry_quality_unchanged']
        assert result['states_per_split']==({'fit':64,'development':64} if mode=='sanity' else {'fit':2544,'development':495})
    return [r for _,r in children]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False);started=time.time()
    sanity=phase(a.output,'sanity');full=phase(a.output,'full')
    result=dict(status='complete_M114_response_pair',optimizer_steps=0,sanity=sanity,full=full,
        started=started,finished=time.time(),elapsed_seconds=time.time()-started,
        no_automatic_training_or_promotion=True)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.output/'driver.exit').write_text('0\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
