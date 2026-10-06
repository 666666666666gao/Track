"""Two useful Train shards, with both sanity stages required before full collection."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path


def phase(root,mode):
    launches=[];children=[]
    for shard in [0,1]:
        args=[sys.executable,'-u',str(Path(__file__).with_name('collect_m117_dense_region.py')),
            '--cache','/root/autodl-tmp/sttrack_m90_train_states_20260928',
            '--full152-spec','/root/autodl-tmp/sttrack_full152_paired_20260925/M82/training_spec.json',
            '--repository','/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1',
            '--bank','/root/autodl-tmp/sttrack_m113_human_initialization_20261006/human_text.pt',
            '--labels','/home/SUTrack_RGBD_L/.aris/m113_human_initialization_20261006/human_train_labels.json',
            '--clip-weight','/root/autodl-tmp/sutrack_assets/weights/ViT-L-14.pt',
            '--baseline','/root/autodl-tmp/sttrack_m116_region_evidence_20261007',
            '--author-model-source','/home/SUTrack_RGBD_L/.aris/m117_dense_region_20261007/clip_surgery_model.py',
            '--output',str(root/(mode+'_shard'+str(shard))),'--mode',mode,'--shard',str(shard)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        with (root/(mode+'_shard'+str(shard)+'.log')).open('w') as log:
            child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env)
        receipt=dict(args=args,pid=child.pid,gpu=shard,mode=mode,started=time.time())
        launches.append(receipt);children.append(child)
    path=root/(mode+'_launch.json');path.write_text(json.dumps(launches,indent=2)+'\n')
    for child,row in zip(children,launches):
        row.update(exit=child.wait(),finished=time.time())
        (root/(mode+'_shard'+str(row['gpu'])+'.exit')).write_text(str(row['exit'])+'\n')
    path.write_text(json.dumps(launches,indent=2)+'\n')
    assert all(r['exit']==0 for r in launches),'Inspect original logs. No automatic retry.'
    results=[json.loads((root/(mode+'_shard'+str(s))/'result.json').read_text()) for s in [0,1]]
    assert all(r['status']=='complete_M117_'+mode+'_shard' and r['model_updates']==0 for r in results)
    assert all(r['frozen_state_exact'] and r['cls_baseline_same_batches_exact'] and not r['GT_loaded'] for r in results)
    if mode=='sanity':
        assert all(r['events']==3 and r['sequences'][0]['author_check']['author_normalized_patch_default_assert_close'] for r in results)
    else:assert sum(r['events'] for r in results)==3502 and sum(len(r['sequences']) for r in results)==152
    return launches,results


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False);started=time.time();launches,smoke=phase(a.output,'sanity')
    full_launch,full=phase(a.output,'full');launches+=full_launch
    receipt=dict(status='complete_M117_region_pair',started=started,finished=time.time(),
        elapsed_seconds=time.time()-started,launches=launches,full=full,sanity=smoke,
        optimizer_steps=0,no_recursive_or_public_evaluation=True)
    (a.output/'result.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (a.output/'driver.exit').write_text('0\n');print(json.dumps(dict(status=receipt['status'],seconds=receipt['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
