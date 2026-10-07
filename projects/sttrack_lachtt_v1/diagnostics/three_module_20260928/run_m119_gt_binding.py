"""Two cached-Train shards; actual sanity completion precedes full reads."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path


def phase(root, mode):
    launches = []; children = []
    for shard in [0,1]:
        args = [sys.executable, '-u', str(Path(__file__).with_name('collect_m119_gt_binding.py')),
            '--cache', '/root/autodl-tmp/sttrack_m90_train_states_20260928',
            '--dense', '/root/autodl-tmp/sttrack_m117_dense_region_20261007',
            '--bank', '/root/autodl-tmp/sttrack_m113_human_initialization_20261006/human_text.pt',
            '--labels', '/home/SUTrack_RGBD_L/.aris/m113_human_initialization_20261006/human_train_labels.json',
            '--parent-rows', '/root/autodl-tmp/sttrack_m118_regional_decoder_20261007/train_human_text',
            '--output', str(root/(mode+'_shard'+str(shard))), '--shard', str(shard), '--mode', mode]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        with (root/(mode+'_shard'+str(shard)+'.log')).open('w') as log:
            child = subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env)
        launches.append(dict(pid=child.pid,args=args,gpu=shard,mode=mode,started=time.time()));children.append(child)
    path=root/(mode+'_launch.json');path.write_text(json.dumps(launches,indent=2)+'\n')
    for child, row in zip(children,launches):
        row.update(exit=child.wait(),finished=time.time())
        (root/(mode+'_shard'+str(row['gpu'])+'.exit')).write_text(str(row['exit'])+'\n')
    path.write_text(json.dumps(launches,indent=2)+'\n')
    assert all(r['exit']==0 for r in launches), 'Inspect original logs; no automatic retry.'
    reports=[json.loads((root/(mode+'_shard'+str(shard))/'result.json').read_text()) for shard in [0,1]]
    assert all(r['status']=='complete_M119_'+mode+'_shard' and r['optimizer_steps']==0 for r in reports)
    assert all(not r['model_forward_executed'] and not r['tracker_state_committed'] for r in reports)
    if mode=='sanity':
        assert all(r['events']==3 and all(x['original_M117_reader_default_close'] for x in r['sequences']) for r in reports)
    else:
        assert sum(r['events'] for r in reports)==3502 and sum(r['valid_gt'] for r in reports)==3039
        assert sum(len(r['sequences']) for r in reports)==152
    return launches, reports


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(exist_ok=False);started=time.time()
    launches, sanity=phase(args.output,'sanity');full_launches,full=phase(args.output,'full')
    result=dict(status='complete_M119_GT_binding_pair',launches=launches+full_launches,sanity=sanity,full=full,
        elapsed_seconds=time.time()-started,optimizer_steps=0,checkpoints_created=False,no_public_evaluation=True)
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.output/'driver.exit').write_text('0\n');print(json.dumps(dict(status=result['status'],seconds=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':
    main()
