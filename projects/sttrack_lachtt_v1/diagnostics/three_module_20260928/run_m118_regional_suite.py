"""Three matched useful arms on two GPUs, all sanity gates before full training."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path


ARMS = ['human_text', 'visual_query', 'generic']


def phase(root, mode):
    records = []; results = []
    for start in [0, 2]:
        children = []
        for gpu, arm in enumerate(ARMS[start:start+2]):
            args = [sys.executable, '-u', str(Path(__file__).with_name('train_m118_regional_decoder.py'))]
            locations = dict(cache='/root/autodl-tmp/sttrack_m90_train_states_20260928',
                contexts='/root/autodl-tmp/sttrack_m98_train_contexts_20260928',
                origins='/root/autodl-tmp/sttrack_m95_initial_origins_20260928',
                dense='/root/autodl-tmp/sttrack_m117_dense_region_20261007',
                bank='/root/autodl-tmp/sttrack_m113_human_initialization_20261006/human_text.pt',
                labels='/home/SUTrack_RGBD_L/.aris/m113_human_initialization_20261006/human_train_labels.json',
                parent='/root/autodl-tmp/sttrack_m101_ab_native_pair_20260928/train_weight1/final.pt',
                **{'parent-result':'/root/autodl-tmp/sttrack_m101_ab_native_pair_20260928/train_weight1/result.json'},
                output=str(root/(mode+'_'+arm)))
            for key, value in locations.items(): args += ['--'+key, value]
            args += ['--arm', arm, '--mode', mode]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
            with (root/(mode+'_'+arm+'.log')).open('w') as stream:
                process = subprocess.Popen(args, stdout=stream, stderr=subprocess.STDOUT, env=env)
            record = dict(arm=arm,mode=mode,gpu=gpu,pid=process.pid,args=args,started=time.time())
            records.append(record); children.append((process,record))
        (root/(mode+'_launch.json')).write_text(json.dumps(records,indent=2)+'\n')
        for process, record in children:
            record.update(exit=process.wait(),finished=time.time())
            (root/(mode+'_'+record['arm']+'.exit')).write_text(str(record['exit'])+'\n')
        (root/(mode+'_launch.json')).write_text(json.dumps(records,indent=2)+'\n')
        assert all(row['exit']==0 for _,row in children), 'Inspect original logs; no automatic retry.'
        for _, record in children:
            result=json.loads((root/(mode+'_'+record['arm'])/'result.json').read_text())
            assert result['arm']==record['arm'] and result['optimized_parameters']==150528
            assert result['optimizer_steps']==(3 if mode=='sanity' else 480)
            assert result['empty_scores_quality_index_box_all3039_exact'] and result['buffers_exact']
            assert result['coordinates']['original_M117_region_ring_reader_default_close']
            results.append(result)
    assert len({r['initial_state_sha256'] for r in results})==1
    assert all(r['bank_sha256']==results[0]['bank_sha256'] and r['parent_sha256']==results[0]['parent_sha256'] for r in results)
    return records, results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False);started=time.time()
    sanity_launch, sanity=phase(args.output,'sanity')
    full_launch, full=phase(args.output,'train')
    result=dict(status='complete_M118_regional_suite',started=started,finished=time.time(),
        elapsed_seconds=time.time()-started,launches=sanity_launch+full_launch,sanity=sanity,full=full,
        no_recursive_or_public_evaluation=True,all_three_matched_initial_states=True)
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.output/'driver.exit').write_text('0\n')
    print(json.dumps(dict(status=result['status'],seconds=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
