"""Three matched source-bound arms, actual sanity before fresh complete training."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path


ARMS=['human_text','visual_query','generic']


def phase(root,mode):
    launches=[];reports=[]
    for start in [0,2]:
        children=[]
        for gpu,arm in enumerate(ARMS[start:start+2]):
            command=[sys.executable,'-u',str(Path(__file__).with_name('train_m121_dense_target.py'))]
            locations=dict(cache='/root/autodl-tmp/sttrack_m90_train_states_20260928',
                native='/root/autodl-tmp/sttrack_m120_native_grid_20261007',
                dense='/root/autodl-tmp/sttrack_m117_dense_region_20261007',
                bank='/root/autodl-tmp/sttrack_m113_human_initialization_20261006/human_text.pt',
                labels='/home/SUTrack_RGBD_L/.aris/m113_human_initialization_20261006/human_train_labels.json',
                output=str(root/(mode+'_'+arm)))
            for name,value in locations.items():command+=['--'+name,value]
            command+=['--arm',arm,'--mode',mode]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
            with (root/(mode+'_'+arm+'.log')).open('w') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
            row=dict(arm=arm,gpu=gpu,pid=child.pid,args=command,mode=mode,started=time.time())
            launches.append(row);children.append((child,row))
        path=root/(mode+'_launch.json');path.write_text(json.dumps(launches,indent=2)+'\n')
        for child,row in children:
            row.update(exit=child.wait(),finished=time.time())
            (root/(mode+'_'+row['arm']+'.exit')).write_text(str(row['exit'])+'\n')
        path.write_text(json.dumps(launches,indent=2)+'\n')
        assert all(row['exit']==0 for _,row in children),'Read original logs; no automatic retry.'
        for _,row in children:
            result=json.loads((root/(mode+'_'+row['arm'])/'result.json').read_text())
            assert result['status']=='complete_M121_'+('sanity' if mode=='sanity' else 'training')
            assert result['optimizer_steps']==(3 if mode=='sanity' else 640)
            assert result['arm']==row['arm'] and result['seed']==2027
            assert result['Empty_semantic_delta_same_forward_exact'] and result['geometry_has_no_current_text_input']
            assert result['native_response_zero_initialization_exact'] and result['source']['valid_gt']==3039
            if mode=='sanity':assert result['all_architectural_groups_updated']
            reports.append(result)
    assert len(reports)==3 and len({r['initial_state_sha256'] for r in reports})==1
    assert len({r['optimized_parameters'] for r in reports})==1
    assert all(r['bank_sha256']==reports[0]['bank_sha256'] and r['labels_sha256']==reports[0]['labels_sha256'] for r in reports)
    return launches,reports


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False);started=time.time()
    sanity_launch,sanity=phase(args.output,'sanity')
    full_launch,full=phase(args.output,'train')
    command=[sys.executable,'-u',str(Path(__file__).with_name('analyze_m121_dense_target.py')),
        '--root',str(args.output),'--parent-result','/root/autodl-tmp/sttrack_m101_ab_native_pair_20260928/train_weight1/result.json']
    with (args.output/'comparison.log').open('w') as log:
        analysis=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    (args.output/'comparison.exit').write_text(str(analysis.returncode)+'\n')
    assert analysis.returncode==0,'Read original CPU report log; no automatic retry.'
    comparison=json.loads((args.output/'comparison.json').read_text())
    assert comparison['status']=='complete_M121_CPU_paired_report'
    result=dict(status='complete_M121_dense_target_suite',sanity=sanity,full=full,
        launches=sanity_launch+full_launch,elapsed_seconds=time.time()-started,no_recursive_or_public_evaluation=True,
        comparison=comparison)
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.output/'driver.exit').write_text('0\n')
    print(json.dumps(dict(status=result['status'],seconds=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
