"""Two human-text full152 jobs; original actual sanity gates fresh full training."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path


def phase(root,mode):
    children=[];launches=[]
    for weight in [0,1]:
        command=[sys.executable,'-u',str(Path(__file__).with_name('train_m122_full_causal.py'))]
        locations=dict(spec='/root/autodl-tmp/sttrack_full152_paired_20260925/M82/training_spec.json',
            repository='/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1',checkpoint='/root/autodl-tmp/sttrack_checkpoints/STTrack_Vot22.pth.tar',
            **{'clip-weight':'/root/autodl-tmp/sutrack_assets/weights/ViT-L-14.pt'},
            bank='/root/autodl-tmp/sttrack_m113_human_initialization_20261006/human_text.pt',
            labels='/home/SUTrack_RGBD_L/.aris/m113_human_initialization_20261006/human_train_labels.json',
            **{'warm-final':'/root/autodl-tmp/sttrack_m121_dense_target_20261007/train_human_text/final.pt',
                'warm-result':'/root/autodl-tmp/sttrack_m121_dense_target_20261007/train_human_text/result.json',
                'cache-plan':'/root/autodl-tmp/sttrack_m90_train_states_20260928/inference_inputs.json'},
            output=str(root/(mode+'_precision'+str(weight))))
        for key,value in locations.items():command+=['--'+key,value]
        command+=['--mode',mode,'--precision-weight',str(weight)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(weight),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        with (root/(mode+'_precision'+str(weight)+'.log')).open('w') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        row=dict(weight=weight,gpu=weight,pid=child.pid,mode=mode,args=command,started=time.time())
        children.append((child,row));launches.append(row)
    path=root/(mode+'_launch.json');path.write_text(json.dumps(launches,indent=2)+'\n')
    for child,row in children:
        row.update(exit=child.wait(),finished=time.time())
        (root/(mode+'_precision'+str(row['weight'])+'.exit')).write_text(str(row['exit'])+'\n')
    path.write_text(json.dumps(launches,indent=2)+'\n')
    assert all(r['exit']==0 for r in launches),'Read original logs; no automatic restart.'
    reports=[json.loads((root/(mode+'_precision'+str(weight))/'result.json').read_text()) for weight in [0,1]]
    assert len({r['initial_state_sha256'] for r in reports})==1
    assert len({r['bank_sha256'] for r in reports})==1 and len({r['warm_final_sha256'] for r in reports})==1
    assert all(r['seed']==2027 and r['GT_reinitializations_after_first_frame']==0 and r['frozen_before_after_exact'] for r in reports)
    if mode=='sanity':
        assert all(r['status']=='complete_M122_recursive_sanity' and r['native_zero_64frame_trajectory_exact'] and r['optimizer_steps']==2 for r in reports)
    else:
        assert all(r['status']=='complete_M122_full152_training' and r['track_calls']==659406 and r['sequence_runs']==456 for r in reports)
        assert all(r['final_state_roundtrip_exact'] for r in reports)
    return launches,reports


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(exist_ok=False);started=time.time()
    sanity_launch,sanity=phase(args.output,'sanity');full_launch,full=phase(args.output,'train')
    report=dict(status='complete_M122_full152_human_pair',sanity=sanity,full=full,launches=sanity_launch+full_launch,
        elapsed_seconds=time.time()-started,no_public_evaluation_yet=True)
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');(args.output/'driver.exit').write_text('0\n')
    print(json.dumps(dict(status=report['status'],seconds=report['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
