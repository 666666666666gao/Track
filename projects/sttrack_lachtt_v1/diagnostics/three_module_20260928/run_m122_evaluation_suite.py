"""One queued full evaluation; wait for original training, never restart a failed stage."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from prepare_m122_evaluation_suite import checked_sources,read,write


def main(args):
    checked_sources(args.source_gate);args.control.mkdir(exist_ok=False)
    # This passive CPU queue performs no progress/GPU query during the first wait.
    time.sleep(3600)
    while not (args.training_root/'driver.exit').is_file():
        os.kill(args.training_pid,0)
        print(json.dumps(dict(status='waiting_for_original_M122_full_pair',observed=time.time(),next_check_seconds=3600)),flush=True)
        time.sleep(3600)
    assert (args.training_root/'driver.exit').read_text().strip()=='0'
    assert read(args.training_root/'result.json')['status']=='complete_M122_full152_human_pair'
    checked_sources(args.source_gate)
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    resources=[list(map(int,line.split(','))) for line in gpu.strip().splitlines()]
    assert len(resources)==2 and all(r[1]<100 and r[2]==0 for r in resources),gpu
    write(args.control/'training_closed.json',dict(status='actual_original_training_complete_before_evaluation',gpu=gpu,
        original_controller_pid=args.training_pid,observed=time.time()))
    source=Path(__file__).parent;launches=[]
    def run_phase(label,jobs):
        children=[]
        for name,command,device in jobs:
            env=dict(os.environ,PYTHONPATH='/home/SUTrack_RGBD_L',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
            if device is not None:env['CUDA_VISIBLE_DEVICES']=str(device)
            with (args.control/(name+'.log')).open('w') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
            row=dict(stage=label,name=name,pid=child.pid,gpu=device,args=command,started=time.time())
            children.append((child,row));launches.append(row)
        write(args.control/'launches.json',launches)
        for child,row in children:
            row.update(exit=child.wait(),finished=time.time());(args.control/(row['name']+'.exit')).write_text(str(row['exit'])+'\n')
        write(args.control/'launches.json',launches)
        assert all(r['exit']==0 for _,r in children),'Read the original failed stage; no automatic retry/restart.'
    bank=args.control/'official_human_text.pt'
    run_phase('encode',[('encode',[sys.executable,'-u',str(source/'encode_m122_official_human_text.py'),'--binding',str(args.binding),
        '--clip-weight','/root/autodl-tmp/sutrack_assets/weights/ViT-L-14.pt','--output',str(bank)],0)])
    run_phase('prepare',[('prepare',[sys.executable,'-u',str(source/'prepare_m122_evaluation_suite.py'),'--training-root',str(args.training_root),
        '--binding',str(args.binding),'--bank',str(bank),'--source-gate',str(args.source_gate),'--output',str(args.output)],None)])
    selection=read(args.output/'selection.json')
    for mode in ['ope_track','ope_analyze']:
        for dataset in ['depthtrack_test','cdtb']:
            jobs=[]
            for row in selection['models']:
                jobs.append((row['name']+'_'+dataset+'_'+mode,[sys.executable,'-u',str(source/'run_m122_official.py'),'--plan',row['plans'][dataset]['path'],'--mode',mode],row['precision_weight']))
            run_phase(dataset+'_'+mode,jobs)
    for row in selection['models']:
        target=args.output/row['name'];vot=target/'vot';run_phase('vot_track',[(row['name']+'_vot',
            [sys.executable,'-u',str(source/'run_m122_vot_shards.py'),'--root',str(vot/'run')],None)])
        (target/'vot.exit').write_text('0\n')
        manifest=read(vot/'run/shard_manifest.json')
        run_phase('vot_official_analysis',[(row['name']+'_analysis',
            ['/root/miniconda3/envs/mplt/bin/python','-m','vot','analysis','--workspace',str(vot/'run/master'),
                '--format','json','--name',row['name']+'_full127',manifest['tracker']],None)])
    run_phase('vot_metric_seal',[('vot_metric_seal',['/root/miniconda3/envs/mplt/bin/python','-u',str(source/'analyze_m122_vot.py'),'--root',str(args.output)],None)])
    run_phase('collect',[('collect',[sys.executable,'-u',str(source/'collect_m122_official_results.py'),'--root',str(args.output)],None)])
    result=read(args.output/'all_results.json');assert result['status']=='six_M122_full_evaluations_complete'
    write(args.control/'result.json',dict(status='complete_M122_official_full_pair_suite',launches=launches,
        any_joint_pass=result['any_joint_pass'],neural_training_restarted=False,external_optimizer_steps=0))
    (args.control/'driver.exit').write_text('0\n');print(json.dumps(dict(status='complete_M122_official_full_pair_suite',any_joint_pass=result['any_joint_pass'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['training-root','binding','source-gate','output','control']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--training-pid',type=int,required=True);main(p.parse_args())
