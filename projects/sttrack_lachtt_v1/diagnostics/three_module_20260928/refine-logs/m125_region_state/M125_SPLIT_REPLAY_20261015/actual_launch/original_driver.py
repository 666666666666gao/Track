from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,time
root=Path(__file__).resolve().parent
config=json.loads((root/'config.json').read_bytes());plan=config['plan'];source=Path(config['remote_source_dir'])
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
for name,digest in plan['sources'].items():assert sha(source/name)==digest
suite=Path(plan['suite']);suite_before=sha(suite/'result.json')
assert (suite/'controller.exit').read_text().strip()=='0'
results=[];launches=[];started=time.time()
for names in config['waves']:
    children=[]
    for gpu,name in enumerate(names):
        output=root/name
        args=[plan['NN_interpreter'],str(source/'replay_m125_selected_E6_observation.py'),
            '--plan',plan['plan'],'--sequence',name,'--frames']+[str(f) for f in plan['panel'][name]]+['--suite',plan['suite'],'--output',str(output)]
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(source)+':/home/SUTrack_RGBD_L',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with (root/(name+'.NN.log')).open('w') as log:
            child=subprocess.Popen(args,env=environment,stdout=log,stderr=subprocess.STDOUT)
        row=dict(sequence=name,physical_gpu=gpu,pid=child.pid,started_at=datetime.now(timezone.utc).isoformat(),args=args)
        launches.append(row);children.append((child,name,row))
    write(root/'launches.json',launches)
    for child,name,row in children:
        row['NN_exit']=child.wait();row['NN_finished_at']=datetime.now(timezone.utc).isoformat()
        (root/(name+'.NN.exit')).write_text(str(row['NN_exit'])+'\n')
    write(root/'launches.json',launches)
    assert all(row['NN_exit']==0 for _,_,row in children),'Read-only NN stage failed; no automatic retry.'
    for _,name,row in children:
        args=[plan['CPU_interpreter'],str(source/'analyze_m125_selected_E6_observation.py'),'--stage',str(root/name/'prediction_stage.json')]
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(source)+':/home/SUTrack_RGBD_L',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with (root/(name+'.CPU.log')).open('w') as log:result=subprocess.run(args,env=environment,stdout=log,stderr=subprocess.STDOUT)
        row['CPU_exit']=result.returncode;row['CPU_finished_at']=datetime.now(timezone.utc).isoformat()
        (root/(name+'.CPU.exit')).write_text(str(result.returncode)+'\n')
        write(root/'launches.json',launches)
        assert result.returncode==0,'CPU metric stage failed; no NN rerun.'
        value=json.loads((root/name/'result.json').read_bytes())
        assert value['status']=='complete_readonly_M125_selected_E6_CDTB_observation'
        results.append(dict(sequence=name,result_path=str(root/name/'result.json'),sha256=sha(root/name/'result.json'),tracker_calls=value['tracker_calls'],candidate_states=value['candidate_states']))
assert sum(r['tracker_calls'] for r in results)==plan['tracker_calls']
assert sum(r['candidate_states'] for r in results)==plan['candidate_states']
assert sha(suite/'result.json')==suite_before
for name,digest in plan['sources'].items():assert sha(source/name)==digest
write(root/'result.json',dict(status='complete_M125_E6_split_readonly_panel',results=results,tracker_calls=plan['tracker_calls'],candidate_states=plan['candidate_states'],extra_Empty_decoder_calls=plan['extra_Empty_decoder_calls'],optimizer_steps=0,formal_suite_result_sha256=suite_before,formal_suite_unchanged=True,elapsed_seconds=time.time()-started,launches=launches))
(root/'controller.exit').write_text('0\n')
