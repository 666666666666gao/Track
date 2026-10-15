from pathlib import Path
import ast,hashlib,json,os,subprocess,zlib

A=Path(__file__).resolve().parent
remote='''from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,zlib
root=Path('/root/autodl-tmp/m125_selected_E6_split_replay_R2_20261015')
assert (root/'controller.exit').read_text().strip()=='0'
result=json.loads((root/'result.json').read_bytes())
assert result['status']=='complete_M125_E6_split_readonly_panel'
config=json.loads((root/'config.json').read_bytes());plan=json.loads(Path(config['plan']['plan']).read_bytes())
files={}
def capture(path):
    path=Path(path);data=path.read_bytes();files[str(path)]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),text=data.decode())
for name in ['config.json','actual_launch.json','launches.json','controller.exit','result.json','driver.py']:capture(root/name)
for row in result['results']:
    name=row['sequence'];folder=root/name
    for filename in ['prediction_stage.json','replayed_predictions.json','result.json']:capture(folder/filename)
    for suffix in ['NN.exit','CPU.exit','NN.log','CPU.log']:capture(root/(name+'.'+suffix))
    capture(Path(plan['dataset_root'])/name/'groundtruth.txt')
    capture(Path(plan['output'])/(name+'.txt'));capture(Path(plan['output'])/(name+'_all_scores.txt'))
for name in config['plan']['sources']:capture(Path(config['remote_source_dir'])/name)
formal=Path(config['plan']['suite'])/'result.json'
assert hashlib.sha256(formal.read_bytes()).hexdigest()==result['formal_suite_result_sha256']
value=dict(status='actual_complete_M125_R2_replay_primary_inputs',observed_at=datetime.now(timezone.utc).isoformat(),files=files,formal_suite_result_sha256=result['formal_suite_result_sha256'],formal_suite_unchanged=True,NN_calls=0,GPU_queries=0)
sys.stdout.buffer.write(zlib.compress(json.dumps(value).encode()));sys.stdout.buffer.flush()
'''
tree=ast.parse((A/'M124_fiftyfirst_hourly_compressed_native_observer_20261014.py').read_text())
args=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and n.targets[0].id=='args'))[:-1]
environment=dict(os.environ,SSH_ASKPASS=str(A/'rgbd_goal_resume_askpass_20261014.cmd'),SSH_ASKPASS_REQUIRE='force',DISPLAY='codex')
assert environment['RGBD_GOAL_SSH_PASSWORD']
stem='M125_R2_complete_replay_primary_inputs_retry1_20261015'
with (A/(stem+'.zlib')).open('xb') as output,(A/(stem+'_stderr.log')).open('xb') as error:
    result=subprocess.run(args+["CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/envs/sttrack/bin/python -B -"],input=remote.encode(),stdout=output,stderr=error,env=environment)
assert result.returncode==0,result.returncode
raw=zlib.decompress((A/(stem+'.zlib')).read_bytes());value=json.loads(raw)
for entry in value['files'].values():assert hashlib.sha256(entry['text'].encode()).hexdigest()==entry['sha256']
with (A/(stem+'.json')).open('xb') as output:output.write(raw)
receipt=dict(status=value['status'],observed_at=value['observed_at'],returncode=0,files=len(value['files']),bytes=len(raw),compressed_bytes=(A/(stem+'.zlib')).stat().st_size,sha256=hashlib.sha256(raw).hexdigest(),NN_calls=0,GPU_queries=0)
with (A/(stem+'_receipt.json')).open('x',encoding='utf-8') as output:output.write(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
