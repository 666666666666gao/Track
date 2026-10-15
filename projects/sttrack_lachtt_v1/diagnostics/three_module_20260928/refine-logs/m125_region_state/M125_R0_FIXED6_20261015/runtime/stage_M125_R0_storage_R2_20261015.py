from pathlib import Path
import ast,json,os,subprocess

A=Path(__file__).resolve().parent
remote='''from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
root=Path('/root/autodl-tmp/m125_R0_fixed6_20261015')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert not any((root/name).exists() for name in ['real64.actual_launch.json','real64','gate.json','source_review.json','actual_launch.json','passes'])
original=json.loads((root/'candidate_gate.json').read_bytes());staging=json.loads((root/'staging.json').read_bytes())
assert sha(root/'candidate_gate.json')==staging['candidate_gate_sha256']
assert original['required_free_disk_bytes']==20*1024**3
for path,digest in original['source_sha256'].items():assert sha(path)==digest,path
for path,digest in original['input_sha256'].items():assert sha(path)==digest,path
candidate=dict(original,required_free_disk_bytes=2*1024**3)
assert shutil.disk_usage(root).free>=candidate['required_free_disk_bytes']
target=root/'candidate_gate_storage_R2.json'
with target.open('x') as stream:stream.write(json.dumps(candidate,indent=2)+chr(10))
value=dict(staging,status='actual_R0_storage_only_candidate_metadata_R2',at=datetime.now(timezone.utc).isoformat(),candidate_gate_sha256=sha(target),original_candidate_gate_sha256=sha(root/'candidate_gate.json'),candidate_gate_path=str(target),old_required_free_disk_bytes=20*1024**3,required_free_disk_bytes=2*1024**3,actual_free_bytes=shutil.disk_usage(root).free,CPU_imports_reexecuted=False,inherited_CPU_import_receipt_sha256=sha(root/'staging.json'),NN_calls=0,GPU_queries=0,training_launched=False)
with (root/'storage_R2_staging.json').open('x') as stream:stream.write(json.dumps(value,indent=2)+chr(10))
print(json.dumps(value))
'''
t=ast.parse((A/'M124_fiftyfirst_hourly_compressed_native_observer_20261014.py').read_text())
args=ast.literal_eval(next(n.value for n in t.body if isinstance(n,ast.Assign) and n.targets[0].id=='args'))[:-1]
env=dict(os.environ,SSH_ASKPASS=str(A/'rgbd_goal_resume_askpass_20261014.cmd'),SSH_ASKPASS_REQUIRE='force',DISPLAY='codex')
assert env['RGBD_GOAL_SSH_PASSWORD']
r=subprocess.run(args+["CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/envs/sttrack/bin/python -B -"],input=remote.encode(),capture_output=True,env=env)
with (A/'M125_R0_storage_R2_staging_20261015_stdout.json').open('xb') as f:f.write(r.stdout)
with (A/'M125_R0_storage_R2_staging_20261015_stderr.log').open('xb') as f:f.write(r.stderr)
assert r.returncode==0,r.stderr.decode()
x=json.loads(r.stdout);print(json.dumps({key:x[key] for key in ['status','candidate_gate_sha256','required_free_disk_bytes','actual_free_bytes','CPU_imports_reexecuted','NN_calls','GPU_queries']}))
