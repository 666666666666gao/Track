"""Require the actual source review, then launch one isolated real64 witness."""
from pathlib import Path
import ast
import base64
import hashlib
import json
import os
import subprocess
import zlib

A = Path(__file__).resolve().parent
review = A / 'M125_R0_FIXED_SOURCE_AUDIT_20261015'
audit_json = (review / 'EXPERIMENT_AUDIT.json').read_bytes()
audit_md = (review / 'EXPERIMENT_AUDIT.md').read_bytes()
audit = json.loads(audit_json)
candidate_path = A / 'M125_R0_FIXED_SOURCE_20261015/CANDIDATE.json'
staging = json.loads((A / 'M125_R0_candidate_staging_20261015_stdout.json').read_bytes())
assert audit['source_verdict'] == 'PASS_SOURCE_ONLY' and audit['blocking_findings'] == []
assert audit['candidate_sha256'] == hashlib.sha256(candidate_path.read_bytes()).hexdigest()
assert audit['audited_staged_source_sha256'] == staging['source_sha256']
payload = dict(audit_json=audit_json.decode(), audit_json_sha256=hashlib.sha256(audit_json).hexdigest(),
    audit_md=audit_md.decode(), audit_md_sha256=hashlib.sha256(audit_md).hexdigest(),
    candidate_sha256=audit['candidate_sha256'], staging=staging)
remote = '''from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,shutil,subprocess,zlib
p=json.loads(zlib.decompress(base64.b64decode('PAYLOAD')))
root=Path('/root/autodl-tmp/m125_R0_fixed6_20261015')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert sha(root/'candidate_gate.json')==p['staging']['candidate_gate_sha256']
gate=json.loads((root/'candidate_gate.json').read_bytes())
review=json.loads(p['audit_json'])
assert review['source_verdict']=='PASS_SOURCE_ONLY' and review['blocking_findings']==[]
assert review['candidate_sha256']==p['candidate_sha256']
assert review['audited_staged_source_sha256']==p['staging']['source_sha256']
for path,digest in gate['source_sha256'].items():assert sha(path)==digest,path
for path,digest in gate['input_sha256'].items():assert sha(path)==digest,path
assert Path('/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/controller.supervised.exit').read_text().strip()=='0'
assert Path('/root/autodl-tmp/m125_selected_E6_split_replay_R2_20261015/controller.exit').read_text().strip()=='0'
assert shutil.disk_usage(root).free>=gate['required_free_disk_bytes']
query=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,check=True,text=True)
gpu_rows=[[int(value.strip()) for value in line.split(',')] for line in query.stdout.splitlines()]
assert len(gpu_rows)==2 and [r[0] for r in gpu_rows]==[0,1] and all(r[1]<500 for r in gpu_rows),gpu_rows
for name,text,digest in [('source_review.json',p['audit_json'],p['audit_json_sha256']),('source_review.md',p['audit_md'],p['audit_md_sha256'])]:
    target=root/name
    with target.open('x') as stream:stream.write(text)
    assert sha(target)==digest
gate.update(status='reviewed_M125_R0_fixed6_source_gate',source_review_report_path=str(root/'source_review.json'),source_review_report_sha256=p['audit_json_sha256'],source_review_candidate_sha256=p['candidate_sha256'])
with (root/'gate.json').open('x') as stream:stream.write(json.dumps(gate,indent=2)+'\\n')
command=[gate['python'],'-B','-u',str(Path(gate['code_root'])/'check_m125_R0_real64.py'),'--gate',str(root/'gate.json'),'--output',str(root/'real64')]
supervisor=chr(10).join(['import subprocess,json,sys','from pathlib import Path','from datetime import datetime,timezone',
    'root=Path('+repr(str(root))+')','command='+repr(command),
    'row=dict(command=command,started_at=datetime.now(timezone.utc).isoformat())',
    'completed=subprocess.run(command,cwd='+repr(gate['code_root'])+')',
    'row.update(exit_code=completed.returncode,finished_at=datetime.now(timezone.utc).isoformat())',
    '(root/"real64.exit").write_text(str(completed.returncode)+chr(10))',
    '(root/"real64.receipt.json").write_text(json.dumps(row,indent=2)+chr(10))',
    'sys.exit(completed.returncode)',''])
driver=root/'real64_supervisor.py'
with driver.open('x') as stream:stream.write(supervisor)
environment=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONPATH=os.pathsep.join([gate['code_root'],gate['base_code_root'],'/home/SUTrack_RGBD_L']),PYTHONDONTWRITEBYTECODE='1',PYTHONOPTIMIZE='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (root/'real64.log').open('xb') as output:
    child=subprocess.Popen([gate['python'],'-B','-u',str(driver)],cwd=gate['code_root'],env=environment,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
value=dict(status='actual_M125_R0_real64_witness_launched',at=datetime.now(timezone.utc).isoformat(),pid=child.pid,root=str(root),command=command,gate_sha256=sha(root/'gate.json'),supervisor_sha256=sha(driver),GPU_memory_admission=gpu_rows,GPU_query_fields='index,memory.used,memory.total',source_review_sha256=p['audit_json_sha256'],formal_training_launched=False,automatic_retries=0)
with (root/'real64.actual_launch.json').open('x') as stream:stream.write(json.dumps(value,indent=2)+'\\n')
print(json.dumps(value))
'''.replace('PAYLOAD', base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode())
tree = ast.parse((A / 'M124_fiftyfirst_hourly_compressed_native_observer_20261014.py').read_text())
args = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == 'args'))[:-1]
environment = dict(os.environ, SSH_ASKPASS=str(A / 'rgbd_goal_resume_askpass_20261014.cmd'),
                   SSH_ASKPASS_REQUIRE='force', DISPLAY='codex')
assert environment['RGBD_GOAL_SSH_PASSWORD']
result = subprocess.run(args + ["CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/envs/sttrack/bin/python -B -"], input=remote.encode(),capture_output=True,env=environment)
with (A / 'M125_R0_real64_actual_launch_20261015_stdout.json').open('xb') as stream:stream.write(result.stdout)
with (A / 'M125_R0_real64_actual_launch_20261015_stderr.log').open('xb') as stream:stream.write(result.stderr)
assert result.returncode == 0, result.stderr.decode()
print(result.stdout.decode())
