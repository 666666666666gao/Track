"""Stage an isolated candidate and CPU-import it; never launch training or NN."""
from pathlib import Path
import ast
import base64
import hashlib
import json
import os
import subprocess
import zlib

A = Path(__file__).resolve().parent
payload = (A / 'M125_R0_staging_payload_20261015.json').read_bytes()
remote = '''from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,shutil,subprocess,sys,zlib
p=json.loads(zlib.decompress(base64.b64decode('PAYLOAD')))
gate=p['candidate'];root=Path(gate['root']);assert not root.exists()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
for path,digest in gate['source_sha256'].items():
    if path not in gate['new_source_local_paths']:assert sha(path)==digest,path
for path,digest in gate['input_sha256'].items():assert sha(path)==digest,path
for row in p['frozen_dependencies']:assert sha(row['remote'])==row['sha256'],row['remote']
root.mkdir();code=root/'code';code.mkdir()
copied={}
for row in p['frozen_dependencies']:
    target=code/Path(row['remote']).name
    assert not target.exists();shutil.copy2(row['remote'],target)
    assert sha(target)==row['sha256'];copied[str(target)]=sha(target)
for name,row in p['sources'].items():
    target=code/name;assert not target.exists();target.write_bytes(row['text'].encode())
    assert sha(target)==row['sha256'];copied[str(target)]=sha(target)
gate.pop('new_source_local_paths');gate['source_sha256'].update(copied)
(root/'candidate_gate.json').write_text(json.dumps(gate,indent=2)+'\\n')
environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=os.pathsep.join([str(code),gate['base_code_root'],'/home/SUTrack_RGBD_L']),PYTHONDONTWRITEBYTECODE='1',PYTHONOPTIMIZE='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
rows=[]
for name in p['sources']:
    command=[gate['python'],'-B','-c','import runpy;runpy.run_path('+repr(str(code/name))+',run_name="M125_CPU_import_preflight")']
    completed=subprocess.run(command,cwd=code,env=environment,capture_output=True)
    row=dict(source=name,command=command,exit_code=completed.returncode,stdout=completed.stdout.decode(),stderr=completed.stderr.decode());rows.append(row)
    (root/(name+'.CPU_import.json')).write_text(json.dumps(row,indent=2)+'\\n')
    assert completed.returncode==0,row
value=dict(status='actual_M125_R0_candidate_staged_and_CPU_imported_only',at=datetime.now(timezone.utc).isoformat(),root=str(root),source_sha256=copied,candidate_gate_sha256=sha(root/'candidate_gate.json'),CPU_imports=rows,NN_calls=0,training_launched=False,GPU_queries=0)
(root/'staging.json').write_text(json.dumps(value,indent=2)+'\\n');print(json.dumps(value))
'''.replace('PAYLOAD', base64.b64encode(zlib.compress(payload)).decode())
tree = ast.parse((A / 'M124_fiftyfirst_hourly_compressed_native_observer_20261014.py').read_text())
args = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == 'args'))[:-1]
environment = dict(os.environ, SSH_ASKPASS=str(A / 'rgbd_goal_resume_askpass_20261014.cmd'),
                   SSH_ASKPASS_REQUIRE='force', DISPLAY='codex')
assert environment['RGBD_GOAL_SSH_PASSWORD']
stem = 'M125_R0_candidate_staging_20261015'
result = subprocess.run(args + ["CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/envs/sttrack/bin/python -B -"],
    input=remote.encode(), capture_output=True, env=environment)
with (A / (stem + '_stdout.json')).open('xb') as stream:stream.write(result.stdout)
with (A / (stem + '_stderr.log')).open('xb') as stream:stream.write(result.stderr)
assert result.returncode == 0, result.stderr.decode()
value = json.loads(result.stdout)
print(json.dumps(dict(status=value['status'],root=value['root'],files=len(value['source_sha256']),
    CPU_imports=len(value['CPU_imports']),NN_calls=0,GPU_queries=0,training_launched=False)))
