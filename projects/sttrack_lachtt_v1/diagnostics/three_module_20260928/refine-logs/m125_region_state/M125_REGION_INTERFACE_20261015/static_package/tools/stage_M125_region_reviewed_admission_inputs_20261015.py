"""Stage immutable regional launch inputs and check CPU imports, without admission."""
from pathlib import Path, PurePosixPath
import ast
import base64
import hashlib
import json
import os
import subprocess
import zlib

A = Path(__file__).resolve().parent
S = A / 'M125_REGION_SOURCE_20261015'
source_review_folder = A / 'M125_REGION_SOURCE_AUDIT_20261015'
deployment_review_folder = A / 'M125_REGION_DEPLOYMENT_AUDIT_R2_20261015'
operator = A / 'M125_REGION_DEPLOYMENT_SOURCE_R2_20261015/deploy_m125_region_fixed.py'
output = A / 'M125_region_reviewed_admission_staging_20261015.json'
error = A / 'M125_region_reviewed_admission_staging_20261015_stderr.log'
assert not output.exists() and not error.exists()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def item(path):
    return dict(text=path.read_bytes().decode('utf-8'), sha256=sha(path))


source_review = json.loads((source_review_folder / 'EXPERIMENT_AUDIT.json').read_bytes())
deployment_review = json.loads((deployment_review_folder / 'EXPERIMENT_AUDIT.json').read_bytes())
for review in [source_review, deployment_review]:
    assert review['source_verdict'] == 'PASS_SOURCE_ONLY' and review['blocking_findings'] == []
assert deployment_review['audited_input_hashes'][str(operator)] == 'sha256:' + sha(operator)
assert deployment_review['audited_input_hashes'][str(source_review_folder / 'EXPERIMENT_AUDIT.json')] == 'sha256:' + sha(source_review_folder / 'EXPERIMENT_AUDIT.json')
files = {}
for path in sorted(S.iterdir()):
    assert path.is_file()
    assert source_review['audited_input_hashes'][str(path)] == 'sha256:' + sha(path)
    files['inputs/' + path.name] = item(path)
for name in ['EXPERIMENT_AUDIT.json', 'EXPERIMENT_AUDIT.md']:
    files['inputs/' + name] = item(source_review_folder / name)
    files['operator/' + name] = item(deployment_review_folder / name)
files['operator/deploy_m125_region_fixed.py'] = item(operator)
assert len(files) == 22
new_names = sorted(path.name for path in S.glob('*.py'))
assert len(new_names) == 11
maps = []
for label in ['ROI', 'R1', 'V']:
    candidate = json.loads((S / ('CANDIDATE_' + label + '.json')).read_bytes())
    code = PurePosixPath(candidate['code_root'])
    maps.append({PurePosixPath(name).name:digest for name,digest in candidate['source_sha256'].items()
                 if PurePosixPath(name).parent == code})
assert maps[0] == maps[1] == maps[2] and len(maps[0]) == 64
payload = dict(files=files, new_names=new_names, code_sha256=maps[0])
remote = '''from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,runpy,subprocess,zlib
p=json.loads(zlib.decompress(base64.b64decode('PAYLOAD')))
root=Path('/root/autodl-tmp/m125_region_reviewed_admission_inputs_20261015')
assert not root.exists()
root.mkdir()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
for relative,row in p['files'].items():
    target=root/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as stream:stream.write(row['text'].encode())
    assert sha(target)==row['sha256'],relative
source_review=json.loads((root/'inputs/EXPERIMENT_AUDIT.json').read_bytes())
deployment_review=json.loads((root/'operator/EXPERIMENT_AUDIT.json').read_bytes())
for review in [source_review,deployment_review]:
    assert review['source_verdict']=='PASS_SOURCE_ONLY' and review['blocking_findings']==[]
functions=runpy.run_path(str(root/'operator/deploy_m125_region_fixed.py'),run_name='M125_static_admission_import_only')
arm_rows=[]
for arm in ['roi','language','visual']:
    candidate,inventory=functions['verify_inputs'](root/'inputs',arm)
    arm_rows.append(dict(arm=arm,authentic_review_and_candidate_inputs_match=True,candidate_status=candidate['status'],
        future_root=candidate['root'],actual_admission_executed=False))
code=root/'code';code.mkdir()
for name,digest in p['code_sha256'].items():
    source=root/'inputs'/name if name in p['new_names'] else Path('/root/autodl-tmp/m125_R0_fixed6_20261015/code')/name
    data=source.read_bytes();assert hashlib.sha256(data).hexdigest()==digest,name
    with (code/name).open('xb') as stream:stream.write(data)
assert len(list(code.iterdir()))==64
environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=os.pathsep.join([str(code),'/home/SUTrack_RGBD_L']),
    PYTHONDONTWRITEBYTECODE='1',PYTHONOPTIMIZE='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
imports=[]
for name in p['new_names']:
    program='import json,runpy,sys;from pathlib import Path;runpy.run_path('+repr(str(code/name))+',run_name="M125_packaged_CPU_import_only");print(json.dumps({key:str(Path(value.__file__).resolve()) for key,value in sys.modules.items() if getattr(value,"__file__",None) and Path(value.__file__).resolve().parent==Path('+repr(str(code))+')}))'
    command=['/root/autodl-tmp/envs/sttrack/bin/python','-B','-c',program]
    started=datetime.now(timezone.utc).isoformat()
    completed=subprocess.run(command,cwd=code,env=environment,capture_output=True)
    row=dict(source=name,command=command,started_at=started,finished_at=datetime.now(timezone.utc).isoformat(),
        exit_code=completed.returncode,stdout=completed.stdout.decode(),stderr=completed.stderr.decode())
    imports.append(row)
    assert completed.returncode==0,row
    origins=json.loads(completed.stdout)
    for path in origins.values():assert sha(path)==p['code_sha256'][Path(path).name],path
    row['verified_packaged_module_origins']=origins
for name,digest in p['code_sha256'].items():assert sha(code/name)==digest,name
receipt=dict(status='actual_static_region_admission_package_CPU_imports_complete',at=datetime.now(timezone.utc).isoformat(),
    root=str(root),files_sha256={name:row['sha256'] for name,row in p['files'].items()},
    code_sha256={str(code/name):digest for name,digest in p['code_sha256'].items()},
    authentic_candidate_checks=arm_rows,imports=imports,CUDA_VISIBLE_DEVICES='',
    PYTHONPATH=environment['PYTHONPATH'],packaged_code_files=64,CPU_import_entries=11,
    original_R0_immutable_source_files_read=53,original_R0_training_progress_queries=0,original_R0_source_modified=False,
    predecessor_completion_checked=False,actual_resource_admission_checked=False,
    real_regional_GPU_deployment=False,actual_deployment_operator_main_executed=False,
    decoder_forward_calls=0,real_tracking_calls=0,GPU_query_commands=0,formal_training_result=False)
with (root/'staging_receipt.json').open('x') as stream:stream.write(json.dumps(receipt,indent=2)+chr(10))
print(json.dumps(receipt))
'''.replace('PAYLOAD', base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode())
tree = ast.parse((A / 'M124_fiftyfirst_hourly_compressed_native_observer_20261014.py').read_text())
args = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == 'args'))[:-1]
environment = dict(os.environ, SSH_ASKPASS=str(A / 'rgbd_goal_resume_askpass_20261014.cmd'),
                   SSH_ASKPASS_REQUIRE='force', DISPLAY='codex')
assert environment['RGBD_GOAL_SSH_PASSWORD']
completed = subprocess.run(args + ["CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/envs/sttrack/bin/python -B -"],
                           input=remote.encode(), capture_output=True, env=environment)
with output.open('xb') as stream:stream.write(completed.stdout)
with error.open('xb') as stream:stream.write(completed.stderr)
assert completed.returncode == 0, completed.stderr.decode()
receipt = json.loads(completed.stdout)
print(json.dumps(dict(status=receipt['status'],root=receipt['root'],files=len(receipt['files_sha256']),
    code_files=receipt['packaged_code_files'],CPU_import_entries=receipt['CPU_import_entries'],
    authentic_candidate_arms=len(receipt['authentic_candidate_checks']),
    real_tracking_calls=receipt['real_tracking_calls'],original_R0_progress_queries=0)))
