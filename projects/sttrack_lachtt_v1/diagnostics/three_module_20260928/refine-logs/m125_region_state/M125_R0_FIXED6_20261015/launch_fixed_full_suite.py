"""One reviewed, real64-checked full fixed-six-pass launch; no automatic retry."""
from pathlib import Path
import ast
import json
import os
import subprocess

A = Path(__file__).resolve().parent
remote = '''from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess
root=Path('/root/autodl-tmp/m125_R0_fixed6_20261015')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
gate=json.loads((root/'gate.json').read_bytes())
assert gate['status']=='reviewed_M125_R0_fixed6_source_gate'
assert gate['fixed_budget_passes']==gate['fixed_final_pass']==6
assert not gate['external_metric_checkpoint_selection'] and not gate['Test50_CDTB80_used_for_checkpoint_selection']
assert not (root/'actual_launch.json').exists() and not (root/'passes').exists()
for path,digest in gate['source_sha256'].items():assert sha(path)==digest,path
for path,digest in gate['input_sha256'].items():assert sha(path)==digest,path
assert sha(gate['source_review_report_path'])==gate['source_review_report_sha256']
review=json.loads(Path(gate['source_review_report_path']).read_bytes())
assert review['source_verdict']=='PASS_SOURCE_ONLY' and review['blocking_findings']==[]
assert review['candidate_sha256']==gate['source_review_candidate_sha256']
for path,digest in review['audited_staged_source_sha256'].items():assert sha(path)==digest,path
assert (root/'real64.exit').read_text().strip()=='0'
receipt=json.loads((root/'real64.receipt.json').read_bytes());assert receipt['exit_code']==0
sanity=json.loads((root/'real64/result.json').read_bytes())
assert sanity['status']=='complete_M125_R0_real64_interface_witness'
assert sanity['gate_sha256']==sha(root/'gate.json') and sanity['source_sha256']==sha(Path(gate['code_root'])/'check_m125_R0_real64.py')
assert sanity['track_calls']==64 and sanity['AB_optimizer_steps']==sanity['C_optimizer_steps']==2
assert sanity['initial_parameters_equal_common_warm'] and sanity['initial_optimizer_empty']
assert sanity['frozen_before_after_exact'] and sanity['saved_model_and_optimizer_roundtrip_exact']
assert not sanity['formal_training_resume_uses_this_sanity']
assert sha(gate['warm_final'])==sanity['warm_sha256']
assert shutil.disk_usage(root).free>=gate['required_free_disk_bytes']
query=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,check=True,text=True)
gpu_rows=[[int(value.strip()) for value in line.split(',')] for line in query.stdout.splitlines()]
assert len(gpu_rows)==2 and [r[0] for r in gpu_rows]==[0,1] and all(r[1]<500 for r in gpu_rows),gpu_rows
environment=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONPATH=os.pathsep.join([gate['code_root'],gate['base_code_root'],'/home/SUTrack_RGBD_L']),PYTHONDONTWRITEBYTECODE='1',PYTHONOPTIMIZE='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
command=[gate['python'],'-B','-u',str(Path(gate['code_root'])/'m125_R0_fixed_process_supervisor.py'),str(root)]
with (root/'controller.log').open('xb') as stream:
    child=subprocess.Popen(command,cwd=gate['code_root'],env=environment,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
value=dict(status='actual_M125_R0_fixed6_full_training_and_final_suite_launched',started_at=datetime.now(timezone.utc).isoformat(),supervisor_pid=child.pid,command=command,root=str(root),gate_sha256=sha(root/'gate.json'),source_review_sha256=gate['source_review_report_sha256'],real64_result_sha256=sha(root/'real64/result.json'),GPU_memory_admission=gpu_rows,GPU_query_fields='index,memory.used,memory.total',fixed_budget_passes=6,fixed_final_pass=6,planned_sequence_runs=912,planned_training_track_calls=1318812,external_metric_checkpoint_selection=False,automatic_retries=0)
with (root/'actual_launch.json').open('x') as stream:stream.write(json.dumps(value,indent=2)+chr(10))
print(json.dumps(value))
'''
tree = ast.parse((A / 'M124_fiftyfirst_hourly_compressed_native_observer_20261014.py').read_text())
args = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == 'args'))[:-1]
environment = dict(os.environ, SSH_ASKPASS=str(A / 'rgbd_goal_resume_askpass_20261014.cmd'),
                   SSH_ASKPASS_REQUIRE='force', DISPLAY='codex')
assert environment['RGBD_GOAL_SSH_PASSWORD']
result = subprocess.run(args + ["CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/envs/sttrack/bin/python -B -"],
    input=remote.encode(), capture_output=True, env=environment)
with (A / 'M125_R0_fixed6_actual_launch_20261015_stdout.json').open('xb') as stream:stream.write(result.stdout)
with (A / 'M125_R0_fixed6_actual_launch_20261015_stderr.log').open('xb') as stream:stream.write(result.stderr)
assert result.returncode == 0, result.stderr.decode()
print(result.stdout.decode())
