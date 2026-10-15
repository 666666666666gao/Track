"""One planned observation of the original full training, no NN/GPU work."""
from pathlib import Path
from datetime import datetime,timezone
import ast,json,os,subprocess

A=Path(__file__).resolve().parent
node=json.loads((A/'M125_R0_first_training_node_20261015.json').read_bytes())
assert datetime.now(timezone.utc)>=datetime.fromisoformat(node['not_before'])
remote='''from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
root=Path('/root/autodl-tmp/m125_R0_fixed6_20261015')
node=NODE
assert datetime.now(timezone.utc)>=datetime.fromisoformat(node['not_before'])
files={}
for name in ['actual_launch.json','controller.supervisor.launch.json','controller.supervised.exit','controller.supervisor.terminal.json','current_training_state.json','result.json','passes/pass-01/train_fixed_pass.launch.json','passes/pass-01/train_fixed_pass.exit','passes/pass-01/train_fixed_pass.receipt.json','passes/pass-01/fit/result.json']:
    path=root/name
    if path.is_file():
        raw=path.read_bytes();files[name]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),text=raw.decode())
def committed_rows(path):
    if not path.exists():return []
    return [json.loads(line) for line in path.read_bytes().split(b'\\n')[:-1]]
records=committed_rows(root/'passes/pass-01/fit/sequence_log.jsonl')
trace=committed_rows(root/'passes/pass-01/fit/sampled_state_trace.jsonl')
logs={}
for name in ['controller.log','passes/pass-01/train_fixed_pass.log']:
    path=root/name
    if path.is_file():logs[name]=path.read_bytes()[-12000:].decode()
process=Path('/proc')/str(node['original_supervisor_pid'])/'cmdline'
command=process.read_bytes().replace(bytes([0]),b' ').decode() if process.is_file() else ''
latest=root/'passes/pass-01/fit/latest.pt'
checkpoint=dict(exists=latest.is_file())
if latest.is_file():checkpoint.update(bytes=latest.stat().st_size,mtime_ns=latest.stat().st_mtime_ns)
print(json.dumps(dict(status='actual_M125_R0_first_committed_training_observation',observed_at=datetime.now(timezone.utc).isoformat(),files=files,sequence_records=records,sampled_trace=trace,log_tails=logs,original_supervisor_cmdline=command,latest_checkpoint_metadata=checkpoint,NN_calls=0,GPU_queries=0,automatic_retries=0)))
'''.replace('NODE',repr(node))
t=ast.parse((A/'M124_fiftyfirst_hourly_compressed_native_observer_20261014.py').read_text())
args=ast.literal_eval(next(n.value for n in t.body if isinstance(n,ast.Assign) and n.targets[0].id=='args'))[:-1]
env=dict(os.environ,SSH_ASKPASS=str(A/'rgbd_goal_resume_askpass_20261014.cmd'),SSH_ASKPASS_REQUIRE='force',DISPLAY='codex')
assert env['RGBD_GOAL_SSH_PASSWORD']
r=subprocess.run(args+["CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/envs/sttrack/bin/python -B -"],input=remote.encode(),capture_output=True,env=env)
with (A/'M125_R0_first_training_commit_observation_20261015.json').open('xb') as stream:stream.write(r.stdout)
with (A/'M125_R0_first_training_commit_observation_20261015_stderr.log').open('xb') as stream:stream.write(r.stderr)
assert r.returncode==0,r.stderr.decode()
value=json.loads(r.stdout)
print(json.dumps(dict(status=value['status'],observed_at=value['observed_at'],sequence_records=value['sequence_records'],sampled_trace_last=value['sampled_trace'][-1:] ,original_supervisor_cmdline=value['original_supervisor_cmdline'],checkpoint=value['latest_checkpoint_metadata'],terminal_files=[name for name in value['files'] if name.endswith('.exit') or name=='result.json']),indent=2))
