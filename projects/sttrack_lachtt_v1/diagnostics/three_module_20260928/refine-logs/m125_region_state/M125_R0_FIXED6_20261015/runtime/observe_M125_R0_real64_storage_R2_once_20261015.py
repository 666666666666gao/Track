"""One terminal witness observation at least three minutes after actual launch."""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import ast
import hashlib
import json
import os
import subprocess

A = Path(__file__).resolve().parent
launch = json.loads((A / 'M125_R0_real64_actual_launch_R2_20261015_stdout.json').read_bytes())
not_before = datetime.fromisoformat(launch['at']) + timedelta(seconds=180)
assert datetime.now(timezone.utc) >= not_before
remote = '''from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
root=Path('/root/autodl-tmp/m125_R0_fixed6_20261015')
assert datetime.now(timezone.utc)>=datetime.fromisoformat('NOT_BEFORE')
files={}
for name in ['real64.actual_launch.json','real64_supervisor.py','real64.exit','real64.receipt.json','real64/result.json','gate.json']:
    path=root/name
    if path.is_file():
        raw=path.read_bytes();files[name]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),text=raw.decode())
log=(root/'real64.log').read_bytes()
value=dict(status='actual_M125_R0_real64_scheduled_observation',at=datetime.now(timezone.utc).isoformat(),files=files,log_tail=log[-20000:].decode(),NN_calls=0,GPU_queries=0)
print(json.dumps(value))
'''.replace('NOT_BEFORE', not_before.isoformat())
tree = ast.parse((A / 'M124_fiftyfirst_hourly_compressed_native_observer_20261014.py').read_text())
args = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == 'args'))[:-1]
environment = dict(os.environ, SSH_ASKPASS=str(A / 'rgbd_goal_resume_askpass_20261014.cmd'),
                   SSH_ASKPASS_REQUIRE='force', DISPLAY='codex')
assert environment['RGBD_GOAL_SSH_PASSWORD']
result = subprocess.run(args + ["CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/envs/sttrack/bin/python -B -"],
    input=remote.encode(),capture_output=True,env=environment)
with (A / 'M125_R0_real64_terminal_observation_R2_20261015.json').open('xb') as stream:stream.write(result.stdout)
with (A / 'M125_R0_real64_terminal_observation_R2_20261015_stderr.log').open('xb') as stream:stream.write(result.stderr)
assert result.returncode == 0, result.stderr.decode()
value = json.loads(result.stdout)
for row in value['files'].values():assert hashlib.sha256(row['text'].encode()).hexdigest() == row['sha256']
print(json.dumps(dict(status=value['status'],at=value['at'],files=list(value['files']),
    log_tail=value['log_tail'][-700:]),indent=2))
