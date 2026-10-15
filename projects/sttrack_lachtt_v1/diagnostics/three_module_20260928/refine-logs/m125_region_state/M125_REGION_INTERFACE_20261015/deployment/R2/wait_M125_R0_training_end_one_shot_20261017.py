"""One absolute-time background wait, then the guarded original R0 observation."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import subprocess
import sys
import time

A = Path(__file__).resolve().parent
node = json.loads((A / 'M125_R0_training_end_node_20261017.json').read_bytes())
started = datetime.now(timezone.utc)
target = datetime.fromisoformat(node['not_before'])
assert target > started
command = [sys.executable, '-B', '-X', 'utf8', str(A / 'observe_M125_R0_training_end_R2_once_20261017.py')]
owner = dict(status='one_absolute_time_wait_owned', pid=os.getpid(), started_at=started.isoformat(),
             not_before=node['not_before'], command=command, NN_calls_before_node=0,
             progress_polling_before_node=0, automatic_retries=0)
with (A / 'M125_R0_training_end_wait_owner_20261017.json').open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(json.dumps(owner, indent=2) + '\n')
time.sleep((target - started).total_seconds())
with (A / 'M125_R0_training_end_timer_observer_stdout_20261017.log').open('xb') as output:
    with (A / 'M125_R0_training_end_timer_observer_stderr_20261017.log').open('xb') as error:
        completed = subprocess.run(command, cwd=A, stdout=output, stderr=error)
terminal = dict(owner, status='one_absolute_time_observation_attempt_terminal',
                observer_exit_code=completed.returncode, finished_at=datetime.now(timezone.utc).isoformat())
with (A / 'M125_R0_training_end_wait_terminal_20261017.json').open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(json.dumps(terminal, indent=2) + '\n')
sys.exit(completed.returncode)
