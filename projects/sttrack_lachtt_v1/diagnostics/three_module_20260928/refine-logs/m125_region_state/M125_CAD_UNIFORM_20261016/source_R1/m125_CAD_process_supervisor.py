"""Preserve the real terminal exit of the fixed uniform-profile inference suite."""
from datetime import datetime,timezone
from pathlib import Path
import json
import os
import subprocess
import sys


def main():
    root=Path(sys.argv[1]);gate=json.loads((root/'gate.json').read_bytes())
    command=[gate['python'],'-B','-u',str(Path(gate['code_root'])/'run_m125_CAD_uniform_suite.py'),str(root)]
    launch=dict(supervisor_pid=os.getpid(),controller_command=command,started_at=datetime.now(timezone.utc).isoformat(),automatic_retries=0)
    (root/'controller.supervisor.launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    completed=subprocess.run(command,cwd=gate['code_root'])
    terminal=dict(launch,exit_code=completed.returncode,finished_at=datetime.now(timezone.utc).isoformat())
    (root/'controller.supervised.exit').write_text(str(completed.returncode)+'\n')
    (root/'controller.supervisor.terminal.json').write_text(json.dumps(terminal,indent=2)+'\n')
    print(json.dumps(terminal),flush=True)
    return completed.returncode


if __name__=='__main__':sys.exit(main())
