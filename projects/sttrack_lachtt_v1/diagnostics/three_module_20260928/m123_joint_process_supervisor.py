"""Record the real controller exit, including an unhandled assertion failure."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import subprocess
import sys


def main():
    assert sys.flags.optimize == 0
    root = Path(sys.argv[1])
    gate = json.loads((root / 'gate.json').read_text())
    command = [gate['python'], '-B', '-u',
               str(Path(gate['code_root']) / 'run_m123_joint_full_suite.py'), str(root)]
    started = datetime.now(timezone.utc).isoformat()
    launch = dict(supervisor_pid=os.getpid(), controller_command=command,
                  started_at=started, progress_queries=0, automatic_retries=0)
    (root / 'controller.supervisor.launch.json').write_text(json.dumps(launch, indent=2) + '\n')
    completed = subprocess.run(command, cwd=gate['code_root'])
    receipt = dict(launch, exit_code=completed.returncode,
                   finished_at=datetime.now(timezone.utc).isoformat(),
                   stderr_destination=str(root / 'controller.log'))
    (root / 'controller.supervised.exit').write_text(str(completed.returncode) + '\n')
    (root / 'controller.supervisor.terminal.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)
    return completed.returncode


if __name__ == '__main__':
    sys.exit(main())
