"""Run one declared training command and preserve its actual exit receipt."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def main():
    root = Path(sys.argv[1])
    plan = json.loads((root / 'launch.json').read_text())
    environment = dict(os.environ, **plan['environment'])
    started = datetime.now(timezone.utc).isoformat()
    with (root / 'train.log').open('x') as log:
        child = subprocess.Popen(plan['command'], cwd=plan['cwd'], env=environment,
            stdout=log, stderr=subprocess.STDOUT)
        identity = dict(supervisor_pid=os.getpid(), child_pid=child.pid,
            command=plan['command'], started_at=started)
        (root / 'child.json').write_text(json.dumps(identity, indent=2) + '\n')
        code = child.wait()
    (root / 'train.exit').write_text(str(code) + '\n')
    terminal = dict(identity, exit_code=code, finished_at=datetime.now(timezone.utc).isoformat())
    (root / 'terminal.json').write_text(json.dumps(terminal, indent=2) + '\n')
    return code


if __name__ == '__main__':
    sys.exit(main())
