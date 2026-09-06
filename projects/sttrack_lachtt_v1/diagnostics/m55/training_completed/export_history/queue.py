"""CPU-only completion export after both existing training controllers exit."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path('/root/autodl-tmp/sttrack_m55_tsg_direction_v2_20260906')
output = Path('/root/autodl-tmp/sttrack_m55_training_completed_20260906')
not_before = datetime.datetime.fromisoformat('2026-09-06T12:36:00+08:00')
markers = [root / ('train_' + arm + '_controller.exit') for arm in ['control', 'clone']]
receipt = dict(pid=os.getpid(), not_before=not_before.isoformat(), poll_seconds=240,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    exporter_sha256=hashlib.sha256((root/'export_training_completion.py').read_bytes()).hexdigest(),
    output=str(output), gpu_execution=False)
(root/'completion_export_queue.json').write_text(json.dumps(receipt, indent=2)+'\n')
with (root/'completion_export_queue.log').open('w') as log:
    log.write(json.dumps(dict(status='waiting_for_completed_training', **receipt))+'\n')
    log.flush()
    time.sleep(max(0., not_before.timestamp()-time.time()))
    while not all(path.exists() for path in markers):
        log.write(json.dumps(dict(status='waiting_for_completed_training',
            terminal_controllers=[p.name for p in markers if p.exists()],
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n')
        log.flush()
        time.sleep(240)
    assert hashlib.sha256((root/'export_training_completion.py').read_bytes()).hexdigest()==receipt['exporter_sha256']
    result = subprocess.run(['/root/autodl-tmp/envs/sttrack/bin/python',
        str(root/'export_training_completion.py'), '--root', str(root), '--output', str(output)],
        stdout=log, stderr=subprocess.STDOUT)
    (root/'completion_export.exit').write_text(str(result.returncode)+'\n')
raise SystemExit(result.returncode)
