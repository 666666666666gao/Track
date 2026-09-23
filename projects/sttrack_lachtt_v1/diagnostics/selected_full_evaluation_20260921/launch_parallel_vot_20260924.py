"""Start M67 VOT while the paused controller's CDTB child continues."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path('/root/autodl-tmp/sttrack_selected_full_evaluation_20260921')
script = ROOT / 'parallel_m67_vot_20260924.sh'
assert not (ROOT / 'parallel_vot_launch.json').exists()
assert not (ROOT / 'M67/vot').exists()
assert (ROOT / 'M67/depthtrack/predictions/metrics.json').exists()
assert not (ROOT / 'M67/cdtb/predictions/receipt.json').exists()
parent = subprocess.check_output(['ps', '-p', '2437', '-o', 'stat='], text=True).strip()
child = subprocess.check_output(['ps', '-p', '2647', '-o', 'stat='], text=True).strip()
assert 'T' in parent and 'T' not in child and 'Z' not in child, (parent, child)
subprocess.run(['bash', '-n', str(script)], check=True)

with (ROOT / 'parallel_m67_vot_controller.log').open('x') as log:
    process = subprocess.Popen(['bash', str(script)], stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
result = dict(status='M67_VOT_started_in_parallel_with_M67_CDTB',
              observed_utc=datetime.now(timezone.utc).isoformat(),
              old_controller_pid=2437, old_controller_status=parent,
              cdtb_child_pid=2647, cdtb_child_status=child,
              vot_controller_pid=process.pid, actual_vot_poll_seconds=3600,
              script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
              planned_new_training_steps=0,
              result_order='M67 full metrics before M82 full metrics')
(ROOT / 'parallel_vot_launch.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
