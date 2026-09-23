"""Start M82's two OPE evaluations on separate GPUs after M67 completes."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path('/root/autodl-tmp/sttrack_selected_full_evaluation_20260921')
script = ROOT / 'parallel_m82_20260924.sh'
assert not (ROOT / 'parallel_m82_launch.json').exists()
assert not Path('/proc/2437').exists()
assert (ROOT / 'M67/cdtb_metrics.exit').read_text().strip() == '0'
assert (ROOT / 'M67/vot_metrics.exit').read_text().strip() == '0'
assert (ROOT / 'parallel_m67_vot.exit').read_text().strip() == '0'
assert json.loads((ROOT / 'M67/cdtb/predictions/metrics.json').read_text())['status'] == 'complete'
assert json.loads((ROOT / 'M67/vot/result.json').read_text())['status'] == 'complete_full127'
assert not (ROOT / 'M82/depthtrack').exists()
assert not (ROOT / 'M82/cdtb').exists()
used = [int(x) for x in subprocess.check_output(
    ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
    text=True).splitlines()]
assert len(used) == 2 and max(used) < 500, used
subprocess.run(['bash', '-n', str(script)], check=True)

with (ROOT / 'parallel_m82_controller.log').open('x') as log:
    process = subprocess.Popen(['bash', str(script)], stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
result = dict(status='M82_full_evaluations_launched_after_M67_complete',
              observed_utc=datetime.now(timezone.utc).isoformat(),
              pid=process.pid, depthtrack_gpu=0, cdtb_gpu=1,
              vot_gpu_count=2, actual_vot_poll_seconds=3600,
              script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
              M67_depthtrack_metric_sha256=hashlib.sha256(
                  (ROOT / 'M67/depthtrack/predictions/metrics.json').read_bytes()).hexdigest(),
              M67_cdtb_metric_sha256=hashlib.sha256(
                  (ROOT / 'M67/cdtb/predictions/metrics.json').read_bytes()).hexdigest(),
              M67_vot_result_sha256=hashlib.sha256(
                  (ROOT / 'M67/vot/result.json').read_bytes()).hexdigest(),
              new_training_steps=0)
(ROOT / 'parallel_m82_launch.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
