"""Seal the completed CDTB child after the old controller was paused."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess


ROOT = Path('/root/autodl-tmp/sttrack_selected_full_evaluation_20260921')
assert not (ROOT / 'finish_m67_cdtb.json').exists()
assert Path('/proc/2437/stat').read_text().split()[2] == 'T'
assert Path('/proc/2647/stat').read_text().split()[2] == 'Z'
receipt = json.loads((ROOT / 'M67/cdtb/predictions/receipt.json').read_text())
assert receipt['status'] == 'complete'
assert len(receipt['sequences']) == 80 and receipt['frames'] == 101956

# The CDTB child has finished. Prevent the original shell from starting a
# second VOT run, then let the frozen analyzer check every sealed output.
os.kill(2437, signal.SIGKILL)
environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
with (ROOT / 'M67/cdtb_metrics.log').open('x') as log:
    result = subprocess.run(['/root/miniconda3/envs/mplt/bin/python', '-u',
                             'interface/run_semantic_ope.py', '--plan',
                             str(ROOT / 'M67/cdtb/plan.json'), '--mode', 'analyze'],
                            cwd=ROOT, env=environment, stdout=log,
                            stderr=subprocess.STDOUT)
(ROOT / 'M67/cdtb_metrics.exit').write_text(str(result.returncode) + '\n')
assert result.returncode == 0
(ROOT / 'finish_m67_cdtb.json').write_text(json.dumps(dict(
    status='sealed_complete_and_analyzed', observed_utc=datetime.now(timezone.utc).isoformat(),
    old_controller_pid=2437, old_controller_terminated_after_child_zombie=True,
    cdtb_child_pid=2647, child_exit_code_not_directly_observed=True,
    sealed_receipt_sequences=80, sealed_receipt_frames=101956,
    frozen_analyzer_exit=0, new_training_steps=0), indent=2) + '\n')
