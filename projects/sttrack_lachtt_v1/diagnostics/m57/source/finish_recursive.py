"""Analyze M57 only after both full-trajectory queues finish successfully."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

from run_recursive import bound
from train import sha, write


root = Path('/root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906')
spec, frozen, training, cases = bound(root)
assert not (root / 'recursive_result.json').exists()
write(root / 'analysis_wait_record.json', dict(status='waiting_for_both_queues',
    started_utc=datetime.now(timezone.utc).isoformat(), interval_seconds=240,
    recursive_spec_sha256=sha(root / 'recursive_spec.json'), scheduler_sha256=sha(__file__),
    full_frame_gt_opened=False))
while True:
    done = []
    for gpu in [0, 1]:
        path = root / ('recursive_queue%d.exit' % gpu)
        if path.exists():
            assert path.read_text().strip() == '0', str(path)
            done.append(gpu)
    if len(done) == 2:
        break
    time.sleep(240)
for arm in spec['variants']:
    assert (root / (arm + '_recursive.exit')).read_text().strip() == '0'
    receipt = json.loads((root / (arm + '_recursive_receipt.json')).read_text())
    assert receipt['status'] == 'complete' and receipt['frames'] == 33130
command = ['/root/miniconda3/envs/mplt/bin/python', str(root / 'run_recursive.py'), '--root', str(root), '--analyze']
write(root / 'analysis_started.json', dict(status='all_four_complete_predictions_ready',
    observed_utc=datetime.now(timezone.utc).isoformat(), command=command,
    recursive_spec_sha256=sha(root / 'recursive_spec.json'),
    receipts={arm: sha(root / (arm + '_recursive_receipt.json')) for arm in spec['variants']}))
subprocess.run(command, cwd=str(root), check=True)
result = json.loads((root / 'recursive_result.json').read_text())
assert result['status'] == 'complete_recursive_development'
print(json.dumps(dict(status='complete_recursive_analysis',
    result_sha256=sha(root / 'recursive_result.json'), primary_pass=result['primary_pass'])), flush=True)
