"""Actual local pre-deadline rejection check; no remote invocation is permitted."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess
import sys

A = Path(__file__).resolve().parent
assert datetime.now(timezone.utc) < datetime.fromisoformat('2026-10-17T12:45:00+08:00')
folder = A / 'M125_REGION_STATIC_PACKAGE_VALIDATION_20261015'
folder.mkdir()
source = A / 'invoke_M125_region_reviewed_phase.py'
stem = 'M125_region_roi_real64_actual_invocation'
paths = [A / (stem + suffix) for suffix in ['_attempt.json', '_stdout.json', '_stderr.log', '_terminal.json']]
assert not any(path.exists() for path in paths)
command = [sys.executable, '-B', '-X', 'utf8', str(source), '--arm', 'roi', '--phase', 'real64']
completed = subprocess.run(command, cwd=A, capture_output=True)
(folder / 'actual_local_rejection.stdout.txt').write_bytes(completed.stdout)
(folder / 'actual_local_rejection.stderr.txt').write_bytes(completed.stderr)
assert completed.returncode == 1
assert b'assert datetime.now(timezone.utc)' in completed.stderr and b'AssertionError' in completed.stderr
assert not any(path.exists() for path in paths)
result = dict(status='actual_local_predeadline_guard_rejection_verified',
    at=datetime.now(timezone.utc).isoformat(), command=command, child_exit_code=completed.returncode,
    evaluation_type='local_invocation_boundary_only', rejected_before_transport=True,
    phase_attempt_or_receipts_created=False, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    SSH_calls=0, original_R0_progress_queries=0, GPU_query_commands=0, NN_calls=0)
(folder / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
