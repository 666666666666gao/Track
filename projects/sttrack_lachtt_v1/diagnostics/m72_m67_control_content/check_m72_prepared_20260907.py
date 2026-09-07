from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import subprocess

BASE = Path('/root/autodl-tmp')
ROOT = BASE / 'sttrack_m72_m67_control_content_20260907'
SOURCE = BASE / 'm72_m67_control_content_20260907.py'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
read = lambda path: json.loads(Path(path).read_text())
spec = read(ROOT / 'spec.json')
assert sha(SOURCE) == spec['source_sha256'] == '6d1b1cbcaf236590cd5c3bab5c5bb8f5ef5aa45088c8939fbb4ad7ef7b6657a2'
assert sha(ROOT / 'spec.json') == 'cb220985b1b8d2bd910ddf1dc9a7dbd9bb5eae563fd591740fbf1afc952b357b'
assert not (ROOT / 'wait.exit').exists()
environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
subprocess.run(['/root/autodl-tmp/envs/sttrack/bin/python', str(SOURCE), 'check'], check=True, env=environment)
for name in ['run_controls.sh', 'launch_after_M71.sh']:
    subprocess.run(['bash', '-n', str(ROOT / name)], check=True)
attempt = subprocess.run(['/root/autodl-tmp/envs/sttrack/bin/python', str(SOURCE), 'eligible'],
    env=environment, capture_output=True, text=True)
assert attempt.returncode == 1
assert 'FileNotFoundError' in attempt.stderr and str(ROOT / 'wait.exit') in attempt.stderr
assert not attempt.stdout
for name in ['prefix', 'empty', 'swapped', 'result.json']:
    assert not (ROOT / name).exists(), name
parent = spec['parent_identity']
proc = Path('/proc') / str(parent['pid'])
fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
assert fields[0] != 'Z' and fields[19] == parent['start_ticks']
assert str((proc / 'cwd').resolve()) == parent['cwd']
assert [part.decode() for part in (proc / 'cmdline').read_bytes().split(b'\0') if part] == parent['argv']
(ROOT / 'premature_eligibility_rejection.log').write_text(attempt.stderr)
result = dict(status='M72_actual_CPU_preparation_and_parent_wait_gate_checked', observed_utc=datetime.now(timezone.utc).isoformat(),
    checker_sha256=sha(__file__), source_sha256=sha(SOURCE), spec_sha256=sha(ROOT / 'spec.json'),
    preparation_result_sha256=sha(ROOT / 'preparation_result.json'), both_shell_scripts_syntax_valid=True,
    premature_eligibility_exit_code=attempt.returncode, expected_rejection='Registered predecessor has not completed; wait.exit does not exist.',
    rejection_log_sha256=sha(ROOT / 'premature_eligibility_rejection.log'), parent_live_identity=parent,
    model_output_directories_created=False, GPU_forward_calls=0, full_recursive_parity_verified=False,
    new_parameters=0, public_evaluation_allowed=False, independent_model_review_pass=False)
(ROOT / 'cpu_check_result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
