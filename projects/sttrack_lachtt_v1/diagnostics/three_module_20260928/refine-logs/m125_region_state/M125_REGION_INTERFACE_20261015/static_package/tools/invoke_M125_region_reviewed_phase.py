"""One reviewed remote real64/full request; invoke after the R0 suite is sealed."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import subprocess

A = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--arm', choices=['roi', 'language', 'visual'], required=True)
parser.add_argument('--phase', choices=['real64', 'full'], required=True)
parsed = parser.parse_args()
assert datetime.now(timezone.utc) >= datetime.fromisoformat('2026-10-17T12:45:00+08:00')
staging_path = A / 'M125_region_reviewed_admission_staging_20261015.json'
staging = json.loads(staging_path.read_bytes())
assert staging['status'] == 'actual_static_region_admission_package_CPU_imports_complete'
assert staging['packaged_code_files'] == 64 and staging['CPU_import_entries'] == 11
assert all(row['exit_code'] == 0 for row in staging['imports'])
operator_path = A / 'M125_REGION_DEPLOYMENT_SOURCE_R2_20261015/deploy_m125_region_fixed.py'
review_path = A / 'M125_REGION_DEPLOYMENT_AUDIT_R2_20261015/EXPERIMENT_AUDIT.json'
operator_sha = hashlib.sha256(operator_path.read_bytes()).hexdigest()
review_sha = hashlib.sha256(review_path.read_bytes()).hexdigest()
assert operator_sha == staging['files_sha256']['operator/deploy_m125_region_fixed.py']
assert review_sha == staging['files_sha256']['operator/EXPERIMENT_AUDIT.json']
review = json.loads(review_path.read_bytes())
assert review['source_verdict'] == 'PASS_SOURCE_ONLY' and review['blocking_findings'] == []
assert review['audited_input_hashes'][str(operator_path)] == 'sha256:' + operator_sha
stem = 'M125_region_' + parsed.arm + '_' + parsed.phase + '_actual_invocation'
output = A / (stem + '_stdout.json')
error = A / (stem + '_stderr.log')
assert not output.exists() and not error.exists()
attempt = dict(status='one_remote_phase_request_attempt_started', arm=parsed.arm, phase=parsed.phase,
    started_at=datetime.now(timezone.utc).isoformat(), staging_receipt_sha256=hashlib.sha256(staging_path.read_bytes()).hexdigest(),
    operator_sha256=operator_sha, R2_source_review_sha256=review_sha, automatic_retries=0)
with (A / (stem + '_attempt.json')).open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(json.dumps(attempt, indent=2) + '\n')
request = dict(root=staging['root'], arm=parsed.arm, phase=parsed.phase,
               files_sha256=staging['files_sha256'], code_sha256=staging['code_sha256'])
remote = '''from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,subprocess,sys
p=REQUEST
assert datetime.now(timezone.utc)>=datetime.fromisoformat('2026-10-17T12:45:00+08:00')
root=Path(p['root'])
for relative,digest in p['files_sha256'].items():
    assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==digest,relative
for path,digest in p['code_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
command=['/root/autodl-tmp/envs/sttrack/bin/python','-B','-u',str(root/'operator/deploy_m125_region_fixed.py'),
         '--inputs',str(root/'inputs'),'--arm',p['arm'],'--phase',p['phase']]
completed=subprocess.run(command,cwd=root/'operator')
sys.exit(completed.returncode)
'''.replace('REQUEST', repr(request))
tree = ast.parse((A / 'M124_fiftyfirst_hourly_compressed_native_observer_20261014.py').read_text())
args = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == 'args'))[:-1]
environment = dict(os.environ, SSH_ASKPASS=str(A / 'rgbd_goal_resume_askpass_20261014.cmd'),
                   SSH_ASKPASS_REQUIRE='force', DISPLAY='codex')
assert environment['RGBD_GOAL_SSH_PASSWORD']
completed = subprocess.run(args + ["CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/envs/sttrack/bin/python -B -"],
                           input=remote.encode(), capture_output=True, env=environment)
with output.open('xb') as stream:stream.write(completed.stdout)
with error.open('xb') as stream:stream.write(completed.stderr)
terminal = dict(attempt, status='one_remote_phase_request_attempt_terminal',
                remote_exit_code=completed.returncode, finished_at=datetime.now(timezone.utc).isoformat())
with (A / (stem + '_terminal.json')).open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(json.dumps(terminal, indent=2) + '\n')
assert completed.returncode == 0, completed.stderr.decode()
print(completed.stdout.decode())
