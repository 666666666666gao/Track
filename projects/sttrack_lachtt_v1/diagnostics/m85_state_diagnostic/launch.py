"""Launch one reviewed read-only diagnostic queue, never restart an existing run."""
import hashlib,json,shutil,subprocess,time
from pathlib import Path
R=Path(__file__).parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
spec=json.loads((R/'spec.json').read_text())
review=json.loads((R/'review_receipt.json').read_text())
assert review['blocking_findings']==0 and review['spec_sha256']==sha(R/'spec.json')
assert review['report_sha256']==sha(R/'CODE_REVIEW.md')
for key,name in [('source_sha256','same_state.py'),('analysis_sha256','analyze_saved.py'),('queue_sha256','run_replay.sh'),('plan_sha256','EXPERIMENT_PLAN.md')]:
    assert spec[key]==sha(R/name)
assert shutil.disk_usage(str(R)).free>=200*1024*1024
assert not (R/'launch.json').exists() and not (R/'preflight').exists() and not (R/'predictions').exists()
with (R/'controller.log').open('wb') as log:
    p=subprocess.Popen(['/bin/bash',str(R/'run_replay.sh')],cwd=str(R),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    time.sleep(1)
    assert p.poll() is None
cmd=Path('/proc/{}/cmdline'.format(p.pid)).read_bytes().replace(b'\0',b' ').decode()
assert str(R/'run_replay.sh') in cmd
record=dict(pid=p.pid,command=cmd,started_unix=time.time(),spec_sha256=sha(R/'spec.json'),review_receipt_sha256=sha(R/'review_receipt.json'),free_disk_bytes=shutil.disk_usage(str(R)).free)
(R/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
