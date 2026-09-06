"""Finalize sealed M59 Train controls after both persistent workers finish."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

ROOT=Path('/root/autodl-tmp/sttrack_m59_fixed_head_content_diagnostic_20260906')
PYTHON='/root/autodl-tmp/envs/sttrack/bin/python'


def main():
    assert (ROOT/'worker0.exit').read_text().strip()=='0'
    while not (ROOT/'worker1.exit').exists():
        time.sleep(240)
    worker1=int((ROOT/'worker1.exit').read_text())
    if worker1!=0:
        (ROOT/'finalization_result.json').write_text(json.dumps(dict(status='analysis_not_run_worker1_failed',worker1_exit=worker1,public_evaluation_allowed=False),indent=2)+'\n')
        return worker1
    with (ROOT/'analysis.log').open('w') as log:
        done=subprocess.run([PYTHON,str(ROOT/'run_controls.py'),'--analyze'],stdout=log,stderr=subprocess.STDOUT)
    (ROOT/'analysis.exit').write_text(str(done.returncode)+'\n')
    (ROOT/'finalization_result.json').write_text(json.dumps(dict(status='completed' if done.returncode==0 else 'analysis_failed',
        analysis_exit=done.returncode,observed_utc=datetime.now(timezone.utc).isoformat(),public_evaluation_allowed=False),indent=2)+'\n')
    return done.returncode


if __name__=='__main__':
    raise SystemExit(main())
