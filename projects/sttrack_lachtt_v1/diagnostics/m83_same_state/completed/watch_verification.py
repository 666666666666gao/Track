from pathlib import Path
import subprocess,time
r=Path('/root/autodl-tmp/sttrack_m83_same_state_20260920')
while not (r/'controller.exit').exists():time.sleep(240)
assert (r/'controller.exit').read_text().strip()=='0'
with (r/'verification.log').open('w') as f:
    rc=subprocess.call(['/root/autodl-tmp/envs/sttrack/bin/python',str(r/'verify_m83_saved.py')],stdout=f,stderr=subprocess.STDOUT)
(r/'verification.exit').write_text(str(rc)+'\n')
raise SystemExit(rc)
