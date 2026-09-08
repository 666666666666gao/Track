"""CPU-only saved-artifact audit after the already scheduled training finishes."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,os,subprocess,time

B=Path('/root/autodl-tmp')
OUT=B/'sttrack_training_audit_watch_20260908'
PY=B/'envs/sttrack/bin/python'
AUDITOR=B/'audit_m80_m81_training_20260908.py'
ROOTS={'m80':B/'sttrack_m80_block_text_dropout_20260908','m81':B/'sttrack_m81_multistart_20260908'}
EXPECTED='99ccfaaa56d135ce6db75cd0cc491259243b8554a261cc2c97df5dc9448d146c'

def state(status,**extra):
    value=dict(status=status,pid=os.getpid(),observed_utc=datetime.now(timezone.utc).isoformat(),poll_seconds=240,**extra)
    (OUT/'state.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value),flush=True)

def check_live(screen):
    result=subprocess.run(['screen','-ls'],stdout=subprocess.PIPE,check=True,text=True)
    assert any(line.split()[0].endswith('.'+screen) for line in result.stdout.splitlines() if line.strip()),screen

def wait_until(timestamp):
    remaining=timestamp-time.time()
    if remaining>0:time.sleep(remaining)

def audit(name):
    assert hashlib.sha256(AUDITOR.read_bytes()).hexdigest()==EXPECTED
    root=ROOTS[name];arms=['category'] if name=='m80' else ['category','empty']
    while True:
        exits=[root/('training_'+arm+'.exit') for arm in arms]
        for p in exits:
            if p.exists():assert p.read_text().strip()=='0',str(p)
        if all(p.exists() for p in exits):break
        check_live('sttrack_m80_dropout_20260908' if name=='m80' else 'sttrack_m81_pair_20260908')
        state('waiting_for_training_terminal',experiment=name)
        time.sleep(240)
    assert not (root/'saved_training_audit.json').exists()
    state('auditing_saved_training',experiment=name)
    with (OUT/(name+'_audit.log')).open('x') as log:
        result=subprocess.run([str(PY),str(AUDITOR),'--experiment',name,'--complete'],stdout=log,stderr=subprocess.STDOUT)
    (OUT/(name+'_audit.exit')).write_text(str(result.returncode)+'\n')
    assert result.returncode==0,name
    receipt=json.loads((root/'saved_training_audit.json').read_text())
    assert receipt['formal_training_completion_verified']
    state('saved_training_verified',experiment=name,receipt=str(root/'saved_training_audit.json'))

assert OUT.exists() and not (OUT/'state.json').exists()
check_live('sttrack_m80_dropout_20260908')
state('waiting_near_M80_estimated_completion',not_before_utc='2026-09-08T10:55:00+00:00')
wait_until(datetime(2026,9,8,10,55,tzinfo=timezone.utc).timestamp())
audit('m80')
while not (ROOTS['m81']/'launch.json').exists():
    check_live('sttrack_m81_wait_m80_20260908')
    state('waiting_for_scheduled_M81_launch')
    time.sleep(240)
check_live('sttrack_m81_pair_20260908')
# Training is estimated at about four hours. Begin terminal polling after three.
state('waiting_near_M81_estimated_completion',earliest_check_utc=datetime.fromtimestamp(time.time()+10800,timezone.utc).isoformat())
time.sleep(10800)
audit('m81')
state('both_saved_training_audits_completed')
