from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,time
R=Path(__file__).parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (R/'launch.json').exists() and not (R/'training').exists()
frozen=json.loads((R/'frozen.json').read_text())
assert frozen['training_spec_sha256']==sha(R/'training_spec.json')
assert frozen['recursive_spec_sha256']==sha(R/'recursive_spec.json')
with (R/'controller.log').open('xb') as log:
    process=subprocess.Popen(['bash',str(R/'run_m84.sh')],cwd=str(R),stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(5)
assert process.poll() is None
children=subprocess.check_output(['pgrep','-P',str(process.pid)],text=True).strip().splitlines()
assert len(children)==1
training_pid=int(children[0]);cmd=Path('/proc/%d/cmdline'%training_pid).read_bytes().replace(b'\0',b' ').decode()
assert 'train_causal.py --arm category' in cmd
receipt=dict(status='training_started',observed_utc=datetime.now(timezone.utc).isoformat(),
    controller_pid=process.pid,training_pid=training_pid,frozen_sha256=sha(R/'frozen.json'),
    source_sha256=sha(Path(__file__)),queue_sha256=sha(R/'run_m84.sh'),seed=2027,
    trained_arms=['category'],performance_result_available=False,
    queue='Category fit -> Category and Empty parallel recursive -> Swapped recursive -> sealed GT analysis')
(R/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
