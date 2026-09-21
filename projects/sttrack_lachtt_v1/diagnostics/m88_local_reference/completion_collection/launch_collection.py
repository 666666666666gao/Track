from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,time

R=Path(__file__).resolve().parent
parent=R.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (R/'launch.json').exists() and not (R/'collection_result.json').exists()
launch=json.loads((parent/'launch.json').read_text())
assert launch['controller_pid']==48174
state=subprocess.check_output(['ps','-p','48174','-o','stat=,args='],text=True).strip()
assert not state.split()[0].startswith('Z') and str(parent/'run_m88.sh') in state
review=json.loads((R/'code_review_receipt.json').read_text())
assert review['blocking_issues']==0
assert sha(R/'CODE_REVIEW.md')==review['report_sha256']
for n,h in review['reviewed_sources'].items():assert sha(R/n)==h,n
subprocess.run(['bash','-n',str(R/'run_collection.sh')],check=True)
with (R/'launcher.log').open('xb') as log:
    process=subprocess.Popen(['bash',str(R/'run_collection.sh')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(2)
assert process.poll() is None
receipt=dict(status='waiting_on_live_original_controller',observed_utc=datetime.now(timezone.utc).isoformat(),
    collector_pid=process.pid,controller_pid=48174,source_sha256=sha(Path(__file__)),collector_source_sha256=sha(R/'collect_completed.py'),
    review_sha256=sha(R/'code_review_receipt.json'),new_training_steps=0,new_tracking_calls=0)
(R/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
