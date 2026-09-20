from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess
R=Path(__file__).parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
assert not (R/'frozen.json').exists() and not (R/'training').exists()
review=read(R/'code_review_receipt.json')
assert review['blocking_issues']==0 and review['review_model']=='gpt-6-astra' and review['reasoning_effort']=='max'
assert sha(R/'CODE_REVIEW.md')==review['report_sha256']
for n,h in review['reviewed_sources'].items():assert sha(R/n)==h
spec=read(R/'training_spec.json');rec=read(R/'recursive_spec.json');pre=read(R/'preflight_result.json')
assert (R/'preflight.exit').read_text().strip()=='0' and pre['status']=='M84_native_parity_and_causal_smoke_pass'
assert pre['training_spec_sha256']==sha(R/'training_spec.json') and pre['source_sha256']==sha(R/'preflight_m84.py')
assert pre['formal_optimizer_steps']==0 and pre['smoke_optimizer_steps']==3 and not pre['smoke_weights_saved']
assert rec['training_spec_sha256']==sha(R/'training_spec.json') and rec['runner_sha256']==sha(R/'run_recursive.py')
assert spec['run_queue_sha256']==rec['queue_sha256']==sha(R/'run_m84.sh')
assert sha(R/'EXPERIMENT_PLAN.md')==spec['plan_sha256']
subprocess.run(['bash','-n',str(R/'run_m84.sh')],check=True)
frozen=dict(status='frozen_before_training',observed_utc=datetime.now(timezone.utc).isoformat(),
    training_spec_sha256=sha(R/'training_spec.json'),recursive_spec_sha256=sha(R/'recursive_spec.json'),
    preflight_sha256=sha(R/'preflight_result.json'),review_receipt_sha256=sha(R/'code_review_receipt.json'),
    seed=2027,trained_arms=['category'],no_multiseed=True,final_checkpoint_only=True)
(R/'frozen.json').write_text(json.dumps(frozen,indent=2)+'\n')
print(json.dumps(frozen))
