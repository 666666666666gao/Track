from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
ROOT=Path('/root/autodl-tmp/sttrack_m67_supervised_semantic_support_20260907')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
assert not (ROOT/'training_spec.json').exists() and not (ROOT/'training').exists()
assert (ROOT/'causal_smoke.exit').read_text().strip()=='0'
c=read(ROOT/'causal_smoke_result.json');s=read(ROOT/'prepared_training_spec.json');p=read(ROOT/'preparation.json')
assert c['status']=='fit_smoke_pass' and c['prepared_spec_sha256']==sha(ROOT/'prepared_training_spec.json')
assert c['checker_sha256']==sha(ROOT/'check_m67.py') and c['preparation_sha256']==sha(ROOT/'preparation.json')
assert c['boxes_scores_queries_templates_exact'] and c['zero_weight_original_loss_on_same_outputs_exact'] and c['original_localization_loss_source_identical'] and c['same_output_loss_checks']==96
assert not c['smoke_weights_saved'] and c['formal_optimizer_steps']==0
for arm in ['control','support']:
    assert c['arms'][arm]['base_parameters_and_buffers_unchanged'] and c['arms'][arm]['optimizer_steps']==3
    assert sha(ROOT/'native_parity'/(arm+'_zero.pth'))==s['initial_checkpoint_sha256'][arm]
assert sha(ROOT/'causal_training.py')==s['causal_script_sha256'] and sha(ROOT/'support_loss.py')==s['support_loss_sha256']
assert sha(ROOT/'train_causal.py')==s['training_script_sha256'] and sha(ROOT/'run_m67.sh')==s['run_queue_sha256']
for n,h in read(ROOT/'integration.json')['source_sha256'].items():assert sha(ROOT/'code'/n)==h
s.update(status='frozen_before_formal_training',observed_utc=datetime.now(timezone.utc).isoformat(),
    preparation_sha256=sha(ROOT/'preparation.json'),causal_smoke_result_sha256=sha(ROOT/'causal_smoke_result.json'),
    freeze_script_sha256=sha(__file__),runtime_estimate_seconds_per_arm=13700,
    monitoring='Initial real coverage/numerical check after about 240 seconds; subsequent checks near runtime estimate or observed anomaly.')
write(ROOT/'training_spec.json',s)
r=read(ROOT/'prepared_recursive_spec.json');assert r['runner_sha256']==sha(ROOT/'run_recursive.py')
r.update(status='frozen_before_formal_training',observed_utc=datetime.now(timezone.utc).isoformat(),training_spec_sha256=sha(ROOT/'training_spec.json'))
write(ROOT/'recursive_spec.json',r)
print(json.dumps(dict(training_spec_sha256=sha(ROOT/'training_spec.json'),recursive_spec_sha256=sha(ROOT/'recursive_spec.json'),
    support_loss_weights=s['support_loss_weights'],track_calls_each=s['total_training_track_calls'],protected_prior_control_sequences=s['protected_prior_control_sequences']),indent=2))
