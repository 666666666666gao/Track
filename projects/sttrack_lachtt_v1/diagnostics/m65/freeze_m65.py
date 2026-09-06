"""Freeze the matched experiment after real Train smoke, before formal optimization."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
ROOT=Path('/root/autodl-tmp/sttrack_m65_category_null_support_20260907')
PARENT=Path('/root/autodl-tmp/sttrack_m58_semantic_spatial_v2_20260906')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
assert not (ROOT/'training_spec.json').exists() and not (ROOT/'training').exists()
p=json.loads((ROOT/'preparation.json').read_text());smoke=json.loads((ROOT/'causal_smoke_result.json').read_text())
assert smoke['status']=='paired_fit_smoke_complete' and smoke['causal_and_deployed_boxes_scores_queries_templates_exact']
assert smoke['preparation_sha256']==sha(ROOT/'preparation.json') and smoke['checker_sha256']==sha(ROOT/'check_m65.py')
assert smoke['source_unchanged'] and smoke['formal_optimizer_steps']==0 and not smoke['smoke_weights_saved']
assert smoke['initial_adapter_state_sha256']['control']==smoke['initial_adapter_state_sha256']['null']
assert all(x['base_parameters_and_buffers_unchanged'] for x in smoke['arms'].values())
integration=json.loads((ROOT/'integration.json').read_text())
for n,h in integration['source_sha256'].items():assert sha(ROOT/'code'/n)==h
s=json.loads((PARENT/'training_spec.json').read_text());parent_sha=sha(PARENT/'training_spec.json')
assert parent_sha==p['parent_inputs']['training_spec.json']
for key in ['preceding_training_spec_sha256','supervision_diagnostic_sha256','text_preparation_sha256','runtime_estimate_seconds_per_arm']:s.pop(key)
s.update(status='frozen_before_formal_training',observed_utc=datetime.now(timezone.utc).isoformat(),revision='m65_category_null_support_v1',
    parent_training_spec_sha256=parent_sha,primary_arm='null',control_arm='control',architecture='semantic_spatial_support_v1',
    hypothesis='A fixed zero-logit, zero-value no-support alternative improves dense category-conditioned recursive tracking over identical category-trained Control.',
    integration_sha256=sha(ROOT/'integration.json'),causal_script_sha256=sha(ROOT/'causal_training.py'),
    training_script_sha256=sha(ROOT/'train_causal.py'),freeze_script_sha256=sha(__file__),
    inventory_sha256=sha(ROOT/'data_inventory.json'),text_fit_sha256=sha(ROOT/'text_fit.pt'),text_development_sha256=sha(ROOT/'text_development.pt'),
    preparation_sha256=sha(ROOT/'preparation.json'),initial_checkpoint_sha256=p['initial_checkpoint_sha256'],
    causal_smoke_result_sha256=sha(ROOT/'causal_smoke_result.json'),run_queue_sha256=sha(ROOT/'run_m65.sh'),
    text_control='Both arms keep the same generated category slot0; valid attrs1..4 become CLIP empty in training and development. Original 5-slot masks and padding retained.',
    architecture_control='Same 289154 learned parameters and zero-residual parameter tensors; only null arm adds fixed zero logit/value to the attribute normalization. No learned null vector.',
    promotion_gates=dict(null_pooled_mean_vs_native_minimum=.002,null_pooled_mean_vs_control_minimum=.001,
        null_macro_mean_no_less_than_native_and_control=True,null_low_frames_no_more_than_native_and_control=True,
        null_H10_no_more_than_native_and_control=True,no_new_failure_on_native_or_control_zero_H10_sequences=True),
    selection_protocol='One predeclared seed and final checkpoint after one complete 130-sequence pass. No early checkpoint or development-driven selection.',
    historical_comparator='M59/M64 category deployment used full-attribute-trained M58. It remains a reported historical comparator, not evidence that train/deploy mismatch caused M64 failure.',
    scope_limitations=['Control versus Null identifies this normalization intervention, not the independent contribution of category semantics.',
      'No claim of fine-grained same-class attribute disambiguation from category-only inputs.',
      'Same-head empty/swapped-category tests are required after development gates and before another VOT low22.',
      'Reused Train development22 and VOT low22 are development sets; one seed does not establish statistical significance.'],
    after_training='Seal both final-head development22 predictions before GT. Apply frozen paired gates. If passed, perform content attribution before low22. No automatic full-dataset evaluation.',
    runtime_estimate_seconds_per_arm=12756.4,monitoring='First coverage/numerical check after 240 seconds; subsequent checks near estimated finish or for an observed fault.',
    independent_review_pass=False,independent_reviewer_status='No fresh independent review; previously observed Astra/max quota unavailable. Executor validation only.',
    final_three_dataset_metrics_exist=False)
assert s['total_training_track_calls']==186694 and len(s['sequence_order'])==130
assert set(s['development_sequences']).isdisjoint(x['sequence'] for x in s['sequence_order'])
write(ROOT/'training_spec.json',s)
r=json.loads((PARENT/'recursive_spec.json').read_text())
r.update(status='frozen_before_formal_training',observed_utc=datetime.now(timezone.utc).isoformat(),
    training_spec_sha256=sha(ROOT/'training_spec.json'),runner_sha256=sha(ROOT/'run_recursive.py'),
    queue_sha256=sha(ROOT/'run_m65.sh'),metric_sha256=sha(ROOT/'recursive_metric.py'),variants=['control','null'],public_evaluation=False)
write(ROOT/'recursive_spec.json',r)
print(json.dumps(dict(status=s['status'],training_spec_sha256=sha(ROOT/'training_spec.json'),recursive_spec_sha256=sha(ROOT/'recursive_spec.json'),
    track_calls_each=186694,maximum_optimizer_steps=s['maximum_optimizer_steps'],estimated_training_hours_each=12756.4/3600),indent=2))
