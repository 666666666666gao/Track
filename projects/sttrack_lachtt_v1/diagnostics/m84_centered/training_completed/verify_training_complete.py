from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib, json, math
import torch

R=Path(__file__).parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
spec=json.loads((R/'training_spec.json').read_text())
frozen=json.loads((R/'frozen.json').read_text())
result=json.loads((R/'training/category/result.json').read_text())
rows=[json.loads(x) for x in (R/'training/category/sequence_log.jsonl').read_text().splitlines()]
assert (R/'training_category.exit').read_text().strip()=='0'
assert result['status']=='one_full_causal_fit_pass_complete'
assert sha(R/'training_spec.json')==frozen['training_spec_sha256']==result['training_spec_sha256']
assert sha(R/'recursive_spec.json')==frozen['recursive_spec_sha256']
assert len(rows)==result['sequences']==130
assert result['total_track_calls']==spec['total_training_track_calls']==186694
assert result['optimizer_steps']==spec['expected_optimizer_steps']==5798
assert result['base_parameters_and_buffers_unchanged'] and not result['evaluation_metrics_computed']
labels=Counter(); calls=0; previous_steps=0
for i,(row,planned) in enumerate(zip(rows,spec['sequence_order'])):
    assert row['sequence']==planned['sequence'] and row['sequence_index']==i
    assert row['track_calls']==planned['rgb_frames']-1
    assert sum(row['label_counts'].values())==row['track_calls']
    assert row['supervised_frames']==row['track_calls']-row['label_counts'].get('invalid',0)
    assert math.isfinite(row['maximum_preclip_gradient_norm'])
    assert row['mean_training_loss'] is None or math.isfinite(row['mean_training_loss'])
    calls+=row['track_calls']; labels.update(row['label_counts'])
    assert row['total_track_calls']==calls and row['total_optimizer_steps']>=previous_steps
    previous_steps=row['total_optimizer_steps']
assert calls==result['total_track_calls'] and previous_steps==result['optimizer_steps']
assert dict(labels)==result['training_label_counts']
for key in ['competition_frames','competition_loss_sum','competition_negatives','native_eligible_frames','preservation_kl_sum']:
    assert rows[-1]['cumulative_'+key]==result[key]
for filename,key in [('final.pth','final_checkpoint_sha256'),('sequence_log.jsonl','sequence_log_sha256'),('sampled_state_trace.jsonl','sampled_trace_sha256')]:
    assert sha(R/'training/category'/filename)==result[key]
assert sha(Path(spec['native_checkpoint']))==spec['native_checkpoint_sha256']
source_mapping={'train_causal.py':'training_script_sha256','causal_training.py':'causal_script_sha256','support_loss.py':'support_loss_sha256','window_competition.py':'window_loss_sha256','native_preservation.py':'preservation_loss_sha256','run_m84.sh':'run_queue_sha256','integration.json':'integration_sha256'}
for name,key in source_mapping.items():assert sha(R/name)==spec[key]
integration=json.loads((R/'integration.json').read_text())
for name,digest in integration['source_sha256'].items():assert sha(R/'code'/name)==digest
cp=torch.load(R/'training/category/final.pth',map_location='cpu')
latest=torch.load(R/'training/category/latest.pth',map_location='cpu')
initial=torch.load(R/'native_parity/category_zero.pth',map_location='cpu')
assert cp['status']=='complete' and cp['architecture']=='semantic_spatial_centered_v1'
assert cp['seed']==2027 and cp['completed_sequences']==130 and cp['frame_count']==186694
assert cp['optimizer_steps']==cp['actual_dataset_optimizer_steps']==5798
assert cp['training_spec_sha256']==frozen['training_spec_sha256']
assert cp['base_checkpoint_sha256']==spec['native_checkpoint_sha256']
assert cp['model'].keys()==latest['model'].keys()==initial['model'].keys()
for key,value in cp['model'].items():
    assert bool(torch.isfinite(value).all()) and torch.equal(value,latest['model'][key])
assert torch.equal(cp['model']['empty_text'],initial['model']['empty_text'])
assert any(not torch.equal(v,initial['model'][k]) for k,v in cp['model'].items())
for state in cp['optimizer']['state'].values():
    for value in state.values():
        if torch.is_tensor(value):assert bool(torch.isfinite(value).all())
trace=[json.loads(x) for x in (R/'training/category/sampled_state_trace.jsonl').read_text().splitlines()]
expected=[(p['sequence'],i) for p in spec['sequence_order'] for i in range(1,p['rgb_frames']) if i%50==0 or i==1 or i==p['rgb_frames']-1]
assert [(x['sequence'],x['frame_index']) for x in trace]==expected
out=dict(status='PASS_SAVED_TRAINING_ARTIFACTS',observed_utc=datetime.now(timezone.utc).isoformat(),
         verifier_sha256=sha(Path(__file__)),result_sha256=sha(R/'training/category/result.json'),
         final_checkpoint_sha256=result['final_checkpoint_sha256'],sequences=len(rows),track_calls=calls,optimizer_steps=previous_steps,
         checked_integration_sources=len(integration['source_sha256']),sampled_trace_rows=len(trace),training_label_counts=dict(labels),
         final_latest_model_tensors_identical=True,empty_buffer_unchanged=True,all_saved_tensors_finite=True,
         verification_scope='Saved artifacts, sequence totals, source hashes and checkpoint metadata; no independent replay of all training frames or reconstruction of the full visited-state digest; base immutability is asserted by the hash-verified trainer at completion.',
         development_metrics_computed=False)
(R/'training_saved_verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
