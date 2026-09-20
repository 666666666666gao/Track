"""Prepare an isolated single-arm experiment, without launching or freezing it."""
from pathlib import Path
import hashlib,json,shutil,torch

R=Path(__file__).parent
P=Path('/root/autodl-tmp/sttrack_m82_native_preservation_20260909')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
assert R.name=='sttrack_m84_centered_20260920' and not (R/'training_spec.json').exists()
assert sha(P/'recursive_result.json')=='823d0560db3acfdd594c53ecfae239ebb59ef77b3b2776bc9037dcdaa3a259ea'
parent=read(P/'training_spec.json');integration=read(P/'integration.json')
for name,value in integration['source_sha256'].items():assert sha(P/'code'/name)==value
shutil.copytree(P/'code',R/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ['causal_training.py','support_loss.py','window_competition.py','native_preservation.py','recursive_metric.py','data_inventory.json']:
    shutil.copyfile(P/name,R/name)
shutil.copyfile(R/'centered_semantic_adapter.py',R/'code/lib/models/sttrack/centered_semantic_adapter.py')
path=R/'code/lib/test/tracker/sttrack_semantic.py'
source=path.read_text()
source=source.replace('from lib.models.sttrack.semantic_spatial_adapter import SemanticSpatialAdapter',
    'from lib.models.sttrack.centered_semantic_adapter import CenteredSemanticSpatialAdapter')
source=source.replace("'semantic_spatial_support_v1'","'semantic_spatial_centered_v1'")
source=source.replace("adapter = SemanticSpatialAdapter(null_support=checkpoint['null_support'])",
    "adapter = CenteredSemanticSpatialAdapter(checkpoint['model']['empty_text'].float(), null_support=checkpoint['null_support'])")
assert 'adapter = CenteredSemanticSpatialAdapter' in source
path.write_text(source)
source=(P/'train_causal.py').read_text().replace("choices=['empty', 'category']","choices=['category']")
source=source.replace(str(P),str(R)).replace("'semantic_spatial_support_v1'","'semantic_spatial_centered_v1'")
source=source.replace("assert frame_count == spec['total_training_track_calls']",
    "assert frame_count == spec['total_training_track_calls']\n    assert steps == spec['expected_optimizer_steps']")
source=source.replace('Single-seed category/empty causal training','Single-seed centered Category causal training')
(R/'train_causal.py').write_text(source)
integration['source_sha256']['lib/test/tracker/sttrack_semantic.py']=sha(path)
integration['source_sha256']['lib/models/sttrack/centered_semantic_adapter.py']=sha(R/'code/lib/models/sttrack/centered_semantic_adapter.py')
integration['parent_M82_integration_sha256']=sha(P/'integration.json')
(R/'integration.json').write_text(json.dumps(integration,indent=2)+'\n')

proof=[];fit=None
for split,arm in [('fit','category'),('development','category'),('development','swapped')]:
    item=parent['banks'][split][arm];assert sha(Path(item['path']))==item['sha256']
    bank=torch.load(item['path'],map_location='cpu')
    assert bool(bank['mask'][:,0].all())
    difference=(bank['tokens'][:,0].float()-bank['empty'].float()).abs().amax(dim=1)
    assert bool((difference>0).all())
    proof.append(dict(split=split,arm=arm,sequences=len(bank['sequences']),all_category_slots_nonempty=True,
        minimum_max_abs_difference_from_empty=float(difference.min()),bank_sha256=item['sha256']))
    if split=='fit':fit=bank
    else:assert torch.equal(bank['empty'].float(),fit['empty'].float())
zero=torch.load(P/'native_parity/category_zero.pth',map_location='cpu')
original={k:v.clone() for k,v in zero['model'].items()}
assert all(bool((v==0).all()) for k,v in original.items() if k in ['delta.2.weight','delta.2.bias'])
zero['model']['empty_text']=fit['empty'].float().clone()
zero['architecture']='semantic_spatial_centered_v1'
(R/'native_parity').mkdir();torch.save(zero,R/'native_parity/category_zero.pth')
assert all(torch.equal(v,zero['model'][k]) for k,v in original.items())

keys=['supervision_coordinate_convention','epochs','learned_parameters','visual_base_frozen_eval',
    'native_checkpoint','native_checkpoint_sha256','dataset_root','development_sequences','sequence_order',
    'total_training_image_frames','total_training_track_calls','maximum_optimizer_steps','optimizer',
    'learning_rate','weight_decay','gradient_accumulation_frames','gradient_clip','loss','invalid_gt',
    'target_centre_outside_crop','state_protocol','backprop_through_time_or_discrete_crops',
    'native_update_interval','native_update_threshold','caption_protocol','native_result_sha256',
    'checkpoint_retention','optimizer_stop_conditions','seed','banks','window_loss_weight',
    'window_hard_negative_count','window_negative_maximum_iou','preservation_weight',
    'preservation_teacher_minimum_iou','preservation_teacher','preservation_eligibility','preservation_distribution']
spec={k:parent[k] for k in keys}
spec.update(status='prepared_not_frozen',architecture='semantic_spatial_centered_v1',primary_arm='category',
    trained_arms=['category'],support_loss_weights={'category':0.},stored_parameters=289154,cancelled_final_bias_dimensions=768,
    expected_optimizer_steps=5798,
    primary_control='M82 Category as nonempty-equivalent hard-empty mask control; no extra Empty training',
    parent_training_spec_sha256=sha(P/'training_spec.json'),parent_result_path=str(P/'recursive_result.json'),
    parent_result_sha256=sha(P/'recursive_result.json'),plan_sha256=sha(R/'EXPERIMENT_PLAN.md'),
    inventory_sha256=sha(R/'data_inventory.json'),integration_sha256=sha(R/'integration.json'),
    initial_checkpoint_sha256={'category':sha(R/'native_parity/category_zero.pth')},
    training_script_sha256=sha(R/'train_causal.py'),causal_script_sha256=sha(R/'causal_training.py'),
    support_loss_sha256=sha(R/'support_loss.py'),window_loss_sha256=sha(R/'window_competition.py'),
    preservation_loss_sha256=sha(R/'native_preservation.py'),run_queue_sha256=sha(R/'run_m84.sh'),
    preflight_source_sha256=sha(R/'preflight_m84.py'),all_development_predictions_before_GT=True,
    public_evaluation='No automatic public evaluation or promotion',fixed_final_checkpoint_only=True,
    acceptance_definition='4 M82 increment + 5 native including zero-H10 protection + 4 Swapped content + 1 per-sequence Empty/native metric parity; all required')
assert spec['total_training_track_calls']==186694 and spec['maximum_optimizer_steps']==5896 and spec['seed']==2027
(R/'training_spec.json').write_text(json.dumps(spec,indent=2)+'\n')
old_rec=read(P/'recursive_spec.json')
rec=dict(status='prepared_not_frozen',training_spec_sha256=sha(R/'training_spec.json'),
    runner_sha256=sha(R/'run_recursive.py'),metric_sha256=sha(R/'recursive_metric.py'),
    queue_sha256=sha(R/'run_m84.sh'),native_result_path=old_rec['native_result_path'],
    variants=['category','category_empty','category_swapped'],cases=old_rec['cases'],
    full_prediction_families_sealed_before_metric_GT=True,fixed_final_heads_only=True,poll_interval_seconds=240,
    public_evaluation=False)
(R/'recursive_spec.json').write_text(json.dumps(rec,indent=2)+'\n')
proof=dict(status='prepared',nonempty_inputs=proof,initial_all_M82_trainable_tensors_exact=True,
    added_nontrainable_buffer='empty_text',base_and_all_other_runtime_sources_unchanged=True,
    training_spec_sha256=sha(R/'training_spec.json'),recursive_spec_sha256=sha(R/'recursive_spec.json'))
(R/'preparation_receipt.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof,indent=2))
