"""Prepare one fixed-seed pair with a new training loss and unchanged inference."""
from pathlib import Path
from datetime import datetime, timezone
import argparse,ast,difflib,hashlib,json,shutil,subprocess
B=Path('/root/autodl-tmp');P=B/'sttrack_m73_paired_lexical_replication_20260907/seed2027'
R=B/'sttrack_m77_window_competition_20260907'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def now():return datetime.now(timezone.utc).isoformat()
def replace(s,a,b):
    assert s.count(a)==1,a
    return s.replace(a,b)

def prepare():
    import torch
    assert not R.exists()
    assert sha(P/'training_spec.json')=='0f90ad42fc4084271e8ad6153714d0252531b1c08f71e2646cb94697c8f506be'
    assert sha(P/'recursive_spec.json')=='0dcb0dfec7da7c0bc0398cd99dcaf56b6747b176294a35a109fc9c72218280c9'
    assert sha(P/'recursive_result.json')=='4e9a52ead8a8ae5fb05250b95bf156d9fed1045d9e400fe29f140d2942a0a546'
    M76=B/'sttrack_m76_same_state_content_20260907'
    assert sha(M76/'completed_evidence_audit.json')=='3297d636109949ca08325d6d48c9ea98b7686c4211fb18480d0cdae16eb39239'
    t=read(P/'training_spec.json');rs=read(P/'recursive_spec.json')
    assert sha(P/'train_causal.py')==t['training_script_sha256']
    assert sha(P/'run_recursive.py')==rs['runner_sha256']
    R.mkdir();(R/'native_parity').mkdir()
    for n,h in read(P/'integration.json')['source_sha256'].items():
        src=P/'code'/n;assert sha(src)==h
        dest=R/'code'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
    for n in ['integration.json','data_inventory.json','causal_training.py','support_loss.py','recursive_metric.py']:
        shutil.copyfile(P/n,R/n)
    shutil.copyfile(B/'m77_window_competition_20260907.py',R/'window_competition.py')
    for arm in ['category','empty']:
        shutil.copyfile(P/'native_parity'/(arm+'_zero.pth'),R/'native_parity'/(arm+'_zero.pth'))
        assert sha(R/'native_parity'/(arm+'_zero.pth'))==t['initial_checkpoint_sha256'][arm]
    for split in ['fit','development']:
        cat=torch.load(t['banks'][split]['category']['path'],map_location='cpu')
        empty=torch.load(t['banks'][split]['empty']['path'],map_location='cpu')
        assert cat['sequences']==empty['sequences'] and torch.equal(cat['mask'],empty['mask'])
        assert torch.equal(cat['tokens'][~cat['mask']],empty['tokens'][~cat['mask']])
        assert torch.equal(empty['tokens'][cat['mask']],cat['empty'].to(cat['tokens'].dtype).expand(int(cat['mask'].sum()),-1))
        for arm in ['category','empty']:assert sha(t['banks'][split][arm]['path'])==t['banks'][split][arm]['sha256']
    old=(P/'train_causal.py').read_text();train=old.replace(str(P),str(R))
    train=replace(train,'from causal_training import CausalTrainingTracker, supervision','from causal_training import CausalTrainingTracker\n    from window_competition import supervision')
    train=replace(train,"    assert sha(root / 'support_loss.py') == spec['support_loss_sha256']","    assert sha(root / 'support_loss.py') == spec['support_loss_sha256']\n    assert sha(root / 'window_competition.py') == spec['window_loss_sha256']")
    train=replace(train,"support_weight=spec['support_loss_weights'][args.arm])","output_window=tracker.output_window)")
    train=replace(train,"    with (output / 'sequence_log.jsonl')", "    rank_frames = rank_negative_sum = 0\n    rank_loss_sum = 0.\n    with (output / 'sequence_log.jsonl')")
    train=replace(train,"                labels[diagnostic['label']] += 1","                labels[diagnostic['label']] += 1\n                if diagnostic['competition_loss'] is not None:\n                    rank_frames += 1\n                    rank_loss_sum += diagnostic['competition_loss']\n                    rank_negative_sum += diagnostic['competition_negatives']")
    train=replace(train,"                mean_training_loss=loss_sum/supervised if supervised else None, template_writes=writes,","                mean_training_loss=loss_sum/supervised if supervised else None, template_writes=writes,\n                cumulative_competition_frames=rank_frames, cumulative_competition_loss_sum=rank_loss_sum, cumulative_competition_negatives=rank_negative_sum,")
    train=replace(train,"        learned_parameters=289154, backward_through_crops_or_time=False,","        learned_parameters=289154, backward_through_crops_or_time=False,\n        competition_frames=rank_frames, competition_loss_sum=rank_loss_sum, competition_negatives=rank_negative_sum, window_loss_sha256=spec['window_loss_sha256'],")
    train=replace(train,'Matched category versus empty-content causal training; identical Null architecture and zero auxiliary loss.','Single-seed category/empty causal training with final-window hard-negative competition.')
    (R/'train_causal.py').write_text(train)
    (R/'training_change.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),train.splitlines(True),fromfile='M73/train_causal.py',tofile='M77/train_causal.py')))
    runner=(P/'run_recursive.py').read_text().replace(str(P),str(R))
    runner=replace(runner,"    result=dict(status='complete_recursive_development'", "    parent_path=Path(training['parent_recursive_result_path'])\n    assert sha(parent_path)==training['parent_recursive_result_sha256']\n    parent=json.loads(parent_path.read_text())['aggregates']\n    prior=parent['category']\n    gates.update(mean_vs_M73_category=primary['mean_iou']>=prior['mean_iou']+.001,\n        macro_vs_M73_category=primary['macro_sequence_mean_iou']>=prior['macro_sequence_mean_iou'],\n        low_frames_vs_M73_category=primary['low_iou_frames']<=prior['low_iou_frames'],\n        H10_vs_M73_category=primary['failure_episodes']<=prior['failure_episodes'])\n    result=dict(status='complete_recursive_development'")
    runner=replace(runner,"        aggregates=aggregates, per_sequence=per, gates=gates, primary_pass=all(gates.values()),","        aggregates=aggregates, per_sequence=per, gates=gates, primary_pass=all(gates.values()),\n        parent_M73_aggregates=parent, parent_M73_result_sha256=sha(parent_path),")
    runner=replace(runner,"next='Fixed-weight category versus empty and swapped category controls before low22' if all(gates.values()) else 'Stop this frozen revision; diagnose complete trajectories'","next='Complete preregistered fixed-head Empty/Swapped content diagnostics regardless of primary gate status; no automatic public evaluation'")
    (R/'run_recursive.py').write_text(runner)
    queue=(P/'run_pair.sh').read_text().replace(str(P),str(R))
    (R/'run_pair.sh').write_text(queue);subprocess.run(['bash','-n',str(R/'run_pair.sh')],check=True)
    # Preserve the actual training protocol while replacing obsolete two-seed wording.
    t.update(status='prepared_before_checks',revision='m77_final_window_competition_v1',seed=2027,
        hypothesis='Explicit final-Hann competition training improves causal tracking; category-content evidence must separately exceed matched empty training and same-weight counterfactuals.',
        architecture_control='Same M73 289154-parameter Null adapter, frozen base, initial tensors, caption banks, sequence order and optimizer. Only the added training loss differs from M73; only text content differs within the M77 pair.',
        selection_protocol='Only fixed seed2027 category final; no additional seeds or checkpoint selection.',
        loss=t['loss']+' + one unit of log-probability competition at the rounded GT cell versus up to nine highest Hann-score competitors whose decoded centers are outside GT and decoded IoU<=0.1. Native raw loss remains.',
        window_loss_weight=1.,window_hard_negative_count=9,window_negative_maximum_iou=.1,
        window_loss_sha256=sha(R/'window_competition.py'),training_script_sha256=sha(R/'train_causal.py'),run_queue_sha256=sha(R/'run_pair.sh'),
        parent_training_spec_sha256=sha(P/'training_spec.json'),parent_recursive_result_path=str(P/'recursive_result.json'),parent_recursive_result_sha256=sha(P/'recursive_result.json'),
        parent_M76_audit_sha256=sha(M76/'completed_evidence_audit.json'),
        after_training='Complete both arms and development22, then same-head Empty/Swapped diagnostic regardless of development gate status. Content results cannot override failed development gates. No automatic public evaluation.',
        scope_limitations=['One user-required fixed seed, reused Train development22; no across-seed variance claim.',
            'This changes the optimization target, not the inference architecture. It is not an established new language contribution.',
            'Inherited eleven gates remain; four comparisons additionally require Category improvement over M73 Category.',
            'In-crop ranking does not solve reappearance outside the crop. Data and head-coordinate semantics are unchanged.'],
        runtime_estimate_seconds_per_arm=15000,causal_check_sha256=None,observed_utc=now())
    write(R/'prepared_training_spec.json',t)
    rs.update(status='prepared_before_checks',runner_sha256=sha(R/'run_recursive.py'),queue_sha256=sha(R/'run_pair.sh'))
    rs.pop('training_spec_sha256')
    write(R/'prepared_recursive_spec.json',rs)
    # Checker independently measures gradient behavior and real fit-sequence training.
    check=(B/'check_m73_causal_20260907.py').read_text()
    a=check.index("p = argparse.ArgumentParser();")
    b=check.index("assert not (ROOT / 'causal_check.json')")
    check=check[:a]+"ROOT = Path('"+str(R)+"')\n"+check[b:]
    check=replace(check,'from causal_training import CausalTrainingTracker, supervision, base_supervision','from causal_training import CausalTrainingTracker, base_supervision\nfrom window_competition import supervision, competition_loss')
    check=check.replace('args.seed','2027').replace('support_weight=0.)','output_window=tracker.output_window)')
    check=replace(check,"        assert (loss is None and original is None) or (loss is not None and original is not None and torch.equal(loss, original))", "        if original is None:\n            assert loss is None\n        elif diagnostic['label']=='centre_inside':\n            assert abs(float((loss-original).detach())-diagnostic['competition_loss'])<1e-4\n            assert diagnostic['competition_negatives']>0\n        else:\n            assert torch.equal(loss,original)")
    check=check.replace('same_loss_checks','loss_contract_checks').replace('same_output_original_loss_exact=True','combined_loss_contract_verified=True').replace('same_output_loss_checks=','combined_loss_checks=')
    check=check.replace('completed_M73_native_parity_and_causal_smoke','completed_M77_native_parity_and_causal_smoke')
    # Add exact checks for inference absence and for duplicate target hypotheses.
    marker="started = time.time(); native.initialize"
    tests="""# Synthetic contract: same-instance decoded hypotheses must not become negatives.
sample=torch.full((1,1,16,16),.01,device='cuda',requires_grad=True)
with torch.no_grad():
    sample[0,0,12,12]=.8;sample[0,0,8,8]=.6;sample[0,0,7,7]=.9
sizes=torch.full((1,2,16,16),.05,device='cuda',requires_grad=True)
offsets=torch.zeros((1,2,16,16),device='cuda',requires_grad=True)
with torch.no_grad():offsets[0,:,7,7]=5.
synthetic={'score_map':sample,'size_map':sizes,'offset_map':offsets}
target=sample.new_tensor([.7,.7,.1,.1])
rank,diagnostic=competition_loss(synthetic,target,native.output_window)
rank.backward()
assert diagnostic['competition_positive_index']==204 and diagnostic['competition_negatives']==9
assert sample.grad[0,0,12,12]<0 and sample.grad[0,0,8,8]>0 and sample.grad[0,0,7,7]==0
assert sizes.grad is None and offsets.grad is None
synthetic_check=dict(loss=float(rank.detach()),positive_gradient=float(sample.grad[0,0,12,12]),false_peak_gradient=float(sample.grad[0,0,8,8]),same_instance_gradient=float(sample.grad[0,0,7,7]),no_regression_mask_gradients=True)
del sample,sizes,offsets,synthetic,rank

"""
    check=replace(check,marker,tests+marker)
    check=replace(check,"    formal_optimizer_steps=0, smoke_weights_saved=False,", "    synthetic_competition_contract=synthetic_check, formal_optimizer_steps=0, smoke_weights_saved=False,")
    (R/'check_causal.py').write_text(check)
    for p in list(R.glob('*.py')):ast.parse(p.read_text())
    spec=dict(status='prepared_M77_single_seed_before_training',observed_utc=now(),source_sha256=sha(__file__),
        training_prepared_sha256=sha(R/'prepared_training_spec.json'),recursive_prepared_sha256=sha(R/'prepared_recursive_spec.json'),check_source_sha256=sha(R/'check_causal.py'),
        seed=2027,additional_seeds=[],primary_arm='category',paired_arm='empty',
        claim='A loss aligned to final spatial competition can turn existing response evidence into recursive benefit. Text contribution is a separate paired/content test.',
        literature_reference='https://github.com/OpenSpaceAI/UVLTrack/blob/6ca34055c3447cd69b032eeb0f7cf6af6c9f3728/lib/train/actors/uvltrack.py',
        literature_boundary='Borrow the hard-negative competition idea, not code or head architecture. Use native STTrack round/crop labels and decoded severe negatives, not UVLTrack floor labels.',
        public_evaluation_allowed=False,independent_model_review_pass=False,
        expected_training_calls_per_arm=186694,expected_development_calls_per_arm=33108,content_controls_after_development_regardless_of_gate=True,
        parent_M73_result_sha256=sha(P/'recursive_result.json'),parent_M76_audit_sha256=sha(M76/'completed_evidence_audit.json'))
    write(R/'spec.json',spec);print(json.dumps(spec,indent=2))

def freeze():
    s=read(R/'spec.json');assert s['source_sha256']==sha(__file__)
    assert (R/'causal_check.exit').read_text().strip()=='0'
    p=read(R/'causal_check.json')
    assert p['checker_sha256']==s['check_source_sha256']==sha(R/'check_causal.py')
    assert p['status']=='completed_M77_native_parity_and_causal_smoke' and p['combined_loss_contract_verified']
    assert p['zero_residual_public_state_exact'] and p['base_frozen_all_arms'] and not p['smoke_weights_saved']
    assert p['formal_optimizer_steps']==0 and all(v['optimizer_steps']==3 for v in p['arms'].values())
    t=read(R/'prepared_training_spec.json');rs=read(R/'prepared_recursive_spec.json')
    assert sha(R/'prepared_training_spec.json')==s['training_prepared_sha256']==p['prepared_spec_sha256']
    assert sha(R/'prepared_recursive_spec.json')==s['recursive_prepared_sha256']
    assert t['training_script_sha256']==sha(R/'train_causal.py') and t['window_loss_sha256']==sha(R/'window_competition.py')
    assert not (R/'frozen.json').exists() and not (R/'training').exists()
    t.update(status='frozen_before_training',causal_check_sha256=sha(R/'causal_check.json'))
    write(R/'training_spec.json',t);rs.update(status='frozen_before_training',training_spec_sha256=sha(R/'training_spec.json'))
    write(R/'recursive_spec.json',rs)
    f=dict(training_spec_sha256=sha(R/'training_spec.json'),recursive_spec_sha256=sha(R/'recursive_spec.json'),spec_sha256=sha(R/'spec.json'),check_sha256=sha(R/'causal_check.json'),seed=2027,public_evaluation_allowed=False)
    write(R/'frozen.json',f);print(json.dumps(f))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','freeze']);a=p.parse_args()
    {'prepare':prepare,'freeze':freeze}[a.action]()
