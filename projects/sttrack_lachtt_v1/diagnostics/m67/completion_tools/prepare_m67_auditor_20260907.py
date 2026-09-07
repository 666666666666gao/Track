from pathlib import Path
import hashlib
BASE=Path('/root/autodl-tmp');OLD=BASE/'audit_m65_completed_20260907.py';OUT=BASE/'audit_m67_completed_20260907.py'
assert hashlib.sha256(OLD.read_bytes()).hexdigest()=='ef6a3f5f9f7b8635497fc993fded0924e7f37f5b67d14c3fab46fc27b7e596db'
s=OLD.read_text()
s=s.replace("ROOT=BASE/'sttrack_m65_category_null_support_20260907'","ROOT=BASE/'sttrack_m67_supervised_semantic_support_20260907'")
s=s.replace("PARENT=BASE/'sttrack_m58_semantic_spatial_v2_20260906'","PARENT=BASE/'sttrack_m65_category_null_support_20260907'")
s=s.replace("TRAIN_SHA='fc04a897f3d3e246203981a2cb3b83ea50075392e30c0c960c8908eaaeabb93b'","TRAIN_SHA='2027b1893343e5e1520c447bd82a5c45468542cecaab0f59b3ec72a345e48b2e'")
s=s.replace("RECURSIVE_SHA='09f3f193de81f9cf91518b2c05497bfea30e8ab88db812e3585f6d3618767723'","RECURSIVE_SHA='d4dd7ca74b8f4b318f8e32f961bc21a10b95aa3802137dd293b49948d83411c5'")
s=s.replace("'null'","'support'").replace('training_null.exit','training_support.exit').replace('null_recursive.exit','support_recursive.exit')
s=s.replace('null_pooled_','support_pooled_').replace('sustained_null_H10','sustained_support_H10').replace('run_m65.sh','run_m67.sh')
s=s.replace('M65','M67')
s=s.replace("'54fd8445297e1eef761fa36e8a17def01afa027f71371984998f08229f6f5ee9'","'0101fa8185855044dda7462df2f9606b6e58c88248e8a7f1cf87497fb055b150'")
s=s.replace("recompute(PARENT,['text','visual']","recompute(PARENT,['control','null']")
s=s.replace("r['null_support']==(arm=='support')","r['null_support']")
s=s.replace("x['null_support']==(arm=='support') and x['use_text']","x['null_support'] and x['use_text']")
s=s.replace("assert r['training_spec_sha256']==TRAIN_SHA and r['base_parameters_and_buffers_unchanged']", "assert r['training_spec_sha256']==TRAIN_SHA and r['base_parameters_and_buffers_unchanged']\n        assert r['support_loss_weight']==training['support_loss_weights'][arm]")
s=s.replace("assert final['base_checkpoint_sha256']==training['native_checkpoint_sha256']", "assert final['base_checkpoint_sha256']==training['native_checkpoint_sha256']\n        assert final['support_loss_weight']==latest['support_loss_weight']==training['support_loss_weights'][arm]")
s=s.replace("calls=steps=0;labels=Counter();scheduled_writes=0", "calls=steps=0;labels=Counter();scheduled_writes=0;support_rows=[]")
needle="                for k in ['previous_bbox','bbox']:assert np.isfinite(x[k]).all() and np.asarray(x[k])[2:].min()>0"
replacement=needle+'''
                if arm=='support' and valid[f]:
                    assert x['support_weight']==.1 and math.isfinite(x['support_loss']) and x['support_loss']>=0
                    assert x['support_positive_cells']==int(x['label']=='centre_inside')
                    assert 0<=x['support_negative_cells']<=256-x['support_positive_cells']
                    assert 0<=x['positive_null_mass']<=1 and 0<=x['negative_null_mass']<=1
                    support_rows.append(x)
                else:
                    assert 'support_loss' not in x and 'support_weight' not in x
'''
assert needle in s;s=s.replace(needle,replacement)
s=s.replace("initial_adapter_state_sha256=r['initial_adapter_state_sha256'])", "initial_adapter_state_sha256=r['initial_adapter_state_sha256'],support_loss_weight=training['support_loss_weights'][arm],support_sampled_rows_checked=len(support_rows))")
s=s.replace("('run_m67.sh','run_queue_sha256')", "('support_loss.py','support_loss_sha256'),('run_m67.sh','run_queue_sha256')")
needle="    gates=dict(mean_vs_native="
replacement="""    parent_result=read(PARENT/'recursive_result.json')
    assert sha(PARENT/'recursive_result.json')=='0101fa8185855044dda7462df2f9606b6e58c88248e8a7f1cf87497fb055b150'
    protected=[seq for seq,x in parent_result['per_sequence']['control'].items() if x['failure_episodes']==0]
    assert protected==training['protected_prior_control_sequences']
    broken['prior_control']=[seq for seq in protected if per['support'][seq]['failure_episodes']>0]
    gates=dict(prior_control_success_protection=not broken['prior_control'],mean_vs_native="""
assert needle in s;s=s.replace(needle,replacement)
s=s.replace("    harms=[]", "    assert broken['prior_control']==result['broken_prior_control_success_sequences']\n    historical_control_delta={k:control[k]-parent_result['aggregates']['null'][k] for k in ['mean_iou','macro_sequence_mean_iou','low_iou_frames','failure_episodes']}\n    harms=[]")
s=s.replace("training=train,families=families,recomputed_aggregates=aggregate", "training=train,families=families,matched_control_minus_historical_M65_Null=historical_control_delta,recomputed_aggregates=aggregate")
assert "r['null_support']==" not in s and 'M65_artifacts' not in s
OUT.write_text(s)
print('M67_AUDITOR_SHA256',hashlib.sha256(OUT.read_bytes()).hexdigest())
