"""Verify M77 final checkpoints, training coverage and independently recomputed scalars."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,math
B=Path('/root/autodl-tmp');R=B/'sttrack_m77_window_competition_20260907';O=R/'content_followup'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def load(name,filename,digest):
    p=B/filename;assert sha(p)==digest
    s=importlib.util.spec_from_file_location(name,str(p));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def gates(per,agg,parent,t):
    c,e,n=agg['category'],agg['empty'],agg['native'];previous=parent['category'];rules=t['promotion_gates']
    return dict(prior_control_success_protection=all(per['category'][s]['failure_episodes']==0 for s in t['protected_prior_control_sequences']),
        mean_vs_native=c['mean_iou']>=n['mean_iou']+rules['category_pooled_mean_vs_native_minimum'],
        mean_vs_control=c['mean_iou']>=e['mean_iou']+rules['category_pooled_mean_vs_control_minimum'],
        macro_vs_native=c['macro_sequence_mean_iou']>=n['macro_sequence_mean_iou'],
        macro_vs_control=c['macro_sequence_mean_iou']>=e['macro_sequence_mean_iou'],
        low_frames_vs_native=c['low_iou_frames']<=n['low_iou_frames'],low_frames_vs_control=c['low_iou_frames']<=e['low_iou_frames'],
        H10_vs_native=c['failure_episodes']<=n['failure_episodes'],H10_vs_control=c['failure_episodes']<=e['failure_episodes'],
        native_success_protection=all(per['category'][s]['failure_episodes']==0 for s,v in per['native'].items() if v['failure_episodes']==0),
        control_success_protection=all(per['category'][s]['failure_episodes']==0 for s,v in per['empty'].items() if v['failure_episodes']==0),
        mean_vs_M73_category=c['mean_iou']>=previous['mean_iou']+.001,
        macro_vs_M73_category=c['macro_sequence_mean_iou']>=previous['macro_sequence_mean_iou'],
        low_frames_vs_M73_category=c['low_iou_frames']<=previous['low_iou_frames'],H10_vs_M73_category=c['failure_episodes']<=previous['failure_episodes'])

def helpers():
    return (load('m77_checkpoint_verifier','audit_m73_completed_20260907.py','6ea04715e50733a0dd41ca7b458f80c6afbd6af826b01e8865ff4674aa427747'),
        load('m77_independent_scalar','audit_m67_completed_20260907.py','1773887deb3be27aa11a4409aa77bd7581f32b8d891ecab2a03454d277e769bf'))

def completed():
    assert not (O/'completed_training_audit.json').exists()
    p=read(O/'spec.json');t=read(R/'training_spec.json');rs=read(R/'recursive_spec.json')
    assert p['auditor_sha256']==sha(__file__) and p['training_spec_sha256']==sha(R/'training_spec.json')
    assert p['recursive_spec_sha256']==sha(R/'recursive_spec.json')
    for name in ['controller.exit','training_category.exit','training_empty.exit','category_recursive.exit','empty_recursive.exit','recursive_analysis.exit']:
        assert (R/name).read_text().strip()=='0'
    for file,key in [('train_causal.py','training_script_sha256'),('window_competition.py','window_loss_sha256'),('causal_training.py','causal_script_sha256'),('run_pair.sh','run_queue_sha256')]:assert sha(R/file)==t[key]
    assert sha(R/'integration.json')==t['integration_sha256']
    for file,digest in read(R/'integration.json')['source_sha256'].items():assert sha(R/'code'/file)==digest
    for split in ['fit','development']:
        for name in ['category','empty']:
            bank=t['banks'][split][name];assert sha(bank['path'])==bank['sha256']
    checker,scalar=helpers();train={arm:checker.training_arm(R,arm,t) for arm in ['category','empty']}
    for key in ['initial_tensor_sha256','base_state_sha256','track_calls','optimizer_steps','supervised_frames']:assert train['category'][key]==train['empty'][key]
    assert train['category']['initial_tensor_sha256']=='56d03acff2972871788dd51f0eff40d0b6d888aad79fe74db4c44cdfb276ca42'
    rank={}
    for arm in ['category','empty']:
        folder=R/'training'/arm;r=read(folder/'result.json')
        assert r['window_loss_sha256']==t['window_loss_sha256']
        records=[json.loads(v) for v in (folder/'sequence_log.jsonl').read_text().splitlines()]
        frames=0;loss=0.;negatives=0
        for v in records:
            frames+=v['label_counts'].get('centre_inside',0)
            assert v['cumulative_competition_frames']==frames
            assert math.isfinite(v['cumulative_competition_loss_sum']) and v['cumulative_competition_loss_sum']>=loss
            assert negatives<=v['cumulative_competition_negatives']<=9*frames
            loss=v['cumulative_competition_loss_sum'];negatives=v['cumulative_competition_negatives']
        assert frames==r['competition_frames'] and frames>0 and loss==r['competition_loss_sum']>0
        assert negatives==r['competition_negatives']>0
        for v in [json.loads(x) for x in (folder/'sampled_state_trace.jsonl').read_text().splitlines()]:
            if v['label']=='centre_inside':
                assert math.isfinite(v['competition_loss']) and v['competition_loss']>=0
                assert 0<=v['competition_negatives']<=9 and 0<=v['competition_positive_index']<256
                assert 0<=v['competition_hann_top1_index']<256 and 0<=v['competition_positive_probability']<=1
            else:assert v['competition_loss'] is None and v['competition_negatives']==0
        rank[arm]=dict(supervised_competition_frames=frames,competition_loss_sum=loss,severe_negatives_used=negatives)
    result=read(R/'recursive_result.json')
    spec,training,per,agg,overlap,intervals,families=scalar.recompute(R,['category','empty'],result)
    parent_path=Path(t['parent_recursive_result_path']);assert sha(parent_path)==t['parent_recursive_result_sha256']
    parent=read(parent_path)['aggregates'];g=gates(per,agg,parent,t)
    assert len(g)==15 and g==result['gates'] and all(g.values())==result['primary_pass']
    harms=[]
    for seq,segments in intervals['category'].items():
        for start,end in segments:
            for ref in ['native','empty']:
                if bool((overlap[ref][seq][start:end]>=.5).all()):
                    harms.append(dict(sequence=seq,start=int(start),end_exclusive=int(end),frames=int(end-start),reference=ref))
    output=dict(status='completed_M77_checkpoint_training_coverage_and_independent_scalar_audit',observed_utc=datetime.now(timezone.utc).isoformat(),auditor_sha256=sha(__file__),
        followup_spec_sha256=sha(O/'spec.json'),training_spec_sha256=sha(R/'training_spec.json'),recursive_spec_sha256=sha(R/'recursive_spec.json'),result_sha256=sha(R/'recursive_result.json'),
        training=train,competition_training=rank,aggregates=agg,gates=g,development_gates_pass=all(g.values()),families=families,
        strict_category_H10_reference_correct_every_frame=harms,seed=2027,additional_seeds=[],new_training_calls=0,new_optimizer_steps=0,
        content_diagnostic_allowed=True,public_evaluation_allowed=False,independent_model_review_pass=False,
        scope='Checkpoint, scalar and frozen-protocol verification; base-frozen proof uses the recorded end-of-training tensor hash assertion, not a second full training replay.')
    (O/'completed_training_audit.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ['families','strict_category_H10_reference_correct_every_frame']},indent=2))

if __name__=='__main__':completed()
