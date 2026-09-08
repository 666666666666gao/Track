"""Recheck sealed M78 low22 outputs and collect neutral completion evidence."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,importlib.util,io,json,shutil,sys,tarfile

B=Path('/root/autodl-tmp');E=B/'sttrack_m78_raw_competition_20260908/candidate_evaluation'
SOURCE=B/'m78_vot_low22_20260908.py';O=E/'low22_completed_evidence'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
    assert not O.exists()
    for name in ['low22_tracking.exit','low22_analysis.exit','low22_controller.exit']:
        assert (E/name).read_text().strip()=='0'
    module_spec=importlib.util.spec_from_file_location('m78_completed_frozen_evaluator',str(SOURCE))
    m=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(m)
    execution=m.checked_execution();result=read(E/'low22_result.json')
    assert result['status']=='completed_low22_same_bundle_evaluation'
    assert result['execution_sha256']==sha(E/'low22_execution.json')
    assert result['bundle_sha256']==execution['bundle_sha256'] and result['text_protocol_sha256']==execution['text_protocol_sha256']
    sys.path.insert(0,str(m.INTERFACE));from semantic_runtime import checked_plan,text_bank
    plan,bundle=checked_plan(E/'low22_plan.json');bank=text_bank(plan,bundle)
    caption_plan=B/'sttrack_m64_category_candidate_20260907/low22_captions/plan.json'
    assert sha(caption_plan)=='5835a31ff987f9f9f1cd89f346843b3830491f8a966585880e631d022bad0ecf'
    captions=read(caption_plan);assert len(captions['cases'])==len(captions['rows'])==303
    for row in captions['rows']:
        assert sha(row['image'])==row['image_sha256'];bank.info(row['image'],row['init_bbox'])
    run=E/'low22_run';merge=read(run/'merge_result.json')
    assert sha(run/'merge_result.json')==result['merge_sha256'] and merge['source_manifest_sha256']==execution['manifest_sha256']
    assert merge['anchor_count']==303 and merge['result_file_count']==len(merge['result_sha256'])==909
    for rel,digest in merge['result_sha256'].items():assert sha(run/'master'/rel)==digest,rel
    analysis=run/'master/analysis/m78_category_low22_analysis.json'
    assert sha(analysis)==result['analysis_sha256']
    data=read(analysis)['results']['baseline']['results']
    metrics=dict(EAO=float(data[0][0][0])*100,ACC=float(data[2][0][0])*100,ROB=float(data[2][0][1])*100)
    assert metrics==result['metrics_percent']
    sys.path.insert(0,'/home/SUTrack_RGBD_L');from tools.finalize_vot_transaction_low22 import collect_confirmed_failure_outcomes
    outcomes,failures,per,settings=collect_confirmed_failure_outcomes(run/'master',m.TRACKER,expected_anchors=303)
    assert outcomes==result['failure_outcomes'] and failures==result['confirmed_failures']
    assert per==result['per_sequence_failures'] and settings==result['failure_settings']
    assert sum(v['run_length'] for v in outcomes.values())==execution['planned_frame_positions']==220483
    old=read(m.M39/'m39_result.json')['arms']['default']
    assert set(outcomes)==set(old['failure_outcomes']) and set(per)==set(old['per_sequence_failures'])
    native_metrics={k:old['metrics_percent'][k.lower()] for k in metrics}
    rescued=[k for k,v in outcomes.items() if not v['failed'] and old['failure_outcomes'][k]['failed']]
    new=[k for k,v in outcomes.items() if v['failed'] and not old['failure_outcomes'][k]['failed']]
    assert rescued==result['rescued_anchors'] and new==result['newly_failed_anchors']
    assert failures-old['confirmed_failures']==len(new)-len(rescued)
    protected=[n for n,v in old['per_sequence_failures'].items() if v['confirmed_failures']==0]
    assert protected==result['protected_native_sequences'] and len(protected)==7
    gate=execution['gate'];checks={k+'_minimum':metrics[k]>=gate[k+'_min'] for k in metrics}
    checks.update(failure_reduction=failures<=gate['confirmed_failures_max'],native_zero_failure_protection=all(per[n]['confirmed_failures']==0 for n in protected),all303_bound_and_complete=True)
    assert checks==result['gate_checks'] and all(checks.values())==result['full_three_dataset_evaluation_allowed']
    report=dict(status='completed_low22_saved_evidence_verified',observed_utc=datetime.now(timezone.utc).isoformat(),auditor_sha256=sha(__file__),
        result_sha256=sha(E/'low22_result.json'),execution_sha256=sha(E/'low22_execution.json'),bundle_sha256=execution['bundle_sha256'],head_sha256=bundle['adapter_checkpoint_sha256'],
        all303_initialization_keys_rebound=True,all909_saved_output_hashes_verified=True,all303_trajectory_lengths_and_failures_recomputed=True,
        trajectory_frame_positions=220483,official_analysis_sha256=sha(analysis),official_analysis_values_verified=True,official_analysis_rerun=False,
        metrics_percent=metrics,native_metrics_percent=native_metrics,delta_metrics_percent={k:metrics[k]-native_metrics[k] for k in metrics},
        native_confirmed_failures=old['confirmed_failures'],confirmed_failures=failures,rescued_count=len(rescued),new_failure_count=len(new),
        broken_native_success_sequences=[n for n in protected if per[n]['confirmed_failures']>0],frozen_gate_checks=checks,
        full_three_dataset_evaluation_allowed=all(checks.values()),full_evaluation_directory_exists=(E/'full_evaluation').exists(),
        followup_stage_snapshot=read(E/'full_followup/stage.json'),new_tracking_calls=0,new_caption_calls=0,new_training_steps=0,
        seed=2027,additional_seeds=[],independent_model_review_pass=False,goal_achieved=False,
        scope='Complete reused VOT low22, not full127. Existing official analysis values and all stored trajectories are checked without rerunning tracking or changing any frozen condition.')
    O.mkdir();write(O/'audit.json',report)
    stream=io.StringIO(newline='');writer=csv.writer(stream)
    writer.writerow(['sequence','anchors','native_failures','M78_failures','rescued','newly_failed'])
    for seq in sorted(per):
        writer.writerow([seq,per[seq]['anchors'],old['per_sequence_failures'][seq]['confirmed_failures'],per[seq]['confirmed_failures'],
            sum(outcomes[k]['sequence']==seq for k in rescued),sum(outcomes[k]['sequence']==seq for k in new)])
    (O/'per_sequence_failures.csv').write_bytes(stream.getvalue().encode('utf-8'))
    stream=io.StringIO(newline='');writer=csv.writer(stream)
    writer.writerow(['anchor_key','sequence','anchor','direction','run_length','native_failed','M78_failed','native_progress','M78_progress'])
    for key,v in outcomes.items():
        previous=old['failure_outcomes'][key]
        writer.writerow([key,v['sequence'],v['anchor'],v['direction'],v['run_length'],previous['failed'],v['failed'],previous['progress'],v['progress']])
    (O/'anchor_comparison.csv').write_bytes(stream.getvalue().encode('utf-8'))
    files={name:O/name for name in ['audit.json','per_sequence_failures.csv','anchor_comparison.csv']}
    for name in ['low22_result.json','low22_execution.json','low22_tracking.exit','low22_analysis.exit','low22_controller.exit','toolkit_analysis.log']:
        files[name]=E/name
    files.update({'merge_result.json':run/'merge_result.json','toolkit_analysis.json':analysis,SOURCE.name:SOURCE,Path(__file__).name:Path(__file__)})
    published=O/'files';published.mkdir();manifest=[]
    for name,src in files.items():
        dst=published/name;shutil.copyfile(src,dst);manifest.append(dict(path=name,bytes=dst.stat().st_size,sha256=sha(dst)))
    write(published/'manifest.json',manifest)
    archive=O/'published.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(published.iterdir()):
            assert p.is_file() and p.stat().st_size<3000000
            tar.add(p,arcname=p.name)
    print(json.dumps(dict(audit=report,archive=str(archive),archive_sha256=sha(archive)),indent=2))
if __name__=='__main__':main()
