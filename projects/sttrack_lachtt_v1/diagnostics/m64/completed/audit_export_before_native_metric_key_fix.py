"""Audit and export the completed frozen M64 low22 negative result."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,importlib.util,json,shutil,sys,tarfile

BASE=Path('/root/autodl-tmp')
ROOT=BASE/'sttrack_m64_category_candidate_20260907'
OUT=ROOT/'completed_publication_evidence'
SOURCE=BASE/'m64_vot_low22_20260907.py'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def main():
    assert sha(ROOT/'low22_result.json')=='ce417ab1e35433880ffbb19b4b178ea5ad7bf9fcd0bed3a344e85b08b67e8b88'
    for n in ['low22_tracking.exit','low22_analysis.exit','low22_controller.exit']:assert (ROOT/n).read_text().strip()=='0'
    s=importlib.util.spec_from_file_location('m64_frozen_evaluator',str(SOURCE));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    ex=m.checked_execution();r=json.loads((ROOT/'low22_result.json').read_text())
    assert r['status']=='completed_low22_same_bundle_evaluation' and r['execution_sha256']==sha(ROOT/'low22_execution.json')
    assert r['bundle_sha256']==ex['bundle_sha256'] and r['text_protocol_sha256']==ex['text_protocol_sha256']
    sys.path.insert(0,str(m.INTERFACE));from semantic_runtime import checked_plan,text_bank
    plan,bundle=checked_plan(ROOT/'low22_plan.json');bank=text_bank(plan,bundle)
    captions=json.loads((ROOT/'low22_captions/plan.json').read_text())
    assert len(captions['cases'])==len(captions['rows'])==303
    for row in captions['rows']:bank.info(row['image'],row['init_bbox'])
    run=ROOT/'low22_run';merge=json.loads((run/'merge_result.json').read_text())
    assert sha(run/'merge_result.json')==r['merge_sha256'] and merge['source_manifest_sha256']==ex['manifest_sha256']
    assert merge['anchor_count']==303 and merge['result_file_count']==len(merge['result_sha256'])==909
    for rel,h in merge['result_sha256'].items():assert sha(run/'master'/rel)==h,rel
    analysis=run/'master/analysis/m64_category_low22_analysis.json';assert sha(analysis)==r['analysis_sha256']
    data=json.loads(analysis.read_text())['results']['baseline']['results']
    values=dict(EAO=float(data[0][0][0])*100,ACC=float(data[2][0][0])*100,ROB=float(data[2][0][1])*100)
    assert values==r['metrics_percent']
    sys.path.insert(0,'/home/SUTrack_RGBD_L');from tools.finalize_vot_transaction_low22 import collect_confirmed_failure_outcomes
    outcomes,failures,per,settings=collect_confirmed_failure_outcomes(run/'master',m.TRACKER,expected_anchors=303)
    assert outcomes==r['failure_outcomes'] and failures==r['confirmed_failures'] and per==r['per_sequence_failures'] and settings==r['failure_settings']
    assert sum(x['run_length'] for x in outcomes.values())==ex['planned_frame_positions']==220483
    old=json.loads((m.M39/'m39_result.json').read_text())['arms']['default']
    rescued=[n for n,v in outcomes.items() if not v['failed'] and old['failure_outcomes'][n]['failed']]
    newly_failed=[n for n,v in outcomes.items() if v['failed'] and not old['failure_outcomes'][n]['failed']]
    assert rescued==r['rescued_anchors'] and newly_failed==r['newly_failed_anchors']
    protected=[n for n,v in old['per_sequence_failures'].items() if v['confirmed_failures']==0]
    assert protected==r['protected_native_sequences'] and len(protected)==7
    gate=ex['gate'];checks={k+'_minimum':values[k]>=gate[k+'_min'] for k in values}
    checks.update(failure_reduction=failures<=gate['confirmed_failures_max'],native_zero_failure_protection=all(per[n]['confirmed_failures']==0 for n in protected),all303_bound_and_complete=True)
    assert checks==r['gate_checks'] and all(checks.values())==r['full_three_dataset_evaluation_allowed']
    assert not r['full_three_dataset_evaluation_allowed'] and not (ROOT/'full_evaluation').exists()
    report=dict(status='completed_negative_low22_evidence_verified',observed_utc=datetime.now(timezone.utc).isoformat(),
        auditor_sha256=sha(__file__),result_sha256=sha(ROOT/'low22_result.json'),execution_sha256=sha(ROOT/'low22_execution.json'),
        bundle_sha256=ex['bundle_sha256'],text_protocol_sha256=ex['text_protocol_sha256'],head_sha256=bundle['adapter_checkpoint_sha256'],
        all303_initialization_keys_rebound=True,all909_saved_output_hashes_verified=True,all303_trajectory_lengths_and_failures_recomputed=True,
        trajectory_frame_positions=220483,official_analysis_sha256=sha(analysis),official_analysis_values_verified=True,official_analysis_rerun=False,
        metrics_percent=values,native_metrics_percent=old['metrics_percent'],delta_metrics_percent={k:values[k]-old['metrics_percent'][k] for k in values},
        native_confirmed_failures=old['confirmed_failures'],confirmed_failures=failures,rescued_count=len(rescued),new_failure_count=len(newly_failed),
        broken_native_success_sequences=[n for n in protected if per[n]['confirmed_failures']>0],frozen_gate_checks=checks,
        full_three_dataset_evaluation_allowed=False,full_evaluation_directory_exists=False,new_tracking_calls=0,new_caption_calls=0,new_training_steps=0,
        independent_model_review_pass=False,scope='Executor audit of the complete reused VOT low22 benchmark. No full127 or same-bundle three-dataset result. The frozen candidate is stopped for promotion, with evidence retained.')
    write(ROOT/'completed_evidence_audit.json',report)
    OUT.mkdir()
    names=['low22_result.json','completed_evidence_audit.json','low22_tracking.exit','low22_analysis.exit','low22_controller.exit','low22_analysis.log','toolkit_analysis.log','low22_execution.json']
    for n in names:shutil.copyfile(ROOT/n,OUT/n)
    shutil.copyfile(run/'merge_result.json',OUT/'merge_result.json');shutil.copyfile(analysis,OUT/'toolkit_analysis.json')
    for p in [SOURCE,Path(__file__)]:shutil.copyfile(p,OUT/p.name)
    write(OUT/'evidence_manifest.json',[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.iterdir())])
    archive=ROOT/'completed_publication_evidence.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for p in sorted(OUT.iterdir()):t.add(p,arcname=p.name)
    print(json.dumps(dict(audit=report,archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size),indent=2))


if __name__=='__main__':main()
