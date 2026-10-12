from pathlib import Path
import hashlib,json
R=Path(r'C:\Users\gb\.codex_track_publish_m29_20260902')
A=R/'.aris/m122_full_causal_20261007'
D=R/'projects/sttrack_lachtt_v1/diagnostics/three_module_20260928'
review_path=A/'M124_epoch01_CDTB_CPU_diagnostic_source_review_R1_20261012.json'
review=json.loads(review_path.read_bytes())
assert review['status']=='WARN_SOURCE_ONLY' and review['blocking_issues']==0
for path,digest in review['audited_input_hashes'].items():
    assert 'sha256:'+hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
native=json.loads((A/'M124_epoch01_CDTB_CPU_diagnostic_native_receipts_20261012.json').read_bytes())
assert native['M124_epoch01_CDTB_CPU_diagnostic_native_terminal_20261012']['exit_code']==0
source=D/'analyze_m124_epoch01_cdtb_saved_trajectory.py'
raw=A/'M124_epoch01_CDTB_CPU_diagnostic_original_stdout_20261012.json'
value=json.loads(raw.read_bytes())
metrics=json.loads((A/'M124_epoch01_actual_cdtb_metrics_20261012.json').read_bytes())
assert value['status']=='actual_complete_M124_epoch01_CDTB_saved_trajectory_diagnosis'
assert value['analysis_source_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()=='508802fa97a762aa8e9f6309ac38c10290f44ce1bdc97dcfe2ad573199454143'
assert value['saved_metrics_sha256']==hashlib.sha256((A/'M124_epoch01_actual_cdtb_metrics_20261012.json').read_bytes()).hexdigest()
assert value['final_sha256']==metrics['final_sha256']=='75aee7ac912b799f9160baadf95ccc6ff406b6b338760a239da6249bfce505db'
assert value['metric_source_sha256']==metrics['metric_source_sha256']=='05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc'
assert value['evaluation_type']=='real_gt' and value['NN_calls']==value['GPU_queries']==value['training_progress_queries']==0
assert value['sequences']==len(value['per_sequence'])==80 and value['frames']==sum(row['frames'] for row in value['per_sequence'])==101956
for short,key in [('P','precision_percent'),('R','recall_percent'),('F','f_score_percent')]:assert abs(value['aggregate'][short]-metrics['metrics'][key])<1e-8
receipt=dict(status='actual_M124_epoch01_CDTB_CPU_diagnostic_intake_bound',raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),source_sha256=value['analysis_source_sha256'],source_review_sha256=hashlib.sha256(review_path.read_bytes()).hexdigest(),source_review_status=review['status'],source_review_blocking_issues=0,source_review_independence='same-family',source_review_acceptance='provisional',actual_reviewer_model='UNATTESTED',native_exit_code=0,final_sha256=value['final_sha256'],saved_metrics_sha256=value['saved_metrics_sha256'],source_review_inputs_verified=len(review['audited_input_hashes']),captured_input_sha256_count=len(value['inputs_sha256']),sequences=80,frames=101956,aggregate=value['aggregate'],NN_calls=0,GPU_queries=0,training_progress_queries=0,server_CPU_analysis_invocations=1,training_source_changed=False,training_restarted=False,full_nine_completion_verified=False)
(A/'M124_epoch01_CDTB_CPU_diagnostic_actual_intake_20261012.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2))
