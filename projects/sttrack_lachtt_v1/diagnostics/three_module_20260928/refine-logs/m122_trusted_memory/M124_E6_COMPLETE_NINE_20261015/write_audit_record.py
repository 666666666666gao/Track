"""Write audit verdict and forensic trace; never modify primary experiment inputs."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

OUT = Path(__file__).parent
REMOTE = '/root/autodl-tmp/m124_ABC_joint_epoch20_20261011'
CODE = REMOTE + '/code/'
EVAL = REMOTE + '/epochs/epoch-06/evaluation/'
TOOLKIT = '/root/miniconda3/envs/mplt/lib/python3.8/site-packages/vot/'
now = datetime.now(timezone.utc).isoformat()
agent = '/root/audit_m124_e6_full_suite'
report = (OUT / 'EXPERIMENT_AUDIT.md').read_text(encoding='utf-8')
hashes = json.loads((OUT / 'audited_input_hashes.json').read_text(encoding='utf-8'))
deterministic = json.loads((OUT / 'deterministic_checks.json').read_text(encoding='utf-8'))
assert deterministic['status'] == 'PASS' and deterministic['total_checks'] == 35167
local_paths = [Path(p) for p in hashes if p.startswith('C:')]
unchanged = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p)].split(':', 1)[1] for p in local_paths}
assert all(unchanged.values())


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


trace = OUT / 'trace'
trace.mkdir(exist_ok=True)
common = dict(skill='experiment-audit', agent_id=agent, requested_model='gpt-6-astra',
              requested_reasoning_effort='max', actual_model_route='UNATTESTED', actual_reasoning_effort='UNATTESTED',
              reviewer_family='openai', review_independence='same-family', acceptance_status='provisional')
write(trace / 'run.meta.json', dict(common, run_id='2026-10-15_m124_e6', generated_at=now, output_dir=str(OUT), trace_scope='Full assigned request and saved review; parent spawn invocation parameters unavailable to this reviewer.'))
write(trace / '001-experiment-audit.request.json', dict(common, call_number=1, purpose='M124 E6 full-suite experiment integrity review',
      tool='delegated native Codex agent; parent invocation not visible', prompt=(OUT / 'AUDIT_REQUEST.txt').read_text(encoding='utf-8'),
      parent_task='Perform a fresh read-only experiment integrity audit under the installed experiment-audit skill. Read and follow the complete assignment in AUDIT_REQUEST.txt. Read every listed primary input directly; do not rely on author interpretations. Save the requested Markdown/JSON outputs in that assignment directory. Do not run NN, SSH, install software, modify inputs, or create agents. Report same-family/provisional independence and actual routing limits. Return the full audit report as your final response.',
      files_referenced=[str(p) for p in local_paths if p.name != 'AUDIT_REQUEST.txt']))
(trace / '001-experiment-audit.response.md').write_bytes((OUT / 'EXPERIMENT_AUDIT.md').read_bytes())
write(trace / '001-experiment-audit.meta.json', dict(common, call_number=1, purpose='M124 E6 full-suite experiment integrity review', timestamp=now, status='completed', verdict='WARN', deterministic_status='PASS', blockers=0))
(trace / 'events.jsonl').write_text(json.dumps(dict(event='review_trace', skill='experiment-audit', purpose='M124 E6 full-suite', agent_id=agent, trace_path=str(trace), status='completed', verdict='WARN')) + '\n', encoding='utf-8')

checks = {
 'gt_provenance': dict(letter='A', status='WARN', details='Dataset GT is separate from predictions. Inference uses legal initialization then image-only updates. OPE GT payload coverage is six samples plus all 130 metadata bindings; all 127 VOT GT sequences are represented by 462 verified shard copies. Text input used human multiframe/video aids; actual bank/binding bytes were unavailable.',
   evidence=['/home/SRTrack_RGBD_L/lib/test/analysis/depthtrack_pr.py:40', '/home/SRTrack_RGBD_L/lib/test/analysis/depthtrack_pr.py:95', CODE+'run_m124_epochwise_official.py:22', CODE+'run_m124_epochwise_official.py:45', TOOLKIT+'experiment/multistart.py:112', CODE+'trusted_template_tracker.py:26', CODE+'bind_m122_official_initializations.py:49']),
 'score_normalization': dict(letter='B', status='PASS', details='PR uses selected prediction count and visible GT count; confidence thresholds select curve points. F is harmonic mean, EAO is the installed official curve slice, and AR uses official progress/sequence weights. No own-maximum metric normalization was found. Saved-value F arithmetic, EAO curve reaggregation and ROB outcome reaggregation passed; raw overlaps were not recomputed.',
   evidence=['/home/SRTrack_RGBD_L/lib/test/analysis/depthtrack_pr.py:117', '/home/SRTrack_RGBD_L/lib/test/analysis/depthtrack_pr.py:133', TOOLKIT+'analysis/longterm.py:65', TOOLKIT+'analysis/multistart.py:129', TOOLKIT+'analysis/multistart.py:175', TOOLKIT+'analysis/multistart.py:442']),
 'result_existence': dict(letter='C', status='WARN', details='All nine E6 scalars match captured metric/analysis/result/CSV/draft files. Bundle, plan, receipt, analysis, merge and final-hash records agree; 53 stage receipts and four shards record zero exits. Raw OPE/VOT output bytes, actual selected weights, full training reports/logs and historical baseline primary result were not supplied.',
   evidence=[EVAL+'all_results.json:26', EVAL+'all_results.json:103', EVAL+'selection.json:3', EVAL+'vot/result.json:16419', EVAL+'vot/run/launch.json:16', REMOTE+'/controller.supervisor.terminal.json:13', str(OUT/'DRAFT_NINE_METRIC_REPORT.md')+':11']),
 'dead_code': dict(letter='D', status='PASS', details='Each reported score has a connected call-and-write path with completed stage evidence. The unused format_metrics convenience formatter and nonselected installed analyses do not underlie claimed scores. Reviewer did not execute the original metric pipeline.',
   evidence=[CODE+'run_m124_epochwise_suite.py:48', CODE+'run_m124_epochwise_official.py:56', CODE+'run_m124_epochwise_suite.py:88', CODE+'analyze_m124_epochwise_vot.py:22', '/home/SUTrack_RGBD_L/tools/finalize_vot_transaction_low22.py:193', EVAL+'vot/run/master/analysis/ABC_joint_epoch06_full127.json:666']),
 'scope': dict(letter='E', status='WARN', details='One seed 2027; eight sequential checkpoints, E6 selected by mean Test50/CDTB80 F, stop after E8 dual declines. Fixed human text used multiframe assistance. VOT is 127 sequences/80741 source frames, 1765 anchors and 1327004 protocol frame slots. Stale no-text/M39 manifest label contradicts sealed ABC/text entry. Initial 37-sequence legacy handoff lacks RNG equivalence. No individual-module, seed-robustness, future-utility, efficiency or proposed-route claim follows.',
   evidence=[CODE+'m124_epochwise_selection.py:4', CODE+'run_m124_epochwise_suite.py:58', REMOTE+'/best.json:2', REMOTE+'/actual_launch.json:17', CODE+'train_m124_joint_epoch.py:79', CODE+'train_m124_joint_epoch.py:230', EVAL+'vot/run/shard_manifest.json:3', EVAL+'vot/selected_m124_epochwise_vot.py:2', CODE+'m124_epochwise_official_runtime.py:61']),
 'eval_type': dict(letter='F', status='PASS', classification='real_gt', by_dataset={'depthtrack_test':'real_gt','cdtb':'real_gt','vot':'real_gt'}, details='All reported metrics compare predictions with dataset GT. Human text and test-informed checkpoint selection qualify interpretation; they do not change GT classification.',
   evidence=['/home/SRTrack_RGBD_L/lib/test/analysis/depthtrack_pr.py:97', TOOLKIT+'analysis/multistart.py:115'])
}
warnings = [
 dict(id='W1', type='selection', blocking=False, summary='Test50 and CDTB80 determine checkpoint selection and early stopping; cannot claim untouched held-out testing.'),
 dict(id='W2', type='input_protocol', blocking=False, summary='Human text used multiframe/video aids; no unqualified automatic or initialization-frame-only comparison.'),
 dict(id='W3', type='metadata', blocking=False, summary='Inherited shard_manifest.native_source M39/no-text label is stale. Actual frozen wrapper loads joint ABC/text path; live tensor execution was not attested.'),
 dict(id='W4', type='audit_scope', blocking=False, summary='Raw OPE/VOT predictions, weight/bank/binding bytes, full training reports/logs, base-plan and source-gate payloads absent. Sealed arithmetic and digest linkage are not raw metric reproduction.'),
 dict(id='W5', type='baseline_provenance', blocking=False, summary='No supplied primary record authenticates draft baseline 77.321654/82.471190/93.669109.'),
 dict(id='W6', type='reproducibility', blocking=False, summary='One seed, warm starts and a 37-sequence legacy handoff without saved RNG; no exact uninterrupted-run equivalence or total discarded-work accounting.')]
claims = [
 dict(id='C1', claim='Captured E6 nine-metric row and same recorded final', impact='supported_with_qualifiers'),
 dict(id='C2', claim='5 of 9 project predicates pass, joint target unmet', impact='supported'),
 dict(id='C3', claim='Original pipeline and four shards terminal zero', impact='supported_by_captured_receipts'),
 dict(id='C4', claim='Untouched tests or first-frame-only automatic text', impact='unsupported'),
 dict(id='C5', claim='Unmodified M39/no-text model evaluated', impact='unsupported'),
 dict(id='C6', claim='Full raw prediction metric reproduction by this reviewer', impact='unsupported'),
 dict(id='C7', claim='Historical STTrack baseline authenticated by this audit', impact='unsupported_pending_primary_result'),
 dict(id='C8', claim='Independent module effects, seed robustness, future memory utility, speed superiority or new planned-route gains', impact='unsupported')]
supporting = ['EXPERIMENT_AUDIT.md','deterministic_checks.json','audited_input_hashes.json','captured_file_line_counts.json','verify_captured_evidence.py','write_audit_record.py','audit_validation_failure.json']
output_hashes = {name:'sha256:'+sha(OUT/name) for name in supporting}
output_hashes.update({str(path.relative_to(OUT)).replace('\\','/'):'sha256:'+sha(path) for path in trace.iterdir() if path.is_file()})
result = dict(audit_skill='experiment-audit', verdict='WARN', reason_code='qualified_captured_results_with_selection_input_and_provenance_limits',
    summary='35167 deterministic checks passed with zero failures. E6 full-suite captured values and terminal records agree; semantic acceptance remains provisional with disclosed selection/input/provenance limitations.',
    generated_at=now, date='2026-10-15', auditor='Fresh Codex reviewer; gpt-6-astra/max requested, actual route unattested',
    agent_id=agent, verdict_id=agent+'@2026-10-15', executor_model='UNATTESTED', executor_family='openai',
    reviewer_model='UNATTESTED', reviewer_reasoning='UNATTESTED', reviewer_model_requested='gpt-6-astra', reviewer_reasoning_requested='max',
    reviewer_family='openai', actual_routing='UNATTESTED', actual_routing_evidence='No tool-visible provider/backend or reasoning attestation is available to this child.',
    review_independence='same-family', acceptance_status='provisional', overall_verdict='warn', integrity_status='warn',
    deterministic_status='PASS', deterministic_acceptance_status='accepted_for_the_stated_checks_only', blocker_count=0, blockers=[], warnings=warnings,
    checks=checks, claims=claims, metrics_percent=deterministic['metrics_percent'], project_gate_predicates=deterministic['gates'],
    joint_target_met=False, selected_final_sha256='4d5c7e411a047fa996835647d1efa3bb94f4184fdc3a592380523be8c04156eb',
    scope=deterministic, action_items=[
      'Retain test-selection, multiframe-text and legacy-handoff disclosures; report these as captured E6 results.',
      'Annotate inherited native_source error separately; keep all original experiment files immutable.',
      'Attach the historical STTrack primary result or mark that comparison as not authenticated in this audit.',
      'Keep original independent_completed_audit=false and link this separate report.',
      'Full independent metric reproduction requires existing raw predictions/confidences/trajectories and complete relevant GT/image metadata; this audit authorizes no new NN work.'],
    audited_input_hashes=hashes, audited_input_hashes_manifest='audited_input_hashes.json', supplied_local_inputs_unchanged=unchanged,
    output_hashes=output_hashes, output_hash_manifest='OUTPUT_HASHES.json', output_hash_policy='This JSON hash is recorded in the final sidecar to avoid self-reference.',
    trace_path=str(trace), report_path=str(OUT/'EXPERIMENT_AUDIT.md'), no_input_edits=True,
    auditor_execution_notes='First final validation caught newline translation in the auditor trace copy; exact-byte trace copy fixed it. Original failed-check evidence is retained in audit_validation_failure.json. Experiment inputs and Markdown report were unchanged.',
    actions=dict(NN_calls=0, SSH_calls=0, software_installations=0, spawned_agents=0, live_tensor_attestation=False, model_weights_loaded=False))
write(OUT/'EXPERIMENT_AUDIT.json', result)
output_hashes['EXPERIMENT_AUDIT.json']='sha256:'+sha(OUT/'EXPERIMENT_AUDIT.json')
write(OUT/'OUTPUT_HASHES.json', dict(generated_at=now, hashes=output_hashes, scope='Final reports, supporting audit evidence and local forensic trace. This manifest does not hash itself.'))
print(json.dumps(dict(verdict=result['verdict'], checks=deterministic['total_checks'], blockers=0,
                     local_inputs_unchanged=len(unchanged), output_hashes={k:v for k,v in output_hashes.items() if k in ['EXPERIMENT_AUDIT.md','EXPERIMENT_AUDIT.json']}, trace_path=str(trace)),indent=2))
