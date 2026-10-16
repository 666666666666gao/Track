"""Write the R2 review only; preserve all original review/source inputs."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
OUT=Path(__file__).resolve().parent;BASE=OUT.parent
CAD=BASE/'M125_CAD_UNIFIED_SOURCE_R2_20261016'
INV=BASE/'M125_CAD_DEPLOYMENT_SOURCE_20261016/invoke_M125_CAD_reviewed_suite.py'
OP=BASE/'M125_REGION_DEPLOYMENT_SOURCE_R2_20261015/deploy_m125_region_fixed.py'
OLD_AUDIT=BASE/'M125_CAD_UNIFORM_SOURCE_AUDIT_20261016'
TRACE=OUT/'.aris/traces/experiment-audit/2026-10-16_run02';TRACE.mkdir(parents=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def refs(p,*lines):return [str(p)+':'+str(n) for n in lines]
now=datetime.now(timezone.utc).isoformat()
before1=read(OUT/'input_hashes_before.json');before2=read(OUT/'additional_dependency_hashes_before.json')
inputs=before1['files']+before2['files']
after=[dict(path=r['path'],bytes=Path(r['path']).stat().st_size,sha256=sha(r['path'])) for r in inputs]
assert all(a==b for a,b in zip(inputs,after))
save(OUT/'unchanged_input_receipt.json',dict(status='UNCHANGED',generated_at=now,files_checked=len(inputs),changed_files=[],files=after,
    original_inputs_and_original_report_preserved=True,scope='120 sealed local files; no remote/current R0 state inspection'))
det=read(OUT/'deterministic_checks.json');fixtures=read(OUT/'fixture_results.json');byte=read(OUT/'review_byte_transport_reproduction.json')
layout=read(BASE/'M125_CAD_R2_final66_actual_CPU_layout_20261016.json')
old=read(OLD_AUDIT/'EXPERIMENT_AUDIT.json')
blocker=dict(id='R2-B1',severity='blocking',component='new one-shot admission/launch operator',
    title='Report text normalization breaks the original-byte SHA binding',
    finding='The launcher reads JSON/MD with Path.read_text(encoding="utf-8") and hashes their original raw file bytes. Universal-newline decoding converts the actual Windows CRLF reports to LF. JSON serialization/compression/base64 transport preserves that altered text; remote .encode() therefore cannot equal the original raw hash. The exact remote assertion is reproduced failing with the existing original report. MD is also changed byte-for-byte. This is an observed artifact-format incompatibility, not a hypothetical fallback requirement.',
    evidence=refs(INV,43,44,45,62,85,89),
    reproduction_path=str(OUT/'review_byte_transport_reproduction.json'),
    actual_original_JSON=dict(raw_bytes=byte['files'][0]['raw_bytes'],transported_bytes=byte['files'][0]['transported_bytes'],
        CRLF_count=byte['files'][0]['CRLF_count'],raw_sha256=byte['files'][0]['raw_sha256'],transported_sha256=byte['files'][0]['transported_sha256']),
    actual_original_MD=dict(raw_bytes=byte['files'][1]['raw_bytes'],transported_bytes=byte['files'][1]['transported_bytes'],
        CRLF_count=byte['files'][1]['CRLF_count'],raw_sha256=byte['files'][1]['raw_sha256'],transported_sha256=byte['files'][1]['transported_sha256']),
    observed_exception='AssertionError at exact embedded remote review hash assertion',
    impact='For reports in the demonstrated format, the remote request fails before predecessor/resource admission or GPU launch, after the exclusive local attempt has already been created. No automatic retry is implemented. An LF-only report happens to avoid this mismatch, but the operator does not preserve the bytes of the actual reviewed artifacts supplied in this workflow.',
    minimal_fix='Read the exact report bytes and transport them losslessly. For these UTF-8 artifacts, read_bytes().decode("utf-8") preserves CRLF while keeping the existing string payload; keep hashing the same raw bytes for JSON and preserve MD bytes too. No fallback, try/except, or unrelated refactor is needed.')
resolved=[
    dict(id='original-B1',status='RESOLVED_SOURCE_AND_FIXTURE',evidence=refs(CAD/'prepare_m125_CAD_evaluation.py',80,93,95),
        details='The undefined args.training access is replaced by sha(bundle["training_result_path"]). All three supported profile prepare mains completed in isolated fixtures, writing selection/bundle/plans/shard configuration; actual CAD make_bundle/checked_plan were exercised against fixture R0 metadata and stubbed predecessor completion.'),
    dict(id='original-B2',status='RESOLVED_SOURCE_AND_FIXTURE',evidence=refs(CAD/'collect_m125_CAD_results.py',12,43,46,48,55),
        details='The event loop now uses event_name. Both U1/U2 complete collector mains ran over synthetic 50/80/1765-shaped receipts, looked up the intended CAD_<profile>_same_R0_fixed6_full127.json, and retained the correct model name in all three rows, gate and CSV.'),
    dict(id='original-W1',status='RESOLVED_BY_CORRECTED_DISCLOSURE',evidence=refs(CAD/'SOURCE_SPEC.json',56,57)+refs(CAD/'PROTOCOL.md',35),
        details='R2 removes the unsupported end-to-end one-time-hash claim and explicitly states that inherited R0->M122 plan checks rehash baseline final/native/CLIP assets per invocation. The frozen call chain is unchanged; no performance improvement is claimed or measured.'),
    dict(id='original-W3',status='OPERATOR_NOW_IMPLEMENTED_BUT_NOT_ACCEPTED',evidence=refs(INV,17,41,53,79,80,88,92,96),
        details='A CAD-specific admission/staging/gate/launch operator is now supplied, rather than the regional main. Its byte-binding defect R2-B1 prevents source acceptance of the combined launch path. Actual admission and launch remain unperformed.')]

checks=[
    dict(id='A',status='PASS_SOURCE_ONLY',subject='Ground-truth provenance and training scope',
        details='No GT path or scientific runtime changed. The same fixed six-pass Train152 R0 final, legal initialization bank and subsequent RGB/depth images feed all profiles. OPE GT opens only after prediction receipts; VOT uses official initial rectangles. Historical test-aware engineering and human multi-frame annotation assistance remain disclosed.',
        evidence=refs(CAD/'run_m125_CAD_official.py',23,27,55,56,72,79)+refs(CAD/'PROTOCOL.md',3,7,23)),
    dict(id='B',status='PASS_SOURCE_ONLY',subject='Metric normalization and official protocol',
        details='OPE metric call/hash/resolution, VOT official analysis/extraction, all nine pass thresholds, DT50/76373, CDTB80/101956, VOT127/1765/5295 scope, and the frozen shard runner remain unchanged. No prediction-based denominator or score substitution is added.',
        evidence=refs(CAD/'run_m125_CAD_official.py',15,16,57,59)+refs(CAD/'analyze_m125_CAD_vot.py',19,24)+refs(CAD/'collect_m125_CAD_results.py',52,53,54)),
    dict(id='C',status='PASS_RECEIPT_SCOPE',subject='Results/receipt existence and R2 version binding',
        details='The R2 receipt records 12 successful imports in its distinct R2 layout, 66 exact source hashes and 16 locked-input hashes, with zero main executions and zero real tracking. All are independently matched against the local frozen sources. Only prepare/collect changed from the original 12 modules, consistent with the primary revision record. Older four-source policy/tensor evidence remains applicable to those unchanged sources; it does not test the revised mains.',
        evidence=[str(BASE/'M125_CAD_R2_final66_actual_CPU_layout_20261016.json')+':1']+refs(CAD/'SOURCE_SPEC.json',18,24,27,37,38)+refs(BASE/'M125_CAD_R2_initial_source_revision_20261016.json',39,60)),
    dict(id='D',status='PASS_SCIENTIFIC_SOURCE_FAIL_LAUNCH_OPERATOR',subject='Reachable complete paths',
        details='Both original main-path defects are resolved and the complete prepare/collector functions were exercised in isolated stdlib fixtures. The unchanged scientific coordinator calls all qualification/tracking/analysis/collection stages. The new operator still fails the exact-byte report-binding requirement (R2-B1).',
        evidence=refs(CAD/'prepare_m125_CAD_evaluation.py',80)+refs(CAD/'collect_m125_CAD_results.py',43,46)+refs(CAD/'run_m125_CAD_uniform_suite.py',58,61,84,90,92,93)+refs(INV,43,45,62)),
    dict(id='E',status='WARN_SCOPE',subject='U0 versus new full evaluations and actual scope',
        details='U0 remains the original complete R0 three-dataset result, with only a planned new 64-step/65-row saved-prefix check. U1/U2 each require full three-dataset inference with one final and one profile. Synthetic fixture artifacts carrying the source functions\' normal success strings are explicitly simulation_only and are not real completions or performance evidence. No actual real64, full U1/U2, deployment/resource admission, new metric, or current R0 terminal state is established.',
        evidence=refs(CAD/'run_m125_CAD_uniform_suite.py',61,69,70,71,75,102)+refs(CAD/'check_m125_CAD_real64.py',57,62)+refs(INV,102)),
    dict(id='F',status='PASS_CLASSIFICATION',subject='Evaluation type',
        details='This review: stdlib/AST and simulation_only interface fixtures. Supplied R2 receipt: CPU import/layout evidence. Intended full benchmark: real_gt. None is relabelled as real GPU tracking.',
        evidence=[str(OUT/'fixture_results.json'),str(BASE/'M125_CAD_R2_final66_actual_CPU_layout_20261016.json')+':1']),
    dict(id='G',status='PASS_SOURCE_ONLY',subject='One checkpoint/profile, strict gates, prediction and memory semantics',
        details='Ten unchanged CAD modules are byte-identical to the original sources reviewed directly. Profile definitions, strict > response/C thresholds, native response at ABC-selected cell, final ABC reported confidence, state/query/feature commits, first-template retention and dynamic slot replacement are unchanged. C still executes for U0/U1/U2; optional U4 remains a bypass and unqueued. CAD remains an engineering adaptation, not official CAD network reproduction.',
        evidence=refs(CAD/'cad_template_profiles.py',14,16,22,25,32)+refs(CAD/'profiled_template_tracker.py',41,44,45,60,61,65,70,71)+refs(CAD/'PROTOCOL.md',19,21,27)),
    dict(id='H',status='PASS_SOURCE_ONLY_EXCEPT_R2_B1',subject='Local/remote time and review/candidate/source/input binding',
        details='The first local main statement and first remote non-import statement enforce 2026-10-17 12:45 +08:00. Exact assertion fixtures reject one second before and accept the boundary, without transport. Locally, PASS/no blockers, candidate SHA, invoker SHA, manifest hashes, locked-input hashes and exact 66-source review/layout/candidate maps are required. Remotely all 66 staging source bytes, 16 locked inputs, review/candidate and all outside-code source/input assets are checked. Report raw-byte transport is the remaining broken link.',
        evidence=refs(INV,18,25,26,27,30,32,35,39,40,56,60,61,62,66,67,73,74,76)),
    dict(id='I',status='PASS_SOURCE_ONLY',subject='Original R0 terminal and resources before exclusive staging/launch',
        details='The source invokes the exact frozen verify_predecessor then resource_admission before root.mkdir/code.mkdir and Popen. The predecessor function requires both controller exits=0, terminal exit=0, 15 successful stages, six complete passes with cumulative 152/219802 accounting, and all three same-final result receipts. Resource check requires free disk and exactly GPUs 0/1 below the memory threshold, without temperature/power queries. These checks were not run against the live system.',
        evidence=refs(INV,79,80,81,96)+refs(OP,59,61,68,71,72,76,77,82,83,86,88,112,114,117,118)),
    dict(id='J',status='PASS_SOURCE_ONLY',subject='Exclusive attempt/root, supervisor terminal and no autoretry',
        details='Local attempt is opened x before one SSH call; remote root.mkdir is exclusive and new files use xb/x. Duplicate calls fail rather than retry. The exact transport args parse from the existing static observer and specify ConnectionAttempts=1; askpass references the environment variable and was not executed. The remote child is the CAD supervisor with PYTHONOPTIMIZE=0; its complete main preserves both stubbed child exit 0 and 7 after exactly one call. The scientific coordinator preserves child receipts/exits and asserts nonzero; no automatic retry loop is present. Launch receipt expressly says real64 and formal metrics are unavailable at launch.',
        evidence=refs(INV,51,81,83,87,91,92,94,96,102,103,106,107,111,116,120)+refs(CAD/'m125_CAD_process_supervisor.py',12,15,16,17,18,20)+refs(CAD/'run_m125_CAD_uniform_suite.py',35)),
    dict(id='K',status='WARN_SCOPE',subject='VOT event attribution/counter sealing',
        details='Unchanged instrumentation binds events to legal initialization keys and profile/final/bundle, requires 1765 unique events, sums counts and seals/rechecks hashes outside official result directories. It does not independently reconcile every event key or track-call count with the corresponding trajectory. No actual events were read. This remains a completed-result review limit, not a newly invented defensive-code requirement.',
        evidence=refs(CAD/'run_m125_CAD_official.py',81,84,85)+refs(CAD/'analyze_m125_CAD_vot.py',35,41,43,44,50)+refs(CAD/'collect_m125_CAD_results.py',42,43)),
    dict(id='L',status='PASS_LOCAL_PRESERVATION',subject='Original report and all supplied inputs unchanged',
        details='120 local files, including the complete original audit output tree, original scientific inputs, R2 files/metadata/receipts, invoker, and two static transport dependencies, are unchanged before/after. No supplied source/report was edited; all new audit/fixture writes are confined to the new R2 audit directory.',
        evidence=[str(OUT/'input_hashes_before.json'),str(OUT/'additional_dependency_hashes_before.json'),str(OUT/'unchanged_input_receipt.json')])]

report=dict(audit_skill='experiment-audit',date='2026-10-16',generated_at=now,verdict='FAIL',overall_verdict='FAIL',source_verdict='FAIL_SOURCE_ONLY',
    scientific_source_verdict='PASS_SOURCE_ONLY',deployment_operator_source_verdict='FAIL_SOURCE_ONLY',integrity_status='fail',
    execution_verdict='NOT_EXECUTED',admission_verdict='BLOCKED_BY_SOURCE_DEFECT_NOT_PERFORMED',
    execution_scope='No actual NN/GPU/remote transport/current R0 status inspection; isolated local stdlib fixtures only.',
    admission_scope='Source ordering and simulated boundary checks only; real predecessor/resource/deployment admission not performed.',
    summary='R2 fixes both original scientific main-path blockers, but the new one-shot operator normalizes review report line endings while retaining raw-byte hashes; its remote review-hash assertion fails on the actual original audit artifacts.',
    blocking_findings=[blocker],resolved_original_findings=resolved,checks=checks,
    nonblocking_findings=[dict(id='R2-W1',finding='Real tracking/metrics/admission remain unperformed; CPU imports and synthetic fixtures retain their narrow scope.'),
        dict(id='R2-W2',finding='Inherited repeated asset hashing is now accurately disclosed; runtime/cost is not measured.'),
        dict(id='R2-W3',finding='VOT event counter sealing is implemented; independent trajectory-count reconciliation remains a completed-result review boundary.')],
    candidate_sha256=sha(CAD/'CANDIDATE.json'),audited_staged_source_sha256=layout['source_sha256'],
    audited_staged_source_sha256_origin=str(BASE/'M125_CAD_R2_final66_actual_CPU_layout_20261016.json')+':source_sha256',
    audited_staged_source_map_preserved_exactly=True,
    audited_input_hashes={r['path']:'sha256:'+r['sha256'] for r in inputs},inputs_unchanged=True,
    unchanged_input_receipt_path=str(OUT/'unchanged_input_receipt.json'),unchanged_input_receipt_sha256=sha(OUT/'unchanged_input_receipt.json'),
    original_report_sha256=sha(OLD_AUDIT/'EXPERIMENT_AUDIT.json'),original_report_verdict_preserved='FAIL_SOURCE_ONLY',
    check_counts=dict(local_inputs_sealed_and_unchanged=120,current_deterministic_checks=419,passed=419,failed=0,
        current_R2_python_modules_read=12,prospective_staged_sources_hash_verified=66,staged_sources_AST_parsed=66,new_invoker_AST_parsed=1,
        embedded_remote_AST_parsed=1,R2_layout_import_receipt_rows=12,R2_layout_locked_inputs_verified=16,
        exact_original_report_byte_comparisons=2,exact_remote_byte_assertion_reproductions=1,
        complete_synthetic_prepare_mains=3,complete_synthetic_collector_mains=2,complete_stubbed_supervisor_mains=2,
        time_boundary_assertion_cases=4,new_pure_policy_or_tensor_runs=0),
    fresh_reviewer_execution=dict(stdlib_only=True,actual_NN_calls=0,torch_imports=0,actual_remote_transport_calls=0,GPU_queries=0,
        original_R0_progress_queries=0,installations=0,supplied_input_writes=0,new_agents=0),
    fixture_classification='simulation_only',fixture_results_path=str(OUT/'fixture_results.json'),
    deterministic_checks_path=str(OUT/'deterministic_checks.json'),byte_transport_reproduction_path=str(OUT/'review_byte_transport_reproduction.json'),
    review_independence='same-family',acceptance_status='provisional',reviewer_family='openai',
    requested_routing=dict(backend='native Codex',model='gpt-6-astra',reasoning_effort='max'),
    actual_attestation=dict(status='UNATTESTED',exact_model=None,reasoning_effort=None,agent_id='/root/audit_m125_cad_uniform_source',
        detail='Continuation of the original native reviewer task; no backend model/effort attestation supplied.'),
    reviewer_model=None,reviewer_reasoning=None,agent_id='/root/audit_m125_cad_uniform_source',
    verdict_id='/root/audit_m125_cad_uniform_source:R2:2026-10-16',trace_path=str(TRACE))

md=['# M125 CAD R2 source and one-shot operator audit','',
    '**Overall: FAIL. Combined source: FAIL_SOURCE_ONLY.**  ',
    '**Scientific R2 coordinator: PASS_SOURCE_ONLY. New admission operator: FAIL_SOURCE_ONLY.**  ',
    '**Execution: NOT_EXECUTED. Admission: BLOCKED_BY_SOURCE_DEFECT_NOT_PERFORMED.**  ',
    '**Review:** same-family, provisional; requested Astra/max, actual route **UNATTESTED**.','',
    report['summary'],'',
    'This is a continuation by the original reviewer. All 12 R2 Python files and metadata were read directly; the byte-identical ten modules reuse the already-read original source reasoning. Both changed mains, the new invoker, embedded remote program, R2 primary receipt/revision record and necessary static dependencies were inspected. No running task was read and no SSH, GPU/NN, actual transport, installation or supplied-source/report mutation occurred.','',
    '## Blocking finding R2-B1: report bytes change during transport','',blocker['finding'],'',
    '**Evidence:** '+'; '.join('`'+r+'`' for r in blocker['evidence']),'',
    '| Actual original artifact | Raw bytes | Transported bytes | CRLF converted | Byte exact |',
    '|---|---:|---:|---:|---|',
    '| EXPERIMENT_AUDIT.json | 97123 | 96376 | 747 | No |',
    '| EXPERIMENT_AUDIT.md | 54303 | 54102 | 201 | No |','',
    'JSON raw SHA: `'+byte['files'][0]['raw_sha256']+'`.  ',
    'JSON transmitted-text SHA: `'+byte['files'][0]['transported_sha256']+'`.  ',
    'MD raw SHA: `'+byte['files'][1]['raw_sha256']+'`.  ',
    'MD transmitted-text SHA: `'+byte['files'][1]['transported_sha256']+'`.','',
    'The reviewer executed the exact embedded assertion after the same JSON → zlib → base64 → inverse sequence, using the actual original artifact bytes. It raised `AssertionError`. This was a local simulation, not a remote attempt. JSON semantic equality does not repair the byte mismatch; MD also loses its original line endings.','',
    blocker['impact'],'','**Minimal fix:** '+blocker['minimal_fix'],'',
    '## Original findings revisited','']
for item in resolved:md.extend(['### '+item['id']+': '+item['status'],'',item['details'],'','**Evidence:** '+'; '.join('`'+r+'`' for r in item['evidence']),''])
md.extend(['## Source, protocol and admission checks',''])
for item in checks:md.extend(['### '+item['id']+' — '+item['subject']+': '+item['status'],'',item['details'],'','**Evidence:** '+'; '.join('`'+r+'`' for r in item['evidence']),''])
md.extend(['## Verification and evidence limits','',
    '- **419 deterministic checks passed**, including all 66 staged-source hashes/AST parses, 16 locked R2 inputs, 12 import-receipt rows/origins, exact two-source revision verification, local/remote date boundaries and static admission ordering.',
    '- Complete source main functions exercised in isolated fixtures: **3 prepare**, **2 collector**, **2 supervisor** (stub exit 0 and exit 7, one child call each). These use synthetic R0 metadata, synthetic benchmark-shaped receipts and stubbed external services. They do not claim valid model weights, benchmark predictions, official metric calculation, actual predecessor completion or GPU availability.',
    '- **2 actual original report byte comparisons** and **1 exact embedded remote assertion reproduction** establish the new blocker.',
    '- **120 local inputs unchanged**, including the original report/evidence tree. `audited_input_hashes` retains `sha256:` values; the candidate and 66-stage maps use bare digests.',
    '- New real NN/GPU/remote calls: **0**. No new scalar/tensor check run was needed because all four prior policy/tensor source hashes are unchanged; historical counts remain historical.', '',
    'The current R2 layout receipt (2026-10-16 04:03:47 UTC) names `/root/autodl-tmp/m125_CAD_R2_uniform_layout_CPU_20261016`, contains all 66 matching source entries and 16 matching locked inputs, and explicitly records zero entrypoint main executions and zero real tracking. It does not establish a completed suite. The separate revision record is independently consistent with the actual two-file diff; it was not used as a substitute for reading the sources.', '',
    'The original audit remains FAIL_SOURCE_ONLY with its original bytes. R2 scientific fixes do not retroactively change that verdict. The new operator now closes the previously absent deployment-source boundary at the implementation level, but R2-B1 prevents acceptance. The date gate alone is never admission; real R0 completion/resources, source-bound one-shot deployment, qualification, and full uniform U1/U2 results remain necessary after correction.', '',
    '## Artifact seals','',
    'Candidate SHA-256: `'+report['candidate_sha256']+'`.',
    'Invoker SHA-256: `'+sha(INV)+'`.', '',
    'The JSON report includes the exact original absolute-key, bare-digest `source_sha256` map from the R2 CPU receipt as `audited_staged_source_sha256`. Independent local verification found all 66 sources equal. `input_hashes_before.json`, `additional_dependency_hashes_before.json` and `unchanged_input_receipt.json` seal every actual input. `fixture_results.json`, `deterministic_checks.json`, `direct_R2_diffs.patch` and `review_byte_transport_reproduction.json` preserve the diagnostic evidence.', '',
    'Full private trace: `.aris/traces/experiment-audit/2026-10-16_run02/` under this R2 audit directory.',''])
md_text='\n'.join(md)
(OUT/'EXPERIMENT_AUDIT.md').write_text(md_text,encoding='utf-8');save(OUT/'EXPERIMENT_AUDIT.json',report)
prompt='''请对独立R2及新增单次启动操作做后续只读源码/完整流程审查，原稿FINAL及全部原输入保持原字节。仅文件路径及检查要求如下，不预设结论。

R2目录：C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_CAD_UNIFIED_SOURCE_R2_20261016/（读取全部12 Python、CANDIDATE.json、SOURCE_SPEC.json、PROTOCOL.md、INPUT_MANIFEST.json）。
新增操作：C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_CAD_DEPLOYMENT_SOURCE_20261016/invoke_M125_CAD_reviewed_suite.py。
实际R2布局导入回执：C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_CAD_R2_final66_actual_CPU_layout_20261016.json。
修订原始记录：C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_CAD_R2_initial_source_revision_20261016.json。
原稿报告/证据：C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_CAD_UNIFORM_SOURCE_AUDIT_20261016/（原输入继续只读，可复用你实际已读的冻结R0/旧依赖/官方源；复核必要依赖）。

检查：重新判断两处原阻断及完整prepare/collector路径、同checkpoint/统一profile/官方协议、R2 metadata声明与真实调用路径、最终66源码绑定及12入口回执版本。检查新增操作的本地与远端日期边界、审查/候选/全部source/input SHA绑定、原R0全6pass+全3集终态与资源准入、独占attempt/root、实际启动/监督器退出与无自动重试，及其是否与科研coordinator衔接。只允许本地stdlib隔离夹具或AST证明；不得SSH/读现行R0状态、GPU/NN、改任何被审源码或原报告。模拟数据只能标记为夹具，不作真实完成态。

另存输出目录 C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_CAD_UNIFORM_SOURCE_AUDIT_R2_20261016/，MD/JSON及完整私有trace。JSON分别写source_verdict、overall/execution/admission scope、blocking_findings、检查数、所有实际输入前后SHA及unchanged。请给裸candidate_sha256和裸audited_staged_source_sha256（从实际R2布局回执读取后独立核对66本地冻结来源），audited_input_hashes继续sha256:前缀。明确same-family/provisional、请求Astra/max且实际route UNATTESTED。'''
supplement='补充一个操作审查检查项：请核查新增启动入口source_review_json/source_review_md的读取、传输编码与source_review_sha256的原始字节是否一致，结合原稿实际EXPERIMENT_AUDIT.json/MD文件逐字节核对，不仅对JSON语义作比较。被审入口和全部R2输入继续保持不变。'
save(TRACE/'run.meta.json',dict(skill='experiment-audit',run_id='2026-10-16_run02',started_at=before1['generated_at'],finished_at=now,
    reviewer_task=report['agent_id'],review_independence='same-family',acceptance_status='provisional',requested_routing=report['requested_routing'],actual_attestation=report['actual_attestation']))
save(TRACE/'001-R2-followup.request.json',dict(call_number=1,purpose='R2-source-and-new-operator-followup',tool='collaboration.followup_task',
    files_referenced=[r['path'] for r in inputs],prompt=prompt,requested_routing=report['requested_routing'],actual_attestation=report['actual_attestation']))
save(TRACE/'002-byte-check-supplement.request.json',dict(call_number=2,purpose='byte-level-report-transport',tool='collaboration.send_message',prompt=supplement))
(TRACE/'001-R2-followup.response.md').write_bytes((OUT/'EXPERIMENT_AUDIT.md').read_bytes())
save(TRACE/'001-R2-followup.meta.json',dict(timestamp=now,status='complete',source_verdict=report['source_verdict'],overall_verdict='FAIL',
    execution_verdict=report['execution_verdict'],review_independence='same-family',acceptance_status='provisional',
    report_json_sha256=sha(OUT/'EXPERIMENT_AUDIT.json'),report_md_sha256=sha(OUT/'EXPERIMENT_AUDIT.md'),
    requested_routing=report['requested_routing'],actual_attestation=report['actual_attestation']))
save(TRACE/'verification-evidence.json',dict(deterministic_checks=det,fixtures=fixtures,byte_transport=byte,
    unchanged_input_receipt_sha256=sha(OUT/'unchanged_input_receipt.json')))
meta=OUT/'.aris/meta';meta.mkdir(parents=True)
(meta/'events.jsonl').write_text(json.dumps(dict(event='review_trace',skill='experiment-audit',purpose='R2-source-and-new-operator-followup',
    agent_id=report['agent_id'],trace_path=str(TRACE),status='complete',source_verdict=report['source_verdict']))+'\n',encoding='utf-8')
print(json.dumps(dict(source_verdict=report['source_verdict'],scientific_source_verdict=report['scientific_source_verdict'],
    operator_source_verdict=report['deployment_operator_source_verdict'],blocking_findings=1,inputs_unchanged=len(inputs),
    report=str(OUT/'EXPERIMENT_AUDIT.md'),json_report=str(OUT/'EXPERIMENT_AUDIT.json'),trace=str(TRACE)),indent=2))
