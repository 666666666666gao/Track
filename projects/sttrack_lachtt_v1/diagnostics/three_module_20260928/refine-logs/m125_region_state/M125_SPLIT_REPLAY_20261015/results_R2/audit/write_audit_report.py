from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
INPUT = BASE / 'M125_R2_RESULT_INPUTS_20261015'
SOURCE = BASE.parent.parent / 'projects/sttrack_lachtt_v1/diagnostics/three_module_20260928'
inventory = json.loads((OUT / 'input_inventory.json').read_bytes())
det = json.loads((OUT / 'deterministic_checks.json').read_bytes())
coverage = json.loads((OUT / 'independent_coverage.json').read_bytes())
assert det['failed'] == 0
now = datetime.now(timezone.utc).isoformat()
agent = '/root/audit_m125_e6_replay_result'
trace = OUT / '.aris/traces/experiment-audit/2026-10-15_run01'
trace.mkdir(parents=True, exist_ok=True)
hashes = {r['path']: r['sha256'] for r in inventory}

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def ref(name, line):
    path = INPUT / name
    return f'[{name}:{line}]({path.as_posix()}:{line})'

def source_ref(name, line):
    return f'[{name}:{line}]({(SOURCE / name).as_posix()}:{line})'

checks = {
 'gt_provenance': dict(status='PASS', details='The scoring stage reads the archived CDTB dataset groundtruth.txt after sealed predictions, with matching dataset-case/result/archive SHA; initialization equals dataset row zero. No model-generated reference is used. Current-frame GT does not enter the supplied replay loop; imported runtime internals are outside the bounded inputs.', evidence=['047_analyze_m125_selected_E6_observation.py:32','047_analyze_m125_selected_E6_observation.py:44','046_replay_m125_selected_E6_observation.py:61','046_replay_m125_selected_E6_observation.py:74','locked_depthtrack_pr.py:49']),
 'score_normalization': dict(status='PASS', details='Dataset-grounded overlap and continuous GT-box coverage are not normalized by prediction statistics. full_response/quality_probability is an explicitly stored same-state ranking counterfactual, not a performance normalization. Raw response/quality/geometry arrays remain available.', evidence=['046_replay_m125_selected_E6_observation.py:91','047_analyze_m125_selected_E6_observation.py:71','summarize_m125_selected_E6_observation.py:70']),
 'result_existence': dict(status='PASS', details='49 extracted artifacts match declared hashes; 48 archived payloads match their raw bytes, lengths and extracted counterparts. 4 NN and 4 CPU child return-code records are zero, controller completion sentinel is zero. 2894 step actions, 2898 prefix rows, 25 sampled states and 6400 candidates per geometry independently count correctly. Every box component and confidence reproduces saved six-decimal formatting.', evidence=['005_driver.py:25','005_driver.py:33','005_driver.py:39','004_result.json:2','deterministic_checks.json','input_inventory.json']),
 'dead_code': dict(status='WARN', details='Supplied CPU stage calls the locked _vot_overlaps for original/replayed prefixes and both candidate geometries; terminal logs and output arrays support execution. Exact local independent overlap recomputation is blocked: cv2 and vot are missing in the supplied interpreters. Full evaluate_depthtrack_results and threshold/F-score functions are not invoked by this diagnostic and no new benchmark PR/F score is established.', evidence=['047_analyze_m125_selected_E6_observation.py:59','047_analyze_m125_selected_E6_observation.py:71','locked_depthtrack_pr.py:8','locked_depthtrack_pr.py:63','dependency_probes.json']),
 'scope': dict(status='WARN', details='Four posthoc CDTB prefixes at one E6 checkpoint, one replay per sequence, 25 selected states (14 GT-valid/11 excluded); no seed-replicated evaluation. E6 metadata explicitly says DepthTrack Test and CDTB were used for checkpoint selection and human review used multiframe aids. Qualified state-level diagnosis only: no benchmark rates, first-error causality, original hidden-state identity, full-trajectory intervention gain or semantic identity truth.', evidence=['PLAN.json:57','006_prediction_stage.json:209','006_prediction_stage.json:218','006_prediction_stage.json:319','047_analyze_m125_selected_E6_observation.py:111']),
 'eval_type': dict(status='PASS', classification='real_gt', details='Dataset GT evaluates frozen model-derived candidates and continuous box coverage. Same-state counterfactual scores are diagnostic interventions; comparison with saved predictions establishes numerical consistency only. GT validity is not pixel visibility or occlusion evidence.', evidence=['locked_depthtrack_pr.py:49','summarize_m125_selected_E6_observation.py:87'])
}
verification_blockers = [
 dict(id='V1', scope='Exact independent official raster overlap recomputation', reason='Both supplied interpreters lack cv2 and vot; actual locked-metric import and direct vot.region import return exit 1 with ModuleNotFoundError. No installation, substitute metric, continuous-IoU approximation, or remote execution was performed.', permits='Coverage, raw-byte, candidate-array, arithmetic and prefix consistency findings; archived IoU values must retain their verification qualifier.'),
 dict(id='V2', scope='Independent attestation of transitive runtime, checkpoint/image bytes, original hidden state, and full-suite prerequisites', reason='The bounded inputs contain source/weight/image hashes and stage assertions, but not the imported OfficialDenseTracker/trusted/decoder code, raw images, weight tensors, full-suite result/best/all_results/receipt bytes or original hidden-state snapshots. These references cannot be rehashed or reexecuted from this package.', permits='Consistency of supplied hashes, source-visible ordering and recorded assertions; no claim of transitive end-to-end state attestation.')
]
claims = [
 dict(id='C1', claim='The four R2 own-history prefixes reproduce saved six-decimal bboxes and confidences.', impact='supported', boundary='Numeric outputs only, not original hidden-state identity.'),
 dict(id='C2', claim='The selected panel has 25 states, 14 GT-valid and 11 GT-invalid; seven valid boxes are fully covered and seven have zero continuous crop coverage.', impact='supported', boundary='Finite posthoc panel; continuous rectangular coverage, not visibility/occlusion.'),
 dict(id='C3', claim='13 of 14 valid sampled states have no refined candidate with official IoU >= 0.5.', impact='needs_qualifier', boundary='Arithmetic agrees with archived candidate-IoU vectors; exact VOT raster overlap was not independently recomputed.'),
 dict(id='C4', claim='Empty and quality-score alternatives never affect the submitted trajectory in the supplied replay loop.', impact='needs_qualifier', boundary='Source ordering, recorded equality assertions and continuous action history support this; transitive runtime and all hidden-state fields were not independently attested.'),
 dict(id='C5', claim='This panel determines whole-benchmark failure frequency or the cause of the first tracking error.', impact='unsupported'),
 dict(id='C6', claim='An alternative improves the complete trajectory, proves semantic identity, or isolates a full trained-module ablation.', impact='unsupported'),
 dict(id='C7', claim='The panel is an untouched test evaluation with independently selected checkpoint/text.', impact='unsupported', boundary='The supplied metadata declares test-metric checkpoint selection and multiframe human aids.'),
]
attribution = dict(agent_id=agent, verdict_id='M125_R2_RESULT_AUDIT_20261015', reviewer_backend='native_codex_agent',
    reviewer_model='UNATTESTED',reviewer_model_requested='gpt-6-astra',reviewer_model_actual='UNATTESTED',
    reviewer_reasoning='UNATTESTED',reviewer_reasoning_requested='max',reviewer_reasoning_actual='UNATTESTED',
    reviewer_family='openai',executor_model='UNATTESTED',executor_family='openai',review_independence='same-family',acceptance_status='provisional')
report = dict(audit_skill='experiment-audit',verdict='WARN',overall_verdict='warn',integrity_status='warn',
    reason_code='exact_overlap_dependency_boundary_and_posthoc_scope',summary='Bounded evidence is numerically consistent; qualified diagnostic use only. Exact raster IoU remains independently unverified.',
    date='2026-10-15',generated_at=now,**attribution,checks=checks,classification='real_gt',
    audited_input_hashes={k:'sha256:'+v for k,v in hashes.items()},trace_path=str(trace),
    deterministic=dict(status='PASS',check_count=det['check_count'],failed=0,input_count=len(inventory),review_independence='deterministic',acceptance_status='accepted',path=str(OUT/'deterministic_checks.json')),
    integrity_blockers=[],verification_blockers=verification_blockers,claims=claims,counts=coverage['counts'],
    official_overlap_recomputed=False,all_original_inputs_unchanged=True,
    execution_constraints=dict(SSH_calls=0,new_NN_calls=0,optimizer_steps=0,package_installations=0,notifications=0,original_input_content_mutations=0),
    reviewer_generated_byproduct=dict(transient_input_directory_bytecode_files=1,corrected=True,relocation_receipt='reviewer_cache_relocation.json',final_bytecode_location=str(OUT/'reviewer_import_cache'),subsequent_probes_use_python_B=True))
dump(OUT/'EXPERIMENT_AUDIT.json',report)

table = '\n'.join(f"| {r['sequence']} | {r['tracker_step_calls']} | {r['prefix_rows']} | {r['sampled_candidate_states']} | {r['bbox_max_abs_difference']:.12g} | {r['confidence_max_abs_difference']:.12g} |" for r in coverage['sequences'])
coverage_table = '\n'.join(f"| {r['sequence']} | {r['frame']} | {r['GT_box_crop_coverage_fraction']:.0f} | {r['GT_width_in_search']:.6f} × {r['GT_height_in_search']:.6f} |" for r in coverage['states'] if r['GT_valid'])
report_md = f'''# Experiment Audit Report: M125 R2 E6 replay

**Overall verdict: WARN. Integrity status: warn. Evaluation: real_gt, bounded posthoc diagnosis.**

Fresh native Codex reviewer `{agent}`. Requested routing: `gpt-6-astra`, reasoning `max`; actual model and effort are **UNATTESTED** because no host attestation was supplied. `review_independence: same-family`; `acceptance_status: provisional`. Deterministic checks are separately accepted within their stated boundaries. Generated {now}.

The supplied primary artifacts are internally consistent. **{det['check_count']:,} deterministic checks pass, zero fail** across 55 files: the 49 manifest entries, INPUTS.json and five explicitly supplied supplementary artifacts. All 49 input hashes match, all 48 archive text payloads match their declared bytes/SHA and extracted copies, and all 55 audited inputs remain unchanged. This is an audit of the supplied local evidence, not new tracking execution.

| Check | Status | Finding |
|---|---|---|
| A. Ground truth provenance | PASS | Dataset GT, read after sealed predictions; initializer equals dataset row zero. |
| B. Score normalization | PASS | No prediction-statistic normalization of performance metrics. |
| C. Result existence and consistency | PASS | Raw files, child exits, counts and six-decimal outputs agree. |
| D. Metric invocation and verification | WARN | Actual scoring path exists and produced vectors; local exact VOT recomputation is unavailable. |
| E. Scope | WARN | Four posthoc prefixes, one selected checkpoint; test-metric selection and multiframe aids disclosed in metadata. |
| F. Evaluation classification | PASS / real_gt | GT-grounded geometry diagnosis; saved-output agreement is only a consistency check. |

## A. Ground truth provenance — PASS within supplied-code scope

The CPU analyzer opens `dataset_root/sequence/groundtruth.txt` and verifies it against `case.gt_sha256` ({ref('047_analyze_m125_selected_E6_observation.py',44)}). All four actual GT files match the stage/result/manifest hashes. Their row counts are 999, 1,522, 874 and 1,819. Initialization boxes equal dataset row zero and the archived original predictions' initial row. Model candidates are predictions evaluated against these dataset boxes; original saved predictions are only a parity reference.

The replay starts from the bound initialization, calls `actor.trusted.step(image)` using the current image, saves the action and then calculates alternatives. It seals prediction JSON before the CPU process opens subsequent GT ({ref('046_replay_m125_selected_E6_observation.py',61)}, {ref('046_replay_m125_selected_E6_observation.py',74)}, {ref('046_replay_m125_selected_E6_observation.py',121)}, {ref('005_driver.py',24)}). No current-GT read, GT-based reset, optimizer construction, backward call or optimizer update exists in the supplied replay loop. The entire recorded `previous_bbox → bbox` chain is continuous, initialized once per sequence. This is source-visible and recorded-state evidence; the imported runtime internals are not included and cannot receive a transitive no-GT certificate from this audit.

GT validity was independently recomputed as finite coordinates and positive width/height, matching the locked metric's treatment of invalid rows ({ref('locked_depthtrack_pr.py',49)}). Eleven sampled rows are invalid: cart 600/606/624; bottle 560/570/573; jug 280/285/315/325; robot 454. They are excluded from all coverage and identity conclusions. Their stored zero IoUs are placeholders for invalid GT, not evidence of an observed failure. The metric variable `visible` means non-special/valid GT here; it does not establish visible pixels.

## B. Score normalization — PASS

The continuous coverage denominator is dataset GT area inside the image. The locked overlap denominator is governed by VOT region overlap. Neither is the model's maximum, minimum or mean. `response / quality` is stored as an explicitly defined same-state ranking counterfactual, with raw responses and probabilities also retained ({ref('046_replay_m125_selected_E6_observation.py',91)}). All 25 states' argmax selections and score/quality algebra are consistent with their archived arrays. Dividing a ranking score by quality is not reported as normalized tracking performance. These score alternatives are not a full trained-module ablation.

## C. Result existence and consistency — PASS

The four NN child return codes and four CPU child return codes are zero, agree between the standalone exit files and launches.json, and have matching terminal log records. `controller.exit` is a completion sentinel written by driver.py after its checks; it is **not independent OS-level attestation of the controller process exit** ({ref('005_driver.py',25)}, {ref('005_driver.py',33)}, {ref('005_driver.py',45)}). The controller result binds each sequence result by SHA. Launch arguments, selected frames, source hashes, E6 checkpoint hash and stage objects agree.

| Sequence | Tracker step calls | Prefix rows, including initialization | Candidate states | Max bbox difference | Max confidence difference |
|---|---:|---:|---:|---:|---:|
{table}

All **11,592 bbox components and 2,898 confidence values**, formatted to six decimals, equal the original saved files exactly; there are zero formatted mismatches. The supplied analyzer uses a 5.01e-7 tolerance and the maximum observed bbox difference is 5.000000555810402e-7. This audit additionally checked exact six-decimal formatting instead of treating only the tolerance flag as proof. Original numeric hidden states, queries and templates were not saved for comparison, so identical formatted outputs do not prove identical original hidden state.

There are 2,894 recorded tracker steps, four initializations, 25 additional Empty decoder evaluations and 6,400 candidate boxes in each geometry (12,800 across native-float32 and refined). These are distinct counters; the CPU archive/summarizer's `NN_calls: 0` describes those CPU stages, not the replay's total computation. The frozen source plan's `prepared_not_reviewed_not_run` and `NN_calls: 0` are historical plan fields; current execution state is established by the separate terminal records. The plan, config, replay source and stage hashes agree.

Every sampled action equals its corresponding prefix action and selected refined candidate. Native candidate geometry is explicitly float32, not the exact native double-precision output ({ref('046_replay_m125_selected_E6_observation.py',93)}). The record includes 32 C predictions and 30 template writes; these normal own-history updates are not optimizer steps or GT resets. The provided action history and supplied source show no alternative action committed to history.

## D. Metric invocation and verification — WARN

The analyzer actually calls the locked `_vot_overlaps` on replayed/saved prefixes and on both candidate geometries, then writes its returned arrays ({ref('047_analyze_m125_selected_E6_observation.py',59)}, {ref('047_analyze_m125_selected_E6_observation.py',71)}). The four CPU logs and their zero return codes support that execution. From the source loop/counts this is 58 overlap-function invocations covering 18,596 box pairs. It does **not** call the full PR/F-score evaluator; no new complete benchmark PR/F score is established by this diagnostic.

Exact local overlap reproduction is **UNVERIFIED_DEPENDENCIES_UNAVAILABLE**. In the supplied Python 3.12.12 / NumPy 1.26.4 environment, importing the exact metric fails at `import cv2`; directly importing `vot.region` also fails. The supplied standard Python 3.13.12 interpreter has neither package nor NumPy. Actual command, return code and traceback are preserved in [dependency_probes.json](dependency_probes.json). No package was installed and no approximate geometry implementation was substituted.

All per-candidate maxima, counts, chosen-index IoUs, oracle score ranks, quality and semantic-delta summaries were independently recalculated **from archived IoU vectors**, and match. Thus the recorded statement “13 of 14 GT-valid sampled states have no refined candidate with IoU ≥ 0.5” is arithmetically consistent with those arrays. Their underlying raster overlaps remain unverified locally. The native and refined count at robot 909 is 14 and 15 respectively; no other valid sampled state has a recorded candidate ≥ 0.5. Keep this distinction in every downstream numerical claim.

## Independent continuous box coverage — PASS

The audit independently clipped dataset GT to the image, then clipped that rectangle to the recorded search crop. Coverage is `area(GT ∩ image ∩ crop) / area(GT ∩ image)`. All 14 valid rows agree with the supplied summary. The actual tracker source sets `search_size=256` and calls `sample_target(..., output_sz=256)` ({source_ref('full_dense_tracker.py',29)}, {source_ref('full_dense_tracker.py',54)}); this file's hash matches all corresponding stage-bundle source entries. All 2,894 recorded `resize_factor × crop_size` values equal 256 within floating-point precision. Crop arithmetic also agrees with every previous predicted box. GT resized width/height below use that source-confirmed 256, not an assumed resolution.

Frame numbers are **zero-based replay indices**; dataset image filenames use `frame + 1` ({ref('046_replay_m125_selected_E6_observation.py',67)}).

| Sequence | Frame | Continuous coverage | GT width × height after search resize |
|---|---:|---:|---:|
{coverage_table}

Seven valid states have full box coverage and seven have zero. The five covered cart boxes are only 4.26–6.73 pixels wide after resize. This describes rectangular geometry and scale only. It does not establish whether the object is occluded, its pixels are visible, depth is valid, a competing instance is present, or the language identifies the target correctly. Image dimensions and RGB/depth hashes are saved in snapshots, but raw images are outside the supplied input set.

## E. Scope — WARN

The source plan explicitly selects posthoc E6 failures plus a positive sequence with robot quartile frames ([PLAN.json:57]({(BASE/'M125_SPLIT_REPLAY_SOURCE_20261015/PLAN.json').as_posix()}:57)). Four sequences and 25 chosen states are not a whole-benchmark sample. There is one replay per sequence at one selected checkpoint; no seed-replicated experiment is present in this package.

E6 was selected by the mean of DepthTrack Test and CDTB F, and the bundle explicitly marks `Test50_CDTB80_used_for_checkpoint_selection: true` and `human_review_used_multiframe_aids: true` ({ref('006_prediction_stage.json',209)}, {ref('006_prediction_stage.json',218)}, {ref('006_prediction_stage.json',319)}). The two historical F values and their average are selection metadata, not independently revalidated benchmark results in this audit. Describe the panel as posthoc diagnosis of a test-selected checkpoint with the declared human-input protocol; do not imply an untouched test set or independent selection.

At each sampled state, the primary action has already been committed before the Empty decoder call. The code checks the box state, query, selected feature and templates against their before-alternative copies and checks model digests across the run ({ref('046_replay_m125_selected_E6_observation.py',79)}, {ref('046_replay_m125_selected_E6_observation.py',114)}). This supports the stated read-only comparison within the saved runtime evidence, not an exhaustive proof for every hidden field. Same-state changes cannot establish complete-trajectory gains, a first-error cause, training ablation effects or semantic identity truth. Full crop coverage together with poor candidate overlap is a sampled geometric observation, not proof of a particular model mechanism causing failure.

## F. Classification — PASS / real_gt

Dataset boxes ground the coverage/overlap evaluation. Predictions are model-produced candidates, not model-produced labels. Saved-output parity is an implementation-consistency check. Empty and quality alternatives are fixed-state diagnostics whose scores are evaluated against real GT; they are not synthetic-proxy ground truth. No human semantic-identity evaluation or visibility-mask evaluation is present.

## Verification blockers and claim ceiling

1. **Exact raster IoU reproduction is blocked by absent local cv2/vot dependencies.** The archived numerical results may be quoted only with that verification boundary; they must not be described as independently reproduced official overlaps.
2. **Full transitive execution and original-state attestation are outside this input set.** The runtime/decoder/trusted-tracker imports, raw image bytes, checkpoint tensors, full-suite terminal/best/all_results/receipt files and original hidden-state snapshots are not supplied. Hash references and successful-stage assertions are consistent, but they cannot independently certify those absent artifacts. This audit does not revalidate the full training/VOT terminal claims or the parent suite's nine metrics.

No integrity contradiction was found in the supplied artifacts. Qualified coverage, counts and six-decimal parity findings are supported. Whole-benchmark rates, first-error causality, exact original hidden-state identity, intervention trajectory gains and semantic identity truth are unsupported by this panel. No new NN experiment is needed to state these limited findings.

## Reproducibility and actual reviewer failures

- [deterministic_checks.py](deterministic_checks.py): independent CPU checks, with [deterministic_checks.json](deterministic_checks.json) and [independent_coverage.json](independent_coverage.json).
- [input_inventory.json](input_inventory.json): all 55 SHA-256/size entries; [final_input_integrity.json](final_input_integrity.json): unchanged-input verification.
- [dependency_probes.json](dependency_probes.json): exact missing-dependency commands and failures.
- Reviewer-only first checker run used a tab delimiter for comma-delimited prediction files and exited 1. The failed source is preserved as `deterministic_checks_r0_csv_failure.py`. The next run incorrectly assumed phrase cosine was scalar per candidate; 25 shape checks failed because the actual field is 256×5. Its source/output are preserved as `deterministic_checks_r1_shape_assumption.py/.json`. Correcting these reviewer assumptions produced the final {det['check_count']:,}/0 result; no experiment input was changed. Initial broad text reads were truncated, then all primary content was parsed directly and checked; truncation was not treated as a successful semantic review of omitted text.
- The first exact-module import probe automatically generated one `locked_depthtrack_pr.cpython-312.pyc` file in the input directory. Its creation time was 2026-10-15T12:02:35.0704287Z and its SHA was `5ef2a20f85796e1d6e46cb2d7cbbb16ba1956c09f3ecb71d7889dbc186246f7a`. The sole reviewer-created cache directory was moved intact into this audit directory after checking its exact paths, name and creation time. [reviewer_cache_relocation.json](reviewer_cache_relocation.json) preserves the receipt. Future metric probes use Python `-B`; the final full checker rerun passed and no cache remained in the input directory. Original input contents were never changed. This transient output-location mistake is recorded rather than omitted.
- [TOOL_EXECUTION_LOG.json](TOOL_EXECUTION_LOG.json) records actual tool outcomes. The full reviewer request/response and attribution metadata are under `.aris/traces/experiment-audit/2026-10-15_run01/`.

Key SHA-256 values: archive `{hashes[str(BASE/'M125_R2_complete_replay_primary_inputs_retry1_20261015.json')]}`; replay `be2239091b279bebc614a4b77d31887ba4c0f6bd125862e194c502fb1a067cac`; analyzer `81081356adb1af5e5429883aa22a3fcba519204133f70001c013b7e195f5bfe4`; locked metric `05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc`; full dense tracker `{hashes[str(SOURCE/'full_dense_tracker.py')]}`. Full hashes are in the inventory.
'''
(OUT/'EXPERIMENT_AUDIT.md').write_text(report_md,encoding='utf-8')

prompt = '''Use the installed experiment-audit workflow as a fresh same-family provisional auditor. Read C:/Users/gb/.codex/skills/experiment-audit/SKILL.md and its applicable reviewer/tracing instructions. Requested model gpt-6-astra, reasoning max; actual routing UNATTESTED unless host evidence. Audit only the listed primary artifacts and code; do not trust executor summaries. Do not SSH, launch NN, install packages, modify inputs, send notifications, or overwrite any original results. Write audit outputs and deterministic checks/trace/inventory to C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_R2_RESULT_AUDIT_20261015. Files: C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_R2_RESULT_INPUTS_20261015/INPUTS.json (enumerates 49 exact extracted primary files, read all applicable code/result/GT/trajectory/source entries); C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_R2_complete_replay_primary_inputs_retry1_20261015.json; C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_R2_complete_box_coverage_summary_20261015.json; C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/summarize_m125_selected_E6_observation.py; C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/full_dense_tracker.py; C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M125_SPLIT_REPLAY_SOURCE_20261015/PLAN.json. Checklist: experiment-audit A-F provenance, normalization, result existence, metric invocation, scope, evaluation classification. Additionally verify archived raw SHA, terminal exits and actual calls/candidate states/counts, dataset-GT rather than model-generated targets, all full prefix bbox and confidence consistency against original saved outputs at six decimals, no post-init GT action/reset/optimizer, alternative predictions do not commit history, source/plan/stage consistency, GT-invalid samples excluded from coverage/identity conclusions. Independently recompute continuous GT/image/crop box coverage and resized GT dimensions, using actual source search size; preserve distinction between box coverage and occlusion/visibility. Recompute candidate overlap results only if exact official dependencies are actually available; otherwise explicitly state the verification boundary without approximating official raster IoU as exact. Check sampled-state claims cannot imply whole-benchmark rates, first-error causality, exact original hidden-state identity, full-trajectory intervention gains, or semantic identity truth. Report A-F status and overall PASS/WARN/FAIL, concrete blockers and qualifiers, hashes and actual tool execution/failures. Runtime interpreter that has NumPy: C:/Users/gb/AppData/Local/uv/cache/archive-v0/7oLWbeZ6v0oUodecKf4d1/Scripts/python.exe; standard Python C:/Users/gb/AppData/Roaming/uv/python/cpython-3.13-windows-x86_64-none/python.exe. No other assumptions. Keep review bounded to these artifacts; no new experiment or literature review.'''
dump(trace/'run.meta.json',dict(skill='experiment-audit',run_id='2026-10-15_run01',started_at=None,recorded_at=now,
    start_time_note='Exact reviewer spawn wall time was not supplied; no invented timestamp.',project_dir=str(BASE.parent.parent),**attribution))
dump(trace/'001-result-audit.request.json',dict(call_number=1,purpose='result-audit',timestamp=None,recorded_at=now,
    tool='spawn_agent',model='gpt-6-astra',reasoning_effort='max',actual_model='UNATTESTED',actual_reasoning_effort='UNATTESTED',
    files_referenced=[r['path'] for r in inventory],prompt=prompt))
(trace/'001-result-audit.response.md').write_text(report_md,encoding='utf-8')
dump(trace/'001-result-audit.meta.json',dict(call_number=1,purpose='result-audit',timestamp=now,status='ok',substantive_verdict='WARN',duration_ms=None,**attribution))
events=OUT/'.aris/meta';events.mkdir(parents=True,exist_ok=True)
(events/'events.jsonl').write_text(json.dumps(dict(event='review_trace',skill='experiment-audit',purpose='result-audit',agent_id=agent,trace_path=str(trace),status='ok',verdict='WARN'))+'\n',encoding='utf-8')

tool_log = [
 dict(chunk_id='0039c4',tool='exec_command',exit_code=0,action='Read session instructions, daily notes and skill; output truncated. No MEMORY.md loaded in this delegated context.'),
 dict(chunk_id='a63942',tool='exec_command',exit_code=0,action='Read reviewer policy, independence, integrity, tracing and integration instructions.'),
 dict(chunk_id='d33142',tool='exec_command',exit_code=0,action='Read exact file inventory and check ancestor AGENTS.md paths; created only requested audit directory.'),
 dict(chunk_id='92578e',tool='exec_command',exit_code=0,action='Broad archive text read was truncated; replaced with complete JSON parsing and per-entry deterministic checks.'),
 dict(chunk_id='3997db',tool='exec_command',exit_code=0,action='inspect_inputs.py read/hash/decoded all 49 manifest entries and parsed both supplemental JSON artifacts.'),
 dict(chunk_id='1ffb8f',tool='exec_command',exit_code=0,action='Read full driver, replay and analyzer source with line numbers.'),
 dict(chunk_id='0df4f7',tool='exec_command',exit_code=0,action='Read full locked metric, summarizer and full_dense source with line numbers.'),
 dict(chunk_id='9fe73d',tool='exec_command',exit_code=0,action='inspect_structures.py parsed all stage/result/action/snapshot structures and preserved structured readout.'),
 dict(chunk_id='55e347',tool='exec_command',exit_code=0,action='Read remaining staged metadata and all execution logs; terminal display truncated some long snapshot lines, arrays subsequently checked completely by script.'),
 dict(chunk_id='acfc8c',tool='exec_command',exit_code=0,action='Initial supplied-runtime dependency discovery: NumPy 1.26.4 available; cv2/vot/shapely/numba absent.'),
 dict(chunk_id='2c496f',tool='exec_command',exit_code=1,action='First deterministic checker stopped at line 138 due reviewer CSV delimiter assumption.',error="ValueError: could not convert string '954.000000,432.900000,328.000000,299.700000' to float64 at row 0, column 1.",preserved_source='deterministic_checks_r0_csv_failure.py'),
 dict(chunk_id='e5afaa',tool='exec_command',exit_code=0,action='Preserved failed checker, read actual comma-delimited rows and exact dependency failures.'),
 dict(chunk_id='50fe7a',tool='exec_command',exit_code=0,action='Second deterministic checker completed: 3773 checks, 3748 pass, 25 false due reviewer scalar phrase-cosine assumption.',preserved_source='deterministic_checks_r1_shape_assumption.py',preserved_result='deterministic_checks_r1_shape_assumption.json'),
 dict(chunk_id='316163',tool='exec_command',exit_code=0,action='Inspected actual 256x5 phrase-cosine schema and preserved failed-assumption source/result.'),
 dict(chunk_id='d8f727',tool='exec_command',exit_code=0,action='Corrected-schema deterministic checker completed: 3798 pass, zero fail. Both actual official dependency-import probes fail as recorded.'),
 dict(chunk_id='8a4cda',tool='exec_command',exit_code=0,action='Read exact source/metadata evidence lines and independently derived valid-state coverage table.'),
 dict(chunk_id='b2b9a3',tool='exec_command',exit_code=0,action='Wrote initial audit report, machine JSON and full reviewer trace.'),
 dict(chunk_id='dc8058',tool='exec_command',exit_code=0,action='Inspected exact input __pycache__ path; identified the sole reviewer-generated metric bytecode file and its creation time.'),
 dict(chunk_id='cc164e',tool='exec_command',exit_code=0,action='Verified absolute source/destination and sole reviewer-created filename/time; moved cache intact into audit output and wrote SHA relocation receipt.'),
 dict(chunk_id='b5bd63',tool='exec_command',exit_code=0,action='Final bytecode-disabled deterministic checker completed: 3798 pass, zero fail. Expected cv2/vot import failures recorded.',result='deterministic_checks.json'),
]
dump(OUT/'TOOL_EXECUTION_LOG.json',dict(records=tool_log,probe_detail='dependency_probes.json',
    note='Recorded outcomes, not fabricated raw transcripts. Tools with truncated display are explicitly identified. All apply_patch calls wrote reviewer-owned files only. The official import probes were executed once per checker attempt; final exact commands/stdout/stderr are retained. No original experiment was rerun.',
    prohibited_experiment_actions_performed=[],transient_audit_byproduct='One reviewer-created metric .pyc temporarily generated under input directory; relocated into audit output with receipt. No original file contents changed.'))
print(json.dumps(dict(report=str(OUT/'EXPERIMENT_AUDIT.md'),json=str(OUT/'EXPERIMENT_AUDIT.json'),verdict='WARN',checks=det['check_count'],failed=det['failed'],trace=str(trace)),indent=2))
