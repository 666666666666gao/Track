"""Write the completed-results audit and its full same-family reviewer trace."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

A = Path(__file__).resolve().parent
R = A.parent / 'completed'
P = A.parent.parent / 'sttrack_m84_centered_20260920' / 'completed'
data = json.loads((A / 'independent_recompute.json').read_text())
stamp = datetime.now(timezone.utc).isoformat()
trace = A / '.aris' / 'traces' / 'experiment-audit' / '2026-09-21_run01'
trace.mkdir(parents=True, exist_ok=True)

request = r'''Apply C:\Users\gb\.codex\skills\experiment-audit\SKILL.md as fresh same-family/provisional integrity reviewer. Read source artifacts directly; verify rather than trust author claims. Root needs a bounded completed-result audit while independently auditing caption provenance. Read-only experiment; you may write ONLY your audit scripts/reports/traces under D:\Program Files\UserCache\gb\codex\tmp\sttrack_m85_state_diagnostic_20260921\audit. Main evidence directory D:\Program Files\UserCache\gb\codex\tmp\sttrack_m85_state_diagnostic_20260921\completed contains spec.json, EXPERIMENT_PLAN.md, same_state.py, analyze_saved.py, run_replay.sh, launch.py, code_review.json, review_receipt.json, all *.exit/log, predictions/*.json and *.npz, preflight/*, diagnostic_result.json. Parent evidence D:\Program Files\UserCache\gb\codex\tmp\sttrack_m84_centered_20260920\completed contains training_spec.json, recursive_result.json, recursive_spec.json, raw predictions and dataset GT (discover concrete relative paths), code/integration. Check A-F from skill: actual dataset GT provenance and prediction sealing before GT, no prediction-based normalization, file/metric existence, executed code, scope limited 3 posthoc sequences not benchmark/cause proof, evaluation classification. Independently recompute M85 dense decode and primary single-step metrics/H10/candidate counts from saved artifacts and parent GT, hash bindings and exact saved Category replay versus M84 raw. Verify counterfactuals don't commit state and Empty/native parity assertions actual executed. Don't rerun GPU or training, don't access SSH/secrets. Reports distinguish deterministic evidence, actual unverified runtime assertions, same-family semantic verdict provisional. Preserve actual request, full final response and metadata in .aris trace inside audit. Supply exact audit/report paths and remaining limitations, no fabricated independent training reproduction. Do not modify frozen source/results.'''

checks = {
    'A_gt_provenance': {
        'status': 'PASS',
        'details': 'Dataset labels are read from DepthTrack Train groundtruth.txt, then exported unchanged as dataset_gt/<sequence>.txt. All three local GT byte hashes agree with both M84 and M85 frozen cases and M84 export manifest. Prediction JSON/NPZ checks precede label parsing. No model-output-derived GT found. Official benchmark validation and independent dataset-download authentication were not performed.',
        'evidence': ['M85/analyze_saved.py:40-54', 'M85/same_state.py:64-91', 'M84/training_spec.json:8', 'M84/export_completed.py:43-63', 'M84/run_recursive.py:81-105']},
    'B_score_normalization': {
        'status': 'WARN',
        'details': 'Primary IoU, threshold counts, H10 and crop coverage have no prediction-statistic rescaling. Auxiliary native_spatial_kl explicitly normalizes each response by its own spatial sum and uses the native prediction as reference; it is a self-supervised distribution diagnostic, not dataset accuracy. Raw maxima and masses are retained. No primary performance manipulation found, but a blanket claim of no prediction normalization would be false.',
        'evidence': ['M85/analyze_saved.py:24-27', 'M85/analyze_saved.py:62-93', 'M85/same_state.py:120-129']},
    'C_result_existence': {
        'status': 'PASS',
        'details': 'All 2616 positions exist. All diagnostic per-frame float metrics, candidate indices/counts, crop flags, summaries and H10 intervals reproduce exactly; maximum reported metric error is 0.0. Four stage exit codes are 0 and logs match receipts/results. The 101-position preflight JSON and dense arrays exactly match the full-run prefix. 481 hash bindings across 313 observed inputs pass, including 283 M84 manifest entries and 161 integrated source files.',
        'evidence': ['M85/predictions/receipt.json:1-34', 'M85/preflight/receipt.json:1-20', 'M85/replay.log:2-5', 'M85/analysis.log:1-56', 'audit/independent_recompute.json']},
    'D_executed_code': {
        'status': 'PASS',
        'details': 'The bound runner calls the Category tracker once per chronological frame; alternate adapter/head calls follow and state checks precede row emission. Dense decoding, IoU and low_runs are called by the analyzer and their outputs are present and independently reproduced. Normal Python queue invocations and complete receipts support runtime assertion completion. Reviewer did not rerun model execution or directly inspect historical query/template tensors; do not interpret this as an independent GPU replay.',
        'evidence': ['M85/run_replay.sh:5-13', 'M85/same_state.py:86-149', 'M85/analyze_saved.py:67-96', 'M84/code/lib/models/sttrack/sttrack.py:175-205', 'M84/code/lib/test/tracker/sttrack.py:96-139']},
    'E_scope': {
        'status': 'PASS',
        'details': 'The plan and result explicitly restrict conclusions to three posthoc selected DepthTrack Train development sequences, one M84 Category checkpoint, seed 2027, and Category-carried history. 2616 noninitial positions include 2200 valid GT frames and 416 invalid ones. Hindsight candidate capacity and single-step alternatives cannot establish independent alternative trajectories, earlier-state causality, unbiased failure prevalence, or a benchmark improvement. M84 all_gates_pass remains false.',
        'evidence': ['M85/EXPERIMENT_PLAN.md:3-20', 'M85/spec.json:55-58', 'M85/diagnostic_result.json:2-6', 'M84/recursive_result.json:964', 'M84/recursive_result.json:1049-1051']},
    'F_evaluation_type': {
        'status': 'PASS',
        'primary_type': 'real_gt',
        'auxiliary_types': {'native_spatial_kl': 'self_supervised_proxy', 'Empty_native_parity': 'engineering_invariance_check'},
        'details': 'Selected-box IoU, H10, crop coverage and dense oracle capacity use exported dataset GT. The KL reference is model-generated and supports only distribution consistency. Map parity is a numerical invariance assertion. No human evaluation, simulation, benchmark evaluator or independent training reproduction was done.',
        'evidence': ['M85/analyze_saved.py:51-77', 'M85/same_state.py:102-103', 'M85/same_state.py:120-129']}
}

limitations = [
    'No GPU inference, training, SSH access, external dataset retrieval, or checkpoint execution was performed by this reviewer.',
    'Native backbone weights, development token banks, and original RGB/depth images are referenced by the executed source and successful runtime receipts but were not directly byte-audited from this local export. The final M84 adapter checkpoint itself was locally hash-verified.',
    'Full Empty dense maps and before/after query or template-content snapshots were not exported. All saved Empty/native boxes and scalar records match exactly; all-map equality and noncommit remain source-plus-completion-receipt runtime evidence rather than an independent replay.',
    'Template noncommit assertions check object identities and query values, not serialized tensor snapshots. Direct source inspection finds alternate paths stateless and no tracker.track/template/query update calls in them; receipt wording should not imply broader forensic proof.',
    'Prediction-before-GT ordering is verified in the bound analysis source and sealed-file consistency. No external trusted timestamp or OS file-access trace independently attests historical access order. Cases were selected posthoc with prior result knowledge.',
    'GT provenance is an internally consistent dataset-path/export/hash chain, not a new authentication against a separately downloaded official archive.',
    'Crop inclusion is geometric in the Category-carried crop. No correct candidate means none among the 256 decoded boxes from each of the three saved heads at that state, not target absence or impossibility under a different state/search policy.',
    'Zero rescue/harm counts use the stated severe thresholds: low <=0.1 and correct >=0.5. They do not imply identical outputs or no moderate quality changes.',
    'Semantic interpretation is a fresh same-family Codex review with provisional acceptance; deterministic saved-artifact checks are accepted within their stated scope.'
]

claims = [
    {'id': 'C1', 'claim': 'M85 saved Category trajectory exactly matches the three M84 parent trajectories.', 'impact': 'supported', 'basis': 'All 2616 bbox/score/prior-state records exactly match; hashes bound to parent receipts.'},
    {'id': 'C2', 'claim': '646 Category low frames partition as 627 center-outside, 18 center-inside/no Category-correct dense candidate, and 1 center-inside/has candidate.', 'impact': 'supported', 'basis': 'Independent all-frame GT computation; 2200 valid frames.'},
    {'id': 'C3', 'claim': '645 of those 646 low frames have no correct candidate in any saved Category/Swapped/native head.', 'impact': 'supported_with_scope', 'basis': '256 dense candidates per head, Category history, correct >=0.5.'},
    {'id': 'C4', 'claim': 'Raw, Swapped or native selected boxes rescue none of the Category-low frames.', 'impact': 'supported_with_scope', 'basis': 'Same-state single-step thresholds only; no independent alternative recurrence.'},
    {'id': 'C5', 'claim': 'Earlier template contamination or query updates caused the failures.', 'impact': 'unsupported', 'basis': 'No state intervention separates earlier-state causes.'},
    {'id': 'C6', 'claim': 'Removing Hann, changing text or widening search improves benchmark performance.', 'impact': 'unsupported', 'basis': 'No changed policy recurrence or formal benchmark evaluation here.'},
    {'id': 'C7', 'claim': 'M84 can now be promoted after M85.', 'impact': 'unsupported', 'basis': 'M84 failed gates remain false; M85 is a selected diagnostic.'},
    {'id': 'C8', 'claim': 'Empty/native full-map parity and counterfactual state noncommit were independently reproduced by the reviewer.', 'impact': 'unsupported_as_worded', 'basis': 'Historical source/receipt evidence plus saved scalar parity, no GPU replay.'}
]

table_rows = []
for s in data['sequences']:
    v = s['summary']
    table_rows.append(f"| {s['sequence']} | {s['positions']} | {v['valid_frames']} | {v['category_low_frames']} | {v['low_center_outside']} | {v['low_center_inside_no_correct_dense']} | {v['low_center_inside_correct_dense']} | {v['low_all_heads_no_correct_dense']} |")
metric_rows = []
for arm, values in data['aggregate_single_step_metrics'].items():
    metric_rows.append(f"| {arm} | {values['raw']['mean_iou']:.12f} | {values['hann']['mean_iou']:.12f} | {values['raw']['low_iou_frames']} | {values['hann']['low_iou_frames']} |")

report = f'''# M85 completed-result experiment integrity audit

**Date:** 2026-09-21. **Auditor:** gpt-6-astra, reasoning max, fresh context (`fork_turns=none`), native Codex agent `/root/m85_completed_integrity`. Root confirmed actual spawn parameters. **Review independence:** same-family. **Acceptance:** provisional for semantic judgment; deterministic saved-artifact checks accepted within scope.

## Overall verdict: WARN

The saved M85 diagnostic is internally consistent and its primary numbers independently reproduce. No fake GT, altered primary metric normalization, missing claimed result, or numerical mismatch was found. The WARN records the auxiliary KL's self-reference and the limits of historical runtime-assertion evidence; it is not evidence that the primary results are false.

**Integrity status:** warn. **Deterministic recomputation:** PASS. No experiment files were modified, no GPU work or training was run, and no SSH or secrets were accessed. This is not an independent reproduction of training or model inference.

## Inputs and binding

M85 root: `{R}`.

M84 root: `{P}`.

The independently executed `{A / 'independent_recompute.py'}` imports neither the author's analyzer nor tracker. It uses NumPy on CPU, transcribes the head decoding and image-coordinate arithmetic, reads only saved arrays/predictions and exported labels, and compares every row.

- 313 observed input files remained byte-identical across the verification pass; 481 hash bindings passed.
- All 283 entries of the parent export manifest and all 161 integration source hashes passed.
- M84 Category final adapter SHA-256: `c63605ebcb1f702de66f96967255e5301bfdca1e3c40450b6d4b8a952791b1e0`.
- M85 spec SHA-256: `748b11527aa7302490a37cfdf9b2b7643239e604943880e675d2e480b4e533ec`.
- Each of the three local GT files matches M84 cases, M85 cases, and the M84 export manifest. Each parent Category prediction matches M85's predeclared hash and the M84 Category receipt.
- Preflight, replay, controller, and analysis exit files contain `0`; replay log receipts equal saved receipt JSON and analysis.log equals independently checked result summaries. The preflight's 101 rows and all three dense arrays exactly equal the full replay prefix.
- Source hashes, review report/JSON/receipt hashes, launch-to-spec binding, and analyzer-to-prediction-receipt binding all pass. The M83 source digest is a checked historical reference, not a claim that M83 code executed in M85.

Runtime: Python {data['runtime']['python_version']}, NumPy {data['runtime']['numpy_version']}, `{data['runtime']['python']}`. Invocation: `uv run --no-project --with numpy python <audit>/independent_recompute.py`, exit 0. The base E:\\python.exe lacked NumPy; the existing uv-managed isolated NumPy runtime was used without changing project environments.

## A. Ground truth provenance: PASS within the archived chain

`M84/training_spec.json:8` names `/root/autodl-tmp/depthtrack/train/sequences`. `M85/analyze_saved.py:53-54` reads `<dataset>/<sequence>/groundtruth.txt`; no model outputs create these labels. `M84/export_completed.py:59-63` exports each GT byte stream as `dataset_gt/<sequence>.txt` only after checking all three parent prediction families. The three local files directly used here are:

- `M84/dataset_gt/mobilephone02_indoor.txt`: SHA-256 `d27bba52e1ebe7f816b9d2ce264d332404530765bcce830029070b146c52e45c`, 701 rows.
- `M84/dataset_gt/car02_indoor.txt`: SHA-256 `ed1171be55b7782858af063f174b2ece551f35b6ee2b5a4d0e81b5fc53f87ec7`, 753 rows.
- `M84/dataset_gt/ghostmask_indoor.txt`: SHA-256 `5aba29f532dd42f9df5bbfd1c6c5558c8632f9d62fbf5fc5d52470d799ef9406`, 1165 rows.

Initialization row zero is excluded. Validity is finite coordinates and positive width/height; 416 later rows are invalid and break H10 runs (`M85/analyze_saved.py:29-37,62-66`). No assumption of absence or rotation is made from invalid labels.

`M85/same_state.py:67-91` loads RGB/depth frames, bank values, an initialization box, and sealed parent predictions, without opening subsequent GT. `M85/analyze_saved.py:40-50` verifies complete receipt/source/spec and all prediction JSON/NPZ hashes before the GT read at line 54. This establishes the intended and bound executed code order plus artifact consistency. The three sequences were selected posthoc; this is not a blind evaluation. No external dataset download or trusted file-access timeline was obtained.

These are custom continuous-box diagnostics, not official DepthTrack Test/VOT scoring. The scope makes official benchmark-script use inapplicable to the claimed result.

## B. Score normalization: primary PASS; auxiliary WARN

Primary box IoU uses intersection divided by geometric union (`M85/analyze_saved.py:24-27`), with ordinary counts and fixed thresholds. Means added by this audit divide IoU sums by valid-frame counts. No primary metric is divided by the model's own max, minimum, or mean.

There **is** prediction-sum normalization in auxiliary `native_spatial_kl` (`M85/same_state.py:120-129`): native and alternative dense response scores are each normalized into spatial distributions, then KL(native || alternative) is measured. This is appropriate as an explicitly labeled **self_supervised_proxy** distribution diagnostic; it is not dataset accuracy. Raw maxima and response mass are saved alongside it. The main analyzer never uses KL to produce performance metrics, thresholds or candidate counts. Do not describe all outputs as unnormalized or all evaluations as real-GT accuracy.

Independent float64 KL differs from saved GPU float32 KL by at most `3.6840784099073653e-07`; native/Empty saved KL and every saved selected-box/scalar field agree exactly. This check is numerical consistency, not proof of semantic caption correctness.

## C. File and numerical result existence: PASS

All 2616 noninitial rows and three `[frames-1,5,16,16]` float32 finite dense arrays exist. Stored map channels match `score,width,height,offset_x,offset_y`, and the saved 16×16 Hann window agrees with its analytic definition within `9.880329604472493e-08`.

All 2616 saved Category Hann boxes and scores equal the M84 raw Category trajectory exactly, and all previous boxes equal the prior M84 row. The audit independently reproduces selected raw/Hann peak indices, maxima, crop origin/side, template-write eligibility, selected-box IoU, dense best IoU, dense correct counts, dense best indices, crop flags, summaries, and H10 intervals. The per-frame reported metric maximum absolute error is **0.0**. Dense decoded selected boxes differ from online saved boxes by at most `3.051757818184342e-05` pixels, below the frozen `1e-4` tolerance; this is float32 mapping roundoff, not an IoU discrepancy.

| Sequence | Tracking calls | Valid GT | Category low | Center outside | Center inside, no Category correct dense | Center inside, has Category correct dense | No correct dense in any saved head |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}
| Total | 2616 | 2200 | 646 | 627 | 18 | 1 | 645 |

“Low” is Category selected Hann IoU ≤ 0.1. “Correct dense” means a decoded candidate box with GT IoU ≥ 0.5 among 256 candidates per head. All raw-selection, Swapped-selection, and native-selection rescue counts are 0. All severe harm counts (Category ≥0.5, alternative ≤0.1) are also 0. Those threshold counts do not imply the outputs are identical.

The one low frame with any correct dense candidate is `mobilephone02_indoor`, frame index **291**. All three heads have three correct candidates; dense best IoU is Category 0.5483026527, Swapped 0.5437777909, native 0.5556136229, while selected Hann IoUs are approximately 0.09066, 0.09000, 0.09990. Its GT center is in the crop, but its full GT box is not. Candidate existence is hindsight capacity and no selected head recovers that frame.

H10 is identical across the three saved heads under Category state, using zero-based, half-open frame intervals:

- mobilephone: `[497,701)`.
- car: `[215,550)`, `[644,675)`.
- ghostmask: `[847,902)`.

For completeness the audit computed single-step mean IoU over the 2200 valid selected frames. These are not unbiased development or benchmark scores, and the alternative rows do not carry their own state:

| Saved head | Raw selected mean IoU | Hann selected mean IoU | Raw low frames | Hann low frames |
|---|---:|---:|---:|---:|
{chr(10).join(metric_rows)}

Full per-sequence metrics and all hash bindings are in `independent_recompute.json`. The source diagnostic's summary/H10 locations are `M85/diagnostic_result.json:9-46`, `:18371-18420`, and `:36119-36156`; its analysis log is only 56 lines and carries the same summaries.

## D. Execution and state assertions: PASS with explicit runtime limits

The queue calls normal Python on `same_state.py --preflight`, then full `same_state.py`, with a nonzero preflight stopping the run (`M85/run_replay.sh:5-13`). Complete receipts/logs are consistent with all 101 preflight and 2616 full positions reaching the code after assertions; they are execution evidence, not independent GPU reruns by this reviewer.

The only state-committing call is `tracker.track(image)` at `M85/same_state.py:89`. The capture occurs at the adapter boundary. Category map references are retained before alternative head hooks overwrite the capture dictionary (`:92-101`). Empty/Swapped call only the adapter and head; native calls only the head. The frozen adapter returns new feature values (`M84/code/lib/models/sttrack/centered_semantic_adapter.py:12-20`; `semantic_spatial_adapter.py:25-52`), and `STTrack.forward_head` performs head computation without a tracker update (`M84/code/lib/models/sttrack/sttrack.py:175-205`). Query update and template-write paths are in the Category forward/track path, not these alternatives.

`M85/same_state.py:102-103` asserts exact Empty/native score, size and offset maps. `:95-110` snapshots bbox, cloned query values and template identities and checks them after counterfactual reads. All exported Empty and native selected boxes, indices, maxima, masses and KL values match exactly across 2616 rows. The full Empty dense maps are intentionally omitted, and historical query/template snapshots are not exported; all-map parity and noncommit therefore remain source-plus-completion-receipt assertions. This report does not promote them to independently replayed claims.

The `actual_template_write` field is deterministically computed eligibility, not a separately saved write event. Its 18 true rows match the frozen two-template tracker branch with interval 50 and strict Hann score >0.75 (`M84/code/lib/test/tracker/sttrack.py:129-139`; config lines 16,67-68). The audit verified this predicate from the parent scores and labels it as eligibility.

The M85 analyzer calls dense decoding, IoU and H10; their returned values appear in the saved results and reproduce. The parent's `recursive_metric.py` also contains an older unused CLI main, but the parent runner imports its `statistics` function (`M84/run_recursive.py:80,105`). This audit does not imply that unrelated legacy entry point executed.

## E. Scope: PASS for the stated diagnostic, no promotion

One M84 Category adapter checkpoint, one seed (2027), three posthoc selected DepthTrack Train development sequences. Empty/Swapped/native alternatives are reads of Category-derived state, not separately trained models or independent recurrences. Dense candidates are oracle hindsight capacity. `M85/EXPERIMENT_PLAN.md:3,7,13,20` explicitly states these limits; `M84/recursive_result.json:964` remains `all_gates_pass=false` and public evaluation remains false at line 1051.

The evidence supports the observed current-crop and candidate conditions. It does not isolate earlier template/query contributions, prove captions caused a failure, estimate population prevalence, or validate a new policy. “Inherited observation/state failure” should be read as the present state-conditioned observation, not a proven earlier causal mechanism.

## F. Evaluation classification

- **real_gt:** selected-box IoU, low-overlap/H10, geometric crop coverage, and dense oracle candidate capacity.
- **self_supervised_proxy:** native_spatial_kl, since its reference is a native model response distribution.
- **Engineering invariance checks:** exact Category replay, Empty/native map assertion, counterfactual noncommit assertion. These are not benchmark performance evaluations.

No simulation-only, human-evaluation, official public benchmark, independent training reproduction or semantic caption-ground-truth assessment occurred in this audit.

## Action items and claim impact

1. Carry the selected-three, same-Category-state, valid-GT and threshold qualifications into every summary; retain both negative and positive conditions, including the one candidate-capacity exception.
2. Label native spatial KL as an auxiliary distribution consistency proxy. Keep it separate from the real-GT metrics; no frozen-result modification is necessary.
3. Describe Empty full-map parity and counterfactual noncommit as checked by the completed original replay, supported by bound source/receipts. Describe saved Empty/native scalar equality and exact Category trajectory matching as independently reverified here.
4. Preserve the failed M84 performance verdict. Any claim about repairing earlier state, Hann removal, wider search or a changed text policy requires a separately defined state intervention or chronological policy evaluation; this audit supplies no such result.

No blocking numerical correction is requested for the frozen diagnostic.

## Remaining limitations

{chr(10).join('- ' + x for x in limitations)}

## Audit artifacts

- `EXPERIMENT_AUDIT.md`: this report.
- `EXPERIMENT_AUDIT.json`: structured A–F verdict, claims, limitations, input hashes and trace path.
- `independent_recompute.py` / `independent_recompute.json`: CPU verifier and full evidence.
- `write_audit_report.py`: report and trace creation source.
- `.aris/traces/experiment-audit/2026-09-21_run01/`: actual request, full final reviewer response and route metadata.
'''

final = f'''M85 completed-result audit is finished. **Semantic verdict: WARN, same-family/provisional. Deterministic saved-artifact checks: PASS.** No fake GT, primary metric rescaling, missing claimed output or numerical mismatch was found.

- Independently recomputed all 2616 saved positions from NPZ/JSON plus exported dataset GT. Every reported per-frame diagnostic metric, candidate count/index, crop flag, summary and H10 interval matches exactly; maximum reported metric error is 0.0. Selected dense-decode error is at most 3.051757818184342e-05 pixels, within the frozen 1e-4 tolerance.
- Category bbox, score and previous-state rows exactly match all three M84 raw trajectories. The 101-position preflight is an exact prefix of saved rows and dense arrays. All 283 parent manifest files, 161 integrated source hashes and 481 binding checks pass.
- There are 2200 valid GT frames and 646 Category-low frames: 627 have the GT center outside the current crop, 18 have the center inside without a correct Category dense candidate, and 1 has a correct dense candidate. Of the 646 low frames, 645 have no correct candidate in any of the three saved heads. Raw, Swapped and native selected-box rescue counts are all zero; severe-harm counts are also zero under the stated thresholds.
- H10 is identical across Category/Swapped/native under Category history: mobilephone [497,701), car [215,550) and [644,675), ghostmask [847,902). The one candidate-capacity exception is mobilephone frame index 291; none of the selected heads chooses its correct dense candidates.

The WARN preserves two distinctions. Primary IoU/H10/candidate metrics are **real_gt**; auxiliary `native_spatial_kl` uses prediction-sum normalization and a native-response reference, so it is a **self_supervised_proxy**, not accuracy. The original replay's Empty/native full-map equality and noncommit assertions have bound source plus complete receipt/log evidence; all saved Empty/native scalar/box records were independently verified equal, but full Empty maps and query/template snapshots were not exported. I did not rerun GPU inference or training. Native backbone weights, development text-bank bytes and original RGB/depth frames were not independently checked from this local export; the final Category adapter checkpoint was hash-verified.

The evidence remains a three-sequence posthoc state diagnostic. It does not prove earlier template/query causality, independent alternative trajectories, unbiased prevalence, benchmark improvement or M84 promotion. M84's failed gate verdict remains unchanged.

Reports and reproducible evidence:

- `{A / 'EXPERIMENT_AUDIT.md'}`
- `{A / 'EXPERIMENT_AUDIT.json'}`
- `{A / 'independent_recompute.py'}`
- `{A / 'independent_recompute.json'}`
- Full request/response and actual reviewer metadata: `{trace}`.

No frozen source or result was modified. All experiment conclusions above were rechecked against current local artifacts; the memory lookup supplied only the pre-existing evidence-bound reporting convention.

<oai-mem-citation>
<citation_entries>
MEMORY.md:411-418|note=[Prior evidence binding and prediction sealing convention rechecked in current source]
</citation_entries>
<rollout_ids>
01a06f16-fd85-7da3-a232-113efe970f88
</rollout_ids>
</oai-mem-citation>'''

structured = {
    'audit_skill': 'experiment-audit', 'verdict': 'WARN', 'overall_verdict': 'warn', 'integrity_status': 'warn',
    'reason_code': 'auxiliary_proxy_normalization_and_historical_runtime_verification_limits',
    'summary': 'All primary saved-artifact numbers reproduce; preserve proxy classification and runtime-evidence limits.',
    'date': '2026-09-21', 'generated_at': stamp,
    'agent_id': '/root/m85_completed_integrity', 'verdict_id': '/root/m85_completed_integrity',
    'executor_model': 'gpt-6-astra', 'executor_family': 'openai',
    'reviewer_model': 'gpt-6-astra', 'reviewer_family': 'openai', 'reviewer_reasoning': 'max',
    'fork_turns': 'none', 'spawn_parameter_attribution': 'Actual native spawn parameters confirmed by root in message.',
    'review_independence': 'same-family', 'acceptance_status': 'provisional',
    'deterministic_evidence': {'status': 'PASS', 'review_independence': 'deterministic', 'acceptance_status': 'accepted',
        'report': str(A / 'independent_recompute.json'), 'script': str(A / 'independent_recompute.py')},
    'checks': checks, 'claims': claims, 'limitations': limitations,
    'aggregate_summary': data['aggregate_summary'], 'audited_input_hashes': data['audited_input_hashes'],
    'trace_path': str(trace), 'gpu_or_training_reproduced': False, 'source_or_result_modified': False,
    'report_sha256': hashlib.sha256(report.encode()).hexdigest(),
}

def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')

(A / 'EXPERIMENT_AUDIT.md').write_text(report, encoding='utf-8', newline='\n')
save(A / 'EXPERIMENT_AUDIT.json', structured)
(trace / '001-completed-result-audit.response.md').write_text(final, encoding='utf-8', newline='\n')
save(trace / 'run.meta.json', {'skill': 'experiment-audit', 'run_id': '2026-09-21_run01', 'started_at': '2026-09-21',
    'timestamp_precision_note': 'Exact native spawn clock not provided; report timestamp is actual write time.',
    'generated_at': stamp, 'executor': 'codex', 'executor_model': 'gpt-6-astra', 'executor_family': 'openai',
    'review_independence': 'same-family', 'acceptance_status': 'provisional', 'project_dir': str(R)})
save(trace / '001-completed-result-audit.request.json', {'call_number': 1, 'purpose': 'completed-result-audit',
    'timestamp': None, 'timestamp_note': 'Historical spawn timestamp unavailable to reviewer; request captured verbatim.',
    'tool': 'spawn_agent', 'model': 'gpt-6-astra', 'reasoning_effort': 'max', 'fork_turns': 'none',
    'files_referenced': [str(R), str(P)], 'prompt': request,
    'followup_messages': [
        'Actual native spawn explicit model=gpt-6-astra, reasoning_effort=max, fork_turns=none, task_name=m85_completed_integrity. No overrides/fallback. Please save those as root-confirmed spawn parameters; your independent interpretation remains same-family/provisional.',
        r'Use uv run --no-project --with numpy python <audit script> (existing uv available, its cache redirected to D:). Prior M84 audit may also identify known runtime; inspect D:\Program Files\UserCache\gb\codex\tmp\sttrack_m84_centered_20260920\audit. Bounded NumPy environment via uv allowed; do not modify project environments.']})
save(trace / '001-completed-result-audit.meta.json', {'call_number': 1, 'purpose': 'completed-result-audit',
    'timestamp': stamp, 'agent_id': '/root/m85_completed_integrity', 'model': 'gpt-6-astra', 'reasoning_effort': 'max',
    'reviewer_family': 'openai', 'review_independence': 'same-family', 'acceptance_status': 'provisional',
    'duration_ms': None, 'duration_note': 'Exact call start not exposed; no duration fabricated.', 'status': 'ok',
    'full_response_sha256': hashlib.sha256(final.encode()).hexdigest(), 'audited_input_hashes': data['audited_input_hashes']})
events = A / '.aris' / 'meta' / 'events.jsonl'
events.parent.mkdir(parents=True, exist_ok=True)
with events.open('a', encoding='utf-8') as f:
    f.write(json.dumps({'event':'review_trace','skill':'experiment-audit','purpose':'completed-result-audit',
        'agent_id':'/root/m85_completed_integrity','trace_path':str(trace),'status':'ok','timestamp':stamp})+'\n')
print(json.dumps({'status':'AUDIT_AND_FULL_TRACE_WRITTEN','report':str(A/'EXPERIMENT_AUDIT.md'),
    'structured_report':str(A/'EXPERIMENT_AUDIT.json'),'trace':str(trace)}, indent=2))
