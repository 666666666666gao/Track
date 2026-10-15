# M125 deployment and future-observer source follow-up

**Source verdict: FAIL_SOURCE_ONLY. Overall verdict: FAIL. Concrete blockers: 2.**

This is a separate read-only follow-up of the draft deployment operator and one-shot observer. The original regional scientific-source audit remains unchanged and retains its original `PASS_SOURCE_ONLY`/`WARN` verdict. No actual deployment, real64, training, tracking, resource query, or remote observation was performed. The local fixtures below establish source behavior only.

Reviewer: `/root/audit_m125_region_fixed_source`, continued native Codex reviewer. Requested policy route: `gpt-6-astra`, effort `max`; actual backend model/effort remain **UNATTESTED** because the runtime does not expose reliable attestation. `review_independence: same-family`; `acceptance_status: provisional`.

## Concrete blockers

### B1 — Authentic source-audit hashes are rejected by the deployment gate

`M125_REGION_DEPLOYMENT_SOURCE_20261015/deploy_m125_region_fixed.py:39` compares `audited[original]` directly to a bare hexadecimal manifest digest. The immutable source audit stores the documented `sha256:<digest>` representation: `M125_REGION_SOURCE_AUDIT_20261015/EXPERIMENT_AUDIT.json:380`, with the first manifest source entry at `:443`. `M125_REGION_SOURCE_20261015/INPUT_MANIFEST.json:2` contains the corresponding bare digest.

Thus an unmodified, authentic report and unmodified source inputs fail on the first entry. All three arm calls to the exact `verify_inputs` function were reproduced in an isolated local copy and raised `AssertionError` at the audit comparison. This happens before predecessor admission, GPU/resource reads, copying, or any launch (`deploy_m125_region_fixed.py:176`–`:183`); the deployment path cannot proceed with its intended inputs.

**Required correction:** compare against the actual audit schema, e.g. `'sha256:' + digest`. Preserve the original audit. A fallback or rewriting the original audit is unnecessary.

### B2 — The one-shot observer does not claim its attempt before transport

`observe_M125_R0_training_end_once_20261017.py:15` only checks whether output/error files already exist. The SSH transport runs at `:63`–`:64`; the first exclusive attempt-file creation happens afterwards at `:65`. Therefore a second invocation beginning while the first transport is in progress sees both files absent and starts a second remote read. An interruption during the first transport similarly leaves no local attempt marker.

The isolated overlap fixture executes the observer statements with the single `subprocess.run` call replaced by a local stub and credentials replaced by a dummy map. While the first stub is active, it invokes a second copy of the same observer. Both pass the file guard, giving **two stub transport calls**. The inner call completes; when the first returns it raises `FileExistsError` at the late exclusive error-file open. Actual SSH/subprocess/GPU/NN calls in this fixture: **0**. This is a direct violation of the requested one-attempt/no-duplicate-read behavior, not a suggested general retry framework.

**Required correction:** reserve the attempt atomically before the transport. The existing error file can be opened with exclusive creation before `subprocess.run`, kept open, and then filled with stderr. Preserve the attempt even if transport fails; add no automatic retry.

## Other requested source checks

These checks identify no additional blocker; they do not turn the draft into executed evidence.

| Topic | Source result |
|---|---|
| Candidate identity and future source staging | Apart from B1, the operator requires the arm-specific candidate, exact root/base root, fixed6/final6, seed2027 and no external checkpoint selection (`deploy_m125_region_fixed.py:40`–`:53`). `verify_sources` requires 11 new and 64 total future siblings and checks both copied source and all declared inherited/input hashes (`:95`–`:109`, `:187`–`:195`). All 192 prospective hashes (64 per arm) match the immutable local files. |
| Import roots and sibling files | Launch sets new `code_root` before `base_code_root`, disables bytecode and optimized Python, and uses the code root as cwd (`deploy_m125_region_fixed.py:133`–`:156`). The mappings include the inherited tracker sibling used in the trainer's source receipt and the inherited VOT shard script invoked by path (`M125_REGION_SOURCE_20261015/train_m125_region_fixed_pass.py:245`; `run_m125_region_fixed_suite.py:86`). Actual deployed resolution is still untested. |
| Entire original R0 completion admission | The operator requires both actual controller exits, supervisor terminal exit0, a completed R0 suite result with 15 successful stages, all six pass exits/counts, fixed final6, and three dataset result files bound by SHA to the selected final (`deploy_m125_region_fixed.py:57`–`:92`). The immutable R0 runner really produces six training stages plus nine preparation/tracking/analysis/collection stages, then writes its result and controller exit (`M125_R0_FIXED_SOURCE_20261015/run_m125_R0_fixed_suite.py:50`–`:101`). The supervisor terminal records `subprocess.run`'s returned exit (`m125_R0_fixed_process_supervisor.py:16`–`:19`). The scheduled **training-end** observation is not treated as proof that the later VOT/entire suite has completed. Actual R0 terminal state was not read. |
| Resource query scope/order | Predecessor and source/input checks precede resource admission (`deploy_m125_region_fixed.py:178`–`:183`). Full phase checks real64 before resource admission (`:203`–`:217`). The one GPU command requests only index and memory used/total, checks exactly indices `[0,1]`, and requires memory below500 MiB; disk space is checked on the predecessor filesystem (`:112`–`:119`). It requests no temperature, power or utilization fields. No resource query was executed by this review. |
| Actual current-arm real64 before full startup | Full phase requires the existing exact gate/source-review/candidate copies, no prior full attempt, exit0 and real64 receipt0; it checks result status, arm, gate hash, checker-source hash, 64 calls/two updates, warm/frozen/serialization evidence and final-batch positive finite regional group norms (`deploy_m125_region_fixed.py:199`–`:216`). These fields match the immutable witness producer (`M125_REGION_SOURCE_20261015/check_m125_region_real64.py:102`–`:111`). Actual witness output remains absent from this review. |
| Fresh common warm versus sanity state | The full command starts the existing fixed-six-pass supervisor (`deploy_m125_region_fixed.py:218`), whose first pass supplies no resume (`M125_REGION_SOURCE_20261015/run_m125_region_fixed_suite.py:55`–`:60`). The initializer loads the common warm and constructs a fresh optimizer/RNG state (`train_m125_region_fixed_pass.py:49`–`:88`). The sanity checkpoint is not passed into formal training. |
| Deployment one-attempt behavior | Real64 requires a nonexistent root and exclusive creation of copied/driver/log files; full phase checks no prior launch/passes and opens `controller.log` exclusively **before** `Popen` (`deploy_m125_region_fixed.py:146`, `:154`–`:156`, `:182`–`:194`, `:202`). There is no automatic retry loop. This prelaunch exclusivity is present in the deployment path; B2 is in the separate observer. |
| Observer deadline/timezone and bounded reads | Both the local program and generated remote program enforce the timezone-aware `not_before` (`observe_M125_R0_training_end_once_20261017.py:16`, `:21`). The node's `2026-10-17T12:45:00+08:00` is `2026-10-17T04:45:00Z` / `2026-10-16T22:45:00-06:00`. An isolated early-call fixture rejected before its stub transport. The observer has one transport statement, reads only declared original-root progress/artifacts and the recorded supervisor PID, and creates no timer or NN work. Its inherited transport arguments specify `ConnectionAttempts=1` (`M124_fiftyfirst_hourly_compressed_native_observer_20261014.py:3`). B2 prevents claiming that this is currently a reliably one-shot observer. |

## A–F classification and evidence boundary

- **A, GT provenance:** this draft introduces no training target or inference GT path. It admits the already reviewed scientific source and preserves the source/arm/warm identities; the original GT/action-separation findings are not reclassified as actual execution.
- **B, normalization:** no new metric normalization is introduced. Predecessor metrics are read as completion/provenance evidence, not optimized or rescaled.
- **C, existence/provenance:** the source report, manifests and all local source files exist and match; B1 is a concrete consumer/schema failure. No deployment or real64 result file is claimed to exist remotely.
- **D, live paths:** the deployment source comparison is live in both phases and blocks them; the observer's late attempt-file guard is live and permits overlapping reads. Neither is dead-code speculation.
- **E, scope:** source inspection plus explicitly synthetic admission fixtures only. Actual exit values, currently running processes, GPU memory, deployed imports, actual real64, six-pass completion and performance were not observed.
- **F, type:** `source_only_admission_review`; local fixtures are `synthetic_control_flow_only`, not `real_gt`, NN validation, or scientific result evidence.

The original audit's raw hashes, current source/configs, and the immutable R0 producers were used directly for field/path verification. The historical transport helper was read only to resolve its literal argument list; its embedded old command was neither executed nor reused as an observation result. No SSH, real remote metadata/progress/process/GPU reads, NN, package installation, input edit, or new agent was used.

## Receipts

The original seal contains89 files, including all13 files in the completed source-audit directory. One derived transport-argument source was additionally hashed before its bounded read, giving **90 audited inputs**. The final before/after receipt shows **90/90 unchanged**, including all13 files in the complete original audit directory; 0 changed and 0 missing.

The reviewer-owned `check_draft.py` ran under `python -B`, uses stdlib only, and writes exclusively under this new output directory. It has **220 successful verification assertions**, including assertions that reproduce the two failures; this does **not** mean the draft passed. The fixture record explicitly reports zero real SSH/subprocess/GPU/NN calls. Original code inputs were never patched. Synthetic copied files and outputs are clearly confined to `synthetic_fixtures/` and are not admissible as real64, deployment or benchmark evidence.

Full request/response and metadata are preserved under `.aris/traces/experiment-audit/2026-10-15_run01/`, with separate original/additional before seals, after hashes and fixture evidence. The machine report lists both blockers and separates the source verdict from `actual_execution_verdict: NOT_EXECUTED` and `performance_verdict: NOT_ASSESSED`.
