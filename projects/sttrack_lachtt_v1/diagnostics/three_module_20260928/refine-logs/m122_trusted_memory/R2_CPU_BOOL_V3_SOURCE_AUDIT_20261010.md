# Bounded v3 boolean serialization source and launch audit

Date:2026-10-10. Reviewer:`/root/m122_r2_rawcheck_source_integrity_20261010`. Requested Codex `gpt-6-astra` / reasoning `max`; actual backend/model/effort **UNATTESTED**. This is the original same-family reviewer continuing source work, **provisional**, not a new independent terminal runtime review.

**Source and exact prepared launch: PASS,0 blocking findings. Overall integrity: WARN.** The actual v2 serialization failure is verified and reproduced in isolation; the six-byte v3 change closes that source defect. No v3 target execution outcome or terminal raw-audit certificate is issued. The executor's message about deployment activity is not treated as runtime evidence.

## Actual failure, preserved without revision

The supplied native join records original session45634, initial chunk70ef51, authentication3ee502 and terminal chunk03c2e3 with exit1 and an assertion at original CPU bootstrap54. The actual CPU evidence contains `CPU_readback.exit`=`1`, an execution record with exit1 and the bound v2 source SHA, and a1457-byte log showing v2 main327 failed during `json.dumps(..., allow_nan=False)`:

```text
TypeError: Object of type bool_ is not JSON serializable
```

The original child ran from2026-10-10T10:12:31.255046Z to10:12:57.937919Z. Its recorded argv equals the prior reviewed full152 CPU argv; `PYTHONOPTIMIZE=0`, empty CUDA visibility and original teacher aggregate SHA are recorded. The remote wrapper then rejected the returned child at line54 after saving log, exit and execution. These are actual failed execution artifacts, not a simulated failed experiment.

The failure-readback stdout embeds an execution object and log exactly equal to the extracted files. It binds the original1487690-byte failure ZIP SHA `79f99b3e4547655ef0c2277cdf80c3133009c3e83c705f4312069ba5421a2372`. The ZIP has39 unique members:38 payloads plus `files.json`; CRC checks and every size/hash/extracted-byte comparison pass. All38 payloads are snapshotted in this audit. The archived v2 verifier equals the original local v2 byte-for-byte.

Evidence: [CPU_readback.log:1](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/R2_CPU_failure_evidence_actual_20261010/CPU/CPU_readback.log:1), [execution.json:1](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/R2_CPU_failure_evidence_actual_20261010/CPU/execution.json:1), [R2_terminal_CPU_actual_join_01_20261010.json:1](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/R2_terminal_CPU_actual_join_01_20261010.json:1), [R2_CPU_failure_readback_original_stdout_20261010.txt:1](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/R2_CPU_failure_readback_original_stdout_20261010.txt:1). The traceback reaches final serialization after the preceding readback code; that location is evidence about this failure, not a replacement for a durable successful raw-readback report or fresh semantic verdict.

**Earlier review limitation:** the previous source audit did not exercise actual NumPy-scalar JSON serialization and therefore missed this defect. Its original source PASS remains preserved as a historical source judgment. The observed v2 CPU outcome is **FAIL/exit1**, never retroactively a runtime PASS. The original v1 FAIL and its two blockers, v2 PASS and B1/B2 fixes, deployment review, configuration supplement, failed checks and archives remain unchanged.

## Minimal verifier change and independent local checks

Only line44 differs between the334-line verifier files. The exact diff is:

```diff
-    return all(math.isfinite(float(x)) for x in box) and box[2] > 0 and box[3] > 0
+    return bool(all(math.isfinite(float(x)) for x in box) and box[2] > 0 and box[3] > 0)
```

AST comparison confirms the sole change is a `bool` call around the unchanged original predicate. No guard, GT row, branch, count, formula, tolerance, label filter, import, tensor path or output strictness is changed. `valid()` is actually called on NumPy GT rows at249 and265, and Python-list prediction boxes at180/259. Python's`and` expression returns an operand, so finite NumPy GT comparisons could produce `numpy.bool_`; these flowed into `current_GT_valid` at292 and, for false values, potentially `common_C_training_eligible` at293. The v3 wrapper normalizes the predicate result to the native JSON boolean domain at its origin ([verify_template_write_teacher_raw_v3.py:43](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/verify_template_write_teacher_raw_v3.py:43)–44; [verify_template_write_teacher_raw_v3.py:249](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/verify_template_write_teacher_raw_v3.py:249)–295).

I independently used the explicitly authorized local `E:/anaconda/python.exe`, observed **CPython3.7.6 / NumPy1.21.6**, and executed only AST-extracted `valid()` functions plus local scalar/JSON fixtures. **Torch was not imported** and no experiment module or main ran.30 predicate cases cover Python lists and NumPy float32/float64 arrays with valid boxes, zero/negative sizes, NaN/infinite coordinates/sizes and legal negative coordinates. All truth values match v2; all v3 returns have exact Python`bool` type.60 nested event JSON cases preserve true/false eligibility, null unknowns, negative and zero values under `allow_nan=False`.

The test retained12 expected reproductions of the original NumPy bool TypeError and3 expected nonfinite-float ValueErrors, including their messages. Strict serialization remains strict; no `default=...`, permissive conversion fallback, exception suppression or `allow_nan=True` was added. The executor's separate four-case regression artifact and script were also read and snapshotted. The independent local test returned exit0 in chunk`adb709`; this is a local test receipt, not target execution.

The Python3.8 grammar check passed for8 reviewed Python input files under the explicit UV CPython3.13 stdlib interpreter. Local CPython3.7.6 NumPy tests and grammar checks do not constitute a target Python3.8 v3 import/end-to-end witness. Original v2 execution is separately evidenced by its actual failed Python3.8 traceback.

## Exact v3 launch, teacher and source binding

| Object | SHA-256 |
|---|---|
| Preserved v2 verifier | `fc5789875562bdb5efe8427c95ae40a2223828f02505cd8474b40f92fdebbe82` |
| Prepared v3 verifier | `bf3c6402d4efebb4f82cb19cb6c18fdb86f12f9ceabece79b635759fe5a2a187` |
| Exact v3 bootstrap / decoded command | `a1843cfee5c6a16d6b1a71545b6c2f7a9e89a763ca4e71447db356956afc418c` |
| Original terminal teacher aggregate | `24c8461b4d71c573fbaa4bf4f91c4e359cca1614c7fe81246ac0504a325c836c` |
| Actual teacher gate77 | `26f555c0d1eb13ddac5469d333926780c2fc760a1e45866e51a497533874fbd7` |
| Exact SSH wrapper | `026b86269bee776c5bf0c6f25d7f7b3bf1de39d0e55f00fceb57ae2834174029` |
| Exact SCP wrapper | `b410a9c7f7fbd55d80f922ec95aba87a5968b6d1e2000c8e7dd2f1c7bef18a65` |

This launch transfers the new source file directly; the prior deployment ZIP is retained unchanged and is not repackaged. SCP's exact source is `verify_template_write_teacher_raw_v3.py` and destination is `/root/autodl-tmp/verify_template_write_teacher_raw_v3_20261010.py`. SSH uses the unchanged host/port/proxy/strict-host-key/authentication options, with leading `PYTHONOPTIMIZE=0 CUDA_VISIBLE_DEVICES=''` assignments. Shell tokenization and base64/zlib decoding prove its compile/exec body equals all3694 bytes of the supplied v3 bootstrap and the preparer's literal template after exact source-SHA substitution. Neither transport wrapper nor bootstrap was executed by this reviewer ([R2_CPU_v3_native_ssh_20261010.py:3](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/R2_CPU_v3_native_ssh_20261010.py:3)–6; [R2_CPU_v3_native_scp_20261010.py:2](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/R2_CPU_v3_native_scp_20261010.py:2)–3).

The actual v3 child argv retains `/root/autodl-tmp/envs/sttrack/bin/python -B -u` and the original teacher/spec/human-bank/human-labels/fixed-split arguments. The only child argv differences are the v3 source filename and output root `/root/autodl-tmp/m122_R2_terminal_CPU_readback_v3_20261010/readback`. The new local stdout is `R2_CPU_v3_original_stdout_20261010.txt`, opened`xb`. Existing failure evidence and its output directory are not reused.

Before subprocess21, bootstrap4–11 requires Python3.8/assertions, controller.exit0 and all4 stage exits0, the **actual original aggregate SHA**, actual gate SHA and new source SHA. Exact-byte aggregate binding includes its terminal status,152/219802 counts, inputs and0-optimization/no-C fields. The unchanged verifier independently checks these structured fields and all77 protected file lengths/hashes at67–84. The outer prefix enables bootstrap assertions before interpreter startup; the child environment at21 independently forces optimization0 and hides CUDA.

The now-supplied teacher aggregate records152 sequences,219802 native calls,1297 events and81908 future branch calls. All5 extracted exit files are0 and the4 stage receipts equal the aggregate stage objects. Both collection/replay metadata files join in original event order,76 sequence indices per shard, matching inputs and aggregate counts. The actual19261-byte gate is now supplied with the previously pinned SHA and77 unique bound paths. These observations establish the provenance used by this source review; this failure archive does **not** contain all raw tensor payloads, shared references, native prefixes and dataset GT needed for a fresh full terminal semantic audit.

Bootstrap pins the aggregate before the child and checks the resulting readback's teacher SHA afterwards. The unchanged verifier independently rehashes the aggregate at307 before emitting output. Bootstrap checks v3 source SHA and zero NN/constructor/optimizer/experiment-import counters in the returned readback result, plus CUDA-uninitialized and independent-certificate-false flags ([R2_CPU_v3_actual_bootstrap_20261010.py:4](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/R2_CPU_v3_actual_bootstrap_20261010.py:4)–33; [verify_template_write_teacher_raw_v3.py:307](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/verify_template_write_teacher_raw_v3.py:307)–325). The new bootstrap uses direct pinned-source admission rather than parsing the prior ZIP review metadata; this bounded review binds that new source and launch explicitly. Archived v2 source PASS is not used as a v3 runtime certificate.

## Failure persistence and actual called CPU path

The v3 remote root uses`mkdir()` without`exist_ok`; the child log uses`xb`; verifier output existence rejection and final`mkdir()` remain unchanged. A returned nonzero child saves its combined stdout/stderr log, exact `CPU_readback.exit` at22 and `execution.json` at27, then fails the exit assertion at28. It cannot reach the result checks, bootstrap result or success markers. The receipt's `CPU_exit=0` occurs only after this assertion. A prelaunch/spawn failure does not manufacture a completed-child receipt; native SSH stderr and exit still need to be retained. These are inspected source branches, not a claim that v3 actually failed or succeeded ([R2_CPU_v3_actual_bootstrap_20261010.py:20](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/R2_CPU_v3_actual_bootstrap_20261010.py:20)–40).

The CPU import/call graph remains stdlib plus NumPy/Torch operations on saved tensors, with no tracker/teacher/trainer/experiment imports, model construction, forward or optimizer call. The v3 change does not relax recursive CPU/no-grad/finite tensor checks; complete152 prefix ranges/order/continuity; every native-qualified opportunity; raw-reference checks including no-event sequences; human-mask/first-box joins; cached519 section checks; W/native-prefix equality and W/K history continuity; K finite/positive box and no-future-write guards at259–260; actual aggregate/shard/prefix digests; or common fit120/dev32 accounting. Strict JSON at327 remains `allow_nan=False`.

Those source checks retain their known ceiling: cached previous/reference features are not independently reconstructed by NN, full per-frame prefix features are absent, and original teacher-state utility labels do not label changed C-policy histories. The code explicitly emits `independent_terminal_audit_certified=False`; it never produces a fresh `review_call_status`, `R2_terminal_raw_audit_complete` or `audited_teacher_*` certificate ([verify_template_write_teacher_raw_v3.py:309](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/verify_template_write_teacher_raw_v3.py:309)–330).

## A–F findings

| Check | Status | Evidence and implication |
|---|---|---|
| A — ground-truth provenance | PASS, source | GT SHA/raw-used row/first-box and human-bank checks at83–105/188–214 are unchanged. The exact toy07_indoor_3201406-to1367 exception is preserved. `bool()` creates no GT or proxy target. |
| B — score normalization | PASS, source | Scalar IoU union at47–51 and signed valid-frame W-K mean at265–280 are unchanged. Negative/zero/unknown preservation and common eligibility remain; local tests verify only Python type normalization. |
| C — result existence | WARN | Original v2 failure is real and retained. Actual teacher aggregate/gate/exits are supplied. No v3 successful result or full raw semantic-audit evidence is reviewed. |
| D — called code | PASS | `valid()` is called in the real GT path; exact line44 correction resolves its JSON-type defect. Exact SSH/SCP/bootstrap/source pins and new output root verify; all existing guards remain active. |
| E — scope | WARN | Supplied teacher metadata is full Train152/seed2027; C fit120/dev32 is a C-only holdout after A+B saw all152. This source/failure review does not certify changed-policy C, final Full152 C, formal metrics or goal completion. |
| F — evaluation type | PASS | Dataset labels remain prospective real_gt Train diagnostics; trajectory equality is a consistency proxy. Local scalar/JSON cases are implementation verification, not experimental performance results. |

The score-denominator taxonomy is preserved: geometric union for IoU, valid-frame count for signed utility, sample-count reduction for C MSE, and valid-frame/equal-sequence reduction for development. Decoder probability/native and prior-bbox motion denominators remain inference/input transforms. Raw unknown-reason counters can overlap; trainer exclusions use disjoint priority reasons. No label sign or unknown-event selection rule changes.

## Preservation, trace and limits

All13 manifest inputs, manifest, root request and15 additional provenance inputs were directly read and snapshotted:30 direct snapshots total. The original failure ZIP and38 extracted payloads are retained. All47 original v1 seal entries,43 v2 entries,68 deployment entries and35 assertion-supplement entries remain exact; all45 original deployment-review inputs remain unchanged. The original v1 archive and deployment ZIP keep their pinned hashes. Existing reports are not edited to erase an error or revise historical runtime assumptions.

The stdlib source/launch checker returned exit0 in chunk`74ca8d`. The NumPy checker records all15 expected errors; there are no new unexpected auditor-check failures. Original PATH interpreter failure and prior missing optional filename errors remain preserved in prior sealed evidence. No input or experiment source was changed by this reviewer.

Actual reviewer actions: one local NumPy import,0 Torch imports,0 experiment imports,0 main calls,0 SSH/SCP/GPU/NN-progress queries,0 model construction/forward/optimizer/training/inference,0 installations and0 new agents. No timer was inspected, waited on or modified. The executor may separately perform authorized CPU work, but this review does not observe or infer that work's outcome.

No further source correction is necessary for the reviewed v3 serialization fix and exact prepared launch. A genuine v3 exit/log/execution/result and full raw teacher evidence still need a fresh terminal semantic review before any terminal certificate or downstream C/formal claim. **`R2_terminal_raw_audit_complete` remains false for this review.**

Full request/follow-up, verbatim report response and attribution metadata are under`private_trace/`; this report and all local checks/snapshots/errors are bound by the new`AUDIT_SEAL.json`.
