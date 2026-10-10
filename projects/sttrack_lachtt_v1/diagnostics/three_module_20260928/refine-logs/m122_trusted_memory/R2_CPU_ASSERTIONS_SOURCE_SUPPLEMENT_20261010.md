# Assertions launch configuration supplement

Date:2026-10-10. Reviewer:`/root/m122_r2_rawcheck_source_integrity_20261010`. Requested Codex `gpt-6-astra` / reasoning `max`; actual backend/model/effort **UNATTESTED**. Same original fresh reviewer in bounded continuation, **same-family / provisional**; no fresh child agent.

**New invocation source verdict: PASS,0 blocking findings. Overall: WARN because target execution, terminal teacher artifacts and a terminal semantic certificate remain absent.** N3's *outer bootstrap environment* component is closed at configuration level for the exact new wrapper. Its target Python3.8/Torch1.13.1 compatibility/runtime component remains unproven. The original report remains unchanged and conditional for its original wrapper.

## Exact change verified

The new wrapper's line3 changes only the final SSH argument by prepending:

```text
PYTHONOPTIMIZE=0 CUDA_VISIBLE_DEVICES='' 
```

The remaining command is byte-for-byte the original `/root/autodl-tmp/envs/sttrack/bin/python -B -c ...` command. All other SSH args, host, port, proxy, authentication and timeout settings are identical. The only other source change is line4's unique stdout filename:

`R2_terminal_CPU_deployment_assertions_original_stdout_20261010.txt`

The stdout file is opened with`xb`; it did not exist at this local check. Subprocess invocation at5 and native exit propagation at6 are unchanged. There is no extra command, retry, shell invocation by local Python or status rewriting. The original stdout and wrapper remain intact. `EXACT_WRAPPER_DIFF.patch` preserves the full raw diff; after replacing only those two literal values, both ASTs are identical.

Evidence: [R2_terminal_CPU_deployment_native_ssh_assertions_20261010.py:3](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/R2_terminal_CPU_deployment_native_ssh_assertions_20261010.py:3), [R2_terminal_CPU_deployment_native_ssh_assertions_20261010.py:4](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/R2_terminal_CPU_deployment_native_ssh_assertions_20261010.py:4); [prepare_R2_CPU_assertions_launch_20261010.py:7](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/prepare_R2_CPU_assertions_launch_20261010.py:7)–21. The preparation script parses the old literal args, adds the prefix, writes a separate wrapper and records an unexecuted receipt. It does not invoke SSH or the bootstrap. Its `Path.write_text(newline=...)` is local preparation code, not a target Python3.8 API claim.

## Environment conclusion

POSIX shell tokenization of the exact new remote command yields the two leading assignment words `PYTHONOPTIMIZE=0` and `CUDA_VISIBLE_DEVICES=` followed by the original executable,`-B`,`-c` and compressed body. The assignments therefore apply to the **outer Python interpreter before startup**, overriding inherited values for this invocation. No`-O`,`-OO`,`-E` or`-I` appears in that bootstrap argv. The compiled bootstrap inherits optimization level0, so its controller/stage/hash assertions are active; empty CUDA visibility also starts at outer launch. The existing child environment at bootstrap43 independently sets the same two values.

This removes the old source-level dependence on an unspecified inherited outer `PYTHONOPTIMIZE` **for this new wrapper**. It does not prove that the command was executed, that the target executable/version exists or imports correctly, that CUDA remained uninitialized in an actual process, or that runtime checks passed. No shell command or target interpreter was run in this review.

Evidence: [R2_terminal_CPU_deployment_native_ssh_assertions_20261010.py:3](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/R2_terminal_CPU_deployment_native_ssh_assertions_20261010.py:3); [CPU_BOOTSTRAP.py:5](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/R2_terminal_CPU_deployment_prepared_20261010/CPU_BOOTSTRAP.py:5), [CPU_BOOTSTRAP.py:43](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/R2_terminal_CPU_deployment_prepared_20261010/CPU_BOOTSTRAP.py:43). The exact compressed body, interpreter and child argv are unchanged.

## Byte and archive closure

I read and snapshotted all8 requested files plus11 provenance inputs.6 Python input files pass Python3.8 grammar under the explicit local UV CPython3.13 interpreter; no target imports were tested. AST/literal parsing, shell tokenization, base64/zlib decoding and hashes show that the new remote command decodes to the same5293 bootstrap bytes, including the prior audit's decoded snapshot.

| Object | SHA-256 |
|---|---|
| New wrapper | `5864188a02ae4202d65ee459f4d6f1010f6da66cb8c4fe898cba6814ac8d9741` |
| Original wrapper | `aefc1bd6949146c8cfbb8b7114ae059d8e87aac87bed89faf432b24d5fa28148` |
| Decoded/original bootstrap | `bc0eaec10dea9dfbc188cd2d1c41ec2b8fee1c372ed8c7b2c17fb8336f33d95e` |
| Original deployment ZIP | `6b99aae715a65bf06d2353ed60f88cf7de175bdbad8700407283ea439eb1e923` |
| Current verifier v2 | `fc5789875562bdb5efe8427c95ae40a2223828f02505cd8474b40f92fdebbe82` |
| Original conditional deployment report | `cb7f9d63d100ae57d2122a8f1a5b1fa13281d9ff2fdf4f75793e0049cf138658` |
| Original deployment seal | `2fb01ddca7a00cb494e94f0553be93ca437f0567de274753e1e6a526014fbc7f` |

The deployment ZIP still has5 unique entries, passes CRC checks and its4 payloads match their original bytes and `files.json`. This includes v1 FAIL/2 blockers and v2 source PASS/0 blockers; neither report is rewritten. All68 prior deployment-sealed files, all45 original inputs, all47 v1-sealed files, all43 v2-sealed files and all48 members of the original v1 archive remain unchanged. This supplement is a new subdirectory and does not reseal or overwrite the old audit.

## A–F consequences

| Check | Status | Bounded finding |
|---|---|---|
| A — GT provenance | PASS, source | Unchanged pinned verifier still reads actual dataset GT and fixed human bank/labels, including the sole toy07 row-alignment exception. No GT path, tensor or label changed. |
| B — score normalization | PASS, source | No metric code changed: geometric-union IoU, signed valid-frame future mean, negative/zero/unknown preservation and shared eligibility remain. |
| C — result existence | WARN | Configuration/ZIP/source and sealed reviews exist; actual terminal teacher and CPU readback evidence remain absent. |
| D — called source | PASS | Only outer environment assignments and unique stdout literal differ. Bootstrap and child CLI, all terminal gates, digests, B1/B2 guards, failure-log/exit/execution ordering and write-once roots are unchanged. |
| E — scope | WARN | Full152/219802/seed2027 and C fit120/dev32 remain intended Train scope. No C training, formal metrics or goal completion is certified. |
| F — type | PASS | This is launch/source verification; prospective dataset labels remain real_gt and trajectory equality remains a consistency proxy. No new performance evaluation occurred. |

Detailed immutable evidence remains in the [original deployment review](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/R2_terminal_CPU_deployment_source_audit_20261010/DEPLOYMENT_AUDIT.md) and its source-v2 references. In particular, CPU_BOOTSTRAP7–14 checks controller/all4 exits and full aggregate/gate before line46 subprocess;45–54 saves the returned child's combined log, exact exit and execution record before rejecting nonzero;55–68 binds actual result/aggregate hashes;18–20 and verifier68/326 preserve new output roots. The new prefix enables outer assertions without changing any of these statements.

## Remaining limits and required evidence

The original N1/N2/N4/N5 boundaries remain: terminal artifacts/runtime are unproven; saved-state readback cannot reconstruct model features or changed-C-policy labels; failure receipts cover returned subprocesses and require native SSH stderr/exit retention for prelaunch failures; downstream experimental claims remain unsupported. N3 is **partly resolved**: outer launch environment is now explicit, target compatibility/import/runtime remains absent. No further source correction is necessary for this supplement's scope.

The original SCP wrapper/package is unchanged. A future actual execution still requires genuine terminal teacher evidence and authorized launch, preserving original stdout/stderr/exit, CPU log/execution/result and actual raw digests. A fresh terminal semantic review must assess those artifacts. No `R2_terminal_raw_audit_complete=True`, runtime PASS or terminal certificate is issued here.

Actual review actions: local stdlib parsing/hashing/ZIP inspection only. SSH/SCP, GPU/NN-progress queries, Torch/NumPy/experiment imports, model construction/forward/optimization/training/inference, installations, source/input mutations and new agents all0. Original timer PID30632/native21826 was not inspected or modified; no NN observation was made. The17:52:45.946171 BJT lower bound remains intact.

Full request, verbatim response and metadata are in`private_trace/`. Input snapshots/hashes, exact diff, decoded bootstrap and deterministic checks are sealed separately by this supplement's`AUDIT_SEAL.json`. Actual backend attribution remains UNATTESTED and acceptance same-family/provisional.
