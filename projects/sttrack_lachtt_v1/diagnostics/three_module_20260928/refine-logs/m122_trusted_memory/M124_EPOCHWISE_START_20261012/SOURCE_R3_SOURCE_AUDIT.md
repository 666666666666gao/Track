# M124 source audit — R3

**SOURCE_ONLY_PASS — 0 blockers.** Requested reviewer: `gpt-6-astra`, reasoning `max`; actual model/effort/backend **UNATTESTED**. Same-family, provisional continuation of R1/R2.

The remaining R2 blocker is closed. In `migrate_m123_to_m124_epochwise.py`, canonical save is at line 88, CPU reload at 89, complete A_B/C and Adam roundtrip checks at 90–97, and the sole SIGTERM at 100. The post-stop evidence flag is added at 114 after copying and SHA checking original files. No additional source correction is requested.

The new staging preparer binds 34 exact archived old code files plus 10 reviewed current files. Its static receiver has no process/GPU/NN action; migration is a separate command. The original next-NN timestamp and 3600-second interval are preserved. This conclusion permits the exact sealed source set to be used for subsequent preparation; it is not evidence that any remote staging, migration, training or evaluation has run.

## Checks

- **R3-C1 Pre-stop canonical recovery and complete model/Adam roundtrip: PASS_SOURCE.** Stable checkpoint bytes are read once and CPU-deserialized at 52. Only the committed sequence-log prefix is parsed at 55-57 and checked at 59-65. Raw checkpoint/log/prefix are retained at 66-71. Canonical resume.pt is constructed, saved at 88, CPU-reloaded at 89, and every A_B/C tensor plus optimizer param_groups and each Adam state value is compared at 90-97. All of this precedes the sole SIGTERM at 100. Failures in this recovery preparation/validation do not reach SIGTERM. Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/migrate_m123_to_m124_epochwise.py:46-100`.

- **R3-C2 Correct original process identity, post-stop evidence and single launch: PASS_SOURCE.** The requested trainer PID 496828 matches the supplied original launch record. Before signalling, /proc cmdline must contain the original trainer path and --output original/fit. There is one os.kill(SIGTERM), followed by required trainer -15, nonzero supervisor exit and PID disappearance. Original files are copied and SHA checked at 108-113; only then is post_stop_checkpoint_retained_not_used_for_resume added to launch provenance at 114. Post-stop latest is never loaded for recovery. There is one supervisor Popen at 132, exclusive controller.log creation and pre-existing handoff/actual_launch refusal; no automatic retry/fallback. Actual live identity/exit remains unobserved. Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/migrate_m123_to_m124_epochwise.py:39-44,100-140`.

- **R3-C3 Hourly observation restriction and separate static staging: PASS_SOURCE.** The request preserves 2026-10-11T15:34:49.891541+00:00, exactly 3600 seconds after the original captured hourly observation. Migration checks this deadline before process/progress/torch/GPU operations. Static receiver imports stdlib only and performs source/input hashes, extraction and a staging receipt; it has no process/GPU/NN action. Migration is emitted as a separate SSH wrapper; its single GPU query occurs after stopping the old process. The short bounded exit-file wait is termination verification, not NN metric polling. No new monitoring schedule or actual observation is claimed. Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/prepare_M124_epochwise_deployment_20261011.py:32-35,44-73; C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/migrate_m123_to_m124_epochwise.py:28,123-137`.

- **R3-C4 34 archived old sources plus 10 reviewed new sources: PASS_SOURCE_AND_BYTES.** All 34 code members are flat, unique, byte-identical to old-gate source pins and the previously reviewed local source files. The ten protocol-listed new filenames are distinct, exactly match current R3 intake SHA values and produce 44 unique new-root paths. All 44 sources parse and compile with the Python 3.8 grammar without importing/executing their modules. The immutable historical protocol's old migration SHA is not used as a current source pin; exact revised bytes come from R3 intake and the freshly built request source map. Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/prepare_M124_epochwise_deployment_20261011.py:13-31; C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epochwise_source_audit_R3_20261011/archive_comparison.json; C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epochwise_source_audit_R3_20261011/deployment_binding_validation.json`.

- **R3-C5 Staging and dispatch source binding: PASS_SOURCE.** The preparer requires SOURCE_ONLY_PASS/zero blockers and rechecks every intake hash. Package payload hashes form all 44 request pins, and the actual final report SHA is bound into the request. Receiver checks package SHA, a fresh root, all new source pins, old source/input pins and the exact report SHA/verdict. Migration rechecks request/old source/input/report hashes, and the controller rechecks the combined source and input maps. The package and actual deployed paths are not instantiated or observed in this audit. AUDIT_SEAL is read/copied as evidence; the admission code uses report verdict, intake hashes and explicit source/report SHA binding rather than independently validating every seal entry. Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/prepare_M124_epochwise_deployment_20261011.py:11-15,31-63; C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/migrate_m123_to_m124_epochwise.py:31-38,115-130; C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m124_epochwise_suite.py:19-23`.

- **R3-C6 Exact SSH route and generated wrapper behavior: PASS_SOURCE.** SCP/SSH flags match the captured M123 native wrappers: connect.nmb2.seetacloud.com:43811 through SOCKS5 127.0.0.1:7898 using E:/python.exe and C:/Users/gb/.ssh/proxy_socks5.py, StrictHostKeyChecking=yes, ConnectionAttempts=1 and password authentication. Proxy helper source is captured/hashed. Local preparer itself does not invoke network; emitted static and migration wrappers each contain one subprocess.run. Static uses CUDA_VISIBLE_DEVICES='' and migration uses 0,1; both set PYTHONOPTIMIZE=0 and -B. Syntax reconstruction used a labelled zero digest placeholder because no actual package/report was built; no connection/authentication result is claimed. Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/prepare_M124_epochwise_deployment_20261011.py:64-73; C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epochwise_source_audit_R3_20261011/deployment_binding_validation.json`.

- **R3-C7 Training, generator, selection and evaluation protocol preserved: PASS_HASH_INHERITED_SOURCE_REVIEW.** Only migration changed among R2's 68 tracked files; the staging preparer is newly added. The actual trainer, generator fixed in R2, A+B+C model/loss, Train152 seed2027/frozen STTrack+CLIP, per-complete-epoch Test50/CDTB80, strict two-consecutive dual decline, equal-mean earliest-tie best and selected-best full VOT remain byte-identical. R2's 6/6 generator byte-parity and committed-prefix fixtures therefore remain applicable. Legacy seed reset/repeated partial sequence/no bitwise-continuation claim, subsequent RNG/Adam/model/counter checkpoint storage, two-device reconstruction and no GT before current state action retain the bounded R1 findings. Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epochwise_source_audit_R3_20261011/R2_comparison.json`.

- **R3-C8 Historical preservation and input identity: PASS_HASH.** 33 R3 intake files match declared hashes. 78 total current input/dependency/context files are snapshotted and checked before/after. R1's 159 artifacts and R2's 105 artifacts, plus both original seals, remain byte-identical. New dependencies added for this round are the archived deployment zip, SOCKS5 helper and two historical native route wrappers. Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epochwise_source_audit_R3_20261011/hashes_before.json; C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epochwise_source_audit_R3_20261011/hash_afterchecks.json`.

- **R3-C9 A-F integrity boundaries: SOURCE_ONLY_INHERITED.** A/B/F source-positive findings and C/D/E evidence/scope warnings carry forward: real dataset GT is intended, metric normalization is not self-maximum manipulation, paths/counts/model identity are source-bound, historical helper mains outside dispatch are not new training/evaluation, and no actual M124 outcome exists in this audit. External Test50/CDTB are explicitly used for stopping/checkpoint selection. No untouched-test, multi-seed or nine-metric target-achievement claim is made. Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epochwise_source_audit_20261011/SOURCE_AUDIT.json`.

## Preserved source identity

Only the migration helper changed among R2's 68 tracked files. The deployment preparer is newly added. All training, loss, generator, evaluation, early-stop and best-selection sources remain byte-identical to R2.

- Migration SHA256: `9e796964beab60632551e89d655720cdbf03a6765c30be95d9323ed659e39800`.
- Deployment preparer SHA256: `d5c7fe1fe20e6828e8947844e62c2359541bd315fe8d47a53b4710ae4c7492ad`.
- 33 intake files and 78 total input/dependency/context snapshots receive final before/after SHA checks.
- R1: 159 sealed artifacts plus seal unchanged. R2: 105 sealed artifacts plus seal unchanged.

## Evidence limits

1. R3 is a bounded source-only followup for the remaining pre-stop ordering fix and newly added staging preparer. It is not a new experiment result audit, runtime admission or deployment completion.

2. Requested reviewer is gpt-6-astra with max reasoning; actual model, effort and backend remain UNATTESTED. This is a same-family, provisional continuation with no executor substitution.

3. No SSH/network connection, GPU query, NN forward, training-progress query, remote process query/signal, actual checkpoint deserialization or target package import was performed. Only local source/JSON/zip bytes, AST compilation and pure generator/string expressions were used.

4. The local preparer, static receiver and migration command were not run. No real R3 deployment zip, migration request, remote staged root or launched controller was produced by the reviewer. Reconstruction uses an explicitly labelled zero SHA placeholder solely for syntax/source validation.

5. Source order demonstrates that recovery preparation and complete model/Adam roundtrip checks precede SIGTERM. It does not demonstrate actual checkpoint content validity, model loading, optimizer reconstruction, CUDA behavior or a successful handoff on the remote machine.

6. The PID/command and hourly deadline are checked against historical local records and future source guards. Current process identity, elapsed runtime, live checkpoint stability and actual next observation timing are not observed.

7. R3 retains the explicit legacy RNG/full frozen-digest absence, seed2027 reset and partial-sequence repetition. It cannot retroactively provide bitwise continuation.

8. Installed Torch/CLIP/VOT/CUDA packages, native binaries and actual import resolution remain unattested. Exact 44-file bundle byte identity does not establish the full deployed runtime environment.

9. The preparer loads and copies AUDIT_SEAL but does not independently revalidate all seal entries. Admission is instead guarded by report verdict, current intake hashes and explicit package/source/report SHA checks. This audit's own finalizer verifies all report/snapshot/trace artifacts and all input after-hashes before sealing.

10. Unchanged training/evaluation/import closure inherits R1/R2 review by exact SHA. Historical helper mains and other research modules are not newly redesigned or re-audited.

11. Native exec_command returns combined stdout/stderr; complete returned outputs are preserved and separate stderr marked unavailable. One combined display exceeded the functions output budget; stored native outputs were subsequently re-read by section, with no source-read omission. The final seal-digest line/tool-return envelope stays in the conversation/parent trace to avoid a recursive seal.

12. All inherited R1 evidence limits below continue to apply, with original intake counts and tooling notes explicitly referring to R1. R1 private traces are preserved in place and must not be published.

## Inherited R1 evidence limits

The following are retained verbatim as historical R1 limits; R1-specific counts and tooling notes refer to that audit.

1. Source-only audit: no SSH/network access, GPU query, training/NN forward, checkpoint deserialization, dataset traversal, remote process query/signal, or source/intake mutation was performed.

2. No M124 migration request, deployed new code-root, live PID state, real checkpoint bytes or actual M124 epoch/OPE/VOT artifacts were supplied or executed. Source checks are not runtime admission and not proof of performance.

3. Legacy M123 checkpoints omit RNG and full frozen-model digest. Seed reset plus repeating the partial sequence is explicit; neither this migration nor later RNG storage retrospectively makes it bitwise continuous.

4. Exactly 26 intake files were fully read; local import closure and relevant dynamically dispatched metric/native/bridge dependencies were followed. Whole installed Torch/CLIP/VOT/CUDA packages and every transitive native backbone helper were not attested. Historical helper main bodies outside the dispatched path were parsed/hashed, not exhaustively re-audited.

5. Five initially discovered local overlay/config candidates have hashes different from the pinned remote runtime or no applicable pin. They remain recorded and snapshotted as non-authoritative candidates; matching official/native source snapshots were separately found and read. They are not substituted for pinned runtime code.

6. The pinned OPE implementation is a local VOT-style PR implementation. This review reads its normalization/GT code, but does not establish upstream official-toolkit byte equivalence or recompute current scores from raw 130-sequence predictions/GT.

7. CurrentC captured warm metrics only establish literal agreement with the requested initial reference. Historical CurrentC results are not M124 results and were not freshly numerically recomputed.

8. Test50/CDTB80 are explicitly used to select epoch and stopping. No untouched-test, multiple-seed robustness or nine-metric target-achievement claim is supported.

9. Human-reviewed initialization text uses multiframe aids. It must not be relabeled as blind first-frame-only human review.

10. The exact actual reviewer model/reasoning/backend is UNATTESTED by native receipts available to this child. Requested route is gpt-6-astra/max; review remains same-family and provisional.

11. Native exec_command exposes combined stdout/stderr. Traces preserve it verbatim and explicitly mark separate stderr unavailable. Initial mandatory session-context read exceeded output budget and is preserved as returned, including truncation. Subsequent source reads and deterministic source checks were bounded; transcript contains two audit tooling errors corrected by explicit installed Python and UTF-8, without target code execution.

12. Trace files contain the mandatory initial local user/session context read and must remain private/project-local, not published. Reports contain only task-relevant evidence.

13. Future source seal validation must use the actual dispatched paths and review report identity. Looping over a source map does not independently attest import resolution or an absent deployment package.

14. Per-epoch parameter-change reporting compares final state with warm initialization, not the immediate prior epoch; it supports cumulative change only.
