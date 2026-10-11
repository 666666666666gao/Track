# M124 source audit — R2

**SOURCE_ONLY_FAIL — 1 remaining blocker.** Requested reviewer: `gpt-6-astra`, effort `max`; actual model/effort/backend **UNATTESTED**. Same-family, provisional continuation of R1.

R1 remains unchanged: all 159 sealed artifacts and its seal match. All 29 R2 intake hashes match. Only the migration helper and generator changed among R1's 63 tracked files; 68 current input/dependency/metadata files are captured for R2 before/after checks.

## Remaining blocker: R2-B1

**Canonical recovery checkpoint and model/Adam roundtrip checks still occur after SIGTERM**

The stable raw pre-stop snapshot is now preserved correctly, and the live post-stop latest.pt is no longer used for recovery. However, source order is torch.load(raw bytes) at 52, os.kill(SIGTERM) at 73, torch.save(canonical) at 103, reload at 104, and A_B/C/Adam roundtrip assertions at 105-112. The canonical checkpoint's first A_B/C digest and optimizer structure accesses are also below the signal. Thus failures in preparing or checking the actual handoff resume.pt can still occur after stopping the original trainer. This leaves the explicit R2 requirement to complete recovery preparation/validation before SIGTERM unmet. No actual checkpoint corruption, missing key or failed write is claimed.

Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/migrate_m123_to_m124_epochwise.py:73` and `:87-112`.

Minimum fix: Move the existing provenance/canonical resume.pt construction, save, reload and complete A_B/C/Adam roundtrip validation above SIGTERM; they depend only on the already captured pre-stop bytes and committed log prefix. Preserve post-stop copy/exit verification below SIGTERM, and only record the factual post-stop-copy provenance flag after those copies succeed. Do not add retry/fallback or alter old trainer/model/loss/evaluation sources.

## Fix and regression checks

- **R2-C1 Stable raw checkpoint read and sealed raw snapshot: PASS_SOURCE.** One read is surrounded by inode/size/mtime_ns checks, byte count matches, CPU torch.load consumes BytesIO(checkpoint_bytes), and the retained raw file SHA matches the captured bytes before SIGTERM. Concurrent-read/deserialization failures occur before the signal; no retry/fallback is added. Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/migrate_m123_to_m124_epochwise.py:46-71`.

- **R2-C2 Committed sequence-log prefix only: PASS_SOURCE.** Raw log bytes are retained, but parsing slices to saved.completed_sequences first. Sequence identity, ordering, calls and optimizer counters are checked before SIGTERM. Three pure JSON source-fragment fixtures confirm that an incomplete uncommitted suffix is ignored while missing/malformed committed rows fail. Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/migrate_m123_to_m124_epochwise.py:54-70`.

- **R2-C3 Post-stop latest is historical evidence only: PASS_SOURCE.** Post-stop latest.pt is only copied and SHA checked. Canonical data derives from the saved object deserialized from pre-stop bytes. No post-stop torch.load(old latest) remains. Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/migrate_m123_to_m124_epochwise.py:81-112`.

- **R2-C4 Recovery preparation and model/Adam validation before signal: FAIL.** The stable raw pre-stop snapshot is now preserved correctly, and the live post-stop latest.pt is no longer used for recovery. However, source order is torch.load(raw bytes) at 52, os.kill(SIGTERM) at 73, torch.save(canonical) at 103, reload at 104, and A_B/C/Adam roundtrip assertions at 105-112. The canonical checkpoint's first A_B/C digest and optimizer structure accesses are also below the signal. Thus failures in preparing or checking the actual handoff resume.pt can still occur after stopping the original trainer. This leaves the explicit R2 requirement to complete recovery preparation/validation before SIGTERM unmet. No actual checkpoint corruption, missing key or failed write is claimed. Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/migrate_m123_to_m124_epochwise.py:73,97-112`.

- **R2-C5 Generator parity including legacy frozen/RNG branch: PASS_SOURCE.** All six outputs reconstructed in memory exactly match the actual source bytes. The legacy None check and the resume_RNG_saved branch are now reproduced correctly. R1-B2 is closed. Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/prepare_m124_epochwise_sources_20261011.py:33-34`.

- **R2-C6 Training and user protocol unchanged: PASS_HASH_INHERITED_SOURCE_REVIEW.** Among R1's 63 tracked inputs/dependencies, only migration and generator changed. The actual trainer, A+B+C model/loss, source user protocol, per-epoch OPE controller, strict dual-decline stop, equal-mean earliest-tie selection and best-only full VOT remain byte-identical to R1. R1's positive source checks and all evidence limits carry forward. Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epochwise_source_audit_R2_20261011/R1_comparison.json`.

- **R2-C7 R1 preserved and R2 intake matched: PASS_HASH.** All 29 R2 intake hashes match; all 159 artifacts inside R1's seal plus the R1 seal itself are unchanged. 68 R2 input/dependency/metadata files were snapshotted for final before/after checking. Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epochwise_source_audit_R2_20261011/R1_comparison.json`.

- **R2-C8 A-F integrity boundary: SOURCE_ONLY.** No GT/metric/scope implementation changed. A/B/F positive source classifications and C/D/E warnings from R1 remain applicable: real dataset-GT intended, no self-max metric normalization, no actual M124 result, no untouched-Test50/CDTB claim, one seed and human multiframe aids. This R2 audit does not reclassify an unrun source pipeline as a completed experiment. Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epochwise_source_audit_20261011/SOURCE_AUDIT.json`.

## Evidence limits

1. Continuation of the same-family/provisional R1 reviewer; requested gpt-6-astra/max, actual model/effort/backend UNATTESTED. No executor substitution.

2. Only the two changed source files are substantively re-reviewed. Unchanged protocol/training/evaluation and dependencies are verified by exact SHA and inherit the R1 source review; they were not needlessly rerun.

3. No SSH, GPU/NN action, training-progress query, process query/signal, actual checkpoint deserialization, target package import or model execution occurred. Pure source-string and JSON fragments were the only executed target logic.

4. Snapshot stability checks are source-verified, not tested against the real concurrently written latest.pt. Actual old checkpoint contents, Adam state, live process identity, migration request, deployment paths and installed package versions remain unobserved.

5. The preserved raw checkpoint now protects the saved boundary; the remaining finding is the explicit ordering requirement for preparing and validating canonical recovery state before signalling, not a claim that R2 has already damaged data.

6. Existing legacy RNG/full-frozen-digest absence, seed2027 reset, repeated partial sequence, external metric selection and no bitwise-continuation claim are unchanged.

7. R1's full evidence limitations remain in force, including no raw OPE/VOT recomputation, no performance/result claim, no upstream toolkit byte-equivalence attestation and no actual deployment admission.

8. The source USER_PROTOCOL is immutable intended-protocol history and retains its original source hash map. Exact R2 code identity is carried by R2 INPUT_MANIFEST; a later deployment request must use the reviewed revised source SHA values.

9. Native exec_command exposes a combined stdout/stderr stream; exact returned output is preserved and separate stderr is marked unavailable. The final tool-return envelope is held in the conversation/parent trace because a seal cannot recursively contain itself.

The two modified files compile as Python source; all six generator products exactly match actual files; three committed-prefix JSON fixtures pass. No training, model, GPU or process action was executed. Preserve R1 and R2, move the existing canonical preparation/validation block before the signal, then review that final diff.

