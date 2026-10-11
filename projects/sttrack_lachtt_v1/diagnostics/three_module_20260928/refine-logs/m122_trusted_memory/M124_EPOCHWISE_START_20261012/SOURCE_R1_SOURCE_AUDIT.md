# M124 epochwise source audit — R1

**SOURCE_ONLY_FAIL — 2 blocking findings.** Requested reviewer: `gpt-6-astra`, effort `max`. Actual backend/model/effort: **UNATTESTED**. Fresh delegated review, same-family, provisional; no executor substitution.

This review uses only `M124_epochwise_USER_PROTOCOL_20261011.json` to define intended behavior. It does not report M124 runtime results or authorize migration. The 26 sealed intake files were read, local imports and relevant dynamic dependencies followed, and 63 input/source/dependency files snapshotted. Hash closure is recorded in `hash_afterchecks.json` and `AUDIT_SEAL.json`.

## Blocking findings

### B1 — The legacy latest.pt can be truncated or replaced between validation and SIGTERM (HIGH)

The legacy trainer flushes its sequence log, then torch.save overwrites fit/latest.pt directly. Migration loads that live path at line 45 but does not preserve or resume that loaded object. After SIGTERM, it copies and reloads the live latest.pt at lines 57-65. A save can overlap the pre-stop read, start after that read, or be interrupted by the signal. The only recoverable latest file can then be incomplete. In addition, line 66 parses every post-stop log line before slicing to the saved count, so an incomplete uncommitted trailing line can abort migration after the trainer has already been stopped. This is a concrete conflicting read/write path, not a claim that corruption actually occurred.

Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/migrate_m123_to_m124_epochwise.py:39-76,85-100`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/train_m123_ABC_joint_full152.py:151-157`.

Minimal fix: In the migration helper only, capture one stable raw legacy checkpoint and raw sequence-log snapshot before SIGTERM; keep the raw bytes, fully CPU-deserialize and validate model/Adam/counts and the completed-sequence prefix before stopping. Verify snapshot stability against the same live file (identity/size/mtime and/or repeated byte SHA). If validation/stability fails, exit without signalling, without retry or fallback. After termination, resume only the pre-stop sealed snapshot; retain the post-stop legacy files as historical evidence even if incomplete. Do not edit the running legacy trainer or rewrite its old logs.

### B2 — The generator emits a trainer that rejects the explicitly supported legacy checkpoint (MEDIUM)

The generator replaces the old before-state block with an unconditional frozen == resume['frozen_state_sha256'] assertion. The canonical legacy handoff deliberately sets frozen_state_sha256=None. Therefore a fresh generator-produced trainer fails before training. The actual generated trainer already contains the resume_RNG_saved/legacy-None branch. In-memory reconstruction of all six generated outputs found exactly this one difference: five outputs match byte-for-byte, trainer differs only by this branch.

Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/prepare_m124_epochwise_sources_20261011.py:33-44`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/train_m124_joint_epoch.py:75-82`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/migrate_m123_to_m124_epochwise.py:85-90`.

Minimal fix: Synchronize generator line 33 with the actual trainer's lines 79-82 legacy-None branch. Reconstruct the six outputs in memory and require exact equality. Preserve old M123 sources and historical protocol records.

## A–F integrity review

### A. Ground-truth provenance: PASS

Training targets originate from the pinned official Train152 groundtruth.txt paths, never from predictions. GT is loaded at sequence entry, but frame targets are consumed only after tracker.step has committed bbox/query/template action. C target is selected-frame real GT IoU, not future utility. OPE opens and hashes subsequent GT only after predictions and receipts exist; VOT uses toolkit dataset groundtruth through FrameMapSequence.

Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/train_m122_full_causal.py:14-22`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/train_m124_joint_epoch.py:113-144`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/m123_joint_full_tracker.py:34-84`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m124_epochwise_official.py:12-61`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/native_ope/source_snapshots/depthtrack_pr.py:40-60,95-111`; `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/official_analysis_sources/finalize_vot_transaction_low22.py:193-258`.

### B. Score normalization: PASS

OPE precision averages overlaps over selected frames; recall divides overlap sum by dataset visibility count; curves macro-average by sequence; F uses the harmonic mean and returns raw plus percent values. Confidence-based thresholds and selecting the PR-curve maximum are explicit metric mechanics, not division by the model's own maximum. VOT EAO/ACC/ROB are multiplied by 100. Internal feature, depth-image and response calibration are not reported metric normalization.

Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/native_ope/source_snapshots/depthtrack_pr.py:25-37,117-153`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epochwise_vot.py:22-25`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/dense_target_decoder.py:97-115`.

### C. File identity and result existence: WARN

All 26 intake file SHA values match. CurrentC captured F references exactly equal protocol [66.09145399091419,72.1632445464765] and use warm final cc71e0b4... . No M124 epoch result, migrated checkpoint, real OPE/VOT output or completed run is asserted. Future readers enforce result/receipt/plan/bundle/prediction SHA links. Actual raw remote artifacts and process state were not accessed.

Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epochwise_source_audit_20261011/INPUT_MANIFEST.json:files`; `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epochwise_USER_PROTOCOL_20261011.json:initial_comparison_F_percent,actual_migration_completed,actual_new_controller_launched`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/collect_m124_epochwise_results.py:13-39`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m124_epochwise_suite.py:53-69,77-102`.

### D. Dead code and dispatch: WARN

The M124 controller dispatches training, both OPE trackers/analyzers, selection, selected-best VOT, sealing and collection. OPE evaluate_depthtrack_results and VOT failure collector have reachable call sites. Historical training/evaluation main functions imported for helpers remain main-guarded and are not called by the M124 route. The harmless depthtrack_pr.format_metrics helper is unused by this route. No function's existence is treated as evidence it ran.

Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m124_epochwise_suite.py:30-94`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m124_epochwise_official.py:45-62,81-84`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epochwise_vot.py:28-35`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/train_m121_dense_target.py:202`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/native_ope/source_snapshots/depthtrack_pr.py:157-158`.

### E. Scope: WARN

Intended scope is one seed (2027), up to 20 completed Train152 epochs, complete 50/76373-frame and 80/101956-frame OPE after every completed epoch, and selected-best 127-sequence/1765-anchor VOT. External Test50/CDTB80 explicitly select checkpoints and stopping, so this is not untouched-test evaluation. Human initialization text review used multiframe aids. No training outcome, robustness, nine-metric success or runtime admission is established.

Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/train_m124_joint_epoch.py:32-50,199,214-236`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m124_epochwise_suite.py:18,27-76,79-102`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/collect_m124_epochwise_results.py:15,32,47-48`.

### F. Evaluation type: PASS

Both OPE and VOT are intended real dataset-GT evaluations, with human-reviewed initialization text as an auxiliary input. They are not synthetic/model-reference proxies. M124 has no measured result in this audit.

Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m124_epochwise_official.py:54-58`; `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/official_analysis_sources/finalize_vot_transaction_low22.py:223-235`.

## Protocol and implementation checks

- **P1 Full Train152 own-history A+B+C — PASS_SOURCE.** Pinned spec has 152 unique sequences and 219802 tracked frames per pass. All frames are tracked; valid GT only controls losses. No fit/dev selection or cached feature fit is dispatched. Own state is not reinitialized from subsequent GT. Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/train_m124_joint_epoch.py:37-50,110-199`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/m123_joint_full_tracker.py:21-84`.

- **P2 Frozen pretrained models and two devices — PASS_SOURCE.** A/B and C are reconstructed on CUDA0, native STTrack and CLIP on CUDA1. Observation tensors move to the learning device; native/CLIP eval+requires_grad_(False)+no_grad and before/after digests guard frozen state. This is source reasoning, not a CUDA execution test. Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/train_m124_joint_epoch.py:67-87,199-202`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/full_dense_tracker.py:23-40,53-83`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/m123_joint_full_tracker.py:13-18,23-43`.

- **P3 Resume model/Adam/RNG and complete-sequence counters — PASS_SOURCE_WITH_LEGACY_LIMIT.** Actual generated trainer loads model/Adam, checks sequence boundary, carries cumulative counters/records, restores saved Python/NumPy/CPU/all-CUDA RNG on new checkpoints. Legacy deliberately has no full frozen digest or RNG, and resets seed 2027 after reconstruction. The incomplete sequence is restarted. There is no bitwise-continuation claim. Migration safety remains blocked by B1. Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/train_m124_joint_epoch.py:57-106,186-198`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/migrate_m123_to_m124_epochwise.py:65-100`.

- **P4 Epoch completion and complete OPE counts — PASS_SOURCE.** Epoch completion requires epoch*152 records and epoch*219802 tracking calls before final publication. Both OPE datasets run and counts/SHA are checked before selection or next training epoch. Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/train_m124_joint_epoch.py:199-236`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m124_epochwise_suite.py:27-57`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m124_epochwise_official.py:14-16,22-42,49-61`.

- **P5 Warm comparison, strict dual-decline stop, equal-mean best and ties — PASS_SOURCE.** Initial previous F values are the supplied warm reference; first trained epoch is eligible even if below warm. Two adjacent transitions with both scores strictly lower trigger stop. Any equality or one-score rise resets streak. Best uses equal mean; strict > preserves earliest tie. Five pure-Python fixtures passed. Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/m124_epochwise_selection.py:4-10`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m124_epochwise_suite.py:24-25,58-76`.

- **P6 One selected epoch weight shared across OPE and VOT — PASS_SOURCE.** Best final.pt is copied and SHA checked; VOT plan reuses that epoch's fit/evaluation bundle; OPE/VOT result identities all compare to the same final and bundle SHA. Four VOT shards require all 1765 trajectories and 5295 files. Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m124_epochwise_suite.py:63-94`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/prepare_m124_epochwise_evaluation.py:31-57,77-120`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/collect_m124_epochwise_results.py:13-38`.

- **P7 Old process target and hourly boundary — PASS_SOURCE_LIMITED.** Migration checks its request deadline before live access; the captured original deadline is 2026-10-11T15:34:49.891541+00:00. It reads /proc/<requested PID>/cmdline and checks old trainer script/output before SIGTERM; old stage must record -15 and controller nonzero, and process path must disappear. Actual MIGRATION_REQUEST and PID birth/current parent identity are not supplied or attested, so this is not permission to stop a process. Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/migrate_m123_to_m124_epochwise.py:26-42,49-56`; `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M123_actual_full_joint_launch_20261011.json:23,25-50`.

- **P8 Source seal, dispatch paths and role imports — WARN_DEPLOYMENT_NOT_ATTESTED.** Source/input maps and review report are hashed before launch, inference checks bundle source identities, and generated VOT wrapper/config/metadata are sealed. Metric-role top-level imports remain stdlib/NumPy plus lazy runtime definitions; Torch/CLIP/CUDA construction is inside tracking paths. Exact future deployed copies/import resolution and installed package versions are not inspected. The future request must pin the actual dispatched new code-root dependencies, not merely old source paths. Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/migrate_m123_to_m124_epochwise.py:30-37,101-116`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m124_epochwise_suite.py:19-23`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m123_joint_full_suite.py:14-31`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/m124_epochwise_official_runtime.py:18-39`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/prepare_m124_epochwise_evaluation.py:77-110`.

- **P9 Actual generated model/loss preservation — PASS_SOURCE.** Entire old/new per-frame action, AB loss, C IoU/MSE loss, accumulation, clipping and Adam update loop are structurally identical by AST. Models and tracker helpers are reused. Only lifecycle/resume/checkpoint/evaluation orchestration changes. C inputs are detached, so this is joint optimization of A+B and C, not temporal end-to-end backpropagation. Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/train_m123_ABC_joint_full152.py:94-141`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/train_m124_joint_epoch.py:125-172`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/train_m121_dense_target.py:19-66`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/train_m122_full_causal.py:25-31`.

- **P10 Generator parity — FAIL.** Five of six deterministic generated outputs exactly match actual bytes. The trainer mismatch is precisely B2. Four additional M124 controller/migration/selection/supervisor sources are separately authored, not generated by this script. Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/prepare_m124_epochwise_sources_20261011.py:33-44`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/train_m124_joint_epoch.py:75-82`; `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/migrate_m123_to_m124_epochwise.py:85-90`.

- **P11 Preserve M123 history — PASS_SOURCE_ONLY.** No source mutation is performed by migration; handoff creates separate copies. This audit itself modifies only its designated output directory and snapshots all 63 inspected inputs. Current R1 cannot certify a future actual stop/copy/launch. Evidence: `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/migrate_m123_to_m124_epochwise.py:47-63,77-90`; `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_epochwise_source_audit_20261011/input_snapshots.json:all rows`.

## Deterministic checks

- All 26 intake hashes match the supplied manifest. Final before/after checks cover all 63 captured files and their raw snapshots.
- 58 Python sources parse and compile to in-memory code objects; no target ML modules were imported or executed.
- Five selection fixtures pass: first eligible epoch; second consecutive dual decline; equality reset; one-score rise reset; earliest tie retained.
- The entire old/new training frame loop has equal AST, including state action, AB loss, C target/MSE, accumulation, clipping and optimizer update.
- In-memory generator reconstruction gives five exact outputs and the single trainer difference documented in B2.
- Captured warm F values exactly match 66.09145399091419 and 72.1632445464765, with the same `cc71e0b4...` warm-final identity.

## Evidence limits

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

## Claim impact and next step

The actual generated frame-level training recipe and epoch selection are source-supported. Safe legacy migration and reproducible generation are not yet supported. Correct B1 and B2 in new source only, preserve this R1 evidence, then re-audit the revised hashes. No M124 training, GPU action, remote process action, checkpoint recovery, OPE recomputation or VOT evaluation was performed here.

