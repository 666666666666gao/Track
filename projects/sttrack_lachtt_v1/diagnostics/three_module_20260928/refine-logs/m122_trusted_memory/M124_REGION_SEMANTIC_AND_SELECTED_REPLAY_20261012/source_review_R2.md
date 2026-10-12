# M124 selected-best onset replay: R2 source integrity review

**Source verdict: PASS; zero source blockers. Overall integrity status: WARN for the unverified runtime and claim scope.** R1 B1 is resolved.

Date: 2026-10-12. Requested reviewer `gpt-6-astra` / `max`; actual model/effort **UNATTESTED**. R2 continues the R1 reviewer; `review_independence: same-family`, `acceptance_status: provisional`.

This is a SOURCE ONLY review. No replay, terminal suite result, saved-prefix parity, nine-metric success or raw-server completion is certified. This report is not GPU-release, NN admission or launch authorization.

## Revision findings

The only replay changes are the `selected_best` checkpoint-metadata rename and three receipt assertions binding plan, final, bank and bundle hashes. The candidate index remains `best`. Removing those assertions and normalizing the metadata name back yields an AST identical to preserved R1. The generator mirrors these changes. No tracking, numerical readout, training or metric implementation changed.

| Item | Result |
|---|---|
| R1 B1 metadata shadowing | Resolved; metadata and candidate index have separate variables |
| Final result construction | Original AST constructs a serializable result and reaches an intercepted write with explicit stdlib fixture inputs |
| Receipt identity | Matching fixture passes; wrong plan/final/bank/bundle values each fail the original assertion |
| R1 originals/reports | Exact bytes and original SHA-256 values preserved |
| Generator | R1 and R2 each reproduce their corresponding replay bytes exactly |
| Python 3.8 grammar | 28 source inputs pass under CPython 3.12.12 `ast.parse(feature_version=(3,8))` |
| Unchanged R1 inputs | 28 exact matches: 23 sources and five policy files |

Fixture values are explicitly synthetic audit inputs. No replay main, torch/model import, NN operation, image/GT loading or actual experimental result write occurred.

## A-F checks

### A ground truth provenance: PASS

Unchanged source-level real_gt design. Tracking starts from only the legal initialization bbox and advances step(image) without current GT. Prediction/candidate/action JSON is sealed and hashed before groundtruth.txt is hashed and parsed. Saved-prefix parity is a separate consistency comparison, not a substitute GT. Train152 loss timing and human multiframe label-aid qualifiers remain as in R1, with the supporting bytes unchanged. No dataset files were opened or raw GT provenance attested in this audit.

**Evidence:** `replay_m124_selected_best_cdtb_onsets.py:63-79`; `replay_m124_selected_best_cdtb_onsets.py:123-150`; `train_m124_joint_epoch.py:113-144`; `train_m122_full_causal.py:14-22`; `bind_m122_official_initializations.py:50-54`.

### B score normalization: PASS

The revision changes no score, geometry or metric calculation. IoUs, candidate counts and ranks are unnormalized diagnostic values. Internal odds_residual and removal of the quality factor are model readouts, not self-normalized benchmark performance. The locked full OPE evaluator still uses per-sequence PR curves and macro-averaging. No new metric numbers or calibration claim are supported.

**Evidence:** `replay_m124_selected_best_cdtb_onsets.py:93-111`; `replay_m124_selected_best_cdtb_onsets.py:149-181`; `dense_target_decoder.py:97-123`; `depthtrack_pr.py:117-153`.

### C result existence and identity: WARN

The requested saved-prefix plan/final/bank/bundle bindings are now explicit and correct. Positive and four separate mismatch fixtures exercise the exact new assertions. Box/score bytes are still checked against the sequence receipt. The top-level selected-final/full-suite guards remain before model construction. Actual suite results, controller exit evidence, saved outputs and the complete collection-manifest hash chain were not audited. The entry point still does not itself revalidate receipt.status, receipt artifact hash, all_results selection_sha256 or every row result_sha256. This is a remaining runtime/manifest scope limit, not an observed mismatch or a new source blocker.

**Evidence:** `replay_m124_selected_best_cdtb_onsets.py:35-56`; `replay_m124_selected_best_cdtb_onsets.py:130-146`; `run_m124_epochwise_official.py:40-61`; `collect_m124_epochwise_results.py:13-35`; `collect_m124_epochwise_results.py:46-48`.

### D called metrics and reachable output: PASS

R1 B1 is resolved. selected_best is assigned once at line 40; the separate candidate best is assigned at line 167 and only used as an index in lines 178-180. The final selected_epoch reads selected_best at line 185. A stdlib-only fixture executed the original candidate assignment twice, then the complete final result expression and intercepted final write call: the selected epoch survived, JSON serialization passed, and the write call was reached. This proves removal of the local blocker, not full replay execution. _load_rows and _vot_overlaps are called by the replay; full PR/F and author-reference checks are not run here.

**Evidence:** `replay_m124_selected_best_cdtb_onsets.py:40-49`; `replay_m124_selected_best_cdtb_onsets.py:136-162`; `replay_m124_selected_best_cdtb_onsets.py:167-185`; `replay_m124_selected_best_cdtb_onsets.py:203`; `run_m124_epochwise_official.py:58`; `depthtrack_pr.py:117`; `dense_region_encoding.py:38-56`.

### E scope assessment: WARN

The reviewed program is a caller-chosen one-sequence prefix with selected posthoc candidate states, not all CDTB80 or a final nine-metric success. Test50/CDTB80 participate in checkpoint selection; these cannot be called untouched holdouts. Alternative readouts remain same-state only, without forced saved-prefix parity or trajectory interventions. The review has no NN results and does not validate calibrated probabilities, text truth, model novelty or hidden-state equality.

**Evidence:** `replay_m124_selected_best_cdtb_onsets.py:29-32`; `replay_m124_selected_best_cdtb_onsets.py:53-80`; `replay_m124_selected_best_cdtb_onsets.py:184-201`; `run_m124_epochwise_suite.py:48-75`; `m124_epochwise_selection.py:4-10`.

### F evaluation type: PASS

The program remains real_gt by source design: submitted and candidate boxes are measured against post-seal dataset GT. Saved-prefix equality is replay consistency; Empty/no-quality/native-score alternatives are read-only fixed-state diagnostic readouts with real-GT measurement. This review is source plus stdlib fixtures, not an actual real_gt replay or suite result.

**Evidence:** `replay_m124_selected_best_cdtb_onsets.py:123-181`; `replay_m124_selected_best_cdtb_onsets.py:195-201`; `depthtrack_pr.py:40-60`.

## Detailed source checks

### B1 all uses and reachability: PASS

All seven metadata name occurrences use selected_best (one Store, six Loads). Candidate best has one Store and three Loads, with no best[...] metadata read anywhere. Two different oracle indices leave selected_best unchanged; original result-construction/write-call AST nodes pass with explicit fixture inputs. No result.json for an experiment was written.

**Evidence:** `replay_m124_selected_best_cdtb_onsets.py:40-49`; `replay_m124_selected_best_cdtb_onsets.py:167-185`; `replay_m124_selected_best_cdtb_onsets.py:203`; `r2/source_validation.json:variable_inventory`; `r2/source_validation.json:B1_reachability_fixture`.

### exact revision scope: PASS

Relative to preserved R1, the replay only separates the checkpoint metadata name and adds three assertions binding four receipt fields. Normalizing selected_best back to best and removing those three assertions yields an AST identical to R1 (excluding source location attributes). The generator mirrors those same changes. No tracking, readout, GT, numerical, retry or optimization logic changed.

**Evidence:** `replay_m124_selected_best_cdtb_onsets.py:40-49`; `replay_m124_selected_best_cdtb_onsets.py:140-142`; `replay_m124_selected_best_cdtb_onsets.py:185`; `prepare_m124_selected_best_onset_replay_source_20261012.py:16-33`; `r2/R1_to_R2_replay.diff`; `r2/R1_to_R2_generator.diff`.

### R1 preservation: PASS

Both supplied R1 original files exactly equal the independent R1 input snapshots and their original SHA-256 values. R1 MD and JSON retain their original report hashes. The fixer hashes each original before copying and only then performs the scoped replacements. The fixer was read and syntax-parsed, not executed by the reviewer.

**Evidence:** `fix_m124_selected_best_replay_B1_20261012.py:8-23`; `fix_m124_selected_best_replay_B1_20261012.py:24-37`; `r2/source_validation.json:preserved_R1_originals`; `r2/source_validation.json:R1_reports_original_hashes_unchanged`.

### generator byte reproduction: PASS

Executing only stdlib text-transformation/check AST statements before the destination-path assignment reproduces the current R2 replay byte-for-byte. The same check on the preserved R1 generator reproduces the preserved R1 replay exactly. No generator file-write statement was run.

**Evidence:** `prepare_m124_selected_best_onset_replay_source_20261012.py:1-37`; `M124_selected_best_replay_R1_original_generator_20261012.py:1-34`; `r2/source_validation.json:generator_checks`.

### python38 grammar: PASS

All 28 Python source inputs (the 25 R1 sources at their current bytes, two preserved originals and the fixer) pass ast.parse(feature_version=(3,8)) under installed CPython 3.12.12. This is a grammar check only, with no torch/model import and no Python 3.8 dependency/runtime attestation.

**Evidence:** `r2/source_validation.json:python38_syntax`.

### saved reference identity: PASS

The exact new assertions bind receipt.plan_sha256 to the provided plan bytes, receipt.final_sha256 to the checked bundle final, and receipt.bank_sha256 / bundle_sha256 to the checked plan identities. The checked plan/final already connects to selected_best. Each of the four mismatched fixture fields is rejected. These receipt checks occur after inference/sealing, before the saved prefix is loaded and compared; they are not GPU admission checks.

**Evidence:** `replay_m124_selected_best_cdtb_onsets.py:42-49`; `replay_m124_selected_best_cdtb_onsets.py:139-146`; `m124_epochwise_official_runtime.py:9-41`; `m122_official_runtime.py:7-21`; `r2/source_validation.json:receipt_identity_fixture`.

### retained causal state and metric checks: PASS

The 28 unchanged R1 inputs (23 sources plus five policy files) were byte-compared with their R1 snapshots and rehashed. The unchanged core and retained source evidence support: model guards before construction; own-history tracking without GT resets or parity forcing; Empty alternatives do not submit box/query/template/selected-feature state; no external backward/optimizer; fixed 16x16/256 candidate mapping with explicitly float32 native observations and retained float64 submitted base; post-seal metric calls and limited tracker-step counters. These are source conclusions, not runtime certifications.

**Evidence:** `replay_m124_selected_best_cdtb_onsets.py:35-56`; `replay_m124_selected_best_cdtb_onsets.py:68-127`; `replay_m124_selected_best_cdtb_onsets.py:149-201`; `full_dense_tracker.py:35-85`; `trusted_template_tracker.py:21-68`; `dense_target_decoder.py:51-123`; `m124_epochwise_official_runtime.py:61-70`; `r2/source_validation.json:unchanged_R1_inputs`.

## Remaining warnings

### W1: Runtime and GPU admission remain unavailable

No actual replay, terminal suite, raw server, live process exit, GPU-release or NN admission proof was supplied or queried. The source demands completion metadata but does not establish that it exists. formal_nine_metrics is not any_joint_pass. Source success does not authorize a launch.

**Evidence:** `replay_m124_selected_best_cdtb_onsets.py:35-56`; `run_m124_epochwise_suite.py:95-103`.

### W2: Remaining artifact-chain scope

The R1 gap in four receipt identity fields is fixed. The replay still relies on the upstream terminal collector for receipt completion and the broader manifest chain; it does not independently check receipt.status, receipt hash against metrics.json, selection_sha256 or each all_results row hash. Actual collection artifacts were not inspected. This is a later concrete-artifact verification requirement, not evidence of an actual mismatch.

**Evidence:** `replay_m124_selected_best_cdtb_onsets.py:46-49`; `replay_m124_selected_best_cdtb_onsets.py:139-144`; `collect_m124_epochwise_results.py:13-35`; `collect_m124_epochwise_results.py:46-48`.

### W3: Dependency and runtime parity limits

Python 3.8 grammar does not attest a Python 3.8 interpreter or torch/NumPy/VOT/native cal_bbox compatibility. The stdlib fixtures do not run main, images, GT, model forward, alternative tensor equality or metric code. Saved-prefix parity and hidden recurrent/template equality remain unknown. Last-image dimensions still bound all prefix/candidate overlaps, with no actual sequence image inspection.

**Evidence:** `replay_m124_selected_best_cdtb_onsets.py:82-119`; `replay_m124_selected_best_cdtb_onsets.py:147-162`; `replay_m124_selected_best_cdtb_onsets.py:195-201`.

### W4: Scope and selection qualifiers

Each invocation covers one caller-selected prefix and posthoc sampled states. The selected final was chosen using Test50/CDTB80 F, so those are not untouched held-out measurements. No full-trajectory rescue, final nine-metric success or comprehensive benchmark result follows from this review. tracker_calls counts tracking steps, not all neural forwards.

**Evidence:** `replay_m124_selected_best_cdtb_onsets.py:29-32`; `replay_m124_selected_best_cdtb_onsets.py:53-89`; `replay_m124_selected_best_cdtb_onsets.py:184-201`; `run_m124_epochwise_suite.py:48-75`.

### W5: Readouts remain uncalibrated and semantically limited

Sigmoid quality, cosine similarity, full-minus-Empty support and current-IoU C regression do not establish calibrated probabilities, phrase visibility, physical instance identity, future memory utility or semantic truth. GT-best candidate rank is an after-the-fact oracle diagnostic, not a submitted alternative. Novelty is outside this audit.

**Evidence:** `dense_target_decoder.py:74-123`; `template_write_C.py:13-24`; `train_m124_joint_epoch.py:140-144`; `train_m124_joint_epoch.py:232`; `replay_m124_selected_best_cdtb_onsets.py:156-181`; `replay_m124_selected_best_cdtb_onsets.py:201`.

### W6: Review attribution is provisional

Requested reviewer gpt-6-astra / max. Actual model and effort are UNATTESTED. R2 continues the same reviewer that began fresh in R1; same-family / provisional, with no cross-family or host model-attestation claim.

## Current metric calls and counters

`_load_rows` is called at replay lines 136 and 145-146; `_vot_overlaps` is called at lines 149-150 and 161-162. The full `evaluate_depthtrack_results` / threshold computation is only invoked by the separate complete OPE runner (`run_m124_epochwise_official.py:58`, `depthtrack_pr.py:117`). `format_metrics` and `verify_author_reference` are not called by this replay. No new P/R/F or author-reference validation is thereby earned.

For a successful invocation, `prefix_frames=max(frames)+1`, `tracker_calls=max(frames)`, `candidate_states=extra_Empty_decoder_calls=len(frames)` and `candidates_per_state=256` (replay lines 31, 53-89, 184-188). These are step/readout counts, not the total NN forward count. No actual runtime counts are reported here.

## R2 and preserved-R1 hashes

| Input | SHA-256 |
|---|---|
| `replay_m124_selected_best_cdtb_onsets.py` | `2b88ddffb9c888ef8d1db8dfff44e1c44cc50086a1bc77b2e99c3c7b984d0538` |
| `prepare_m124_selected_best_onset_replay_source_20261012.py` | `e6807834ed4e0359077863ec5aaad3005c4ce6ecd226718abe73b7e1a724b4d6` |
| `M124_selected_best_replay_R1_original_source_20261012.py` | `4915b45543e904f02d4ecd0801c43863a01481f3771739cd95587bb58a420fbb` |
| `M124_selected_best_replay_R1_original_generator_20261012.py` | `98a285a0d82c0fd10e24b8814781c18aa12cd60d510445b6043eb7307fbd94ca` |
| `M124_selected_best_onset_replay_source_review_R1_20261012.md` | `d284908ffc617ca883ea4ffb79a0ad8ae26b3a84deeec024f572ec493d7ca75d` |
| `M124_selected_best_onset_replay_source_review_R1_20261012.json` | `aad8bf521029f94f5cd7dc2084aea7c57a961e6e18a89f62a5a46f9e93a494d3` |
| `fix_m124_selected_best_replay_B1_20261012.py` | `0b7081e5a835ee8ac132d9641fb71bf1be2c868d35e4c63918260d96ecf00649` |

The JSON contains hashes and exact read/reuse scopes for all 35 inputs. All were rehashed unchanged at R2 report serialization. Current sources, preserved originals, R1 reports and fixer bytes are copied under `r2/input_sources/`.

## Forensic artifacts

Trace: `C:/Users/gb/.codex_track_publish_m29_20260902/.aris/traces/experiment-audit/2026-10-12_run01`.

- `r2/source_validation.json`: grammar, generator, variable inventory, reachability fixture, four negative identity cases and original preservation.
- `r2/R1_to_R2_replay.diff` and `r2/R1_to_R2_generator.diff`: exact revision diffs.
- `r2/CurrentC_to_R2_replay.diff`: complete comparison with the original CurrentC entry.
- `002-source-R2.request.json`, `.response.md`, `.meta.json`: the review continuation trace.

No source edits, SSH/network calls, GPU/training queries, model/torch imports, NN calls, optimizer steps, installs or git operations were performed by this reviewer. Actual terminal/manifest proof and separate GPU-release/admission proof remain required before launch; actual artifacts remain required before result claims.
