# R2 CPU raw readback — source audit v2

Date: 2026-10-10. Reviewer: `/root/m122_r2_rawcheck_source_integrity_20261010`, continuing the original fresh-context review. Requested Codex `gpt-6-astra` / `max`; actual backend, model and reasoning: **UNATTESTED**. **Same-family / provisional**, not another fresh or cross-family review.

**Source verdict: PASS, zero source blockers. Overall: WARN because actual terminal teacher artifacts and runtime readback remain absent.** The two v1 findings are closed at source scope by the exact two required assertions. This report is not a terminal teacher semantic certificate and does not establish R2 completion, C training or final metrics.

Reviewed verifier SHA256: `fc5789875562bdb5efe8427c95ae40a2223828f02505cd8474b40f92fdebbe82` (21305 bytes). Original v1 SHA remains `103941e60b238a1025fc2d7f760bbd643b8b36996bd927df509627e52139e81c` in its frozen snapshot and FAIL report.

## Preservation and exact change scope

All 24 current primary inputs were read directly and matched their supplied SHA/size. The new manifest and preparation receipt were also read and snapshotted, making 26 v2 snapshots. All 14 supplied Python files parse with Python3.8 grammar using the explicit existing CPython3.13 stdlib interpreter. All supplied JSON was parsed, the fixed bank archive was read, and the prior semantic review of the 23 unchanged inputs was retained after byte equality was established.

Every one of the **47 v1 sealed files** matches its original SHA/size. The v1 report is still **FAIL / two blockers**, SHA `40f6df29b6d226cfa7b311458d44ef7ec6d6d665207a502fdd1bf7163af3e745`; its seal is still `610e8716f0354e270ed4f9a86225629aa1830b2100113e03892c5e2a3ea1714d`. Its original counterexample, failed PATH-interpreter record, full response and snapshots are preserved.

The archive `R2_raw_readback_source_v1_original_20261010.zip` has SHA `c243cc496cbb088c0d9d49f167bea78ad6de2146ccc715e1c6bfd1e9bc3d2099`. Its 48 entries are exactly the 47 sealed files plus `AUDIT_SEAL.json`; each entry's bytes/hash match the original and ZIP CRC validation succeeds. The archive was read only.

The only primary input change is insertion of these lines in `verify_template_write_teacher_raw.py`, inside the existing future-row loop:

```python
assert valid(k['record']['bbox'])
assert not w['record']['template_write'] and not k['record']['template_write']
```

They are now lines **259–260**, immediately after W/prefix equality and before branch continuity and IoU calculation. Removing exactly those two lines reproduces the frozen v1 source byte-for-byte at the text level; the original and new file SHA/size are independently recorded. The other **23 primary inputs are unchanged**. The running teacher/controller/features/cache/fork sources were not changed by this revision or reviewer. `SOURCE_V1_V2_DIFF.patch` retains the exact diff.

## B1 and B2 closure

| v1 finding | v2 evidence | Decision |
|---|---|---|
| B1: invalid K future bbox could become ordinary zero IoU | Line 259 calls the actual `valid()` on every saved K bbox before `future_valid` (265) and `ki` (267). It requires all coordinates finite and both sizes positive. W retains its independent validation through exact equality with a validated prefix box. | CLOSED_SOURCE_ONLY |
| B2: K future records could assert an extra template write | Line 260 independently rejects either W or K `template_write=True` within every saved future frame. This matches `template_write_forks.py:119` and the original 32-frame window under the 50-frame schedule. | CLOSED_SOURCE_ONLY |

The reviewer compiled **only the exact v2 `valid` function and the two actual AST assertion nodes** into a stdlib fixture. No Torch/NumPy/experiment import, main, model or rollout was run. The assertions are confirmed to be direct statements in the reached future-row loop, preceding GT-validity branching, so invalid K boxes are rejected even when that frame's GT is unknown. Empty future windows have no future records to check and retain their existing null-label treatment.

| Extracted-source fixture | Actual result |
|---|---|
| Ordinary positive-size finite box | Accepted |
| Finite box with negative x/y and positive sizes | Accepted |
| Original v1 counterexample `[100,100,-1,2]` | AssertionError |
| Negative height, zero width, zero height | AssertionError for each |
| NaN coordinate, infinite coordinate, NaN size, infinite size | AssertionError for each |
| W-only future write, K-only future write, both future writes | AssertionError for each |

All **13 cases** behaved as required: **11 invalid cases rejected**, **2 valid cases accepted**. These are verification fixtures, not observed teacher samples or evidence that the historical teacher was corrupt. The original v1 acceptance counterexample remains preserved; it is not rewritten as a historical PASS.

No further source correction is required for B1/B2 or was identified in this bounded continuation. No fallback, extra exception handling or unrelated refactor is requested.

## A–F recheck

| Check | Status | Source conclusion and evidence |
|---|---|---|
| A. Ground-truth provenance | PASS | Real dataset GT SHA/count/first-box checks remain at verifier lines 188–200; teacher prediction serialization precedes GT loading (`run_template_write_teacher.py:137–150`). Human-bank alignment and no-event references remain checked at verifier 96–105,202–214. |
| B. Score normalization | PASS | Scalar IoU denominator is geometric union (47–51); future utility remains signed W−K mean over dataset-valid future frames (265–279). Unknowns and negative/zero outcomes are preserved. New K validation closes the prediction-domain gap without changing any label formula. |
| C. Result existence/provenance | WARN | All reviewed local inputs, v1 evidence and archive exist and match. Verifier terminal gates, source/protected77 checks and actual output hashes remain required by source. Actual terminal teacher/raw artifacts are not supplied. |
| D. Called computations/guards | PASS | Main calls the validator helpers and reaches both new guards inside the future loop before IoU. No unused replacement helper was added; no model/teacher import or false runtime PASS was introduced. |
| E. Scope | WARN | Source intends Train152, seed2027, 219802 prefix calls, C fit120/development32. Actual full-teacher event/sign/eligible counts, C results and final nine metrics remain unproved. |
| F. Evaluation type | PASS | Dataset GT IoUs and signed utility are prospective `real_gt` Train diagnostics. W/native and historical K/K equality are consistency proxies. Human text is fixed input, not `human_eval`; test fixtures are not experimental data. |

### Actual called path and terminal/provenance gates

The new source still runs `main()` under its final guard. Main requires assertions enabled and Python3.8 (line 67), then controller zero exit, complete aggregate status, fixed152/219802 counts, exact model-input digests and no optimization/C evaluation (69–74). It pins the actual runtime gate SHA `26f555c0d1eb13ddac5469d333926780c2fc760a1e45866e51a497533874fbd7`, requires **77** bound files, and checks every path's actual bytes/hash (75–84). Both collect and replay zero exits are required for each shard (137–138).

The supplied local gate contains **75** entries, not the target's77. The unchanged bootstrap adds the two actual native/CLIP checkpoint byte-stat pins (`R2_concrete_bootstrap_20261010.py:19–24`) and records the changed gate digest/count (34,43). The unchanged actual launch receipt records77 and the digest pinned by this verifier. This source link is coherent; the target77 gate bytes and their current runtime verification are not present locally. Neither the 75-entry preparation gate nor the launch receipt can substitute for the terminal gate/readback.

The aggregate, collection, teacher and source input joins remain unchanged. The exact even/odd original indices, shard physical-GPU strings, seed2027 and frozen/decoder before/after equality metadata are required at lines 144–161. These are metadata/byte validations, not an independent reexecution of frozen networks. Original stage launch commands, receipts, logs and terminal controller provenance must still be examined by the later terminal reviewer.

### All opportunities, all sequence prefixes and raw references

Lines 163–169 require one-to-one collection/teacher events, original fields unchanged, unique sorted `(sequence_index,frame)` keys and per-sequence grouping. Lines 172–187 consume every original noninitialization frame for every shard sequence, with exact sequence/frame order, positive finite native boxes and previous-box continuity. The actual native predicate is `frame % 50 == 0 and native_same_position_response > .75`. Lines 298–306 require no trailing prefix rows, exact equality between all observed opportunities and stored events, and exact shard/aggregate call totals. No positive-GT or positive-utility event filter was added.

The unchanged spec gives shard0 **76 sequences /107060 calls**, shard1 **76/112742**, total **152/219802**. These are planned static counts, not completed execution counts. Qualification additionally depends on actual native response, so checkpoint counts are not observed event counts.

Every sequence's GT and shared reference are checked before the no-event `continue` at line214. Per-sequence references are loaded and hashed even when there are no event rows, and their words/mask/empty/initial box are compared to the bound human bank/spec. For event sequences, shared-reference byte/SHA joins and each payload/rollout byte/SHA check remain at215–231. For no-event references there is no original event-row expected digest; actual digests are recorded, and no absent original commitment is claimed.

### GT alignment, bank and first-box evidence

The sole known GT-count exception remains **`toy07_indoor_320`**, original sequence index62, **1406 GT rows →1367 paired image frames**, first box `[298,164,36,45]`, GT SHA `683e8ae7ae401b71b8d10e9bb489c3956a150163606f5bac925a911f395444e2`. The spec, unchanged original loader (`train_m122_full_causal.py:14–22`) and verifier193–197 agree. No new truncation, generic fallback or GT rewriting is introduced. Actual remote GT files/images have not been read in this review.

The supplied fixed bank was freshly inspected with the same reviewer-owned symbolic pickle parser and raw IEEE754/boolean storage reader, executed from the v2 audit directory. No pickle reducer or Torch loader executed. All152 bank sequence and historical split entries match the label file; masks match phrase counts; padding is zero; raw float storage is finite. Stored descriptors remain CPU/no-grad tokens `(152,5,768)`, bool mask `(152,5)` and empty `(768,)`. The spec and bank have different sequence orders, so the unchanged `.index(sequence)` alignment is material and correct. All152 recorded first boxes have finite coordinates and positive sizes.

The immutable human metadata still has151 `conflicting` and1 `supported` historical status strings while declaring all152 human-confirmed. `pine02_wild_320` retains one semicolon-containing attribute phrase occupying one slot. This review verifies bank alignment and retained bytes, not independent human/image correctness or new phrase generation. The source explicitly leaves `human_review_recreated=False`.

### Cached519, W/native correspondence and branch continuity

Lines232–248 still bind the event record/state to its native prefix and check the cached float32 `(1,519)` input: current128, saved previous128, shared initial256, quality/observation/native3 and predicted-motion4. Current, previous, shared reference and quality sections use exact tensor equality. Motion is independently recomputed from consecutive boxes with `rtol=atol=1e-6`. The provenance limitation remains: there are no per-frame prefix query/features to reconstruct every past feature independently, and the shared initial projection is not rerun through a network.

The raw tensor checker at107–118,205,231 recursively requires CPU, no-grad and finite tensors. Future lengths/bounds remain fixed, W records match the native prefix, and the W/K previous-box chains evolve separately from the shared event box. The two new guards now close K bbox validity and the future no-write constraint. Query lists and saved selected-feature shape `(1,128)` remain checked. Record equality and saved-tensor consistency do not prove every historic input use, kernel or unrecorded recurrent state.

### Labels, denominators and common fit/dev accounting

Current/future GT validity still uses finite GT coordinates and positive GT sizes. Current invalid GT yields null current IoU. Future invalid frames keep null W/K labels; no valid future yields null utility. Signed per-frame W−K values and their arithmetic mean are neither clipped nor rectified. All events, including negative, zero, unknown and truncated-tail cases, are retained. Common supervised eligibility remains current GT valid plus at least one valid future GT frame.

The exact pinned split is unchanged and was read directly; its original SHA ranking proof is preserved. It maps the original sequence order to120 fit/32 development sequences. The newly generated bank/spec inventory checks all152 mappings. Verifier278–295 records event-level labels/eligibility and split counts. The unchanged C consumer (`train_template_write_C.py:39–99`) uses the same feature rows/eligibility for both supervision arms and changes only the target. Its exclusion reasons are disjoint by current-unknown-first precedence; the raw verifier's current-unknown and no-valid-future diagnostic counters can overlap and must not be summed as disjoint exclusions. Eligible/excluded totals and per-event flags remain unambiguous.

Denominator taxonomy is unchanged: geometric union for IoU; valid-future count for mean signed utility; sample count for MSE; valid-frame/equal-sequence reduction for development metrics. Decoder `probability/native`, pointwise odds normalization, feature normalization and previous-box-size motion scaling are model/input transforms, not metric self-normalization. The historical preservation-loss denominator is an uncalled training objective in this readback path. H10 remains a geometric development diagnostic, not VOT ROB.

A+B already trained on all152. Development32 is a holdout for new C optimization only, and fixed-teacher utility is valid only on the original teacher states. It is not automatically a label for changed C-policy history.

### CPU helper versus terminal semantic certificate

The actual output path at309–330 records aggregate, both teacher-shard and both prefix digests, raw file hashes and the77 protected-file inventory. It still sets `independent_terminal_audit_certified=False`. AST inspection confirms it does **not** emit `review_call_status`, `R2_terminal_raw_audit_complete` or `audited_teacher_result_sha256`; these remain the fresh terminal review's responsibility. Its status is a CPU readback status, not a semantic PASS.

The unchanged C training/analyzer gates still demand a completed zero-blocker terminal audit plus matching audited aggregate/shard/prefix digests (`train_template_write_C.py:111–116`; `analyze_template_write_C_dev.py:97–116`). Neither this source report nor a successful future CPU helper run alone supplies that certificate.

## Nonblocking limits and claim impact

- **N1 — terminal evidence absent:** the supplied launch receipt explicitly does not prove completion. Actual controller/stage terminal provenance, target gate77 bytes, full teacher/raw artifacts and GT readback remain absent. No runtime PASS, completed event counts or C admission is issued.
- **N2 — saved-evidence ceiling:** prefix record correspondence and cached tensor support do not reconstruct unsaved per-frame histories, first-reference network projections or every historic kernel. The revised source preserves these limits.
- **N3 — static compatibility:** Python3.8 grammar passes on14 files; no target Python3.8/Torch1.13.1 import, ABI, Torch load, NumPy calculation or whole readback execution occurred. Exact version/optimization/CUDA-uninitialized checks remain future runtime gates, not observations made here.
- **N4 — semantic scope:** geometric Train utility, C-only holdout and fixed human-input provenance do not establish identity truth, crop-out recovery, longer-term benefit, changed-policy causal labels, final same-weight nine metrics or goal completion.

Supported now: the exact v2 source closes B1/B2 and has no remaining identified source blocker in the reviewed scope. Its intended comprehensive raw-readback checks, independent label formula and separation from semantic certification remain coherent. No other code correction is necessary from this review.

Still unsupported: actual full R2 completion/readback, terminal semantic validity, real event/sign/eligible counts, C-current/future training or development results, Full152 final C, new benchmark results, or full-goal achievement. Source PASS may support preparing the exact readback version; actual execution requires the genuine terminal inputs and the later semantic result audit remains separate.

## Evidence and execution record

`DIRECT_INPUT_SHA256.json` and `direct_input_snapshots/` bind all26 v2 inputs. `V2_SOURCE_CHECKS.json` contains the47-file/48-entry archive preservation evidence, exact diff,14 AST checks and13 actual-condition fixtures. `BANK_STATIC_CHECKS.json` contains the fresh fixed-bank static checks and all152 inventory. `SOURCE_V1_V2_DIFF.patch` retains the two-line change. Final input/preservation checks and reports are sealed under this directory; the v1 FAIL evidence remains untouched.

No new reviewer check failed in this continuation. The v2 preparation receipt retains the executor's separate nonexistent `INPUT_SHA256_AFTER.json` read error (`bfa3e7`, exit1), and the old PATH-interpreter failure remains in the untouched v1 evidence. Neither is converted into an experimental failure or silently removed.

Counters for this continuation: SSH0, GPU/NN progress queries0, Torch imports0, NumPy imports0, experiment imports0, model constructors0, forward0, optimizer0, training/inference0, installations0, source mutations0, extra agents0. The stdlib fixtures executed only the actual scalar validity function and two assertions. Main was never run. The original timer PID17716/native8217 was not queried or changed, and no remote observation occurred.

This report is the same reviewer's bounded v2 continuation. Requested Astra/max remains routing information only; actual backend identity is UNATTESTED. Semantic acceptance remains same-family/provisional. `R2_terminal_raw_audit_complete=false` and `independent_terminal_audit_certified=false`.
