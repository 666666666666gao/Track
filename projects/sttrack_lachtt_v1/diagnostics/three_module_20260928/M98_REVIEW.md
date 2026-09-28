# M98 predeployment source review

- verdict: **PASS**
- reviewer: fresh Codex reviewer, gpt-6-astra / max
- review_independence: same-family
- acceptance_status: provisional
- reviewed_at: 2026-09-28 08:32 CST
- scope: local source review, saved receipt inspection, Python AST parsing and Bash syntax check only; no deployment, SSH, GPU execution, real tensor-cache loading, private human-answer access, training, or public evaluation

No concrete blocking defect was found in the current M98 implementation. This clears the source-review gate for the planned smoke. It does not establish that the smoke or full collection has passed.

Reviewed `M98_CONTEXT_CACHE_PLAN.md`, `collect_train_contexts.py`, `audit_train_contexts.py`, `queue_train_contexts.sh`, the updated staged experiment plan, M97's collector/prototype/review/completion receipts, M90's preparer/collector/completion receipts, M95's initial-origin collector/receipts, and the local native tracker/model and candidate/spatial observation APIs. The saved M84 native crop and backbone sources were inspected where the overlay does not carry those dependencies.

## Correctness findings

1. **The scope and lineage are preserved.** The collector selects the existing M90 shard rows without changing event selection or the fit/development split. M90 preparation/spec/input identities and every original sequence feature are checked before use. M95's two feature files are checked against their existing receipts before exact t0-region comparison. The saved M90 and M95 receipts share the same checkpoint and preparation; their shard sequence counts are 77/75, with 65/65 fit and 12/10 development. M90 records 1801/1701 events and 108335/110859 actual prefix calls, totaling 3502 and 219194. The larger preparation prefix estimates include initialization once per sequence; M98 correctly counts only real tracking calls.

2. **The t0 observation does not commit a new trajectory state.** Native `STTrack` sets the network to eval mode. The auxiliary call supplies `track_query_before=None`, so the model creates its query list locally; the collector does not assign the returned query to the tracker. Backbone patch embedding creates fresh token tensors from the template pixels. The collector checks unchanged frame, box and query immediately afterward, then verifies every real prefix bbox/score at the unchanged 1e-4-pixel/1e-6 tolerances. No optimizer or model update is introduced. The harmless backbone assignment of the same configured keep rate is not a history/template commit.

3. **The missing fields use M97's actual interfaces and arithmetic.** Both t0 and event crops use native `sample_target`, retaining its Python-float resize and actual padding mask. Re-decoding the same ten proposals must reproduce M90's float32 boxes and fp16 local RoIs exactly; t0 local RoIs must reproduce M95 exactly. Context extraction uses the same 2x expanded boxes and twelve `RING` positions as M97. `valid_samples` samples native padding support at those locations. `depth_fraction` reads the original depth PNG and stores the nonzero-pixel fraction inside the region. These are sampling support descriptors, not calibrated depth reliability or validity of an entire contextualized token.

4. **The sequence layout and audit agree.** The preparer emits sorted unique event frames; M98 checks those against the original feature metadata and visits them in the same order. For E events, the saved fields are fp16 `initial_context=[2,12,768]` and `contexts=[E,10,2,12,768]`; boolean masks are `[16]`, `[12]`, `[E,10,16]`, and `[E,10,12]`; float32 depth fractions are scalar and `[E,10]`. The CPU audit checks every field's shape, dtype and finiteness, depth bounds, event order, split, original feature linkage, distinct shard membership, per-sequence call counts and full totals. Existing region tokens, proposals, scores, geometry and initial references remain in M90/M95 rather than being duplicated.

5. **Command failures propagate.** The queue uses the same single-thread environment and unbuffered Python invocation as the native collection. With no `errexit`, both explicit waits run and both worker exits are recorded. Either nonzero worker exit prevents CPU audit and produces a nonzero queue exit. When both workers succeed, the audit's exit becomes the queue's exit. The output directory is created by the required preceding smoke. No new fallback, wrapper, or recovery machinery is needed for this planned execution.

6. **The stated resource arithmetic is consistent.** Event context alone is 1,290,977,280 bytes. All specified full-run tensors total 1,297,706,112 bytes, before small serialization metadata; the three-event smoke tensors add 1,143,776 bytes. This fits the executor-reported 3.08 GB free space without deleting existing artifacts. The saved native shard times are 6260.8/5654.8 seconds, so roughly two hours plus the subsequent audit is a reasonable planning estimate, not a measured M98 runtime.

The collector and auditor do not open subsequent GT, text banks, or private review answers, and do not generate labels. Imports of the M97 helpers/prototype only provide functions and the ring index; their guarded entry points are not run. No semantic training, checkpoint, recursive action, C change, or official result is added.

## Verification and remaining runtime gates

Independent local AST parsing passed for both new Python files using the existing `uv run --no-project python`; `bash -n queue_train_contexts.sh` passed. No M98 tensor or native runtime execution was performed by this reviewer.

Before full collection, run the prescribed shard-0 first-fit smoke on `cube04_indoor` and its first three existing events, `[10,12,14]`. Require its normal prefix/t0/event exact-reference assertions, finite tensors and expected shapes. As added to the plan, compare its t0 context/support fields and first event's four new fields exactly with the completed M97 cube04 smoke slices. This comparison belongs to the executor's one-time smoke launcher; it has not been executed in this review.

Only after that gate passes, launch the two full shards. Require both worker exit codes to be zero and the CPU auditor to finish with `complete_context_input_audit_only`, 152 sequences, 3502 events and 219194 prefix calls. Exclude smoke from those totals. Diagnose any exact-reference failure without weakening the checks.

No code changes are requested by this review. Completion will establish input coverage on the already selected Train states; independently reviewed semantic/instance labels, content-versus-visual evidence and any later recursive/public acceptance remain separate gates. This is same-family provisional acceptance, not cross-family acceptance.
