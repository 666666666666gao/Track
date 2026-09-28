# M99 completed-result analyzer review

**PASS after correction — source and CPU arithmetic review only.** No remaining blocking issue was found. This is not acceptance of a completed M99 run.

- reviewer: fresh Codex gpt-6-astra / max
- review_independence: same-family
- acceptance_status: provisional
- completed: 2026-09-28 10:04 CST; requested record suffix: 20260928_0958
- reviewed analyzer SHA-256: `812fd403a84aea11a0717e63f5c5d79b3e45aba61b4f3d7bd38e553026aa46ee`

Reviewed `analyze_ab_visual_control.py`, `M99_AB_VISUAL_CONTROL_PLAN.md`, `train_ab_visual_control.py`, `M99_ANALYSIS_PREPARATION.md`, the historical arithmetic receipt, and the actual M91 visual/M96 t0-search result JSONs. Applied the workspace's minimal-change rule and the installed local Codex review policy.

**BLOCKING issues:** None remaining. The original terminal expression passed `flush=True` to `json.dumps`, producing `TypeError: JSONEncoder.__init__() got an unexpected keyword argument 'flush'` after artifact writing. I reproduced this by executing the exact isolated AST expression on CPU. The executor moved the argument to `print` (current line 142); the corrected expression passed the same isolated check. No source patch was made by this reviewer.

**NON-BLOCKING issues:** None remaining. Report wording now correctly distinguishes recording the result SHA-256 from verifying the final-weight SHA-256 against the result. No additional verification infrastructure is requested.

Actual checks and findings:

1. Analyzer AST parsing and standard-library-only import passed. The registered result fields match the training producer: final train mode, seed 2027, batch 64, 12 epochs, 40 updates per epoch and 480 total updates, Empty-input/frozen-parameter disclosures, and final-weight digest. The result schema includes event strata; the analyzer's grouping and all four checks match the producer and plan. Failed capacity checks are retained, without promotion or retraining.
2. Recomputed both real historical sets: **495 unique events / 22 sequences each**, 990 records total. Counts, means, rescues and breaks match their saved overall summaries and the preparation receipt. All 22 sequence count totals and weighted means reconcile to each overall result. Both sets have identical event keys, native IoUs and oracle IoUs; native correctness is 268 and oracle correctness is 305. Saved historical result hashes and the updated analyzer-source hash match the receipt.
3. Historical arithmetic was M91 **270 correct, mean IoU 0.5285129019614272, 3 rescues / 1 break**; M96 **271 correct, mean IoU 0.527995017056137, 4 rescues / 1 break**. Deltas are correctly expressed as counts and `100 * mean-IoU difference`: +2 / +0.3742178878511804 pp and +3 / +0.3224293973221659 pp, respectively.
4. Isolated execution of the analyzer and producer's four-check expressions agreed on the archived summaries, including M96's failing transition check. Historical event rows do **not** contain strata, so historical strata were not independently reconstructed from rows. This does not conflict with the analyzer: its historical comparison consumes overall rows only, while future M99 rows contain strata from the training producer.
5. Source inspection confirms preservation of all M99 events, strata and sequences in CSVs; final-weight digest comparison precedes writing; historical comparisons are explicitly descriptive and differ in representation, origin and capacity. The report discloses reused development data and makes no semantic, recursive, C-effectiveness or official-acceptance claim.

No M99 result or final weight was available or loaded. The complete M99 command path, its actual weight verification and its future per-stratum arithmetic remain unexecuted. No SSH, GPU, training, inference, private human-label mapping, deployment or original-source edit was performed. Historical arithmetic checks are not M99 results; the terminal-output probe is only a serialization check.
