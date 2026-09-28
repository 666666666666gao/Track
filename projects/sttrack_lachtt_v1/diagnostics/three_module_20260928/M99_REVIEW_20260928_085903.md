# M99 predeployment source review

**PASS — M99 source-review gate only.** No concrete blocking defect was found, and no code changes are requested.

- reviewer: fresh Codex gpt-6-astra / max
- review_independence: same-family
- acceptance_status: provisional
- scope: local source and saved-receipt review, AST parsing, CPU model instantiation and parameter-freeze inspection. No server connection, actual tensor-cache loading, optimizer step, GPU execution, deployment, or source-file edit was performed.

Reviewed `M99_AB_VISUAL_CONTROL_PLAN.md`, the appended M99 section of `EXPERIMENT_PLAN.md`, `train_ab_visual_control.py`, and the specified M97/M98/M95 dependencies. Also checked M90 preparation, collection and IoU code, native candidate decoding, and saved collection/probe receipts.

**BLOCKING issues:** None in the reviewed source.

**NON-BLOCKING issues:** None requiring an additional code change.

The implementation correctly preserves the following contracts:

1. **Inputs, alignment and splits.** `load_inputs` joins each M98 frame to its M90 event index before assembling RoIs, geometry, scores, boxes and IoU targets. It checks original/context feature receipts and matching split metadata. Full mode requires M98 `queue.exit == 0` and the complete `152/3502/219194` audit. That auditor verifies full event order, membership, shapes, dtypes, finiteness and support bounds. M95 references are loaded through the existing receipt-checked t0-search loader. Smoke mode requires the exact three fit keys `cube04_indoor@10`, `@12`, `@14` and no development panel.

2. **No label or development optimization leakage.** Candidate IoUs come from actual Train GT through the existing `overlaps` helper. Invalid current GT is excluded. IoUs and strata are absent from model inputs; optimization indexes only `panel['fit']`. Development labels may be loaded, but development outputs are evaluated only after the fixed training budget. The saved coverage receipts substantiate 2544 valid fit events and 495 valid development events.

3. **Empty behavior and capacity disclosure.** Only `bank['empty']` is used. All five slots receive the same encoded Empty vector with an all-active mask independent of sequence. No category, attribute or sequence mask enters the model. The unchanged M97 architecture retains local/context tokens, positional and role encoding, modality support, visual cross-reading and candidate interaction. Exact Empty-versus-visual selection equality is asserted during optimization and final evaluation.

4. **Freezing and optimizer scope.** `freeze_unused` freezes the text projection, phrase slots, phrase/text attention, phrase-evidence module, semantic/phrase adapters and observation head. Shared visual modules remain optimized. AdamW receives only parameters with `requires_grad=True`; exact frozen-value equality is checked afterward. Independent CPU instantiation confirmed **241,994 total parameters, 142,086 optimized parameters and 99,908 frozen parameters**. The plan and output explicitly state that this is not an effectively capacity-matched language ablation.

5. **Loss and initialization.** The objective is exactly visual-selection BCE plus visual-quality BCE against candidate GT IoU, with weight 1 each. It supplies no identity, semantic-state or observability supervision. The selection residual's final layer starts at zero. Native decoding emits descending response peaks, and the saved base scores are their log responses, so candidate 0 is the native choice. The script checks exact initial score equality and candidate-zero selection.

6. **Sanity behavior.** Both sanity modes perform exactly two temporary updates on the same initial batch. On the second step they require nonzero visual-read and candidate-relation gradients and finite gradients throughout. They check unchanged frozen parameters and save neither a checkpoint nor development performance. CPU smoke uses three actual examples; subsequent GPU sanity uses 64. GPU peak allocated/reserved memory and optimization time are recorded.

7. **Fixed final training and diagnosis.** Training uses seed 2027, deterministic epoch permutations, batch 64, AdamW at `3e-4`, and exactly 12 epochs—480 optimizer steps for 2544 fit events. No development checkpoint search, seed sweep or loss-weight scan exists. Final evaluation uses candidate-versus-GT IoU and records every valid development event, strata, native/selected/oracle IoU, selected index, rescues and breaks. All four predeclared capacity checks are recorded as booleans; failure does not suppress the complete result or substitute another epoch.

Verification performed: all six specified Python implementation/dependency files passed AST parsing. The existing PyTorch `1.13.1+cpu` environment successfully instantiated the model, applied the actual freeze function, verified its frozen values and confirmed the parameter counts and zero selection head. No actual-input or gradient success is claimed by this review.

**Remaining execution gates:**

- During M98, run only the authorized CPU smoke on the completed `[10,12,14]` data. Require successful exit, two steps, three examples, finite nonzero named gradients, unchanged frozen parameters, and no checkpoint/development evaluation.
- Before any M99 GPU work, require M98's terminal queue exit 0 and full input audit.
- Then require successful two-step GPU batch-64 sanity before the fixed 12-epoch control.

Passing these gates supports the planned visual-capacity diagnosis. It does not clear semantic fitting, effective-capacity matching, recursion, C changes or official/public acceptance.
