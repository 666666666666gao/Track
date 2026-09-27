# M96 predeployment code and scientific-integrity review

- verdict: PASS
- reviewer: fresh Codex reviewer, `/root/m96_origin_control_review`
- model: gpt-6-astra
- reasoning_effort: max
- fork_turns: none
- review_independence: same-family
- acceptance_status: provisional
- scope: local source and saved-artifact review; no deployment, training, or GPU inference performed by this reviewer

No concrete blocking defects found. No code changes requested. The implementation may enter the planned one-epoch t0-search sanity; both 12-epoch arms remain gated on that sanity completing successfully.

Reviewed `M96_ORIGIN_CONTROL_PLAN.md`, the working-tree diff and full source of `train_fixed_visual_selector.py`, the M91 plan/review, the M95 collector and audit, M95 completed receipts and cosine summaries, `preparation.json`, and the existing IoU helper. The parent confirmed the model, effort, and fresh-context spawn route above.

## Findings

1. **The reference substitution is correctly scoped.** `load_initial_search` (lines 80-94) loads both complete non-smoke M95 shards, uses their existing source and feature checks, rejects GT-loaded or committed-query artifacts, and maps each unique sequence to its stored search RoI. In `load_panel` (lines 37-40), that tensor replaces only the initial reference. Both sources use float conversion and pooling over the same 16-sample dimension. M95 produces `(2, 16, 768)` reference tensors, so pooling gives the existing `(2, 768)` input expected by the selector. Candidate tensors, native scores, geometry, ordering, boxes, and targets follow the unchanged path.

2. **The two learned visual arms hold the stated training budget fixed.** The CLI restricts t0-search to the visual variant and requires its origins directory (lines 161-174). Model architecture and initialization, AdamW, batch size 64, learning rate 3e-4, 12-epoch default, and seed 2027 are unchanged. Each epoch's permutation uses the same explicit `2027 + epoch` seed. Loading the alternate cached references introduces no random sampling. AST comparison against repository HEAD confirms that `Selector`, `evaluate`, and the complete epoch-training loop are unchanged.

3. **Optimization and evaluation use the intended split and GT targets.** Only `panel['fit']` supplies inputs and targets inside the optimization loop (lines 185-196). `load_panel` checks label/feature split agreement, skips invalid GT, computes IoU against the dataset's current `xywh` target, and requires 2,544 fit and 495 development events. The pre-training development call only asserts native zero-residual count parity; this model has no BatchNorm or dropout state to update there. Development is then evaluated after the fixed final epoch, with no intermediate development scores or checkpoint selection. Final counts, rescues, breaks, mean IoU, and supplied strata are computed directly from those GT IoUs.

4. **The M95 evidence supports the bounded reference experiment.** Saved full receipts contain 152 unique sequences, split 130 fit / 22 development, with complete non-smoke status, `GT_loaded=false`, `auxiliary_query_committed=false`, and zero recorded first-native-frame box/score errors. The M95 source constructs the search reference from the legal first RGB-D frame and initialization box. Its recorded development cosine counts are 8/495 for first-use template, 9/495 for t0 template, and 178/495 for t0 search, while native is 268/495. These agree with the M96 plan; they justify checking reference origin without establishing that a learned selector will improve tracking.

5. **Reporting and interpretation remain appropriate.** `result.json` now names `initial_origin` and retains the fixed final weights, hyperparameters, fit/development metrics, and training history. The plan explicitly recognizes repeated development use and limits the claim to a fixed-state visual control. No text, physical-identity target, public evaluation, new search region, recursive action, or memory update is introduced by this diff. The intended comparison is final t0-search versus the newly run first-use arm; historical M91 results remain contextual evidence.

## Validation and execution gate

- Python source compilation passed using the installed uv-managed CPython 3.13 interpreter. The default `python` launcher was unusable (`No pyvenv.cfg file`); this was a local launcher issue, not a code failure.
- AST invariance checks passed for the selector, evaluation function, and complete training loop.
- Parsed M95 receipts establish 152 unique references and the 130/22 split; parsed cosine output establishes all three variants' 2,544/495 valid-event counts and 10,506 total variant-event rows.

The M95 tensor files are not present in this local completed-artifact folder, so this review did not execute the new tensor loader or the GPU training path. The planned one-epoch t0-search sanity must confirm actual artifact loading, finite training output, the expected event counts, `initial_origin=t0_search`, and the final report before both full arms run. Use separate M96 outputs and retain the historical M91 artifacts, as planned.

No non-blocking code changes are proposed. Passing this review or the reference-control comparison does not authorize promotion to recursive tracking or replace independently reviewed Module A semantic/identity evidence. No private candidate-answer manifests were opened, and no external model reviewer was used. This PASS is same-family and provisional.
