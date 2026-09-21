# M88: search-conditioned local initialization reference

Date: 2026-09-21. Fixed seed2027 only. Main objective remains one RGB-D-L model meeting the three official dataset targets; this experiment is a development step, not a replacement objective.

## Problem and one hypothesis
The M84/M87 adapter first pools 16 initialization tokens into one bound vector per phrase. All 256 search positions share these bound vectors. Query-dependent template details cannot be selected after that pooling. M87 crop-only text replacement failed; it does not prove this pooling is the failure cause. M88 tests whether retaining local initialization choices for each current search position improves complete tracking and specific text contribution.

For modality m, existing projected search r_p, initial token z_j, phrase query q_k and scale 1/sqrt(64):

  A[p,k,j] = softmax_j((q_k + r_p) dot z_j / sqrt(64))
  B[p,k] = LayerNorm(q_k + sum_j A[p,k,j] z_j)

Use B[p,k] instead of the old spatially constant bound token in evidence and context. Existing modality weights remain computed from the OLD initialization-only bound vectors; no claim of current-depth reliability is made. Existing null normalization, RGB/Depth/context/product inputs and delta MLP remain. The shared centered two-pass computation remains F + D(x,q) - D(x,empty), with gradients through both branches. There are still 289154 stored trainable parameters; 768 final bias entries cancel. Computation increases because every position retrieves local references; not equal-FLOPs.

This is conditional cross-attention, not claimed as unprecedented or caption verification. It introduces no new modality, detector, semantic truth labels, target segmentation, temporal association, memory or recovery. Box-derived initialization RoI is weak spatial evidence and may include background.

## Frozen comparison and data
- Primary structure control: completed M84 Category, exact original text banks, same initial parameter tensors, loss, fit order, optimizer and budget. Do not use M87's changed text banks.
- Category-only training; no Empty training because the centered empty gradient cancels.
- Reuse M84 category, Empty and Swapped banks byte-for-byte. Swapped is a different-string content intervention, not guaranteed semantic contradiction. No caption call and no post-result editing.
- 130 DepthTrack Train fit sequences, 186694 track calls, expected5798 optimization windows; t0 initialization only, forward causal predicted crops, detached state. Raw competition and reliable same-state native spatial KL unchanged; valid/invalid GT policy unchanged. No additional loss.
- Final complete-pass checkpoint only. New architecture tag identifies the changed forward function even though parameter shapes are unchanged.
- Complete three development22 families: Category, same-weightEmpty, same-weightSwapped. Seal all predictions before subsequent GT metric loading. Repeated development data, not unseen external tests.

## Preflight and acceptance
CPU deterministic checks: shapes, explicit per-position loop agrees with vectorization, batch handling, masked slots, reference-token permutation equivariance, search permutation equivariance, dependence of bound vector on current query, finite gradients, exact centered Empty output and numerically cancelling Empty gradients. Use deterministic synthetic tensors for contract checks, never as performance ground truth.

Pre-freeze numerical clarification: the synthetic sum-loss test produced maximum Empty gradient2.384185791015625e-7 for both M88 and the originalM84 graph with identical tensors/weights. Mathematical cancellation does not imply bitwise-zero accumulated parameter gradients. The check records the actual gradient with tolerance1e-6; output Empty equality remains exact, final bias gradient remains exactly0. Search permutation error was2.384185791015625e-7 and is checked with atol/rtol2e-6. This changes test wording only; no model, loss, performance gate or data changes. Initial CPU bank enumeration failed on existing integer metadata before any tracking; enumeration now names only the five actual bank entries.

Real GPU preflight: inherited chair01 fit prefix; zero-Category and nonzero-weightsEmpty each101 frames vs independently initialized native (bbox/score/query/templates); then96 own-state fit frames/3 optimizer updates discarded. Confirm finite loss, gradients in RGB/Depth/text, frozen base and Empty equality after updates. This is correctness/trainability only. It does not select hyperparameters or a checkpoint. Estimate2-5 minutes.

Main run: one full fit (roughly4-6 GPU hours, estimate fromM87 plus additional attention), then three full development recursions (roughly1-2 hours across2GPUs). Poll240 seconds near milestones; do not repeatedly query loss or select an intermediate head.

14 frozen acceptance booleans: relative to M84, pooledIoU strictlyhigher, macroIoU no lower, lowframes no more, H10 no more (4); same four relative to native plus preserve native zero-H10 sequences (5); same four relative to Swapped (4); Empty/native per-sequence parity (1). All required for promotion. M82/M87 remain historical only. Report LOO, strictdamage/improvement and all negative rows descriptively, never change the gate afterward. Empty full bbox/score equality is a supplementary functionality check using the preserved native/M84Empty reference.

If the structure fails, seal it without LR/weight/window sweep. If it passes aggregate gates but real words do not beat controls, do not claim instance semantics. If no-quality candidates or crop-out events remain, this local attention does not supply recovery. Formal external input binding/evaluation is a later separately frozen step only after evidence supports proceeding; all three official datasets ultimately require the same final artifact/protocol.

## Execution tracker
1. Implementation and fresh same-family code review: pending.
2. CPU/GPU preflight: pending, no final training yet.
3. Freeze and launch complete training: only after preflight/review.
4. Three complete recursions, independent result audit and synchronized handoff.

Disk: preserve both Qwen, base and CLIP. Reuse immutable weights/banks by path, no large model copies. New final adapter about3.5MB. No external notifications.
