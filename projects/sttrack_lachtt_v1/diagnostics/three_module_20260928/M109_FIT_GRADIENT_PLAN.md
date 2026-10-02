# M109 fit-only objective diagnostics

M108 did not establish a positive semantic increment. Do not change its finals,
labels or gates. Inspect three distinguishable explanations using only its
130 fit sequences: localization agreement of the 11 weak A/B choices; gradient
magnitude imbalance; opposing loss gradients on the actual last-epoch batches.
IoU disagreement is not sufficient to mark a physical-identity judgment wrong.

Reuse the native environment, cache, bank, parent, finals, masks and loss formulas.
GPU0 reads the generic arm; GPU1 reads the reviewed-text arm. Each reads its
zero-readout M101 initialization and its own fixed final. Recreate epoch12's
seed2038 batch order, and measure gradients of localization, reliable-native
preservation and weak-pair losses with their existing unit coefficients. Preserve
the exact same 95,683 trainable coordinates as M108. Compute norms, cosines and
weak-to-combined-baseline norm ratios. Zero-vector cosines/ratios are undefined
and stored as null, rather than fabricated as alignment or conflict.

No optimizer, backward accumulation, parameter update, new checkpoint, development
metric or public benchmark evaluation. Assert all parameter/buffer bytes unchanged
and parameter.grad remains None. Source-provenance hashes bind inputs. This is a
recomputed initial/final diagnostic, not a reconstruction of historical training
gradients. Keep individual weak choices/IoUs private; publish aggregates/batches.
The common loader constructs both splits; only fit tensors enter the model and
gradient analysis. No development model output is evaluated or used to choose changes.

Expected runtime around one minute for both GPUs, given M108's measured 60.6-second
whole workflow. Inspect near 180 seconds, not repeatedly. Based on measured fit
evidence, decide whether an objective change is warranted; do not tune against
held-out events or assume one local conflict explains all official failures.
