# M96 same-budget visual-control reference comparison

M95 found development pure-cosine correctness 8/495 with the first-use
template, 9/495 with t0 template, and 178/495 with t0 search; native is268/495.
This supports a reference-origin check, not direct cosine deployment. Compare
two learned M91 visual controls using either first-use template or t0 search.
Both arms use identical native candidates, scores, geometry, labels, model,
seed2027, data permutations, batch64, AdamW3e-4, 12epochs, and final checkpoint.
Only the initial-reference tensor changes. Keep historical M91 results intact.

Use former-fit130 (2544 valid Train events) for optimization and report
former-development22 (495 valid events) once at the final epoch. IoU soft
labels supervise localization quality, not physical identity or semantic truth.
The development split is repeatedly used evidence. No public data, text, new
search region, memory, frame submission or official nine-metric evaluation.

Reuse `train_fixed_visual_selector.py`; its new explicit `--initial-origin`
selects the reference source. All other training behavior remains unchanged.
Run one-epoch t0-search sanity first, then two parallel visual arms on GPUs0/1.
Predeclare counts, rescues/breaks, mean IoU, and healthy/transition strata;
compare final t0-search against this run's first-use control and historical
M91 geometry/visual. No development-best checkpoint or hyperparameter sweep.

If t0-search does not improve final choice over this run's first-use control
while keeping development healthy-state breaks at zero, do not promote this
visual prototype to recursive tracking. Even passing that diagnostic cannot
replace independently reviewed Module A phrase/identity labels or prove final
tracking gains. This is a bounded reference-control experiment, not a module
contribution or another full152 residual-loss run.
