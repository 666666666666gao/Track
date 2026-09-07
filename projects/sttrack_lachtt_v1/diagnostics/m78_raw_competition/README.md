# M78 raw-response competition control

Single seed2027. Two new arms: RawCategory and RawEmpty, with the same sealed initialization, data, optimizer, budget and 289154-parameter adapter as M77. The training-only competition uses raw scores (including hard-negative ordering); inference retains native Hann, query and template updates. Sealed M73 and M77 runs provide no-added-ranking and Hann-ranking references. No extra seeds or repeated baseline training.

Causal checks passed before freezing and detached launch. This directory is a launch snapshot, not a completed result. See EXPERIMENT_PLAN.md for ten prospective native/paired-Empty gates, fixed-head content tests, scope and failure interpretation. Historical protection gates are not accumulated or retroactively changed. No public benchmark run is started by this queue.

The master section5.151 records M77 completed negative findings and why this comparison is needed. No inference code or weights are included here; integration.json binds all160 unchanged inference files. Both Qwen models are preserved. Requested independent model review remains unavailable; executable prechecks are not a reviewer PASS.
