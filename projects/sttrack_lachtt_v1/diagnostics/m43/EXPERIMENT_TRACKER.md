# M43 tracker — launch record, 2026-09-05

| Stage | Status | Evidence |
|---|---|---|
| Rationale and protocol | Frozen | M42 spatial superiority failed; separately test the useful trained pooled control |
| Training | No new training | M42 final checkpoints; unchanged runtime and inference settings |
| Recursive inference | Running | 22 existing Train development sequences; 33,130 frames per arm; independent complete state |
| Primary comparison | Pooled versus default | Fixed before new trajectories; no post hoc model switching |
| Secondary comparison | Spatial versus default | Diagnostic only |
| Post hoc analysis | Pending sealed trajectories | Mean IoU, low-overlap frames, failure episodes and successful-sequence protection |
| Low22 / full benchmarks | Not launched | Require the frozen primary performance gate |

The original M42 gate remains failed and its controller launched zero recursive
GPU jobs. M43 is an explicitly separate exploratory performance follow-up on
previously used development sequences, not a fresh holdout or proof of a local
spatial information benefit. Ground truth enters normal initialization and
post hoc analysis; subsequent inference uses observed images and predictions.
