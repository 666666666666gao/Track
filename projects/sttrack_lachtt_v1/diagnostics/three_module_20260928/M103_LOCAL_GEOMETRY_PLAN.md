# M103: cached B geometry diagnosis before visual refinement

Previous M102 is terminal and fails teacher screening; do not repeat it. M101's
Empty visual capacity passes four predeclared checks, while its fixed native
Top-10 selection still misses223/495development events. M102 already decomposes
these into33selection/19Top-10 omissions/65center-in dense misses/106center-out
dense misses. That decomposition is Train-only and must not be restated as
official VOT failure statistics or physical identity labels.

This stage is read-only CPU arithmetic on all3039valid M90 native cached events
(2544fit130,495reused development22). It does not initialize a tracker, load a
network, recompute crops, optimize, write checkpoints or use public-test data.
Read the existing training_labels separately from inference cache. Preserve
every event, split and group. Reuse the actual overlaps and crop-origin helpers.

For every original Top-10 and full256box, compute IoU with the training GT.
Then make two explicitly undeployable geometric counterfactuals:
1. Replace predicted centers by the GT center, retain each predicted size.
2. Replace predicted sizes by the GT size, retain each predicted center.
Apply the original image-boundary/minimum10pixel clipping in both cases.
Also clip the GT box itself with that same rule to expose size/boundary limits.
Compare all maxima at IoU>=0.5, with original boxes unchanged.

These controls measure sensitivity/capacity, not causal module recovery.
They cannot supply predicted corrections. In particular, copying the GT center
is not a learned search mechanism. Center-out does not mean the whole object
is invisible; overlapping boxes can still be useful.

For local full256miss events with GT center inside, summarize both interventions,
Top-10 overlap>=0.1, full GT inside crop, GT center in any current candidate box
or its2x context, and GT side below10pixels. These measurements determine whether
a candidate-local visual refiner has observable overlap to learn from, or needs
a broader search feature readout. Report the intersection of those conditions,
not just each marginal. GT overlap is localization supervision, not semantic
or same-instance annotation.

Predeclared training eligibility for a possible next refiner is:
GT entire box inside the current crop AND native Top-10 maxIoU>=0.1.
Use exactly one highest-IoU native candidate per eligible fit event as a standard
single target assignment; do not train every wrong candidate to move to the GT.
This eligibility is only counted here. No refiner is automatically launched.
If implementing next, retain the frozen M101 visual/selection checkpoint,
separate its geometry branch, initialize corrections to zero and verify native
box/selection equality before any optimization. No new semantic truth, C,
recursive action or public evaluation can follow from this arithmetic alone.

Expected runtime: roughly30-60seconds from the previous22-sequence CPU read
(10.4seconds including connection and Torch imports). No GPU poll is needed.
Fresh source review and actual complete row/parity checks precede acceptance.
Preserve JSON summary and every3039event row; no prompt/seed/threshold scan.
The same final-weight Full152/three-dataset nine-target goal stays unchanged.
