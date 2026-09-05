# M42 execution plan

1. Freeze Train-only sequence ledger, window manifests and matching arms.
2. Verify ROI coordinates, permutation behavior, local-versus-mean information,
   a finite optimizer step and120 frames of exact default replay. **Passed.**
3. Collect1,375 causal observations on two RTX3090 devices. Replay both native
   boxes and confidence on every frame. **Launched 2026-09-05.**
4. After both successful terminal receipts, automatically fit spatial and pooled
   association heads for20 epochs each; save final weights and fixed heldout
   metrics. Any collector failure stops the fitting stage.
5. Audit source/checkpoint bindings, coverage, selected-IoU gains, rescues,
   harmful changes and per-sequence effects. Check the frozen information gate.
6. Only on pass, run paired recursive development tracking with each branch's
   own boxes, queries, templates and reference bank. No future frame or GT may
   enter inference. Healthy windows remain in the denominator.
7. Only after recursive gains, evaluate frozen low22 versus M39 default.
8. Only after clear low22 improvement, evaluate the same resulting checkpoint
   on DepthTrack Test, CDTB and full127 VOT; record one model identity across all
   reports and append only new results to the canonical handoff.

The run is `/root/autodl-tmp/sttrack_m42_local_spatial_v1_20260905`.
Controller: `screen sttrack_m42_train_20260905`, with collection on GPUs0/1 and
subsequent small-head fitting on GPU0. Launch throughput predicts about one
hour of collection plus fitting. Inspect logs at meaningful milestones; do not
poll every few seconds. No automatic public evaluation is attached to this run.
