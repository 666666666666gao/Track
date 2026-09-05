# M49 supplementary training-cache control

Frozen after the VOT onset-only candidate diagnosis, before the following training-cache comparison. Keep the same three fixed rules and designate maximum IoU with the previous predicted box as primary; do not fit a threshold or switch primary after observing results.

Use all2101 sealed M44 pairs from DepthTrack Train (63 fit sequences /1511 pairs,22 repeatedly used development sequences /590 pairs), including healthy, intermediate, transition, late-low and unavailable strata. Verify all85 feature hashes against the original collection receipts. Geometry and previous predicted states only enter selection. Read GT labels only after writing every selected rank and candidate box to a sealed JSON file.

Report separately by split: valid-GT pair count, native-correct and selected-correct counts (continuous box IoU >=0.5, matching existing cache training geometry), native-wrong to selected-correct changes, native-correct to selected-wrong changes, harmful changes to IoU <=0.1, mean IoU, and all sequences. Unavailable GT is excluded from IoU aggregates and counted. The rules have no abstention action; do not claim they recognize absence. Compare source boxes with native predicted states for causal alignment.

This is a cached single-frame feasibility screen, not recursive tracking, not a held-out final test, and not directly comparable to bounded rasterized VOT IoU. If primary has more correct-to-wrong than wrong-to-correct changes or lower mean IoU, do not launch it as an unconditional recursive policy. A next trained/conditional motion model would require its own protocol and the existing recursive protection gates.
