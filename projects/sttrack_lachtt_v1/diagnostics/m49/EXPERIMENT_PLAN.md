# M49: adjacent-frame motion, scale, and template-update diagnosis

Frozen before measurement on 2026-09-05. This follows the user's motion/scale steering after M48; the previously discussed independent-branch M49 was never implemented.

**Claim to test:** small physical motion and gradual scale changes can coexist with tracker-induced center/scale jumps. Determine which signal actually changes at the 124 native STTrack M39 failure onsets before adding a motion or template-update rule.

The baseline is M39 STTrack default: original checkpoint, query window 4, search factor 4, template factor 2, dynamic template update every 50 tracking steps when confidence is strictly above 0.75. No new inference, optimization, architecture, or public benchmark run is performed.

## Measurements

1. For all unique chronological adjacent GT pairs in the frozen low22, measure center displacement in pixels and relative to the previous GT geometric-mean side, linear scale ratio `sqrt(area_t / area_prev)`, and width/height ratios. Invalid GT endpoints are excluded and counted; a visibility gap is never interpreted as one-frame velocity.
2. For all 124 failure anchors, follow the actual forward/backward evaluation direction. Compare the GT motion to the prediction jump and prior localization error, using the same previous-GT normalization. Verify the vector identity `GT_current - pred_previous = (GT_current - GT_previous) + (GT_previous - pred_previous)`.
3. Compare exact factor-4 crop geometry from the actual previous prediction, from perfect previous GT (oracle diagnostic only), and from the fixed causal prediction `center_t = 2*center_prev - center_prevprev`. Width and height stay at the previous predicted values. No thresholds are fitted. Report healthy paired frames as well as failure onsets, including coverage gained AND coverage lost. This is geometry, not evidence that a correct candidate is generated or that recursive tracking improves.
4. Reconstruct deterministic native template writes from the sealed confidence files and tracking-step index. At each onset use only writes strictly before it. Report template age, IoU when written, and GT scale change since that template. Separate writes before/after the first confirmed failure and count how many bad writes occur after an existing run of at least 10 low-IoU frames.

Failures are anchor events; repeated source-frame transitions are also deduplicated for physical-motion summaries. Healthy controls require previous and current bounded VOT IoU at least 0.5. They are descriptive controls, not independent statistical samples. Pixel/frame is not physical speed, and GT axis-aligned box changes may reflect pose, occlusion, or annotation changes.

## Decision and next validation

- Report every low22 sequence, all 124 failure events, and the seven protected sequences; do not select only successful examples.
- Motion extrapolation is worth a training-only recursive trial only if it improves onset geometry without a larger number of lost healthy coverages. Even a positive result does not authorize a hard displacement cap or a VOT-tuned threshold.
- An adaptive template proposal needs evidence that the latest usable template lags appearance/scale before failures, rather than treating writes after tracking loss as the initial cause. Pure uniform scale change is already largely normalized by template cropping and resizing.
- Any new learned module is trained on the existing DepthTrack Train sequence split first. Retain the established training recursion, low22, and full three-dataset promotion gates. Local online text remains a separate variable and receives no new rollout in this diagnosis.

Expected cost: CPU analysis of the existing 303 trajectories (about 220k steps), no GPU hours. Outputs: protocol, source, input/output hashes, per-onset/per-sequence/template-write CSVs, and aggregate JSON. No new EAO/ACC/ROB claim.
