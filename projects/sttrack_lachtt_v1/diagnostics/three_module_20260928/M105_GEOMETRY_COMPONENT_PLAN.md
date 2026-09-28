# M105: same-final learned center/size attribution

M104 actually improved fixed-state development correctness272->286, with
15rescues/1break versus M101. The healthy group lost mobilephone02_indoor@65
(.510501->.448263), so only3of4predeclared checks passed. Fit additionally lost
pine01_indoor@1276. Keep every gain and harm; do not relax the healthy gate.

Use the exact completed M104 final and frozen M101 parent on all3039valid
DepthTrack Train cache events:2544fit and495development. Preserve Emptyfive-slot
inputs, ten native candidates, candidate indices, parent selection, local/context
encoding and original image/minimum10pixel clipping. No optimization, new weights,
GT forward gate, tracker state, template write, semantic teacher or public eval.

For each event compute the learned four-dimensional delta once, then read:
1. parent: unchanged original boxes;
2. full: both learned center and log-size deltas;
3. center_only: learned center, zero log-size delta;
4. size_only: zero center delta, learned log-size delta around original center.

Both interventions use the existing native-clipping decoder. Clipping can alter
the effective center/extent, so these are controlled component interventions,
not an additive decomposition of IoU or a newly trained two-arm ablation.
Ground truth is read only after decoding to calculate overlap; the original
GT-derived split/strata remain evaluation metadata, not inputs to the refiner.

Run fit on GPU0and development on GPU1, preserving original within-split batches64.
Before interpreting components, require exact replay of every existing full-row
selection, native/parent/full/Top10oracle IoU and all original stratum summaries.
Check zero correction on all30390boxes and unchanged frozen parameters/buffers.
Report all/marginal groups plus joint-stratum profiles, per-candidate overlaps,
all selected boxes/deltas and all gains/harms at IoU.5. No favorable variant is
automatically promoted into inference; choose the next Train-only training change
from fitting behavior, and reserve development as a check of the proposed change.

Fresh same-family Codex review is required before deployment, then actual two-GPU
readout and independent local arithmetic acceptance. Runtime should be under a
minute based on M104's32second two-process run. Poll near that estimate rather
than repeatedly. Do not rerun completed M104 or old Full152 experiments.

The independently reviewed initialization labels remain pending user confirmation.
This visual diagnosis can proceed without treating model prefill as human truth.
All official nine metrics and the unified final three-dataset target remain unchanged.
