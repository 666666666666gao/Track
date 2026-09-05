# M42 recursive development protocol — frozen before trained outcomes

The static fitting specification remains unchanged. A separate controller
waits240 seconds between completion checks; it launches no GPU work unless
the original collection/fitting controller exits successfully and all three
static information criteria pass.

Each arm then tracks all available frames of the22 existing fold5 development
sequences from the normal first-frame initialization. The spatial and pooled
arms have independent crop centers, query lists, template images and native
reference banks. The baseline is the already sealed full152 default trace,
with its two source-file hashes bound by `recursive_spec.json`.

The template interval and threshold remain50/.75. When a different candidate
is selected, its own response score controls the template write. This keeps
the confidence attached to the box actually being written. Default choice0
and NONE preserve the original box, confidence, templates and query exactly.

The new runtime passed120 frames of simultaneous comparison with an ordinary
STTrack instance: boxes, confidence, template tensors and query tensors all
matched exactly. Two additional forced rank2 probes verified that a selected
alternative uses its own size/offset box and its own score. These are software
checks with zero trained association updates, not tracking improvements.

GT files are opened only by the posthoc analyzer after both trajectories are
sealed. It reports continuous xywh IoU, severe-low-IoU frames and persistent
failure episodes. Initialization and invalid-GT frames are excluded. Invalid
GT breaks a failure span; an episode requires10 valid consecutive frames with
IoU≤.1. These are Train development proxies, not VOT ROB or toolkit metrics.

All five frozen recursive criteria are required:

1. Spatial global mean IoU exceeds both default and pooled.
2. Spatial has fewer severe-low-IoU frames than default.
3. Spatial has no increase in persistent failure episodes.
4. At least3 sequences have positive mean IoU gain over default.
5. No default sequence with zero failure episodes acquires one.

Per-sequence means and aggregate frame/sequence statistics are retained. A
pass permits freezing the subsequent low22 comparison against M39 default;
no automatic low22 or full benchmark is attached to this controller. A failed
static gate creates a zero-GPU-job receipt and stops this recursive stage.

Both association-final weights remain paired with the identical official base
weight and frozen runtime/config identity. Historical SRTrack or SUTrack
numbers cannot be attributed to this new model.
