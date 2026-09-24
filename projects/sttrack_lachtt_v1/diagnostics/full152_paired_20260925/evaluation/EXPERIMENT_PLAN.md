# Full152 external evaluation plan

The M67-Control and M82-Category final checkpoints come from the paired 152-sequence DepthTrack Train pass (seed 2027). The final checkpoint after the full pass is selected before any external metric is computed.

For each model, use one checkpoint, one frozen category-only text protocol, the original Hann and dynamic-template rules, and the same bfloat16 initialization text observations already prepared for the historical full evaluation. DepthTrack Test50 and CDTB80 report official P/R/F. VOT-RGBD2022 full127 reports EAO/ACC/ROB and confirmed failed anchors over all 1765 start points.

The previous DepthTrack Train development 22 are part of the new training set. They may only be used as training diagnostics for these Full152 weights. Old 130-sequence M67/M82 results stay separate.

Run M67's DepthTrack and CDTB tracking on GPU0/1 in parallel, evaluate both, then run its four VOT shards across GPU0/1. Repeat for M82. Freeze source and bundle hashes before tracking; never choose a checkpoint using external metrics.
