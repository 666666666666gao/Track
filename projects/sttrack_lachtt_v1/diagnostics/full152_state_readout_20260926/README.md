# M82 Full152 same-state head readout

Replay four posthoc CDTB cases with the sealed M82 Full152 category weight and verify every produced bbox and score against the official full-track files. At preselected frames, read four score/size/offset combinations from the same crop, template, query, backbone forward, and category history. Only the original adapted output commits tracker state. GT is not loaded in this readout; IoU can be evaluated afterward against sealed labels.

Run `analyze.py` only after the readout is sealed; it opens the already bound CDTB labels and reports per-frame overlap and a small summary. This diagnostic separates whether a frame's change comes mainly from spatial score selection or size/offset geometry. It cannot prove a causal training mechanism or predict the complete counterfactual trajectory, because alternate readouts do not commit state. The four cases were selected after external results, so they are mechanism examples rather than an unbiased benchmark.

Prepared separately from the frozen Full152 evaluation root. `run_readout.sh` uses GPU 1 only after both VOT shards assigned to that GPU finish; the remaining VOT shards continue on GPU 0. The diagnostic does not write into the VOT evaluation root.
