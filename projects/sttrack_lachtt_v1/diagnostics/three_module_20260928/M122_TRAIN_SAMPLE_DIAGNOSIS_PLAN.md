# M122 historical training-state diagnosis

Status: source review pending; this document contains no new benchmark results.

Run the new stdlib-only `analyze_m122_train_samples.py` on the previously sealed third-pass training logs and separately exported DepthTrack Train ground truth. The two original models, 32 local evaluation sources and the running full VOT schedule remain frozen. This diagnosis does not execute a neural network or query current GPU progress.

Inputs: the SHA-bound original Train152 manifest, all 152 original groundtruth text files, two sets of 456 sequence records and 1998 sparse state records, and dimensions/hashes of their 666 unique sampled RGB frames. The exporter used CPU only; raw RGB images are not in the local diagnosis package. Input ZIP is 1036355 bytes, SHA 998b5d8afae7c14f8dd4c15104f05e2b1f74deceb6bd0a0da0307c2944b8c565. Native SSH59431 and SCP99680 exited 0; source export made zero NN executions/progress queries. The 154 copied input files were verified locally.

The full sequence logs provide exact call, supervision, GT-window-intersection and template-write totals for each of the three training passes. The saved states are frame 1, every 500th frame and each sequence's final frame. Join each state to its original dataset GT, use the recorded crop origin, and check the documented native same-selected-cell template rule. Separate sampled severe localization errors, crop-center exclusion, GT-window nonintersection, and writes with invalid GT. Check learned quality and observation outputs against their corresponding geometric labels in these historical samples.

Outputs: six model/pass summaries, all 912 sequence/pass ledger rows, and all 3996 diagnosed sampled states. The private plan pins every input SHA; files are rechecked before output. No benchmark P/R/F or VOT metrics are calculated, no final checkpoint is selected, and no inference threshold is changed.

Boundaries: training samples come from changing weights during optimization, not evaluation with a fixed final. Their mean IoU/calibration is not held-out performance. Sparse samples cannot establish all-frame failure or all-write error rates. Invalid GT is unknown rather than proof of physical absence. CPU float32 overlap reimplements the loss arithmetic for diagnosis and does not assert GPU bitwise replay. Image shape provenance is the recorded remote CPU export, without independently replayable RGB bytes locally. No text, module or template causal effect is claimed from these logs alone.

Use reviewed findings to define Train-only events for the next mechanism experiment after both current VOT evaluations finish. Do not choose thresholds or sequence rules from public test failures. Source review precedes execution; result review precedes claims/publication of new diagnostic numbers.
