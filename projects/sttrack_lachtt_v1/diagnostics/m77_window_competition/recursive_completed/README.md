# M77 completed fit and recursive development evidence

Both seed2027 training arms completed130 sequences,186694 calls and5798 optimizer steps. The two final heads completed the entire reused DepthTrack Train development22. Sealed outputs were independently recomputed and all15 frozen conditions were checked; only2 passed for the Category head. Numerical integrity is not method promotion or independent model review.

Category pooled/macro IoU:0.658378645835/0.668491236846;7188 low-overlap frames,80 H10 episodes. The same-budget Empty training control:0.699518640826/0.707317747098;5905 low-overlap frames,70 H10 episodes. There are4 distinct Category H10 intervals totaling250 frames where Native and Empty remain correct on every frame; the two reference rows must not be double counted.

See handoff_append.md, recursive_result.json, completed_training_audit.json and per_sequence.csv. The exact3x102 Category prefix passed; fixed-Category-head Empty/Swapped content runs were live at the timestamp in process_snapshot.json. Their results remain pending in this snapshot and cannot override failed development conditions. No public evaluation result or additional seed is claimed. Later content results will be published separately.

This directory's EXPERIMENT_TRACKER.md supersedes preserved launch snapshots. Master section5.149 records architecture scope, completed training, metric definitions, negative results and the next diagnostic. Original weights, full traces and predictions remain on the server; this export does not copy them into Git.
