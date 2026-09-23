# Active goal supplement — 2026-09-24

User authorization supersedes the earlier experiment scope.

1. Finish the frozen M67 Control evaluation first, then M82 Category: DepthTrack Test50, CDTB80, VOT-RGBD2022 full127/1765 anchors. Preserve completed M67 DepthTrack outputs. Use each model's fixed final checkpoint, shared text protocol, and bound inference configuration across all three datasets.
2. After both complete, compare official full metrics. Stop research if at least one single complete model meets every target: DepthTrack P/R/F >=65.2/64.9/65.1; CDTB >=72.9/75.6/74.2; VOT EAO>77.9, ACC>82.1, ROB>93.7. Never combine results from different models.
3. If neither model meets all targets, inspect per-sequence deficits and failure mechanisms before implementing new ideas. Preserve original results; no retrospective relabeling or selection of external-test checkpoints. New training remains DepthTrack Train only, fixed seed2027; no multi-seed. Existing M67 seed2026 remains explicitly authorized.
4. Monitor long training/evaluation once per hour, after a single startup health check. No frequent diagnostic polling. VOT runner polling set to 3600 seconds.
5. Maintain the remote and local master handoff and publish authorized code/results to the existing Track repository. Report complete official metrics and unresolved deficits honestly. Current goal remains active; do not mark complete on launch, partial metrics, or development gains.
6. Current authorized server is connect.nmb2.seetacloud.com port43811. Credentials are not stored in this supplement.