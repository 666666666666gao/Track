# M43 tracker — terminal record, 2026-09-05

Primary pooled-versus-default performance gate: **FAIL**. Both arms completed
all 22 previously used Train development sequences, 33,130 frames per arm.
The source/weights and original primary choice are unchanged. No new optimizer
steps or public benchmark evaluations occurred.

| 路径 | mean IoU | 序列平均IoU | 严重低IoU帧 | 持续失败段 | 有失败段序列 |
|---|---:|---:|---:|---:|---:|
| STTrack default | 0.652226 | 0.684336 | 7397 | 75 | 19 |
| 已训练池化头（主候选） | 0.617064 | 0.676411 | 8553 | 77 | 17 |
| 已训练空间头（辅助） | 0.653608 | 0.688219 | 7298 | 75 | 18 |

These are Train development proxies. M42 spatial superiority remains failed.
Independent terminal auditing verifies all 44 trajectory files, exact default
prefixes before the first override, and all reported IoU/failure counts.
See result.json, terminal_audit.json and per_sequence.csv for complete evidence.

Next: no public launch; inspect actual recursive failure evidence.
The launch tracker is preserved in history/EXPERIMENT_TRACKER.launch.md.
