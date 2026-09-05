# M45 default-priority identity supervision

The original M44 max-IoU targets supervise121 already-correct fitting defaults
away from candidate0. Median immediate oracle gain is.029482;85gains are below
.05. M45 tests one training-target change on the same features and architecture.
This is a hypothesis about unnecessary changes, not proof of a unique cause.

| Stage | Status | Evidence |
|---|---|---|
| Training/runtime/spec | Frozen before fitting | spec0cf2fbd6ef521312b5bade3a52f7dbfb5c1c713571ff3e8a3b6d2e11cdd6c93e |
| Real training | Complete |63fit sequences/1511pairs;448739parameters;20epochs960steps |
| Target changes | Verified |121current and174preceding fitting labels; default preserved ifIoU>=.5, otherwise old max-IoU/NONE |
| Historical control pairing | PASS | Exact same fresh initialization and sample order as sealed M44 geometry; same optimizer and data |
| Final checkpoint reload | PASS | All590development logits exact after strict reload |
| Static development | Diagnostic only |meanIoU.446766,15changes,5rescues,1severe regression |
| Full recursion | Running | Two shards16567/16563frames, all22development sequences; PIDs17379/17380 |
| Pipeline | Running | PID17291; screensttrack_m45_default_priority_20260905 |
| Public evaluation | Not launched | Original STTrack-default recursive and subsequent low22 gates required |

Final association SHA:
`853a25fbc3c9ef12ab54442c30b27bab75f0b1fadcc9bcc82cbec6e8700ed59c`.
It requires the unchanged official STTrack backbone. Training-result SHA:
`e2647ea8d9738632d93f99d0108e638025b1c38894f8560cc9a532eb4f13f39d`.

The runtime remains `STTrackCandidateSet` in geometry mode, including default
template adaptation, full query state and11-way argmax. No inference threshold,
text, wider search or new model block is introduced. Changed matching targets
mean matching-accuracy numbers are not directly comparable with M44 labels.

The full recursive gate requires at least.01meanIoU improvement over STTrack
default, fewer low-overlap frames, no more persistent failure episodes, at
least3positive sequence means and protection of original zero-episode sequences.
Merely beating failed M44 does not pass. Both complete shards are sealed before
GT analysis. Low22 follows only after this gate; the same frozen weight reaches
all three full datasets only after clear low22 improvement.
