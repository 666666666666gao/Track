# M42 tracker — launch record, 2026-09-05

| Stage | Status | Evidence / result |
|---|---|---|
| Source/Train manifest | Frozen |85 sequences;1,375 windows;63 fit/22 development sequences |
| ROI/model contracts | PASS | Geometry, default preservation, permutation, local-information control, finite training step |
| Causal replay smoke | PASS |120 frames;6 observations; exact boxes/confidence;2 real template writes |
| Feature collection | Running | PIDs8097/8098; two RTX3090 GPUs;40/45 sequences by shard |
| Association fitting | Pending after collection | Matched spatial/pooled, fixed20 epochs each |
| Static information gate | Pending | No new accuracy claim |
| Recursive development | Not launched | Requires information-gate pass |
| Low22 / three datasets | Not launched | No new formal tracking metrics |

Data preparation stopped once on the measured toy07 annotation/image count
mismatch. The corrected manifest binds real image/trace frame counts and
records the39 unused annotation rows. No GPU optimization had run at that
point. A transient SSH upload failed before controller launch; the read-only
check found no controller or collector processes, after which the two source
files were transferred and the one controller was started.

New checkpoints and terminal numerical results are absent at this launch
record. Update this file from actual receipts after completion; retain the
launch version in the run history.
