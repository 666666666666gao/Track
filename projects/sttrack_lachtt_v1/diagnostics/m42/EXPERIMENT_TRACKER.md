# M42 tracker — terminal record, 2026-09-05

| Stage | Status | Evidence / result |
|---|---|---|
| Source/Train manifest | Frozen |85 sequences;1,375 windows;63 fit/22 development sequences |
| ROI/model contracts | PASS | Geometry, default preservation, permutation, local-information control, finite training step |
| Causal replay smoke | PASS |120 frames;6 observations; exact boxes/confidence;2 real template writes |
| Feature collection | Complete | 85 sequences; 1,375 windows; 126,382 tracked frames plus 85 initializations; exact default boxes/confidence |
| Association fitting | Complete | Matched spatial/pooled; 20 epochs and 640 optimizer updates per arm; 50,858 parameters each |
| Static information gate | FAIL | Spatial and pooled make identical final choices on all 375 development windows |
| Integrity audit | PASS | Source/base hashes unchanged; all 13 head parameter tensors changed; checkpoint reload logits exact |
| Recursive development under M42 | Not launched | Failed information gate; receipt confirms zero GPU jobs |
| Distinct M43 performance follow-up | Complete: FAIL | Pooled primary; full 22-sequence recursive result in diagnostics/m43; M42 superiority remains failed |
| Low22 / three datasets | Not launched | No new formal tracking metrics |

Data preparation stopped once on the measured toy07 annotation/image count
mismatch. The corrected manifest binds real image/trace frame counts and
records the39 unused annotation rows. No GPU optimization had run at that
point. A transient SSH upload failed before controller launch; the read-only
check found no controller or collector processes, after which the two source
files were transferred and the one controller was started.

Development window mean IoU rose from default 0.414883 to 0.435760 for both
heads. Both rescued 10 windows across six sequences and caused zero newly
severe errors, but the 12 changed choices also include one joint failure and
one slight regression. This is not a recursive or formal benchmark result.
The spatial-versus-pooled hypothesis remains failed. M43 tests the separate
performance question of whether the already trained pooled control improves
default through full recursive tracking on the 22 development sequences.

Evidence: training_result.json, terminal_audit.json and
recursive_gate_receipt.json. Final spatial checkpoint SHA256:
88e37f12f4645a68a09cef135785113bbb85d9b07655728a2a1357192eb10c9a;
pooled checkpoint SHA256:
70908682c8049c1b444fe55949fbe5bb66ddbf9c9895375a017f205c87a73cca.
Both are association heads and require the unchanged official STTrack base.
The launch version is retained in history/EXPERIMENT_TRACKER.launch.md.
