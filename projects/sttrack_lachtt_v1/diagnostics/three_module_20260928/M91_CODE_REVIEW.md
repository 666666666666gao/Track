# M91 fixed-state selector code review

Independent same-family reviewer: `/root/m91_selector_review` (2026-09-28).
Acceptance: provisional; no blocking issues. Code may enter sanity, then the two
full Train-only control variants if sanity passes. The reviewer did not run
training or modify files.

The reviewer checked that candidate 0 is the native Hann choice; collected
bbox/score parity is exact; pooled RoIs and geometry have the expected shapes;
IoU labels and evaluation use GT `xywh`; only former-fit130 events update the
model, while former-development22 is evaluated at the fixed final epoch. The
seed, 12 epochs, batch 64 and AdamW learning rate 3e-4 match the plan. The
estimated CUDA memory fits one RTX 3090 but still requires an actual sanity run.

During review, a protocol mismatch was identified and fixed: the initial
implementation evaluated development events every epoch. It now checks native
zero-residual parity before training and reports development only after the
fixed final epoch; it does not choose a checkpoint from development scores.

Non-blocking limits: logged `train_bce` averages batches rather than weighting
the shorter final batch; the geometry arm has the same declared architecture
but the zero visual input leaves its visual subnetwork inactive, so the paired
result cannot by itself isolate information from effective capacity. The two
arms are preliminary controls, not a parameter-matched final semantic ablation.
