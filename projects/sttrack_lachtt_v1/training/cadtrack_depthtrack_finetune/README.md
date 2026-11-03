# CADTrack DepthTrack fine-tune: metric environment fix

This successor source repairs the observed epoch31 scoring failure in the
2026-10-31 run. It is staged for a future training entry; it is not a resumed
or newly launched experiment. Historical scripts and receipts remain in the
diagnostics archive.

The original trainer completed all50 DepthTrack Test sequences and76373 frames,
then failed importing `vot.region` while scoring in the training interpreter.
The same saved predictions were successfully scored with the existing
`/root/miniconda3/envs/mplt/bin/python`.

The successor keeps tracking in the original training process, preserving its
model instance and existing prediction receipt. It runs only the locked
score command in the existing metric interpreter, with CUDA hidden and CPU
thread counts set to1. A failed scoring subprocess raises immediately; no
automatic retry, replacement metric, or dependency installation is added.
The locked metric import is checked before training starts and in preflight.

The original LR1e-5, milestones[10,30], gamma0.1 and epochs31-35 full-Test50
selection recipe are unchanged. Such selection must be labeled
`test-set-selected`.

This patch does not implement resume or repair the missing optimizer,
scheduler, RNG and data-loader state of the old epoch31 candidate. The existing
nonempty run-directory guard is retained. Do not point this entry at the old
run to repeat epochs1-31 or overwrite its artifacts. A future run or
warm-start continuation needs a separately declared directory and protocol.
No new training may run alongside the active two-GPU ROI VOT suite.

Validation must distinguish the CPU scoring path from training. A successful
score-only regression cannot establish that epochs32-35 completed or that
the missing original state was recovered.

## Validation performed

The actual new helper functions and path constants were extracted from this
source with Python AST and executed on the authorized AutoDL host. The locked
metric import succeeded, CUDA was hidden, and all three CPU thread settings
were1. A deliberately failing scoring child returned17 and propagated as
`CalledProcessError`.

`score_predictions(31, candidate)` then passed the existing evaluator's
checkpoint/case/GT/prediction/confidence/source hash checks on complete
Test50 (50 sequences,76373 frames), exit0. The generated metrics file's SHA256
was identical before and after. This tests the actual repaired subprocess
path, without importing or executing the full GPU training entry.

The first local validation harness attempt used an unavailable Python3.7 AST
helper and stopped before SSH; the corrected harness used AST top-level line
ranges. The product code required no compatibility branch.
