# CADTrack DepthTrack fine-tune: scoring and state preservation

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

The successor uses the separate run root
`/root/autodl-tmp/cadtrack_depthtrack_finetune_20261104`. Every completed
epoch31-35 network checkpoint is atomically retained under `final_five/`;
`best.json` points to the retained checkpoint selected by full DepthTrack
Test50 F-score. This is test-set selection and the selected Test50 score is
not an unbiased validation result. The selected checkpoint is intended to be
evaluated with the same inference profile on full DepthTrack Test50, CDTB80
and VOT-RGBD127; the pure-baseline three-dataset evaluation entry must be
prepared separately and is not supplied by the A/B/C evaluator.

This entry does not implement resume or repair the missing optimizer,
scheduler, RNG and data-loader state of the old epoch31 candidate. The existing
nonempty run-directory guard is retained. Do not point this entry at the old
run to repeat epochs1-31 or overwrite its artifacts. A future run or
warm-start continuation needs a separately declared directory and protocol.
Do not start this baseline training while the separate CADTrack A/B/C run is
active.

The YAML disables native checkpoint saving (`SAVE_EPOCH_INTERVAL: 1000`,
`SAVE_LAST_N_EPOCH: 0`). The successor therefore saves one rolling
`latest_training_state.pth.tar` after every complete training epoch, before
the optional full-Test50 evaluation and the native scheduler step. The snapshot
contains the network, optimizer, scheduler, training statistics, preceding
best-selection state, Python/NumPy/Torch CPU/CUDA RNG states and config/runtime
hashes. It is written through a temporary file and then replaces the preceding
snapshot; no old-run artifact is removed.

Its phase is explicitly `training_complete_before_evaluation_and_scheduler`.
For epochs31-35, evaluation is pending; a scheduler step is pending for every
epoch. It is not a ready-to-start-next-epoch checkpoint. Recovery must account
for any evaluation artifacts already written, complete best selection and
advance the scheduler exactly once before training the next epoch. No recovery
command or automatic retry is added here, and exact restart equivalence has
not been established. This recipe has AMP disabled; the snapshot is scoped to
that unchanged recipe. Before a future launch, budget space for the rolling
snapshot plus its temporary replacement, candidate and best weights.
The final-five set requires additional storage for five network-only
checkpoints; verify free space before launch.

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

The actual state-saving helper was separately exercised on a local CPU model
with AdamW moments and MultiStepLR after the milestone10 step. Saved network,
optimizer moments, scheduler state and Python/NumPy/Torch RNG replay matched.
The epoch31 evaluation-pending marker and an epoch30 rolling replacement were
checked, with no temporary file left behind. CUDA RNG collection returned an
empty list on this CPU-only host; GPU-state restoration, data-loader replay and
end-to-end training continuation have not been verified.
