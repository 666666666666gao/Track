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

Run `train_depthtrack_finetune.py --preflight` before training. The training
output directory must be absent or empty. For a detached run, keep supervisor
receipts and logs in a separate control directory so they do not trip that
empty-directory guard:

```text
python supervise.py \
  --run-root /root/autodl-tmp/cadtrack_depthtrack_finetune_20261104 \
  --control-root /root/autodl-tmp/cadtrack_depthtrack_finetune_20261104_control \
  -- /root/autodl-tmp/envs/sttrack/bin/python -B -u \
  /path/to/cadtrack_depthtrack_finetune/train_depthtrack_finetune.py
```

The wrapper binds GPU0, uses the existing training interpreter, writes the
child log and exit/terminal receipts under the control directory, and refuses
to reuse a nonempty run directory or existing control directory.

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

## Fixed final and complete evaluation

`prepare_evaluation.py` binds all five epoch31-35 network checkpoints, their
five Test50 metric receipts, `best.json`, the user-selected checkpoint SHA,
the native CADTrack inference configuration, full OPE case inventories and
the archived full-127 VOT manifest. Pass `--selected-epoch 31` through
`--selected-epoch 35` and point `--checkpoint` to that exact retained weight.
The automatic Test50 maximum in `best.json` is recorded for comparison but
does not choose the evaluation checkpoint. The user must review all five
candidate scores and choose explicitly. This remains `test-set-selected`;
the choice and resulting three-dataset report must disclose selection bias.
Preparation creates a new evaluation root and four new VOT workspaces; it
does not reuse prediction files.

The evaluator uses the original CADTrack `tracker_for` and RGB+JET-depth input
path. All three datasets bind one selected checkpoint and one inference profile:
template factor/size2.0/128, search factor/size4.0/256, update interval5 and
confidence threshold0.65. Prediction stages do not open subsequent-frame GT;
metric scoring happens only after the dataset's prediction receipt is complete.

Before any VOT tracking, all four archived workspaces must match the pinned
toolkit0.7.1 MultiStart experiment and analysis parameters. VOT runs the
127-sequence/1765-anchor four-shard protocol in two GPU waves, merges and
rehashes all5295 output files, reads official EAO/ACC/ROB from the official
analysis, runs the identity checker and verifies per-anchor trajectory lengths
with the existing read-only coverage auditor. These audits explicitly leave
checkpoint/source binding and `independent_completed_audit` false.

After a successful pure-baseline training run and a fresh resource check, copy
the evaluation tools and official baseline plans to the host, then prepare a
new output directory:

```text
python prepare_evaluation.py \
  --training-root /root/autodl-tmp/cadtrack_depthtrack_finetune_20261104 \
  --base-evaluation /root/autodl-tmp/m125_region_roi_fixed6/evaluation \
  --checkpoint /root/autodl-tmp/cadtrack_depthtrack_finetune_20261104/final_five/epoch_NN.pth.tar \
  --selected-epoch NN \
  --evaluation-source /path/to/cadtrack_depthtrack_finetune/evaluation.py \
  --evaluation-runner /path/to/cadtrack_depthtrack_finetune/evaluation.py \
  --vot-identity-tool /path/to/vot_analysis_identity.py \
  --vot-coverage-tool /path/to/vot_trajectory_coverage.py \
  --vot-reference /root/autodl-tmp/m125_A1_anchored_U2_m025_20261020_r2/vot/master/analysis/A1_anchored_U2_full127.json \
  --output /root/autodl-tmp/cadtrack_depthtrack_finetune_20261104/official-evaluation
```

Both `--evaluation-source` and `--evaluation-runner` intentionally use this
directory's `evaluation.py`: it implements both worker modes and the controller.
Preparation copies the same source to `evaluation.py` and `run_evaluation.py`
in the isolated output directory. The latter is the controller entry, not a
missing source file to obtain from the A/B/C evaluator.

The prepared runner CLI is `run_evaluation.py --evaluation-root ... --tracker-python
/root/autodl-tmp/envs/sttrack/bin/python --metric-python
/root/miniconda3/envs/mplt/bin/python`. Do not start it until the training
terminal receipts are successful, the selected epoch and all five retained
checkpoint hashes validate, the active A/B/C job has ended, and disk/GPU
resources have been freshly checked. CPU tests do not establish model metrics
or authorize a remote run.

Run these tools with assertions enabled (no `-O` or `-OO`). Child environments
pin `PYTHONOPTIMIZE=0`. Preparation seals plan hashes in `selection.json`;
full controller checks verify the native source manifest, selected checkpoint,
all retained weights, evaluator and execution ledger. Anchor workers check the
pinned manifest/ledger digests without rehashing all retained weights on each
startup. This is an execution binding check, not a completed independent audit.

The terminal `all_results.json` reports each of the nine fixed acceptance gates
and sets `project_status` to `ACTIVE_MET` only when every gate passes. DT/CDTB
use inclusive thresholds; VOT EAO/ACC/ROB use strict greater-than thresholds.
Non-finite or out-of-range metrics are rejected.
