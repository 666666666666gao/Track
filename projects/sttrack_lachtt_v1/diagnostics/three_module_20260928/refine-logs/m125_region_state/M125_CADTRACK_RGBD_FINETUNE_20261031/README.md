# CADTrack RGB-D DepthTrack Fine-Tuning

## Architecture and initialization

This experiment uses the CADTrack ViT-B/16 architecture and the epoch-20 CADTrack checkpoint (`d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8`). The checkpoint is loaded strictly with no missing or unexpected keys. The RGB-T model input is adapted as RGB plus a three-channel JET color map of DepthTrack depth; this is a modality transfer, not a new architecture or a claim that thermal and depth are equivalent.

Training uses all 152 DepthTrack Train sequences. The sole annotation-length mismatch is recorded by the training loader and is truncated to the available paired RGB-D frames.

## Training protocol

| Setting | Value |
|---|---:|
| Epochs | 35 |
| Base LR | 1e-5 (one tenth of the original 1e-4) |
| LR schedule | MultiStepLR, milestones [10, 30], gamma 0.1 |
| Batch size | 8 |
| Samples per epoch | 20,000 |
| Checkpoint selection | Highest F-score across epochs 31-35 |
| Validation split | None; use complete test sets as requested |

The selection score is computed on the complete DepthTrack Test50 set (50 sequences, 76,373 frames). Since this is also a test set, the selected score is test-guided and must be labeled as such; do not describe it as an unbiased validation estimate.

## Evaluation coverage

| Dataset/protocol | Required complete coverage | Current status |
|---|---:|---|
| DepthTrack OPE | 50 sequences, 76,373 frames | Training in progress; no evaluation metrics yet |
| CDTB OPE | 80 sequences, 101,956 frames | Pending best checkpoint |
| VOT-RGBD2022 | 127 sequences, 1,765 anchors, 5,295 trajectory outputs | Pending best checkpoint |

The OPE runner checks full-set sequence/frame counts and seals prediction files with per-sequence ground-truth and output hashes. Scoring verifies the checkpoint hash against the prediction receipt. The VOT preparation tool binds an explicitly supplied checkpoint and run name while retaining the frozen full 127-sequence/1,765-anchor manifest.

| Archived input | SHA-256 |
|---|---|
| DepthTrack Test50 cases | `158fe61d7726af5e2c1f30c76175a97a4cba56e30e2f599ec7762be034417b44` |
| CDTB80 cases | `b3170466a090fa1a8949e0297566c9e3ab744e789ca0c7f1bcc71633ccfae887` |
| VOT-RGBD2022 shard/anchor manifest | `99eec1fec6fa3b93fc1d179aecd636130703cf2df69ee99a779b50fb0d2d964e` |
| Training launch record | `3b035a6caa61fd23fd6c350a6e87e43c81426698d1088bc8d7ce903b41a06a6b` |

## Current run and next steps

At 2026-11-01 01:54 UTC, the AutoDL run was live at epoch 1, batch 1,300/2,500 (it was at 900/2,500 at 01:40 UTC). This is training progress only, not an evaluation result. Keep the existing job running. After epochs 31-35, retain the best full-Test50 F-score checkpoint, then evaluate that exact checkpoint on full DepthTrack OPE, full CDTB OPE, and full VOT-RGBD2022; record its SHA-256 with all results.

At 2026-11-01 02:17:29 UTC, one 300-second observation window verified the same process tree was still live and advanced to epoch 1, batch 1,900/2,500. The 300-second interval contained 350 additional logged steps, with finite training losses. No checkpoint or test metrics existed at this observation. The user explicitly confirmed full-test-set evaluation with no fixed subset; retain the complete Test50 selection and full final coverage above.

At 2026-11-01 02:42:42 UTC, the log reached epoch 1 batch 2,500/2,500 and epoch 2 batch 50/2,500. Trainer PID 55501 and workers 59649/59713 were live; checkpoint directory remained empty, with no evaluation metrics or VOT job.

The known modality gap is RGB-T pretraining followed by JET-mapped depth. Do not attribute any metric change to this gap without evidence. If the complete-set results remain low, inspect per-sequence failures first, then plan a controlled depth-input/representation comparison and review relevant open-source RGB-D modules before changing the architecture.
