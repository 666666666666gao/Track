# M67/M82 Full152 paired training

This is a new experiment. It does not replace the completed 130-sequence M67 and M82 checkpoints or their full-dataset results.

- Both arms use the same 152 DepthTrack Train sequences, order, seed 2027, category-only five-slot text bank, zero-residual adapter initial tensors, frozen native STTrack, optimizer settings, Hann selection, and native template update.
- M67 uses its original localization loss with support-loss weight zero. M82 adds its original raw competition and reliable same-state native spatial preservation losses.
- The prior development 22 are now training data. Any subsequent results on those sequences are training diagnostics, not held-out validation.
- The training pass is 219,954 images and 219,802 track calls. Valid-GT masking determines the actual optimization count, which must be read from the completion receipts.
- The final checkpoint after the single full pass is the predetermined selection. Each resulting model must be evaluated unchanged on DepthTrack Test50, CDTB80, and VOT-RGBD2022 full127 using the same external BF16 initialization-text protocol.

`prepare_full152.py` creates the frozen remote inputs under `/root/autodl-tmp/sttrack_full152_paired_20260925`. The exact generated scripts, specs, and preparation receipt are archived here. Source RGB-D data, text tensor bank, pretrained and trained weights remain on the private server.

The first-frame forward/backward preflight passed on GPU1 for both arms with finite losses and gradients on all 23 adapter tensors. No optimizer step or training artifact was produced by that preflight.
