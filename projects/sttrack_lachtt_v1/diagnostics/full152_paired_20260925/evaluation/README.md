# Prepared Full152 external evaluation

This evaluation has not started. It launches only after both Full152 training exit files are zero and both predetermined final checkpoints have completion receipts.

The scripts reuse the previously generated bfloat16 DepthTrack Test50, CDTB80, and VOT-RGBD2022 full127 initialization-text observations through read-only links. They bind new Full152 checkpoints to separate model bundles, run M67 first and M82 second, and collect all six full-dataset results. Each model uses two GPUs concurrently for DepthTrack/CDTB and four disjoint VOT shards across the same two GPUs.

`prepare_full.py` checks the completed 152-sequence, 219,802-call training receipts and model hashes before creating evaluation bundles. `run_full_eval.sh` writes per-stage exit files. The old 130-sequence results remain in their original evaluation directory.

The new evaluation root is `/root/autodl-tmp/sttrack_full152_evaluation_20260925`. Its Python and shell entrypoints have passed static syntax checks. No external metric has been computed for Full152 weights at preparation time.
