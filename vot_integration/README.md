# VOT Toolkit Integration

This directory registers the current RGB-D-L MPLT tracker for the Python VOT toolkit.

Example after a VOT workspace is configured:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mplt
cd /home/OSTrack_RGBD_L_dataset_modified
vot --registry vot_integration evaluate --workspace /path/to/vot_workspace dala_mplt_roberta_ep10
vot --registry vot_integration analysis --workspace /path/to/vot_workspace dala_mplt_roberta_ep10
```

The tracker entry reads these environment variables:

- `MPLT_CONFIG`: experiment yaml name.
- `MPLT_EPOCH`: checkpoint epoch.
- `MPLT_CHECKPOINT_ROOT`: training output root containing `checkpoints/train/...`.
- `MPLT_DATASET_NAME`: usually `votrgbd2022` or `cdtb`.
- `MPLT_LANG_JSONL`: optional language annotation jsonl path, or multiple paths separated by `:`.

For other epochs, copy the entry in `trackers.ini` and change `env_MPLT_EPOCH`.
