# Sealed ROI VOT trajectory coverage

`vot_trajectory_coverage.py` is an additional deterministic read of completed
toolkit0.7.1 full127 output. It does not run tracking, analysis, overlaps,
failure scoring or neural inference. It writes only a new explicitly named
report; all input result files remain unchanged.

It checks all1765 forward/backward anchors derived from actual dataset metadata
against the four441/441/441/442 manifest partitions, real127 sequence identities
and frame counts, every decoded binary trajectory length, both confidence/time
stream lengths, all5295 merged hashes and original shard hashes, original
shard exits, and current files against the supplied execution-source ledger.
The supplied expected execution digest identifies the ledger being checked.
Current ledger agreement alone does not establish historic checkpoint or full
training/source/plan provenance. The report deliberately leaves
`checkpoint_source_binding_checked=false` and
`independent_completed_audit=false`. The separate identity checker is still
required for official analysis descriptors, protocol and sealed metrics.

This version targets the sealed ROI manifest/merge schema, including its `exit`
launch field. It is not an alternative evaluator for the running CADTrack port.
Use the existing mplt interpreter with CUDA hidden, bytecode disabled and one
CPU thread. Run only after all original shard/controller stages terminate.

```text
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
/root/miniconda3/envs/mplt/bin/python -B vot_trajectory_coverage.py \
  --workspace /root/autodl-tmp/m125_region_roi_fixed6/evaluation/vot/run/master \
  --manifest /root/autodl-tmp/m125_region_roi_fixed6/evaluation/vot/run/shard_manifest.json \
  --merge /root/autodl-tmp/m125_region_roi_fixed6/evaluation/vot/run/merge_result.json \
  --execution /root/autodl-tmp/m125_region_roi_fixed6/evaluation/vot/execution.json \
  --execution-sha256 51a3041dadbefb842622ccebd9253f79ea2bd507bc80796b0ac3b1f6244b06b1 \
  --analysis /root/autodl-tmp/m125_region_roi_fixed6/evaluation/vot/run/master/analysis/sttrack_m125_region_roi_fixed6_full127.json \
  --output /path/to/new/roi_vot_trajectory_coverage.json
```

The CPU unit tests exercise direction-dependent lengths, auxiliary-stream
truncation, complete partition cardinality and duplicate/wrong identities.
They do not decode real VOT outputs; that requires running this tool on the
actual sealed workspace.
