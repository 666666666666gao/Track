# Official VOT analysis identity check

`vot_analysis_identity.py` is an additional read-only check for the fixed ROI
full127 run. The existing analyzer extracts values by position without checking
analysis descriptors. This tool checks those descriptors, the complete baseline
protocol against an explicitly supplied official reference, toolkit metadata,
tracker identity, all127 sequence identities/metadata and manifest frame counts,
and agreement between official and sealed metrics.

Use only after the original controller has completed official analysis and
sealing. The reference is the archived official
`A1_anchored_U2_full127.json`, reviewed for toolkit0.7.1 and the full127
MultiStart protocol (EAO115-755, burnin10, grace10, threshold0.1, bounded,
ignore mask `_ignore`). A protocol difference stops the check for investigation;
it must not be repaired by changing an existing result or silently substituting
a different reference. The printed reference hash identifies the actual input.

```text
python -B vot_analysis_identity.py \
  --analysis /root/autodl-tmp/m125_region_roi_fixed6/evaluation/vot/run/master/analysis/<model>_full127.json \
  --reference /path/to/archived/A1_anchored_U2_full127.json \
  --manifest /root/autodl-tmp/m125_region_roi_fixed6/evaluation/vot/run/shard_manifest.json \
  --sealed-result /root/autodl-tmp/m125_region_roi_fixed6/evaluation/vot/result.json
```

This command prints a report and does not write files, run tracking/analysis,
load a neural model or change the live controller. It deliberately does not
claim a complete audit: actual1765 trajectory lengths,5295 result-file hashes,
shard exits and final/bundle/source/plan bindings still require the terminal
audit recorded in the project handoff. Successful metadata checking is not
evidence that the experiment meets the nine metric gates.

Local verification used the archived official A1 analysis (Git blob bytes
SHA256 `73e0b6c15713735240d5d50acd51a95593155f2e3471a9f3d13d38249f5df2e6`)
and the published ROI preflight manifest, with the fixture tracker name changed
to A1 and fixture sealed metrics derived from that historical analysis. The
metadata check passed; altered descriptor order, sequence identity, threshold
and sealed ROB were each rejected. This exercises the added check and does not
constitute an audit or result for the currently running ROI experiment.
