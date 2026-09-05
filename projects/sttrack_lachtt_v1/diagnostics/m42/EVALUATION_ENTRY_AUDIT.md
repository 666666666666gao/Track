# Same-model evaluation entry audit — no benchmark launched

Read-only source/manifest inspection on2026-09-05. No Test/CDTB/VOT images or
subsequent labels were passed through a new model during this inspection.

| Evaluation | Existing authoritative input | Complete coverage |
|---|---|---|
| VOT low22 | `/root/autodl-tmp/sutrack_vot_low22_anchor_identity_v1/run/shard_manifest.json` |22 sequences,303 anchors,4 shards;220,483 estimated frames |
| VOT full127 | `/root/autodl-tmp/sutrack_rgbd_language_safe_template_vot_full127_v1/shard_manifest.json` |127 sequences,1,765 anchors,10 shards;1,327,004 estimated frames |
| DepthTrack Test | `/root/autodl-tmp/depthtrack/test/sequences` |50 sequences,76,373 frames |
| CDTB | `/root/autodl-tmp/CDTB/sequences` |80 sequences,101,956 frames |

The two VOT manifest SHA256 values are respectively
`600b1ebb8b0c2f69b831f954e907e63709fd69afb7ea94c5b58e8c7408a29eed` and
`8bf5271b3cdc0e0f4587657502f0aa4d873c6cfbc8716f88a1fabb55aa5334b3`.
Full127 workload is substantially larger than a short smoke; estimate it from
measured candidate-runtime throughput before launch and schedule long waits.

M39 already has a working RGB-D TraX bridge and direct STTrack construction in
`/root/autodl-tmp/sttrack_lachtt_m39_vot_low22_template_ablation_v1_20260902/wrappers/`.
Future candidate wrappers should use the newly trained tracker with the same
protocol. Keep inference in `/root/autodl-tmp/envs/sttrack/bin/python` and VOT
analysis in `/root/miniconda3/envs/mplt/bin/python`. The native internal state
must not be replaced by rounded TraX output when moving to the next frame.

The retained DepthTrack/CDTB scripts in `/home/SRTrack_RGBD_L/tracking/` select
SRTrack deployment classes and language manifests. Passing an STTrack weight
to them is not a valid model migration. New STTrack output should preserve the
same result format and reuse the independent PR evaluator, whose path is
`/home/SRTrack_RGBD_L/lib/test/analysis/depthtrack_pr.py`, SHA256
`05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc`.

That evaluator uses VOT bounded-region overlap, per-sequence PR curves and
macro averaging, with resolution100. It is different from M42's continuous
Train-window IoU. Existing OPE serialization writes boxes and confidence with
six decimal places, and initialization confidence is1.0. Preserve those
conventions for the new tracker; retain native values internally. The normal
PR curve threshold computation is an evaluation operation, not a change to
the deployed association/template thresholds.

Every new evaluation record must bind the same base checkpoint, trained
association checkpoint, runtime and configuration hashes. Do not select a
different head per dataset. Old SRTrack champion directories contain historical
records; they are not evidence of the current STTrack model's performance.
M39 default is the direct low22 comparator. No public launch is added by this
audit; the training and recursive gates remain in force.
