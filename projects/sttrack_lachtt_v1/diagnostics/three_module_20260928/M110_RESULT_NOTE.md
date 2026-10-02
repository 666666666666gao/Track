# M110 no-weak-rank semantic pair: completed

Both RTX3090 arms completed separate three-update sanity runs and fixed final
training. All six exit files are zero. Driver elapsed48.982646 seconds. Each
final used seed2027,12 epochs,480 AdamW3e-4 updates, batch64, fitting130 Train
sequences /2544 cached states. The same22 Train sequences /495 states were held
out. The M108 bank and parent were reused byte-for-byte; the original sources
and results were retained. The weak pair loss is logged but not optimized.

GPU assertions confirmed identical Empty scores and quality over all3039 states,
unchanged frozen parameters/buffers, finite and nonzero internal text gradients,
and exact final serialization roundtrip. Stored row summaries, gates, source
hashes and weight hashes were additionally recounted by accept_m110_outputs.py.
These executor checks are distinct from an independent reviewer forward replay.

| Final / input | IoU>=0.5 count /495 | Mean IoU | Rescues/breaks against native | Healthy count /264 | Transition count /127 |
|---|---:|---:|---:|---:|---:|
| Native |268|0.524770723|—|264|4|
| Both finals / Empty |272|0.530707698|5/1|264|4|
| Generic final / object |272|0.530402968|5/1|264|4|
| Generic final / reviewed |272|0.530707698|5/1|264|4|
| Reviewed final / object |272|0.530707698|5/1|264|4|
| Reviewed final / reviewed |272|0.530402968|5/1|264|4|

Both own-condition finals pass the existing four fixed-state gates. However,
the reviewed final's reviewed descriptions make6 selections differ from its
Empty condition: no IoU improvements, one IoU decrease, and mean IoU change
−0.000304730. Its generic condition equals the Empty mean. The generic final's
reviewed input equals its Empty mean; its object input has the same small loss.
Fitting correct counts are1608 for generic and1605 for reviewed, versus the
parent's1602. None of this establishes beneficial semantic content on held-out
states or full recursive tracking.

Compared with M108, exclusion removes the observed generic arm's two healthy
breaks and the reviewed arm's transition deficit. This is a matched fixed-state
objective comparison, not a new seed estimate. Gate passage alone does not meet
the additional requirement of a useful reviewed-text increment. No automatic
recursive/public promotion is performed and no new official metric is claimed.

Public archive SHA256:094206e9a16acf6c533c8f7dc6c867bc4b07adb3bcf8d7641ff5397470e78c43.
Finals: generic746ef5fc8d832627b07ffff17538b7d481379b9eac918bf62f9689a475df91e9;
reviewed36f3cdf300522eeb68252f538d04c4b8e02eff6a69a45c09102c3bebf82da4cc.
All28 public artifacts are mirrored under m110_completed and byte-verified.
Descriptions and A/B judgments remain private and model-provisional. CDTB/VOT
review inputs and later-frame-only attributes do not enter this experiment.

Next work should establish an actual semantic discrimination task in fitting
data and separately measure it before another recursive training attempt.
Neither repeated identical fitting nor a larger semantic coefficient is
justified by this result. Formal three-dataset targets remain unmet.
