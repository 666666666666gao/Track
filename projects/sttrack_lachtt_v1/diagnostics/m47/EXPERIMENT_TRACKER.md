# M47 multiple valid correspondence destinations

The fitting-only audit below is complete. The subsequent fixed M47 loss
comparison has now been implemented and trained, with an actual-tensor audit
PASS. Complete22-sequence recursion has finished and failed its frozen
performance gate. No public evaluation was launched.

The current observer extracts ten boxes by3x3 grid suppression. It does not
establish ten distinct physical object instances. The audit recomputes the
existing IoU>=.5 valid-box definition on all1511 fitting pairs from63 sequences,
using SHA-verified M44 caches and the existing labels. Development and public
sequence performance are not calculated.

| Quantity | Count |
|---|---:|
| Current inputs with at least one valid box |872|
| Current inputs with multiple valid boxes |812|
| Previous inputs with at least one valid box |937|
| Previous inputs with multiple valid boxes |906|
| Both frames contain a valid box |836|
| Active forward assignments with other valid previous slots |822|
| Active reverse assignments with other valid current slots |798|
| Extra valid current boxes beyond one per input |3582|
| Fitting sequences containing multiple valid boxes |63/63|

Thus812/872=93.12% of current nonempty fitting inputs contain several IoU-valid
box hypotheses. Candidate-list indices cannot directly mean “the third
similar object.” Such language needs a correspondence to physical instances.

This does **not** prove that unique action supervision is a label bug: the
tracker must ultimately emit one box. It motivates a separate test of the
auxiliary correspondence objective. The fixed subsequent experiment retains
M45's single default-priority action loss, data, initialization,960updates and
runtime, while treating the IoU-valid target boxes as multiple positive slots
only in partial correspondence training. Unknown distractor identities must
remain unlabeled. The original STTrack-default/low22 advancement conditions remain.

Full histograms, all1511pair counts and per-sequence counts are in
`valid_candidate_multiplicity.json`; its source is
`audit_valid_candidate_multiplicity.py`. This is descriptive fitting data,
not evidence of a deployment gain or proof of the unique failure cause.


## Frozen training comparison and actual execution

Only the auxiliary opposite-frame positive destination set changes. The same
M45 selected current/previous target queries are supervised. Matching uses
negative log probability mass over all destinations with IoU>=.5, or unmatched
when the destination has no valid box. Both-empty pairs contribute zero. The
action CE, matching weight.25, architecture, native confidence, inference,
query/template update and search remain unchanged. No text is enabled.

CPU contract checks passed: exact0loss/gradient error against the old loss in
single-positive cases, zero both-empty matching/gradient, no extra direct
distractor-to-distractor supervision, and unchanged action loss. Raising a
valid alternative score decreases the new matching loss, unlike old exact-slot
CE. See `loss_contract.json`. These checks establish semantics, not performance.

| Stage | Verified status |
|---|---|
| Source/data/protocol | Frozen before fitting; spec47f0b9903e845777c406fde350010ac71ecb9689fb3d26b1405af26194b7267b |
| Actual training | Complete, exit0;63fit sequences1511pairs,20epochs960updates |
| Matched control | Exact M45 action labels, initial tensors and sample-order hash |
| Checkpoint reload/audit | PASS;448739parameters,46updated tensors, all finite, strict reload |
| Static development |590inputs; meanIoU.441917896, correct271, changes41, NONE204,7rescues/5severe regressions |
| Full recursion | Complete; two shards16567/16563frames, all22existing development sequences,28897valid frames |
| Public evaluation | Not launched because the fixed recursive gate failed |

Root `/root/autodl-tmp/sttrack_m47_multipositive_v1_20260905`; screen
`sttrack_m47_multipositive_20260905`. Original pipeline22403 and Python
22493/22494 all ended. Terminal status was verified16:24:46CST; training,
both recursive shards and controller exited0. Recursion began about16:07:01.

Final head SHA:016a3898d43ac870761c18424c7627ab35fe4a495bf61579c14629e1607066ab.
Training result SHA:c2df1099344ea29dfea3555647b04aa6f1b1d3ac245df316c76a2b659061065c.
Weight audit SHA:255446a3234a3648aec8d140da86dfdb0072c8ec7675c2a5e71edbf66035ce2d.

The reported matching-membership accuracy uses a larger positive destination
set and must not be compared directly with M45 exact-slot accuracy. M44-relative
action-label-change counts in the training artifact are inherited metadata;
the complete current/previous label maps are exactly identical to M45. Static
results do not select an epoch or bypass the fixed complete recursive run.

## Complete recursive result

| Metric | M47 | Change from native STTrack default |
|---|---:|---:|
| Valid frames |28897|0|
| Mean IoU |.571037773888139|-.081188489288105|
| Low-overlap frames |9946|+2549|
| H10 failure episodes |99|+24|
| Sequences with positive mean change |9/22|Diagnostic count|
| New failures in default-zero-episode sequences |0|Protection condition passes|

Mean gain, fewer low frames and no episode increase all fail. Sequence
coverage and protection of default-zero-episode sequences pass. All trajectory
hashes, full frame counts, finite values, initialization, exact default
prefixes, checkpoint/source bindings, scalar IoU/H10 recomputation and gate
calculations pass the recorded integrity checks. The local22-row aggregate
also reproduces the terminal artifact.

The largest two regressions are notebook02_indoor: meanIoU.388048,
1318additional low frames and17additional episodes; and egg_indoor:
meanIoU.427174,1194additional low frames and12additional episodes. Their
combined low-frame increase2512 nearly accounts for the net2549increase;
other sequences include both gains and losses. These are descriptive
concentrations, not proof of a unique causal mechanism.

Relative to M45, meanIoU falls by11.306460percentage points. Accepting multiple
valid auxiliary destinations does not improve recursive association in this
controlled comparison. The loss-contract and weight-integrity passes do not
override this negative result. Do not promote this checkpoint, loosen the
gate or launch low22/full benchmarks from M47.

Artifacts: `recursive_result.json`, `per_sequence.csv`, and both shard
receipts. Recursive result SHA:
`a669372d8f89614cfc565207c163832f5090385f136f3e1db918219023e60409`.
CSV SHA:`fd4a36c2718c96e9cbcddef3e7aa6d64a66f8cf19ddb8e11769e6043eb077adb`.
These remain DepthTrack Train development metrics, not formal VOT scores.
