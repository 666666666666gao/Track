# M47 fitting-only correspondence audit

This stage has completed a CPU data audit only. There is no M47 trained weight,
neural architecture change, tracker intervention or benchmark score yet.

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
auxiliary correspondence objective. A bounded next experiment could retain
M45's single default-priority action loss, data, initialization,960updates and
runtime, while treating the IoU-valid target boxes as multiple positive slots
only in partial correspondence training. Unknown distractor identities must
remain unlabeled. That experiment has not yet been implemented or trained,
and the original STTrack-default/low22 advancement conditions remain.

Full histograms, all1511pair counts and per-sequence counts are in
`valid_candidate_multiplicity.json`; its source is
`audit_valid_candidate_multiplicity.py`. This is descriptive fitting data,
not evidence of a deployment gain or proof of the unique failure cause.
