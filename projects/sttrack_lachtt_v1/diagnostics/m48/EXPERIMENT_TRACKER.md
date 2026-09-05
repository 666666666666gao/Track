# M48 fixed native-continuity admission

The fitting audit and actual runtime contract are complete. A single fixed
full22-sequence recursive comparison started16:57CST on2026-09-05. There is no
terminal M48 performance result yet. Public evaluation is not running.

## Motivation and fitting audit

At M45's replayed first wrong choices on egg and mobilephone02, the native
candidate RoI features have stronger continuity to the previous selected RoI
for the default than for the learned proposal. This motivates a necessary
admission condition; it does not establish that native cosine identifies the
target reliably. The streams come from the fused tracker, so they are not
independent measurements.

The rule is fixed before recursive evaluation: retain a nondefault M45
proposal only if its unprojected4x4 RoI cosine to the preceding selected RoI
is at least the default candidate's cosine in BOTH RGB and depth. The margin
is zero and is not swept. The learned head still proposes the action; this
does not choose the candidate with the highest confidence or cosine.

The audit reads all1511original fitting pairs from63DepthTrack Train
sequences with feature, label and proposal-artifact SHA checks. All previous
choices in these cached default trajectories are0. No development or public
performance is computed by the audit.

| Fitting-only metric | M45 proposal | After fixed screening |
|---|---:|---:|
| Changes |96|20|
| Mean IoU |.513510402512411|.494053173051319|
| Correct boxes, IoU>=.5 |853|780|
| Rescues, default<=.1 and selected>=.5 |26|5|
| Severe regressions, default>=.5 and selected<=.1 |0|0|
| Beneficial changes |94|19|
| Harmful changes |1|1|

Default fitting meanIoU is.489167940853231. Of76vetoes,75remove a beneficial
change and21remove a rescue; the single harmful fitting change remains.
Accepted changes cover15sequences, and retained rescues cover4. This is a
large loss of fitting utility and provides no demonstrated safety benefit.
The absence of fitting severe regressions in M45 prevents estimating their
rejection rate. The remaining5rescues motivate only one complete recursive
test, not a calibrated safety claim or boundary relaxation.

Audit artifact `native_continuity_audit.json`:883988bytes, SHA
`969a982bf7633b31d25569b2cdd4bcf6e957b45934705fceb808085b3fa24950`.
Its source is `audit_native_continuity.py`. IoUs are calculated after proposals
are sealed. The runtime rule has no GT access.

## Model and state semantics

The exact M45 geometry head remains448739parameters, already trained on
DepthTrack Train63sequences/1511pairs for20epochs960updates. M48 adds no
parameters and performs zero optimizer steps. Base STTrack, head tensors,
search, confidence, query and default template adaptation remain unchanged.

`sttrack_candidate_continuity.py` wraps the frozen head's inference. A veto
converts the proposal to the existing candidate0 action before the inherited
tracker commits its state. The existing `STTrackCandidateSet.track` still
performs all box/confidence decoding and updates to query, template,
reference bank and previous selection. Accepted proposals follow exactly
the M45 candidate path. Language is OFF in this policy comparison.

## Actual contract checks and execution

All checks ran on the server and passed:

- Four synthetic cases cover weaker evidence, stronger evidence with a
  nonzero preceding selection, disagreement between streams, and equality.
- All1511fitting decisions match between the CPU audit and GPU predicate.
- Two fitting sequences and120tracked frames force118nondefault proposals
  to be vetoed. Native default boxes, confidence, query, templates and mask
  state agree exactly, including2actual default template updates.
- Two forced admitted frames agree exactly with M45 and propagate the
  previous selected candidate from0to1.
- The loaded448739parameter head and every frozen source hash match. No
  optimizer or checkpoint mutation occurs.

Root `/root/autodl-tmp/sttrack_m48_native_continuity_v1_20260905`;
screen `sttrack_m48_native_continuity_20260905`; original controller24424,
Python24427/24428; two shards16567/16563frames. Start16:57:04CST, expected
completion about17:14–17:16. First useful progress check planned near17:11.
Do not relaunch these processes on an observation timeout.

Spec SHA:`f398dc9b4639f15e3f4a69bae067037b5ebc6ac4bac7244ea348be4eb37128a0`.
Runtime contract SHA:`1ce4b05ed965f2342354e46fe71100d0dcc7764f0b79faee2a68004a654b622f`.
M45 head SHA:`853a25fbc3c9ef12ab54442c30b27bab75f0b1fadcc9bcc82cbec6e8700ed59c`.

## Frozen advancement

Complete all22previously used DepthTrack Train development sequences and
compare against the actual native STTrack default. MeanIoU must improve by
at least.01, low frames must decrease, H10episodes must not increase, at least
3sequences must improve and default-zero-episode sequences must be protected.
No boundary sweep or partial-sequence stopping decision is allowed.

Only a pass permits the frozen low22 comparison against M39 STTrack default.
Low22 must increase EAO and ROB by at least1percentage point each, preserve
ACC within.10percentage point, reduce failures and protect its7successful
sequences. A low22 pass is required before full three-dataset validation of
the same base/head/runtime bundle. This is repeatedly used development data,
not a fresh unseen test. No formal metric is claimed by this document.
