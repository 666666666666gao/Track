# M48 fixed native-continuity admission

The fitting audit, actual runtime contract and fixed full22-sequence recursive
comparison are complete. M48 improves aggregate meanIoU and both failure
measures, but fails protection of mobilephone02. Four of five frozen gates
pass; the overall advancement gate fails. No public evaluation is running.

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
Python24427/24428; two shards16567/16563frames. Start16:57:04CST. The17:11:44
milestone verified the original processes and18sealed sequences/28107frames.
Terminal17:16:14check confirmed all three processes ended and all three exits0.
No M48 full-recursion job remains active.

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

## Complete recursive result and residual protection failure

| Complete22-sequence metric | M48 | Change from native STTrack default |
|---|---:|---:|
| Valid frames |28897|0|
| Mean IoU |.714587093266861|+.062360830090617|
| Low-overlap frames |5331|-2066|
| H10 failure episodes |68|-7|
| Positive sequences |7/22|Coverage gate passes|
| New failures in default-zero-episode sequences |1sequence|mobilephone02_indoor|
| Proposed / vetoed / admitted overrides |914/858/56|Causal runtime action counts|

Mean gain, fewer low frames, no episode increase and sequence coverage pass.
Healthy-sequence protection fails, so overall advancement remains false.
M48 improves meanIoU over the same M45 head by3.048472percentage points.
The fixed policy is sealed; there is no low22/full evaluation or gate change.

The largest gains are ball07_indoor:+46.371919percentage points of meanIoU,
1106fewer low frames and3fewer episodes; and colacan01_indoor:+33.293493points,
1115fewer low frames. Losses also remain. Full22rows are in `per_sequence.csv`.
Result/source/checkpoint bindings, all trajectory hashes, frame counts,
initialization, finite values, causal decision records and exact pre-override
default prefixes pass. Scalar IoU/H10 and local22-row aggregate checks pass.

Result SHA:`e3934bd736e9f14b6009cad4c5c05090a2527c3410aa54bb179eab2d84feeb7b`.
CSV SHA:`5d7a26361a522952d89c6a90075ae28ec8e490dcf63c7eb4939c8528c354e70d`.

The post-protocol fitting analysis in `previous_target_diagnosis.json` uses
only the already sealed fitting labels:20of21vetoed rescues lack a valid
previous default, compared with1of5retained rescues. Continuity to a previous
mistake can obstruct recovery. This does not change M48 or establish a
deployable GT condition.

### Exact mobilephone02 replay

`inspect_mobilephone.py` replays all700tracked frames and exactly matches
the sealed boxes, confidence, proposals and continuity deltas. There is only
one admitted override, at zero-based frame383. Default/selected IoU are
.759101/.697773; their mutual boxIoU is.689389. Native confidence drops from
.533021to.060398. Previous-RGB/depth cosine improves by.002610/.000928, so
the fixed rule admits the proposal. Both boxes still have valid target
overlap, which cannot be described as a proven immediate identity switch.

At frame450, M48 reports NONE but native confidence.833475 still permits a
template write. GT is invalid there, so its template IoU is reported asnull.
The native default's confidence at450 is.391428 and does not permit a write.
The complete M48 failure episode occupies497through700,204valid low frames;
the later writes at500,550,600,650and700all have zero GT overlap.

The first admitted crop perturbation is114frames before the confirmed
episode. Neither a correct instantaneous box nor a short confirmation window
proves long-term stability. The write at450 is a candidate causal contributor,
not a confirmed unique cause. A separate diagnostic suppresses only that
write and otherwise retains the complete M48 path. Its selected frame is
post hoc; it must not become a deployment rule or be spliced into full22
results. See `mobilephone_diagnosis.json` and the separate intervention source.

### Single-write intervention and reappearance geometry: complete

The700-frame intervention is complete. Suppressing only the write at450
preserves exact boxes/confidence through450 and restores the native threshold
immediately afterward. It leaves meanIoU.568362728917360,205low frames and
1episode unchanged. Later confidence and boxes differ, so this is metric
equivalence, not identical trajectories. This write alone is insufficient to
explain or repair the protection failure. No NONE-based update gate was added.

At the first valid GT frame after occlusion,497, the target box is
[326,136,26,31]. Exact native factor4crop geometry from the preceding states:

| Path | Crop xyxy | Target area covered |
|---|---|---:|
| Native default |[268,84,411,227]|100%|
| M48 |[411,156,706,451]|0%|
| M48 with only write450suppressed |[376,-9,812,427]|0%|

The wrong search state persists despite the isolated template intervention.
This is a newly introduced DepthTrack Train regression; it does not revise
the M40 census of native STTrack VOT failure onsets. The next investigation
must preserve access to the complete native-default state through tentative
changes and examine recovery after occlusion. It must not deploy a GT-based
recenter, a frame383/450exception, or unvalidated global update suppression.
No new successor runtime, trained head or public evaluation has been created.

Intervention result:`phone_skip_write450/result.json`, SHA
`3e91001fd2752ad389948b4df6995bee8f406f86619a3bac924f3d72e4c40c3d`.
Geometry:`mobilephone_search_geometry.json`, SHA
`2e7648eebeb0e0473da2016ac54e5a87c32f5ddda2057b589110c1ab33a428f0`.
