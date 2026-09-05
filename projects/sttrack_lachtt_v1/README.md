# STTrack LACHTT RGB-D language tracking overlay

This directory publishes only the source and configuration changes developed
for the language-anchored candidate association and protected/tentative RGB-D
tracking experiments. It is an overlay for STTrack, not a model checkpoint or
a copy of the server workspace.

## Upstream base and source identity

- Upstream repository: <https://github.com/NJU-PCALab/STTrack>
- Required upstream commit: `283cd6dd45536636490db8bca1c63c4647be799b`
- Upstream license: [`UPSTREAM_LICENSE.txt`](UPSTREAM_LICENSE.txt)
- Experiment branch: `codex/language-anchored-candidate-transaction-v1`
- Published source commit: `72b6446f5ba0e96c8882001f3286585fd81cff30`
- Overlay manifest SHA256: `485c5a45d4f0268fa743fa95b75e9a8477fbdce532dbb5887b4f2e87ea2abf27`
- Published project files: 77

To reconstruct the source tree:

```bash
git clone https://github.com/NJU-PCALab/STTrack.git
cd STTrack
git checkout 283cd6dd45536636490db8bca1c63c4647be799b
cp -a /path/to/Track/projects/sttrack_lachtt_v1/overlay/. ./
```

Configure datasets, checkpoints and local paths for your own machine. Frozen
experiment utilities may contain provenance hashes and artifact names, but the
published overlay contains no dataset, checkpoint, prediction, cache, VOT
workspace, Qwen model, API credential or private server configuration.

## Included work

- Candidate-own RGB-D observation extraction and strict depth-missing masks.
- Language-anchored target/distractor association prototypes.
- Independent utility and safety outputs with conservative abstention.
- Canonical candidate roles and permutation-invariant set processing.
- Protected/tentative recursive rollout, multicentre recovery and atomic-state
  experiment utilities.
- Fixed-pool, task-tower, richer-RoI and learned bounded-relation capacity
  studies, including their fail-closed runners and audits.
- M17 sequence-disjoint target/split closure that serializes training targets
  while withholding numeric held-out targets.
- M17-1 same-bytes post-audit binding builder and fail-closed sequence-
  disjoint utility/safety/survival runner.
- M18 sequence-disjoint causal-survival target closure with numeric targets
  restricted to training folds.
- M18 causal quantile-survival model and its zero-step, immutable-journal
  architecture smoke runner.
- M19a import-only bootstrap attribution runner, which records the exact
  dependency-generated process and `/dev/null` events before any model code.
- M19b mechanical extractor for a non-expanding, exact bootstrap receipt with
  two byte-identical derivations and linked-worktree-aware Git identity checks.
- M20a receipt-bound zero-step model-runtime smoke runner with exact inner-gate
  closure, immutable attempt journals and explicit failed-publication sealing.
- M21a immutable, phase-closed successor runner. It deep-copies observation
  snapshots, isolates Git/control identity reads in a verified postflight phase,
  derives negative-result metadata from observations and closes binding/runtime
  authorization without modifying the model or fixture.
- M22a sequence-disjoint causal-survival training runner. It trains on
  DepthTrack Train folds 2--5, computes fold-1 predictions before opening the
  delayed held-out labels, enforces exact event/CLIP/native storage-dtype
  contracts and converts the verified tensors to contiguous CPU float32.
- M23a exact five-frame hypothesis deduplication plus a parameter-disjoint
  direct benefit/catastrophe selective router. Its runner uses natural-prior
  sequence-weighted training, fixed non-scanned commit gates, candidate/event
  permutation audits and explicit fold-0/full-target exclusion.
- M24 sequence-fold epistemic committee runner. Four same-initialization
  single-fold models vote only on folds unseen by each voter, with unanimous
  role identity and fixed worst-member benefit/catastrophe/margin gates.
- M25 sequence-pooled leave-one-fold-out runner. Each model trains on three
  sequence-disjoint folds and evaluates only the excluded fourth fold under
  the unchanged M23 direct-selection policy.
- M26 nested sequence-calibrated counterfactual-harm runner. It keeps the full
  candidate-own utility tower, adds a parameter-disjoint 591-parameter signed
  H3/H5/H10 harm head and calibrates harm residuals on a sequence fold that is
  disjoint from both fitting and outer evaluation.
- M27 protected-own RGB-D observation collector for the exact 507 development
  events, with immutable source-feature and runtime-access auditing.
- M28 matched candidate-only versus paired-protected safety-veto model and
  runner. It freezes the 12 M25 actions and compares equal-size 699-parameter
  harm heads under sequence-disjoint fit/calibration/evaluation folds.
- M29 utility-conditioned five-age temporal-harm model and runner. It trains
  and calibrates on exactly one frozen M25 utility-top action per event, keeps
  the original 12-action final policy, and compares matched 8,323-parameter
  candidate-only and paired-protected GRU safety heads.
- M30 utility-conditioned strict-benefit temporal model and runner. It keeps
  the same one-action-per-event substrate but directly predicts the frozen
  H10 gain, H10 branch mean and H5 early-hit-rate components, with nested
  sequence-disjoint lower-bound calibration.

The code includes negative and diagnostic experiments because they are needed
to reproduce why several apparently promising selectors were rejected. Their
presence does not mean that every module is enabled in the best tracker.

## Verified result boundary

The best completed formal VOT-RGBD2022 result remains:

| EAO | ACC | ROB |
|---:|---:|---:|
| 74.020583 | 82.579344 | 89.565651 |

M17-0 and M18-0 are target/split integrity closures, not tracker results. The
sole M18a zero-step architecture execution passed 20 of 21 engineering gates,
including all model-architecture, permutation, monotonicity, gradient-isolation
and exact-state checks. It failed closed at the runtime-side-effect gate because
the observer recorded a subprocess event and a `/dev/null` write without enough
call-site evidence to attribute them safely. Artifact integrity passed, but the
engineering outcome failed; no optimizer step, training result, checkpoint,
prediction or public benchmark metric was produced, and the fixed M18 family is
stopped without rerun. M19a subsequently attributed the bootstrap side effect
to exactly one dependency import-time `uname -p` probe, and M19b extracted a
sealed exact receipt for that single event. M19a/M19b are provenance results
only: neither ran the model, loaded a checkpoint, trained, predicted or evaluated
a benchmark. M20a then executed the frozen causal-survival model with zero
optimizer steps. Its 17 model gates passed, but independent result audit found
mutable observer snapshots and 286 later sensitive-read events that invalidated
the runner's zero-new-runtime-side-effect acceptance. M20a is therefore a sealed
negative engineering result: it produced no checkpoint, prediction or benchmark
metric and cannot authorize training or evaluation. M21a corrected only the two
confirmed observation defects plus review-required journal/authorization gaps.
Its independent Type-A result audit found zero model-runtime side-effect events,
while all 286 repository reads were isolated to the postflight Git identity
phase with exact path and stack validation. All 17 frozen model gates passed,
with zero optimizer construction/step and zero checkpoint write. This is an
engineering-smoke result only; it is not a training, generalization or benchmark
result and does not change the formal VOT/DepthTrack/CDTB metrics. M22a then
completed all 206 frozen optimizer steps with sequence-disjoint training and
held-out evaluation. Its engineering gates passed, but every one of the 121
held-out events abstained: `selected/beneficial/catastrophic = 0/0/0`.
Independent audit therefore sealed the scientific result as FAIL and stopped
the fixed M22 family without threshold scanning or low22 execution. The
held-out data still contained 80 beneficial actions across nine sequences, and
the top dominance candidate was beneficial for 16 of 19 beneficial events;
the failure was absolute survival calibration (the H10 risk gate passed 0/121),
not absence of recoverable candidates. M22 produced no tracking checkpoint or
public benchmark result and does not change the formal metrics.

M23a then collapsed exact duplicate five-frame bbox trajectories and replaced
the miscalibrated quantile gates with direct benefit/catastrophe heads. Its
audited R2 completed 768/768 optimizer steps and moved from M22's total
abstention to four fixed-policy actions: three beneficial, one neutral and zero
strict catastrophic. The mean true H10 gain of selected actions was +0.423557,
but coverage and precision missed the preregistered gates
(`selected=4<5`, `beneficial=3<4`, `precision=0.75<0.95`). In particular,
`file02_indoor` received predicted catastrophe probability 0.001589 despite a
true H10 gain of -0.274504. Independent audit therefore stopped the direct
unique-hypothesis family without threshold scanning, fold-0 access, checkpoint
creation or public evaluation. A subsequent source-only reporting correction
renamed the two negative-polarity data-isolation engineering flags; it did not
rerun or alter the sealed scientific result. M23 produces no formal benchmark
metric change.

M24 then completed its fixed 780-step four-member sequence-OOF committee run.
Engineering and integrity passed, but all 507 events abstained. Only 60 events
had unanimous top-role identity, and no event passed either the fixed
worst-member margin or worst-member benefit gate. Independent audit therefore
stopped the single-fold committee family without scans or public evaluation.

M25 replaced the single-fold committee with four pooled leave-one-fold-out
models while preserving the M23 thresholds. Its audited 2,304-step run restored
12 OOF actions: eight beneficial, three neutral and one catastrophic. The mean
true H10 gain was +0.464147, but precision was only 0.666667. In particular,
`cup14_indoor` received predicted benefit 0.923768 and catastrophe 0.002103
despite true H10 gain -0.343674. M25 is therefore also scientifically stopped;
it produced no checkpoint or formal benchmark result.

M26 then completed its one frozen 1,536-step nested run over 507 OOF events.
The utility head still produced 22 top candidates with benefit probability at
least 0.80, but the empirical sequence-q90 harm calibration made every H3 harm
upper bound positive (minimum +0.076609), so all 507 events abstained. Its
candidate permutation audit was exactly invariant, while event reordering had
a maximum floating discrepancy of 1.9073486328125e-06 and therefore failed the
preregistered bit-exact engineering gate. Independent audit recorded integrity
PASS, engineering FAIL and scientific FAIL. M26 produced no checkpoint or
formal benchmark result and is stopped without threshold scanning or rerun.

M27 then collected five aligned protected-own native RGB, depth and fused
observations for all 507 development events. This was an observation-closure
pass only: no model was trained and no public metric changed.

M28 completed its one frozen 3,072-step matched safety-veto run. The
candidate-only control and paired-protected condition each retained zero of the
12 frozen M25 actions after sequence-q90 harm calibration. Both failed the
scientific gate. The only engineering failure was a candidate-permutation
floating difference of at most `5.960464477539063e-08`; event-order replay was
tensor-exact. Independent audit recorded integrity PASS, engineering FAIL and
scientific FAIL. M28 produced no checkpoint or formal benchmark result and is
stopped without rerun, threshold scanning or public evaluation.

M29 then aligned the safety fit/calibration population with the one M25
utility-top action actually seen per event and retained the full five-age
sequence in a matched GRU head. Its audited 3,072-step run passed every
engineering gate. Both candidate-only and paired-protected conditions retained
the same five original M25 actions: four beneficial, one neutral and zero
catastrophic. Mean true H10 gain was +0.577335, but beneficial precision was
0.8 instead of the frozen 0.95 requirement, and retained actions covered only
folds 2 and 5 instead of at least three folds. The paired condition therefore
did not add causal selection value, and both conditions failed the scientific
gate. M29 is a sealed negative, produced no checkpoint or public metric, and is
stopped without rerun, threshold/q90/architecture scans or public evaluation.

M30 replaced relative harm with direct prediction of the three existing
strict-benefit components. Its audited 1,536-step run passed every integrity
and engineering gate, but sequence-q90 lower calibration made all 12 frozen
actions abstain. The true targets have an eight-action, all-beneficial oracle
boundary, but the raw model would retain two neutral actions and the known
`cup14_indoor@1258` catastrophic action. M30 is therefore a sealed scientific
negative: it produced no checkpoint or public metric and must not be repaired
by relaxing q90, thresholds or the frozen safety gates.

For all experiments, failure analyses, artifact hashes and next-action
restrictions, read the single project master:
[`../../docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md`](../../docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md).

## M39閳ユ彈41 baseline and candidate diagnosis

M39 evaluated STTrack default and global no-update on the same frozen 22 difficult
VOT-RGBD2022 sequences and 303 multi-start anchors. Default achieved
EAO/ACC/ROB **57.135993/75.719622/73.022401**, with 124 confirmed failures;
no-update had 155 failures. These are difficult-subset results, not full127
metrics or evidence that one STTrack checkpoint passed all three datasets.
The preserved full127 VOT result belongs to SUTrack, while the historical
DepthTrack/CDTB results belong to SRTrack.

M40 found that 115 of the 124 failure onsets still contain the target center
inside the default search crop. M41 therefore replays all failure onsets,
exports each response peak's own size/offset box, and compares raw/Hann NMS
top10 and dense candidate capacity. Nine out-of-crop events receive one
independent factor7 shadow. This is GT-timed diagnosis; no ground truth enters
candidate generation beyond normal anchor initialization.

Source and small evidence for M39/M40, new M39 per-sequence metrics, and the
M41 protocol are under [`diagnostics/`](diagnostics/). M41 completed all 124
onsets: 91/115 in-crop failures have an IoU>=0.5 candidate in Hann top10,
including 78 whose first correct candidate ranks second. Dense capacity is
97/115. The nine factor7 shadows have five correct raw-top10 candidates but
only one in Hann top10. These are single-frame oracle capacities, not rescues.

The local Qwen2.5-VL template/relative-position pilot also completed. After
correcting coordinate and causal-reference semantics, its three conditions
localized 4/1/0 of 24 windows correctly; the relative-position condition
abstained throughout. It failed the incremental-information gate and changed
no tracker output. New neural modules must be trained on DepthTrack Train before
the same resulting checkpoint is evaluated on DepthTrack, CDTB and full VOT.
See the master handoff for the latest execution status and limits.

## M42 DepthTrack local association training

M42 completed candidate-own 4x4 RGB/depth collection on 85 existing DepthTrack
Train sequences (63 fitting / 22 development). Two matched 50,858-parameter
heads each received 640 optimizer updates over 20 epochs; the STTrack backbone
and box head stayed frozen and default template adaptation was preserved.
All 126,382 replayed frames reproduced default boxes and confidence exactly.

On 375 development windows, both heads improved mean IoU from 0.414883 to
0.435760, with 10 rescues across six sequences and zero new severe errors.
Their 12 changed choices also include one joint failure and one slight
regression. Their final choices are identical, so the primary hypothesis that
retaining spatial information improves over mean pooling **failed**. The
original recursive gate stopped with zero GPU jobs. See
[the terminal tracker](diagnostics/m42/EXPERIMENT_TRACKER.md) and
[the integrity audit](diagnostics/m42/terminal_audit.json).

M43 completed the separate pooled-versus-default recursive performance test on
all 33,130 frames of 22 development sequences per arm. The frozen pooled primary
gate **FAIL**: mean IoU 0.652226 to 0.617064,
low-IoU frames 7397 to 8553, and persistent
failure episodes 75 to 77.
M42's spatial-information hypothesis remains failed and the spatial arm is
diagnostic only. These are Train proxies; no new formal benchmark score exists.
See [the complete M43 result](diagnostics/m43/EXPERIMENT_TRACKER.md).

## M44 temporal candidate sets

M44 represents all ten current and ten preceding candidates, the two causal
template references and the preceding selected index. A shared native RGB/depth
descriptor and two Transformer layers learn target identity and partial
frame-to-frame target correspondence. The geometry primary includes candidate
center/size coordinates; the equal-parameter appearance control zeroes them.
Both heads have 448,739 parameters. The official backbone, box head, crop and
template schedule remain unchanged; no language benefit is claimed.

Native collection completed on the existing 85-sequence DepthTrack Train
ledger: 2,101 pairs, with 1,511 fitting and 590 previously used development
pairs. All126,382 tracked frames reproduce default boxes/confidence exactly,
including1,420 template writes. Runtime integration checks also passed.
Both fixed20-epoch/960-update training runs completed and their final weights
were strictly reloaded. On590 static development windows, meanIoU is
0.440274/0.441423/0.438411 for default/geometry/appearance. These small static
differences did not transfer to recursive performance. Both complete22-sequence
paths failed: geometry meanIoU.617098,8436lowframes,87episodes;
appearance.640547,7466lowframes,85episodes; default.652226,7397lowframes,75episodes.
All44trajectory/source/weight bindings and independent scalar metrics passed
the terminal audit. No new public metric is claimed. See
[the frozen M44 protocol](diagnostics/m44/EXPERIMENT_SPEC.md) and
[execution tracker](diagnostics/m44/EXPERIMENT_TRACKER.md).

## M45 default-preserving training targets

M45 keeps the same geometry architecture, data, optimizer, seed and runtime.
Its single training intervention labels an already-correct default candidate
as the target, rather than requiring a different box with higher instantaneous
IoU. All-bad sets still use NONE. Current/previous fitting labels change121/174
times. This is a new20-epoch/960-step training from the same fresh initialization,
not added epochs on the failed weights. Initialization/sample-order hashes
match the sealed M44 geometry control.

Static590-window meanIoU is.446766, with15changes,5rescues and1severe regression.
M45 is now complete: mean IoU0.684102374 versus default0.652226263,
6188 versus7397 low-overlap frames, but80 versus75 persistent failure episodes
and a new mobilephone02 failure. Its fixed advancement gate fails; no public
evaluation was launched. All22full trajectories and actual trained tensors
passed integrity checks. See [M45 records](diagnostics/m45/EXPERIMENT_TRACKER.md)
and the master section5.41 for the preserved serialization failure, minimal
analysis-only integer conversion, complete metrics and next training-coverage
hypothesis.

## M46 initialization-frame fitting coverage

M46 adds the omitted fitting frames2through9 from the same63 DepthTrack Train
sequences. It retains the M45 architecture, labels, inference and960-update
budget:20 rounds each sample1511 pairs from the expanded2015-pair fitting pool.
These are sampled rounds, not20complete epochs. Collection uses native default
predicted crops with box/confidence parity checks. Collection and960updates have completed; actual weight audit passed. A
report-only digest-variable error was recovered without extra optimization
or checkpoint changes. Complete22-sequence recursion is now finished: meanIoU.686428, low frames5934,
H10episodes89. Its fixed gate fails because episodes increase and mobilephone02
is broken. No public evaluation was launched. Static590-window meanIoU.445511
has8rescues/4severe regressions and remains diagnostic only. See [the M46 protocol and tracker](diagnostics/m46/EXPERIMENT_TRACKER.md).

## M47 multiple valid correspondence destinations

M47 retains M45's unique action targets and inference, and changes only the
auxiliary opposite-frame positive destination set. Its loss accepts all boxes
meeting the existing IoU criterion for that target. Single-positive loss and
gradient equivalence, empty-target behavior and unlabeled-pair checks passed
before training. The actual20-epoch/960-update run completed with identical M45
action labels, initialization and sample order; strict reload and tensor audit
passed. Static590-window meanIoU is.441918 with7rescues/5severe regressions.
All22development sequences completed the fixed recursive evaluation: meanIoU
.571038, low-overlap frames9946 and H10episodes99. This fails the default
comparison by8.118849percentage points of meanIoU and adds24episodes. M47 is
sealed without public evaluation. No language contribution is claimed. See
[M47 protocol, audit and execution](diagnostics/m47/EXPERIMENT_TRACKER.md).

## M48 native-continuity admission

M48 reuses the exact DepthTrack-trained M45 head without new parameters or
optimization. It accepts a proposed override only when native RGB and depth
RoI continuity to the preceding selected target both meet the default's
values. The fixed fitting audit retains20of96changes and5of26rescues; it
does not establish safety. GPU/CPU decisions and120frames of veto/default
state parity pass, including two native template updates. The complete full22
comparison reaches meanIoU.714587,5331low frames and68H10episodes, improving
on native default. Four of five gates pass, but mobilephone02 gains one
failure, so the protection gate fails and public evaluation is withheld.
See [the M48 audit and protocol](diagnostics/m48/EXPERIMENT_TRACKER.md).

## M49 adjacent-frame motion, scale and template diagnosis

M49 follows the user's motion/scale steering. The native M39 census covers
all303anchors,220180tracking steps and13922unique adjacent GT pairs. Median
GT motion is2pixels; at115in-crop failure onsets, median normalized GT motion
is.088410 while the prediction jumps.865329. Recent templates were correct
when written at117of124onsets. Pure velocity-based crop recentering gains one
failure-center coverage and loses two healthy-center coverages.

GT-free prior-box overlap ranking selects a correct candidate at63of124
onsets, but breaks43native-correct fitting-cache examples while rescuing22,
and breaks15development examples while rescuing8. Its unconditional use fails
the frozen cache screen. These are diagnostics, with no new training, text
calls, recursive run or public benchmark metrics. Motion remains a useful
conditional cue, not a validated standalone selector.
See [M49 measurements, figures and next experiments](diagnostics/m49/REPORT.md).

## M50 scale-triggered template update and full native reference

M50 preserves native periodic updates and adds a dynamic-template write when
native confidence is above0.75 and the predicted linear scale changes by at
least1.25relative to the latest template. The fixed policy adds no learned
parameters. The120-frame disabled-policy comparison matches native boxes,
scores, templates and queries exactly, including a native update. Its full
Train-development22-sequence recursive evaluation is running onGPU0.

The independent native STTrack full127 reference is running onGPU1 with four
VOT workers. It reuses303SHA-verified M39native anchors and evaluates the
remaining1462. Neither run has final metrics at this launch milestone.
See [M50 scope, contracts and launch evidence](diagnostics/m50/LAUNCH_REPORT.md).

## Integrity

`MANIFEST.sha256` binds every file in `overlay/`. Before publication the source
set was checked for common model-weight extensions, files larger than 10 MB and
credentials. The overlay contains no `.pth`, `.pt`, `.ckpt`, `.bin`,
`.safetensors`, `.onnx`, archive, dataset or result payload.
