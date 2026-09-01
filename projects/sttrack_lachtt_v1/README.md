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
- Published source commit: `f5e754d2d804f71124a63a014a3b58d76932b855`
- Published project files: 68

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

For all experiments, failure analyses, artifact hashes
and next-action restrictions, read the single project master:
[`../../docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md`](../../docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md).

## Integrity

`MANIFEST.sha256` binds every file in `overlay/`. Before publication the source
set was checked for common model-weight extensions, files larger than 10 MB and
credentials. The overlay contains no `.pth`, `.pt`, `.ckpt`, `.bin`,
`.safetensors`, `.onnx`, archive, dataset or result payload.
