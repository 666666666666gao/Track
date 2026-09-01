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
- Published source commit: `d83fbbdd0286a535e8ec9c915313bb75de84c7e9`
- Published project files: 61

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
metric and cannot authorize training or evaluation. For all experiments, failure
analyses, artifact hashes
and next-action restrictions, read the single project master:
[`../../docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md`](../../docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md).

## Integrity

`MANIFEST.sha256` binds every file in `overlay/`. Before publication the source
set was checked for common model-weight extensions, files larger than 10 MB and
credentials. The overlay contains no `.pth`, `.pt`, `.ckpt`, `.bin`,
`.safetensors`, `.onnx`, archive, dataset or result payload.
