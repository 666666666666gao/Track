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
- Published source commit: `3426dfc7dd06dc65506bd128a332d15b0b2ec845`
- Overlay archive SHA256: `14a9375fc8ca4eddfc5e9e73245759a77911e90be54f4c1057683613afd8645a`
- Published project files: 54

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

The code includes negative and diagnostic experiments because they are needed
to reproduce why several apparently promising selectors were rejected. Their
presence does not mean that every module is enabled in the best tracker.

## Verified result boundary

The best completed formal VOT-RGBD2022 result remains:

| EAO | ACC | ROB |
|---:|---:|---:|
| 74.020583 | 82.579344 | 89.565651 |

M17-0 is only a target/split integrity closure. It did not train a model or
produce a new public benchmark result. For all experiments, failure analyses,
artifact hashes and next-action restrictions, read the single project master:
[`../../docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md`](../../docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md).

## Integrity

`MANIFEST.sha256` binds every file in `overlay/`. Before publication the source
set was checked for common model-weight extensions, files larger than 10 MB and
credentials. The overlay contains no `.pth`, `.pt`, `.ckpt`, `.bin`,
`.safetensors`, `.onnx`, archive, dataset or result payload.
