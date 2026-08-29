# SUTrack RGB-D Language + Safe Template Update (overlay)

This directory contains only the source/configuration changes developed for the
RGB-D language tracking experiments. It is an **overlay**, not a standalone copy
of SUTrack. Model weights, datasets, VOT workspaces, prediction files, logs,
environment files, and private handoff records are intentionally excluded.

## Upstream base

- Repository: <https://github.com/chenxin-dlut/SUTrack>
- Required base commit: `d65052d1ba3fcf55010e1fb3665ee6616c139a2c`
- Upstream license: [`overlay/LICENSE.txt`](overlay/LICENSE.txt)

To reconstruct the source tree:

```bash
git clone https://github.com/chenxin-dlut/SUTrack.git
cd SUTrack
git checkout d65052d1ba3fcf55010e1fb3665ee6616c139a2c
cp -a /path/to/Track/projects/sutrack_rgbd_language_template/overlay/. ./
```

Configure dataset, language-manifest, checkpoint, CLIP, workspace and Python
paths for your own machine before running. Several frozen experiment utilities
retain the original server paths as provenance defaults; those paths are not
portable and do not point to files included here.

## Included work

- RGB-D frame loading kept independent from training-dataset utilities.
- SHA-bound sequence-level language manifests.
- Fail-closed dynamic-template updates using confidence, response margin, RGB
  identity, raw-depth stability, update spacing and template age.
- Immutable first-frame template plus one bounded dynamic slot.
- Atomic state/template rollback experiments and temporal depth identity.
- Exact VOT multi-start language lookup keyed by `(sequence, anchor_index)`,
  with no sequence-level fallback. The published `identity_only_v1` records
  currently reuse one stable identity sentence within each sequence; they are
  anchor-keyed for integrity, not independently captioned from every anchor
  image.
- A protected/tentative two-frame transaction that separates safe template
  candidates from hard state conflicts. The baseline-first veto variant keeps
  the direct new-template baseline public and uses the old-template branch only
  as a counterfactual veto candidate.
- A metric-blind CUDA preflight and a low-22 machine gate. Transaction full-127
  is never launched automatically, even when the low-22 gate passes.
- VOT/TraX RGB-D bridge updates.
- Train-only tracing, fixed-six causal checks, state-gate training, failure-family
  diagnostics and frozen analyzers.
- Configuration snapshots used by the experiments.

## Verified result scope

The best completed **formal VOT-RGBD2022** result in this project uses the same
DepthTrack-trained SUTrack-L384 checkpoint with sequence-stable identity-only
text:

| Full-127 variant | EAO | ACC | ROB |
|---|---:|---:|---:|
| Structured sequence text | 73.974969 | **82.627562** | 89.455266 |
| Sequence-stable identity-only text | **74.020583** | 82.579344 | **89.565651** |

This is below the SUTrack paper's reported `76.6 / 83.5 / 92.2`; do not describe
the overlay as improving the official baseline. The identity-only cleanup gives
only `+0.045613 EAO / -0.048218 ACC / +0.110385 ROB` percentage points over the
project's structured-text run.

Before any new full-dataset run, the annotation change was evaluated only on a
frozen low-metric subset (`ACC < 0.70 OR ROB < 0.75`): 22 sequences and 303
multi-start anchors. Replacing the structured sequence text with category plus
stable identity-only text changed the subset metrics as follows:

| Low-22 variant | EAO | ACC | ROB | Confirmed failures |
|---|---:|---:|---:|---:|
| Structured sequence text | 42.629281 | 71.827916 | 53.412816 | 200/303 |
| Anchor-keyed sequence-stable identity-only text | **43.274104** | **72.065511** | **54.388022** | **195/303** |
| Qwen current-anchor visual identity text | 42.946692 | 71.910688 | 54.305144 | 202/303 |
| Protected/tentative transaction v1 | 21.412955 | 61.659985 | 28.125866 | 251/303 |
| Template-only transaction v2 | 43.106822 | 72.019219 | 54.162754 | 201/303 |
| Baseline-first template veto v3 | 43.273265 | 72.064522 | **54.388022** | **195/303** |

This passed the pre-registered low-22 gate and therefore authorized a separate
full-127 validation only for the identity-only text. The Qwen visual annotation
and both historical transaction variants failed their own low-22 gates, so no
full-127 run was authorized for them.

The v1 transaction failed because state conflicts froze the recursive bbox. v2
removed that behavior but still made the old-template branch public while the
direct baseline had already committed the new template. The included
`baseline-first counterfactual template veto` fixes this semantic reversal:

```text
protected/public = direct baseline with new template
tentative/shadow = counterfactual old template
promote tentative = veto the template update
rollback/timeout/error = keep the direct baseline
```

Its structural checks pass, and a real 12-frame CUDA replay of
`cube02_indoor_2@450B` opens one template transaction while remaining exactly
aligned with the direct baseline through rollback. The formal low-22 run opened
3,050 transactions and issued five old-template vetoes. All 303 anchor failure
outcomes and failure progress values remained exactly equal to the direct
baseline: there were no new catastrophic failures and no rescues. EAO changed
by `-0.000840` percentage points, ACC by `-0.000989`, ROB by `0.000000`, and the
confirmed-failure count stayed at 195. The strict EAO/ROB improvement gate
therefore failed, no full-127 run was authorized, and no gain is claimed. This
result shows that the baseline-first transaction removes the historical
regression but that online branch utility alone is too weak and inactive to
improve robustness. The next testable path is a train-only survival/template-
veto gate learned from DepthTrack Train recursive rollouts; the code contains
no automatic full-127 launch path.

A frozen six-sequence DepthTrack-Train language ON/OFF study also rejected
unconditional structured language for VOT robustness: language ON improved mean
single-start IoU by `+0.009006`, but increased proxy failure episodes from `16`
to `18`, with only `2/6` sequences non-negative. A pre-registered short
appearance+category prompt reduced the mean gain to `+0.004008` and still
increased failures to `17`, with only `1/6` sequences non-negative. A post-hoc
OFF/structured/short GT oracle reached `+0.025354` mean IoU but did not reduce
the `16` OFF failure episodes. It therefore proves overlap capacity, not the
robustness/recovery capacity needed to justify formal VOT. The next protected
path keeps language code available but prioritizes a shadow/tentative template
rollout; formal VOT remains gated on Train-only failure and cross-sequence safety.

DepthTrack Test and CDTB results are protected references and are not included as
new runs in this publication snapshot.

## Integrity

`MANIFEST.sha256` binds every published overlay file. The repository-level
`.gitignore` rejects common checkpoint and output formats; the commit was also
scanned for files larger than 10 MB and for model-weight extensions before push.
