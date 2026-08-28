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
  candidates from hard state conflicts, then atomically promotes or rolls back
  the complete recursive state.
- A metric-blind CUDA preflight and a low-22 machine gate. Transaction full-127
  is never launched automatically, even when the low-22 gate passes.
- VOT/TraX RGB-D bridge updates.
- Train-only tracing, fixed-six causal checks, state-gate training, failure-family
  diagnostics and frozen analyzers.
- Configuration snapshots used by the experiments.

## Verified result scope

The best completed **formal VOT-RGBD2022** result for the current frozen
SUTrack-L384 + structured-language + safe-v1 configuration is:

| EAO | ACC | ROB |
|---:|---:|---:|
| 73.974969 | 82.627562 | 89.455266 |

This is below the SUTrack paper's reported `76.6 / 83.5 / 92.2`; do not describe
the overlay as improving the official baseline. Paired diagnostics indicate that
the current safe-template policy increases some failure chains.

Before any new full-dataset run, the annotation change was evaluated only on a
frozen low-metric subset (`ACC < 0.70 OR ROB < 0.75`): 22 sequences and 303
multi-start anchors. Replacing the structured sequence text with category plus
stable identity-only text changed the subset metrics as follows:

| Low-22 variant | EAO | ACC | ROB | Confirmed failures |
|---|---:|---:|---:|---:|
| Structured sequence text | 42.629281 | 71.827916 | 53.412816 | 200/303 |
| Anchor-keyed sequence-stable identity-only text | **43.274104** | **72.065511** | **54.388022** | **195/303** |

This passed the pre-registered low-22 gate and therefore authorized a separate
full-127 validation. That full validation is still running at the time of this
source snapshot, so its result must not replace the formal numbers above.
Likewise, the protected/tentative transaction has passed structural checks but
has not yet produced a low-22 VOT result; no robustness gain is claimed for it.

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
