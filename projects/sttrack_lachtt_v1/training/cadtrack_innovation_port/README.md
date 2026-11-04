# CADTrack pretrained baseline with the existing innovations

This separate port starts from official `CADTrack_ep0020.pth.tar`, SHA256
`d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8`.
It does not use or resume the failed epoch31 fine-tune. CADTrack's native
tracker loads all250 state entries strictly before any module is attached.
The archived STTrack ROI experiment and both old CADTrack runs are untouched.

## Architecture

The frozen CADTrack backbone supplies the last256 RGB and auxiliary tokens
separately (each768 channels); a hook on `decode_fuse_search` supplies the
actual768-channel fused search grid before the center head. Hooks return no
replacement outputs. The original native head, Hann response, candidate box
decode and floating-point image mapping are retained. The frozen CLIP ViT-L/14
dense reader observes the same RGB search crop. Padding masks are crop
validity, not sensor reliability or visible-object annotations.

The published `RegionConditionedTargetDecoder('roi')` and `TemplateWriteC`
architectures are reused without edits, via `--legacy-code`. The ROI arm
retains inherited full-minus-null first-frame language conditioning; only
the new regional semantic delta is zero. The existing human-confirmed
Train152 phrase bank and five-slot mask are reused. All new A/B/C parameters
start fresh with seed2027. STTrack-trained module weights are not silently
treated as CADTrack-compatible pretrained weights.

CADTrack returns `track_query_before=None`; no temporal-query list is invented.
Its two-template memory retains the immutable initial template. C gates only
native-qualified writes (every5 frames, native selected-cell Hann confidence
above0.65), with the inherited raw-C threshold0.5. The STTrack cadence50 and
threshold0.75 are not copied. C receives detached current/past selected
features, first-frame reference, predicted quality and motion, and is
supervised against selected-frame current GT IoU. This is not a future
template-utility or physical depth-reliability module.

## Separate Training Protocol

`train.py --mode train` declares six full own-history passes through all152
DepthTrack training sequences,219802 track calls per pass, seed2027. CADTrack
and CLIP remain frozen; A/B/C use AdamW LR3e-5, weight decay0.01, accumulation32,
gradient norm1, the inherited precision-preservation loss and current-IoU C
MSE. Current GT is read for loss after action, never for crop reset or module
inputs. The inherited documented toy07 GT-tail alignment is retained.

The fixed pass06 final, with one common runtime and inference profile, is the
only intended future candidate for complete Test50/CDTB80/VOT127. No test
threshold, best checkpoint, dataset-specific weight or profile selection is
implemented. The old CADTrack fine-tuning recipe LR1e-5/milestones[10,30]
and final-five test-set selection are a different experiment, not this port.
This architecture transplant is not a controlled single-factor optimization
of the completed ROI run. No performance improvement is established.

One rolling `latest.pt` stores A/B/C, optimizer, CPU/CUDA/Python/NumPy RNG,
completed sequence position and input/source bindings after gradient flush.
The native checkpoint/CLIP stay external and frozen. There is no resume
loader and no claim of verified end-to-end exact recovery. Output directories
must be new; no automatic retry or overwrite of an existing run is allowed.
`final.pt` stores A/B/C and the protocol. `load_final_tracker` strictly loads
the common trained final, checks native/config/CLIP binding and rejects smoke
weights; callers supply a dataset's predeclared first-frame-only phrase bank.
The factory deliberately has no metric-selection logic.

## Verification And Commands

Both commands take the same required arguments:

```text
--legacy-code /root/autodl-tmp/m125_region_roi_fixed6/code
--repository /root/autodl-tmp/external_baselines/cadtrack_rgbd_probe/source
--checkpoint /root/autodl-tmp/external_baselines/cadtrack_rgbd_probe/CADTrack_ep0020.pth.tar
--config /root/autodl-tmp/external_baselines/cadtrack_rgbd_probe/source/experiments/cadtrack/cadtrack.yaml
--clip-weight /root/autodl-tmp/sutrack_assets/weights/ViT-L-14.pt
--bank /root/autodl-tmp/sttrack_m113_human_initialization_20261006/human_text.pt
--labels /home/SUTrack_RGBD_L/.aris/m113_human_initialization_20261006/human_train_labels.json
--spec /root/autodl-tmp/sttrack_full152_paired_20260925/M82/training_spec.json
--output NEW_SEPARATE_DIRECTORY
--mode smoke
```

Use `/root/autodl-tmp/envs/sttrack/bin/python -B` with CUDA devices0/1 and
bytecode writes disabled. `verify_native.py` checks hook/no-hook network-output
equality,64-frame native/zero-residual box and confidence equality, and exact
template histories on the first bound training sequence. C is bypassed only
in this explicit native parity control. Half-scale response at zero residual
is inherited from quality sigmoid0.5; it is disclosed, not score parity.

`train.py --mode smoke` makes two32-frame optimizer steps on that same training
sequence, with C enabled, checks finite nonzero A/B and C gradient norms,
frozen-state equality, new-module changes and exact save/reload. It writes
only to a separate smoke directory. Neither check produces official metrics.

The production `--mode train` entry exists for the declared protocol, but
full training must not start until these checks and deployment review finish.
The original client catalog supplied no subagent operation; a subsequent fresh,
ephemeral Codex CLI review of99dfd9f using model gpt-6-astra / max reasoning
found no blocking issues. This is same-family static review with provisional
acceptance, not independent runtime or full-result auditing. Exact report and
provenance are in DEPLOYMENT_REVIEW.md and deployment_review.json. The only
post-review core change updates the review-status metadata. The resource read
showed two idle RTX3090 GPUs,
93MB free on system overlay and8.3GB on the data mount. No cleanup or package
installation is authorized. Small smoke artifacts fit on the data mount;
At that review stage, the full-run disk/log budget and resource admission
still needed a launch-time check; the completed check is recorded below.

## Observed Validation

Both remote commands passed on the bound Train152 `cube04_indoor` sequence.
The native64-frame control included one actual template write. The separate
C-enabled smoke made two A/B and two C steps; both gradient norms were finite
and positive. A subsequent read of the saved AdamW state found positive
first-moment norms in all20 module-prefix groups across71 parameter entries,
including language and regional branches. Frozen CADTrack/CLIP state and
module serialization checks passed. Exact code and receipt SHA256 values
are in `validation.json`; no new formal metric or performance claim follows.
At that validation stage formal six-pass training had not started; see the
subsequent formal launch record below.

## Supervised Full Training

`supervise.py RUN_ROOT` runs the single argv command in RUN_ROOT/launch.json,
using its declared cwd and environment. It creates train.log exclusively,
records the actual child/supervisor PIDs, waits for that child and preserves
train.exit plus terminal.json. It never retries or starts evaluation. Its CPU
test checks actual child success and failure7, stdout and matching receipts.

Before launch, copy this port and the unchanged legacy sources into the new
run's code directories. Point --legacy-code at that snapshot, and place the
new --output, logs and TMPDIR on the data mount. Use start_new_session=True
when starting the supervisor so an SSH disconnect cannot terminate training.
The first full Train152 inventory check found152 sequences,219954 matching
RGB/depth frames and219802 calls/pass, with all GT hashes and initial boxes
matching the bound spec. Six passes must total912 sequence runs/1318812 calls.

Measured smoke sizes: latest5916444 bytes, final1968308 bytes; rolling plus
temporary replacement and final peak13801196 bytes. A256MiB training-output
budget covers these small module states,912 sequence records, source snapshots
and bounded console logs. The observed free data space was8888422400 bytes.
The completed ROI evaluation occupied170436983 bytes as a sizing reference,
not a guarantee of new evaluation size. Do not clear the93MiB system overlay;
disable bytecode writes and direct temporary/output files to the data mount.

Formal training launched2026-11-04T01:28:38Z in the separate data-mount root
`/root/autodl-tmp/cadtrack_abc_train152_fixed6_20261104`, using frozen code from
90e4863 and byte-exact native/legacy source snapshots. The initial actual
process check confirmed supervisor280132 detached with PPID1 and child280133
running the full-training argv. No terminal receipt or final result existed.
FORMAL_LAUNCH.json records this observation; it is not a completion receipt.
Do not change this running snapshot or launch another copy. No new formal
metrics are available; the existing ROI audit limitations remain unchanged.
At2026-11-04T01:36:00Z the same original process chain was live, and the
first complete sequence cube04_indoor recorded2575 calls,81 A/B steps and
81 C steps in262.616683 training-loop seconds. latest.pt was present. This
single-sequence timing does not establish a stable full-training ETA.
