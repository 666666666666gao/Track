# M52 paired training and recursive evaluation implementation

Prepared on 2026-09-05, before M52 fitting-label analysis or optimizer steps.
Collection remains the independently launched job described in `launch.json`.
This implementation record does not claim completed training or performance.

## Frozen comparison

Both arms use a fresh seed-2026, 448,739-parameter candidate association head
with the original M45 absolute candidate geometry, its default-priority labels,
20 epochs, batch size 32, and 1,900 optimizer steps. The control repeats the
original 1,511 fit events twice. The mixed arm pairs those same physical events
with captures from the fixed M45 policy's own recursive trajectory. There are
3,022 logical views per arm, not 3,022 independent physical events.

The trainer verifies capture receipts and hashes before loading existing
DepthTrack Train GT labels. Current and previous labels are recomputed for
each view's own candidate boxes. Candidate zero denotes the native/default
box on that view's state; the actually selected output box is stored separately.
The mixed input uses the recorded previous candidate choice. Both arms must
have identical initialization and logical sample-order digests.

## Execution and validation

`run_training.sh` executes the data audit, control training, mixed training,
the actual tracker-loader contract, four recursive shards, then analysis.
The 22 development sequences contain 33,130 frames per arm, 66,260 in total.
Tracking uses no development GT; after the trajectories are written and
sealed, analysis loads GT and recomputes continuous overlap and H10 metrics.
This is a repeated DepthTrack Train development evaluation, not official
DepthTrack Test, CDTB, or VOT scoring.

`PRE_TRAINING_NOTE.md` distinguishes two outcomes before any M52 result:
mixed data benefit requires passing both native-baseline and paired-control
gates; a control that independently passes every native gate may advance
only as an additional-training improvement. No public evaluation starts
automatically. The original collection specification is unchanged.

The prepared source passed local AST parsing, remote Python 3.8 `--help`
imports, and `bash -n`. Those checks do not establish a completed data audit,
successful training, tracker-loader equality, or performance improvement.

## Advisory review and source completeness

The independent-context GPT-5.5 xhigh review is recorded verbatim in
`EXPERIMENT_AUDIT.md`, with its WARN verdict retained. This is a GPT-family
Type-A advisory review, not a cross-family acquittal. At review time only
collection launch and the 240-frame capture contract were available.
Completion receipts will be added as they are actually produced.

The exact native tracker source has subsequently been supplied in
`source_snapshots/sttrack.py` with the SHA-256 recorded in its README. This
adds review evidence without modifying the running tracker or collection.
