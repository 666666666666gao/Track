# M42 — native spatial candidate association

Frozen on 2026-09-05 before the new feature collection and fitting. M41 and the
template/position-text pilots are recorded in master sections 5.31–5.32.

## Question and isolated change

Do candidate-own 4×4 native RGB/depth tokens add cross-sequence instance
selection information beyond a matched model that immediately averages those
same tokens? STTrack default remains the reference path. Its official frozen
checkpoint, fixed query window, factor4 search, Hann response, top10/NMS3
candidate generation and default template adaptation are preserved during
collection. No new caption, search scale, regression or backbone change is
combined with this test.

Both arms have 50,858 trainable parameters. Candidate input is K×2×16×768;
reference input is 3×2×16×768. A shared 768→16 projection and a 1024→32→8
pair network compare each modality/ROI with the initial template, latest
default dynamic template and preceding prediction. Six relation vectors plus
eight response/geometry scalars feed a 56→32→1 residual candidate score. A
112→32→1 set-level output represents NONE. Candidate logits add the native
log Hann score; NONE preserves the exact default output. The pooled control
averages each ROI, then repeats its mean across 16 cells before the identical
network. Parameter counts, optimizer, data ordering and epochs are matched.

Template representations are captured at their first causal forward use and
retained until the corresponding image changes. The previous-frame ROI is
cropped around the predicted box. No subsequent GT enters this reference bank.
The latest default template is not assumed to be independently verified as
correct. Both arms receive the same potentially imperfect references.

## Data and execution

Only DepthTrack Train is used. The existing M18 training ledger supplies 85
sequences: folds2–4 give63 fitting sequences, fold5 gives22 development sequences.
These folds have previous experiment exposure and are not a fresh test set.
Folds0/1 are excluded. VOT, CDTB and DepthTrack Test are not read by this run.

The manifest selects up to8 uniformly spaced healthy frames (default IoU≥.5),
8 hard frames (IoU≤.1) and2 unavailable-target frames per sequence, from frame10
through length−5. These are stratified training windows, not exclusively first
failure onsets or estimates of deployment prevalence. Selection uses training
GT; initialization and sealed prediction traces are kept in a separate
inference manifest. Training labels are opened only after feature sealing.

The frozen manifest has1,375 events and126,467 prefix frames including85
initializations. The two shards contain63,065 and63,402 frames. Stored features
use FP16 and fitting converts them to FP32. Expected storage is under1GB.
`toy07_indoor_320` has1,367 image pairs but1,406 GT rows; the39 trailing rows
without images are unused and recorded in `spec.json`.

Replay gates compare every native predicted box with the existing full152
trace (≤1e−4px) and confidence (≤1e−6). The initial120-frame smoke had zero
box/confidence differences and exercised one actual dynamic write in each of
two sequences. Geometry/permutation/local-information and finite-gradient
tests must pass before collection.

## Training and decisions

Each arm is fit for exactly20 epochs, batch32, AdamW lr.001, weight decay.0001,
gradient clipping5, seed2026. The target is the highest-IoU candidate if any
candidate reaches.5; otherwise NONE. There is no heldout epoch selection,
class-reweighting or score-threshold scan. Only final association weights are
saved. The pretrained STTrack backbone/box head remain frozen.

The static information gate requires spatial mean selected IoU to exceed both
pooled and default, more rescued default failures than broken default successes,
and positive mean selected-IoU gain on at least3 development sequences. All
three conditions are required. This permits paired prediction-crop recursive
development evaluation; it does not permit a public benchmark launch directly.

After recursive development gains, freeze the trained model and evaluate the
same low22 set against M39 default. Only clear improvements justify the same
resulting checkpoint's DepthTrack Test, CDTB and full VOT-RGBD2022 evaluation.
Static candidate accuracy is never reported as VOT robustness or a trajectory
rescue. A failed frozen M42 condition is recorded without epoch/threshold scans.

The authoritative machine-readable settings and hashes are `spec.json`,
`launch.json`, the two shard receipts and `training_result.json` when complete.
