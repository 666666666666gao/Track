# M44: target identity across complete candidate sets

M43's first-override audit establishes two adjacent-egg instance errors while
the default candidate is correct. M42's head sees individual current candidates
and one preceding prediction RoI; it does not represent the preceding candidate
set or supervise actual frame-to-frame target correspondence. M44 tests this
missing relation using native STTrack observations.

Primary performance question: can the trained temporal candidate-set model
improve complete recursive tracking over STTrack default? Supporting mechanism
question: do explicit candidate positions/sizes improve the same network over
an equal-parameter appearance/response-only control?

The primary is **geometry**, frozen before feature collection and optimization.
The **appearance** control zeroes only explicit center/size coordinates. Both
retain the same RoIs, response scores, preceding selected index, template
references, architecture, initialization, optimizer and sample order. No
post hoc choice of primary is allowed. Existing M42/M43 results stay sealed.

## Architecture and supervision

- Ten current plus ten immediately previous NMS candidates, each with native
  4x4 RGB and depth RoI tokens; immutable initial and actual dynamic template
  RoIs; previous selected-candidate marker.
- Shared per-cell 768-to-32 projection and object projection to128 dimensions;
  two PyTorch Transformer layers, four attention heads, FFN256, dropout0.
- Geometry arm adds normalized image-coordinate center/size embeddings.
- Eleven-way identity prediction: ten candidates plus NONE. NONE preserves
  ordinary default output. Initial identity logits preserve default before any
  optimization; the first tracking frame collects preceding candidates.
- Partial correspondence supervision uses the true target candidate in the
  two training frames (or unmatched when absent). Other instances have no
  fabricated identity labels. Total loss is identity CE plus0.25 times the
  mean available forward/reverse target-match CE, temperature0.1.
- Official STTrack backbone/box head, searchfactor4 and default template
  update schedule stay frozen. No new text, wider crop or update threshold.

## Training and data

Use the existing85-sequence Train ledger: folds2/3/4 (63 sequences) fit;
fold5 (22 sequences) development. These folds have prior development exposure;
the official frozen backbone also predates the new head split. This is not a
claim of entirely unseen sequences or new external data.

Per sequence, select up to12 uniformly spaced healthy frames,4 intermediate
frames,2 late low-IoU frames,2 unavailable frames, and up to6 persistent-failure
onsets with offsets-2/0/+2. Overlapping selections are deduplicated. Each event
stores the native candidate sets at t-1 and t. Sampling GT is Train-only;
the inference file contains only normal initialization GT, frame requests and
expected default predictions. Collection never opens the label file.

Both heads train for fixed20 epochs with seed2026, batch32, AdamW lr0.0003,
weight decay0.01, gradient norm1.0. Final epoch only, no threshold or epoch
selection. Training source/runtime hashes are bound before the first optimizer
step. Collectors bind their source, data plans and official checkpoint first.

## Decisions and scope

Static snapshot results are diagnostic. M43 showed that identical snapshot
choices can lead to different recursive results, so the decisive test runs
both complete22-sequence recursive paths after collection/training integrity,
regardless of the static performance ranking.

The geometry primary must improve global Train meanIoU by at least0.01, reduce
IoU<=0.1 frames, not increase >=10-frame failure episodes, improve at least3
sequence means, and add no episode to a default-zero-episode sequence.
Geometry superiority over appearance is separately required to claim an
explicit-position contribution; it is not a substitute for main performance.

Only the main recursive pass permits a frozen low22 comparison against M39
default. That comparison requires EAO and ROB each improve at least1pp, ACC
drop no more than0.10pp, fewer failures, and preservation of all7 zero-failure
sequences. Only then evaluate the same frozen model on DepthTrack Test, CDTB
and VOT full127. Overall VOT targets remain ROB>93.7, EAO>77.9, ACC>82.1.

This model does not yet generate language or prove an ordinal description
benefit. Reliable candidate identities and current positions are prerequisites
for subsequently testing true/empty/wrong/shuffled language conditions.

## Upstream reference

KeepTrack motivates matching complete candidate sets, including distractors,
with explicit unmatched outcomes and partial target correspondence labels.
Its implementation combines candidate descriptors, coordinates and response
scores using self/cross attention. This is a new native-STTrack implementation
using PyTorch layers; no upstream source or weights are copied.

[Official association model](https://github.com/visionml/pytracking/blob/master/ltr/models/target_candidate_matching/target_candidate_matching.py)
[Official matcher](https://github.com/visionml/pytracking/blob/master/ltr/models/target_candidate_matching/superglue.py)
[Official training](https://github.com/visionml/pytracking/blob/master/ltr/train_settings/keep_track/keep_track.py)
