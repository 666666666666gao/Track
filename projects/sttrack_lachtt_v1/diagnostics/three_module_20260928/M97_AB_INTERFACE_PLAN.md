# M97 A+B actual-input interface prototype

M96 did not justify recursive promotion. Implement the planned A+B interface
in an isolated diagnostic folder, leaving STTrack and all official runtimes
unchanged. This is wiring/gradient validation, not semantic model training.
Human phrase and physical-instance labels remain blank.

Collect only the first eight former-fit130 sequences in the fixed M90 manifest,
at each sequence's earliest existing M90 event (all frame10; total80 native
prefix calls). Do not select by outcomes. Reproduce all prefix native boxes
and scores, and the stored event's ten proposals and local RoIs. In a legal
t0 auxiliary forward, read the initial search RoI and its surrounding ring;
do not commit the returned query. At the current event, sample12 perimeter
locations of a2x expanded candidate box, plus the existing16 local samples.
Record valid sample masks by nearest sampling the native crop's actual
attention-padding mask at the same RoI coordinates, and record
raw-depth nonzero fraction within each proposal. These are input support
descriptors, not calibrated modality reliability or identity labels.
These masks describe support at the sample location, not the entire ViT
token's receptive field. Recompute the native double-precision resize from
sample_target; do not use a float32 cached resize to decode exact proposals.
Collection reads no subsequent GT, text or human answers. Use one-case smoke then two
four-case GPU shards. No full3502-event recache yet.

A keeps local token layout, initial/current and region/context roles, and
separate RGB/Depth projection. Current visual evidence reads initial/context
tokens. Phrase queries then read those visual tokens; candidate visual tokens
read the bound phrases. Emit candidate features, per-phrase3-state evidence
logits, and learned modality weights. The state logits receive no semantic
supervision or calibration; indirect localization gradients in the throwaway
probe do not make them support/conflict/unobservable truth. Shared q-minus-empty computation
makes the semantic increment zero for exactly empty content with unchanged
slots/masks. This reuses M84's boundary idea and is not a novelty claim.

B reads the candidate set through a permutation-equivariant attention layer,
then emits selection logits, visual-only localization-quality logits, and an
observation logit. Selection starts as the native log response through a
zero-initialized residual. Local geometry is frozen at this first interface
stage, as specified in the staged design; no new frame is committed. Box,
selected score, quality and feature are gathered using one final index.

On these eight actual inputs, check native zero-residual choice, candidate
permutation equivariance, empty-versus-visual equality after a nonzero update,
and selected-field alignment. Read the existing **unverified category-only**
bank only in the separate smoke probe (768-dimensional five-slot tokens;
active attribute slots are exactly the encoded empty vector, padding zeros).
Use two optimizer steps of Train-GT IoU selection/quality BCE solely to verify
nonzero gradients reach both interaction directions after zero-head warmup.
Do not supervise physical identity, phrase states or observation from IoU.
Do not save an official/final checkpoint, claim semantic gains, launch a full
training budget, or evaluate any public dataset. The observation head remains
untrained pending a defensible task label.

Deliver source, fresh predeployment review, native-equivalence receipts, and
actual-input sanity JSON. Passing these checks proves wiring only. Next
semantic training still requires independent reviewed Train phrases and
candidate instances, and the content-versus-visual development gate.
