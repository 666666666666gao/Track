# M46 initialization-frame training coverage

M45 improved aggregate recursive mean IoU but failed its fixed stability and
successful-sequence protection criteria. Its earliest fitting input is frame10,
whereas the runtime makes association decisions from frame2. The first egg
regression occurs at frame6. This is an observed input-coverage gap; it does not
establish the cause of the phone regression at frame62 or all later errors.

M46 adds all frames2through9 from the same63 DepthTrack Train fitting sequences.
The504 new consecutive pairs use exact STTrack-default predicted crops, native
RGB/depth candidate-own tokens, and the causal query/template state. The existing
collector checks each box and confidence against the sealed default trace.
It receives initialization boxes and never opens the separate training-label
file. No new development-sequence or public-test frame is added to fitting.

The architecture is the unchanged448739-parameter candidate-set head on the
frozen official STTrack backbone. M45's default-priority target rule, loss,
inference argmax/NONE behavior, crop, query and default template updates remain.
There is no frame-number exclusion, new confidence threshold or language input.

The fitting pool grows from1511 to2015 pairs. Each of20 seeded rounds takes the
first1511 entries of a new permutation of the full fitting pool, without
replacement within a round. Batch32, constant AdamW learning rate.0003,
weight decay.01 and gradient clipping1 give960 updates, matching M45. These are
**20 sampled optimization rounds, not20 complete epochs over2015 inputs**.
Fresh initial weights must exactly match the M45 control. Sample order changes
with the data-pool intervention and is recorded, not claimed to be identical.
Final checkpoint only; no intermediate checkpoint selection.

| Stage | Status at launch |
|---|---|
| Source/protocol | Frozen before collection and fitting; see `spec.json` |
| New data collection | Launched in two shards,288/279 tracked frames plus63 initializations |
| Training | Pipeline follows successful collection; actual completion pending verification |
| Full recursion | Scheduled for all22 existing development sequences,16567/16563 frames |
| Advancement | Same STTrack-default gate, then fixed low22 gate; no automatic public launch |

Root: `/root/autodl-tmp/sttrack_m46_initial_frame_v1_20260905`.
Screen: `sttrack_m46_initial_frames_20260905`.
Experiment spec SHA:
`1e0fd942361b20c16eae377398a11f5e88c68ef3069fec3bdd72dcf497a01987`.
New collection spec SHA:
`87bb8997675d73c2c79e9b36bd7183969bfa8804650ba6591fdd8f969b868b6f`.

The primary recursive gate still requires mean IoU at least.01 above STTrack
default, fewer low-overlap frames, no increase in H10 episodes, at least3
positive sequence means, and preservation of original zero-episode sequences.
The same new model can reach three complete datasets only after clear low22
improvement. M45's failure is not retrospectively promoted.
