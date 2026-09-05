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
| New data collection | Complete:63sequences,567tracked frames,504pairs,545606694bytes; exact default box/confidence |
| Training | Complete960updates; all2015inputs sampled; actual-tensor audit PASS |
| Full recursion | Running since15:13:25CST, Python20236/20237; all22 existing development sequences,16567/16563 frames |
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


The original training completed all960updates and saved the final checkpoint,
then failed while constructing its report because a source-hash loop overwrote
the sample-order digest variable. The original training/controller exits1 are
preserved. `finalize_m46_training.py` reconstructs metadata from actual input
files, the deterministic sampler and20logged rounds, then evaluates the saved
weight. It contains no backward, optimizer step or checkpoint write; the
checkpoint SHA before/after is identical. Its exit is0, followed by a successful
CPU tensor audit. The fixed runtime/spec/trainer source remain untouched.

Final head SHA:10977f0259d246f679351fb976c99c07763c7f1f63d3ebe49a085dc640987153.
Training result SHA:65e76de916d1984309215c1336f7315dff226e1a1b13efdab6d44c322cdbe5cf.
Static590development meanIoU.445511132, correct278, changes68, NONE195,
8rescues/4severe regressions. It is not an advancement result or a reason to
change the precommitted all22recursive run.

Original recursive-stage commands are recorded in `recursive_pipeline.sh`;
its controller20235 records `recursive_controller.exit` separately from the
original failure. The new screen is sttrack_m46_recursion_20260905. Both original
Python handles were verified alive. Expected terminal around15:30CST; next
meaningful check around15:27. No public evaluation has been launched.
