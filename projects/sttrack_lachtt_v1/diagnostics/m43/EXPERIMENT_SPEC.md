# M43 — recursive performance of the trained pooled control

M42 is complete and its primary spatial-versus-pooled information gate failed:
the two trained heads chose identical candidates on all375 development windows.
This result is retained. No M42 gate, weight, epoch count or decision threshold
is changed.

Both heads nevertheless improved the default snapshot predictions, with10
rescues, one joint failure and one slight still-correct regression among12
changes. M43 asks a separate, exploratory performance question: can the already
trained **pooled control** improve full recursive tracking over default?
The pooled head is the primary candidate fixed before these trajectories are
observed. Spatial is run as a secondary diagnostic and cannot replace the
primary after seeing the results.

Run every available frame of all22 existing Train fold5 development sequences:
33,130 frames per arm,66,260 across the two arms including initialization. These
sequences have already been used for development, including M42 snapshots; no
new/unseen-test claim is made. All native tracker, reference-bank, runtime and
checkpoint hashes are frozen in `spec.json`. Both heads were trained only on
M42's63 DepthTrack Train fitting sequences. M43 has zero optimizer steps and no
additional architecture or threshold change.

Each arm owns the full crop/query/template/reference state. Initialization is
normal first-frame GT; subsequent GT is read only after both trajectories are
sealed. Default comparisons use the frozen full152 prediction cache, filtered
to these22 sequences. The two weights and the official base remain unchanged.

The primary pooled-versus-default gate requires all of:

1. Higher global mean continuous xywh IoU.
2. Fewer valid frames with IoU≤.1.
3. No increase in episodes of at least10 consecutive valid low-IoU frames.
4. Positive mean IoU gain on at least3 sequences.
5. No persistent failure added to a default-zero-episode sequence.

Initialization and invalid-GT frames are excluded from IoU averages; invalid
GT breaks an episode. The same proxy calculations already verified for M42's
recursive preparation are reused. They are not VOT EAO/ACC/ROB or formal
DepthTrack Test/CDTB metrics.

A pooled primary pass permits freezing a low22 comparison against M39 default.
No automatic public benchmark is attached. Any later low22 and three-dataset
results must use the same frozen trained model identity. Even if pooled is
useful, M42 does not establish an independent benefit of retained4×4 structure.
