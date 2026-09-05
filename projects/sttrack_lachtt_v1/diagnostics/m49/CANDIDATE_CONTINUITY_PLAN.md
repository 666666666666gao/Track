# M49 supplementary candidate-continuity diagnosis

Frozen after the completed motion/scale census, before measuring candidate selection. This is an adaptively motivated diagnostic, not a preregistered independent validation.

The census found small GT motion but large predicted jumps at many native M39 failure onsets. Use the already sealed M41 factor-4 Hann top10 candidates at every one of the 124 onsets. Select without GT, seal the decisions, then score them with bounded VOT IoU.

Fixed comparisons, no threshold search:

- Native top1, as the reference.
- Nearest candidate center to the previous native predicted center: position-only diagnostic.
- Maximum continuous box IoU with the previous native predicted box: primary position-and-scale continuity diagnostic.
- Maximum continuous box IoU with the constant-velocity predicted box, keeping previous native predicted width/height: velocity diagnostic. Use the M41 native prior state and the sealed preceding trajectory box only.

Ties retain the earliest response rank. The previous box is a ranking reference, never inserted as a new candidate. A candidate is labelled correct only after selection if bounded VOT IoU is at least 0.5. Report all124, inside115/outside9, all sequences, remaining missed capacity, and sealed choice/input hashes. No online acceptance rule or recursive improvement is claimed from these GT-selected failure frames; healthy and recursive controls on DepthTrack Train are still required.

Also create an explanatory figure from fixed examples: cube02_indoor_1@300B, two_tennis_balls_3@0F, toy09_indoor_1@0F, yogurt_indoor_1@1000B. These illustrate large-motion exceptions and small-motion prediction jumps; they do not estimate prevalence.
