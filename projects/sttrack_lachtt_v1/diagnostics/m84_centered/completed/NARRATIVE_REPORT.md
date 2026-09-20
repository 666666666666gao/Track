# M84 completed development evidence

M84 uses a frozen STTrack backbone/head and the centered shared adapter `F + D(x,q) - D(x,empty)`. One Category model was trained from the original zero initialization with seed2027, 130 full sequences initialized from the first frame, 186694 tracking calls and 5798 optimizer steps. Raw competition and reliable-native spatial preservation remain enabled. The two adapter branches add computation, and the canceled final bias reduces effective parameters: this is not a strictly compute-matched or effective-capacity-matched comparison with M82. No new memory, online caption, Hann policy or template threshold was introduced.

All three independent content recursions completed before GT analysis: 22 sequences, 33130 frames each; 28897 valid non-initialization frames contribute to metrics. These are repeatedly used DepthTrack Train development sequences, not official DepthTrack Test, CDTB or VOT results.

| Condition | Frame-weighted mean IoU | Sequence mean IoU | IoU <= 0.1 frames | H10 runs |
|---|---:|---:|---:|---:|
| Native | 0.652226 | 0.684336 | 7397 | 75 |
| M82 control | 0.729521 | 0.744777 | 4952 | 67 |
| M84 Category | 0.708521 | 0.700454 | 5541 | 72 |
| M84 same-head Empty | 0.652226 | 0.684336 | 7397 | 75 |
| M84 same-head Swapped | 0.704142 | 0.705934 | 5796 | 73 |

The frozen criteria pass 8/14: M82 increment 0/4, native comparison 4/5, swapped comparison 3/4 and empty-native per-sequence metric parity 1/1. The failed native success protection is mobilephone02_indoor: native mean IoU 0.851274, H10=0; Category 0.560285, H10=1; Swapped 0.842371, H10=0. M84 is not promoted and no official external evaluation has started.

Category improves pooled IoU over native by 5.6295 percentage points, but loses 2.0999 points against M82 and adds 589 low-overlap frames and 5 H10 runs. Category exceeds Swapped pooled IoU by only 0.4379 points and is lower by 0.5479 points on sequence-equal IoU. Removing colacan01, flower02 or ghostmask separately reverses the pooled Category-minus-Swapped difference; 19/22 leave-one-sequence-out differences remain positive. Against Empty/native all 22 remain positive.

The frozen gate tests per-sequence metric parity. A supplemental independent comparison of the SHA-bound native-reference extraction also verifies all 33130 bbox arrays and all 33108 noninitial scores exactly equal to Empty. The original full native shards remain remote; their hashes and extraction were checked there, not independently rehashed by the local reviewer. The algebraic empty-input identity concerns a current state; switching to empty after a divergent Category history does not reconstruct an independent native trajectory.

The large Category-versus-Empty difference is not a clean decomposition of semantic understanding: the centered nonlinear branch can learn generic nonempty conditioning. Original versus Swapped, sequence distribution and known automatic-caption errors remain material. The direct empty-reference subtraction prior art is documented in [the published prior-art report](https://github.com/666666666666gao/Track/blob/main/projects/sttrack_lachtt_v1/diagnostics/m84_centered/prior_art/EMPTY_REFERENCE_PRIOR_ART.md); this experiment does not establish that subtraction itself is novel.

Next: retain the completed negative result, audit saved predictions, then examine first divergence and sustained damage in mobilephone02/car02 alongside positive sequences. Compare evidence before any new training; separate differences in local observation, raw prediction, windowed selection and later state. Do not change thresholds or restart an architecture search based solely on pooled metrics. Any subsequent experiment must have prospective controls, fixed seed2027 and the same final-checkpoint rule. Official three-dataset targets remain unmet.
