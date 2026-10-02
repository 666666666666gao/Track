# M107 model-first-review weak semantic A+B pair

The paired fixed-state run completed on 2026-10-02 with seed2027, 130 fitting
sequences, 22 held-out Train development sequences, 2,544 fitting states,
495 development states, and 480 optimizer updates per arm. GPU0 received Empty;
GPU1 received text from the externally model-reviewed first-frame labels. The
same eleven model-reviewed candidate A/B choices supervised both arms; thirteen
uncertain choices were ignored. The labels were not human-confirmed.

| Development condition | IoU≥0.5 selected / 495 | mean IoU | rescues | breaks | healthy-state breaks |
| --- | ---: | ---: | ---: | ---: | ---: |
| native candidate 0 | 268 | 0.52477072 | — | — | — |
| M107 Empty arm | 273 | 0.52987699 | 8 | 3 | 2 |
| M107 weak-text arm, weak text | 273 | 0.53144407 | 8 | 3 | 2 |
| same weak-text weight, Empty text | 273 | 0.53341376 | 8 | 3 | 2 |

The weak-text weight changes its selection in 20/495 states when given weak
text instead of Empty: six IoUs improve and six worsen. The weak-text mean IoU
is 0.00196969 lower than its own Empty condition, and neither weight meets the
prespecified no-healthy-break condition. The fixed-state result therefore does
not establish a beneficial semantic contribution or justify a full recursive
promotion. Both weights have eight rescues and three breaks relative to the
same-state native candidate. These are cached-state candidate selections, not
official DepthTrack Test P/R/F or VOT EAO/ACC/ROB.

Source and final output hashes are recorded in `m107_completed/result.json`.
The raw reviewer text bank remains private; only its SHA-256 and the fact that
the first review was model-produced should be published. Human labels will be
treated as a separate later experiment, not retroactively attributed to M107.
