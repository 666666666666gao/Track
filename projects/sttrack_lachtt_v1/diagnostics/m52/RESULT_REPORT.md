# M52 completed paired recursive result

Both fixed heads completed all 22 DepthTrack Train development sequences,
33,130 frames per arm and 66,260 frames in total. All four recursive shards,
analysis, and the paired pipeline exited 0. Neither head passes the frozen
advancement rule; no M52 low22 or full public evaluation is launched.

| Full recursive development | Mean IoU | IoU <= 0.1 frames | H10 failure episodes | Native-zero-episode sequences newly broken |
|---|---:|---:|---:|---:|
| Native STTrack reference | 0.652226263176 | 7,397 | 75 | Reference |
| Control: duplicated native fitting records | 0.643872982661 | 7,429 | 90 | 0 |
| Mixed: native plus fixed-M45 policy states | 0.630894720264 | 8,058 | 80 | 0 |

Both arms have 28,897 valid evaluation frames. Initialization and unavailable
GT are excluded from the mean; unavailable GT interrupts a low-overlap streak.
H10 counts maximal runs of at least ten valid consecutive frames with IoU <=
0.1. It is not a VOT failure-anchor count or a ROB measurement.

Relative to native STTrack, Control loses 0.008353280515 mean IoU and adds 32
low frames and 15 episodes. Mixed loses 0.021331542912 and adds 661 low frames
and 5 episodes. Mixed versus the equal-budget Control loses 0.012978262397
mean IoU and adds 629 low frames, while reducing the episode count by 10.
Thus the paired state-data condition fails, as does the independent
extra-training Control condition. `primary_pass=false`,
`extra_training_control_pass=false`, and `advancing_arm=null`.

The two arms improve the mean on eight and six sequences respectively and
both preserve every originally zero-episode sequence. These two passing
conditions do not override the three failed native aggregate conditions.
Control makes 603 nondefault choices; Mixed makes 418. Fewer choices alone
does not establish safer recursion.

The complete 44-row table is in `per_sequence.csv`. Selected concentrations
illustrate the tradeoffs without replacing that table:

| Sequence | Control mean IoU / low frames / H10 | Mixed mean IoU / low frames / H10 |
|---|---:|---:|
| notebook02_indoor | 0.374269 / 1,396 / 16 | 0.848132 / 76 / 2 |
| egg_indoor | 0.667147 / 796 / 12 | 0.551240 / 1,232 / 15 |
| colacan01_indoor | 0.670929 / 549 / 8 | 0.101102 / 2,464 / 10 |
| mobilephone02_indoor | 0.845581 / 1 / 0 | 0.847083 / 1 / 0 |

Mixed avoids Control's large notebook regression but loses its colacan gain
and worsens egg. These are trajectory outcomes, not proof of the first causal
error. In particular, a positive or negative static action cannot explain the
whole subsequent trajectory without a matched-state intervention.

## What the experiment establishes

The training data and deployment state differ, and the new fixed-M45 captures
contain measured state changes. This particular one-round augmentation did
not improve recursion under the frozen paired design. Extra training of the
old cache also did not help. This result does not establish state mismatch as
the dominant cause, and it does not prove that all on-policy or iterative
training methods are ineffective.

Both heads retain the same 448,739-parameter architecture and frozen STTrack
base. Their initial tensors, logical sample order, and 1,900 training steps
match. The analysis verifies complete frame coverage, trajectory/checkpoint
hashes, valid boxes and scores, exact native prefixes before first override,
and a recomputation of the native reference from its sealed full traces.
These executor checks are distinct from the separately recorded advisory
integrity review.

Result SHA256:
`92e7b0f20883d8ccee6c3911aba70f0d5bc39c5f7a0bc7804aeb868a208953aa`.
Checkpoint bindings and all gate booleans are in `recursive_result.json`.
All four receipts and terminal exit files are retained. No thresholds,
checkpoint epoch, or advancement rule were changed after observing results.

## Next bounded question

The completed [affinity diagnostic](AFFINITY_REPORT.md) does not support simply
replacing the classifier with the existing matching output. The next template
investigation should first compare reads of past native templates from one
frozen search/query state, preserving the initial template and default update
path. It should cover the fixed fitting events, retain both healthy and failed
cases, seal outputs before GT analysis, and label any best-template choice as
an offline oracle. This tests whether past views have useful candidate
capacity before training a new reader; it is not another scale-triggered
write rule or a deployable recovery claim.

The independent native full127 reference continues separately. A future
accepted model must still pass the frozen low22 condition and then use the
same base, learned head, preprocessing and runtime policy on DepthTrack Test,
CDTB and full VOT-RGBD2022. Historical scores from different models cannot
complete that requirement.
