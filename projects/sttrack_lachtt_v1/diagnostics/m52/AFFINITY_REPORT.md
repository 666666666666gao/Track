# M52 auxiliary affinity readout: completed static diagnostic

This exploratory diagnostic was specified after M52 training, while the
unchanged paired recursive evaluation was running. It used no optimization,
changed no runtime policy, and does not enter the frozen advancement rule.
Execution finished with exit code 0. The CPU classifier decisions matched the
recorded GPU decisions on every event in all six arm/scope combinations.

The causal readout selects a current candidate from the affinity column
indexed by the actual previous-choice input. It does not receive GT. The
unmatched row maps to candidate 0 for comparison with existing output
semantics. This is a direct readout, without calibrated matching probabilities,
one-to-one assignment, persistent object IDs, or a separate lost-target state.

| Same 590 development events | Mean IoU | IoU >= 0.5 | Nondefault choices | Static rescues | Static breaks |
|---|---:|---:|---:|---:|---:|
| Default candidate | 0.440273732 | 268 | 0 | 0 | 0 |
| Control classifier | 0.446131557 | 272 | 10 | 4 | 1 |
| Control causal affinity | 0.439262837 | 267 | 8 | 1 | 1 |
| Mixed classifier | 0.446681499 | 274 | 14 | 4 | 0 |
| Mixed causal affinity | 0.438714385 | 268 | 14 | 2 | 1 |

Static rescue means default IoU <= 0.1 and chosen IoU >= 0.5; static break
means the converse. These are individual cached events, not rescued or broken
recursive tracks. Unavailable GT retains the existing static zero convention.
The tiny change in the default mean's last digits versus the earlier training
table comes from reduction arithmetic; classifier decisions were identical.

A separate privileged readout uses the previous GT candidate column. It is
reported only where that previous candidate exists, on the same 336-event
subset for all three readouts. It is not deployable and is not a mathematical
upper bound on association capacity.

| Same previous-GT-available subset | Classifier mean IoU | Causal affinity mean IoU | Privileged previous-GT affinity mean IoU |
|---|---:|---:|---:|
| Control, 336 events | 0.739543378 | 0.732462227 | 0.735241115 |
| Mixed, 336 events | 0.742288411 | 0.739110649 | 0.741889596 |

Directly substituting this existing affinity readout for the classifier has no
benefit in this development cache. Even supplying the previous GT candidate
does not exceed the classifier on the matched development subset. This does
not prove that affinity contains no complementary information, nor that a
newly trained identity propagation mechanism cannot help. It does rule out
claiming an improvement from merely exposing the present auxiliary output.

The fitting caches include native states and fixed-M45 states. Neither is a
new collection from either trained M52 head. No KeepTrack reproduction, long
absence recovery experiment, template intervention, text intervention, or
formal benchmark evaluation was performed here. Full causal recursive M52
results remain the advancement evidence.

Reproduction: run `inspect_affinity_readout.py --root <M52 root>` in the
existing STTrack environment with an empty `CUDA_VISIBLE_DEVICES`. It reuses
the sealed M52 data audit and checkpoint hashes. Per-event choices, all
fitting-scope metrics, and source bindings are in
`affinity_readout_diagnostic.json`; the execution log and zero exit code are
published alongside it.
