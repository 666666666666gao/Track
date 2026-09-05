# Mechanisms checked against primary sources

Read on 2026-09-05 to guide experiments after the frozen M52 comparison.
These notes describe mechanisms, not demonstrated improvements in this project.
No third-party implementation was copied into the tracker.

## KeepTrack: assignment and object bookkeeping are additional operations

The [KeepTrack paper](https://arxiv.org/pdf/2103.16556) describes candidate
embeddings, candidate association, assignment with unmatched entries, and
online object bookkeeping. Its training combines partial target supervision
with synthetic self-supervised correspondences. The existing M52 affinity
loss supervises known target matches; its runtime selects classification
logits. Therefore an auxiliary matrix alone is not the complete mechanism.

The official
[candidate collection implementation](https://github.com/visionml/pytracking/blob/master/pytracking/tracker/keep_track/candidates.py)
stores object IDs and score histories, propagates candidates using matches,
handles uncertain target assignments, and attempts reselection after loss.
The new candidate dictionary contains currently observed candidates; this
code is not evidence of indefinite identity retention for invisible objects.
The implementation also uses score-dependent matching rules that differ from
a single generic probability threshold. Its numerical settings cannot be
assumed calibrated for STTrack's scores or M52's raw affinity.

The official
[training entry](https://github.com/visionml/pytracking/blob/master/ltr/train_settings/keep_track/keep_track.py)
explicitly uses self-supervised and partially supervised sampling, as well as
different frame/subsequence modes and separate validation loaders. Borrowing
only an online ID dictionary would omit this training context. A future test
should identify the added correspondence supervision and compare it under
the same DepthTrack Train data and recursive protocol.

## DAgger: M52 tests one fixed-policy augmentation

[DAgger](https://proceedings.mlr.press/v15/ross11a.html) iterates policy rollout,
expert labeling of visited states, data aggregation, and policy training.
M52 collects one fixed M45 policy at the original physical sampling times and
trains two fresh heads. It is one policy-state augmentation experiment, not
iterative DAgger, not on-policy data from the resulting M52 heads, and not
end-to-end differentiation through cropping. The paper's theoretical result
must not be claimed for this neural experiment.

Only 36 of the 1,511 sampled new events have a nonzero previous-choice input.
However, 1,052 sampled predictions differ from the native trajectory and 271
have a center difference greater than one quarter of the native predicted
scale. Thus the added data has a measured state difference, with sparse direct
coverage of nonzero previous selections. Neither counting all 3,022 logical
views as independent frames nor dismissing all zero-previous-choice events as
unchanged states is justified.

The next intervention must follow the completed M52 recursive outcome. A
positive paired result would motivate additional policy-state collection. A
negative result would limit this particular augmentation; it would not prove
that state distribution is irrelevant. Direct affinity readout is evaluated
separately in `AFFINITY_REPORT.md` and is not a new promotion criterion.
