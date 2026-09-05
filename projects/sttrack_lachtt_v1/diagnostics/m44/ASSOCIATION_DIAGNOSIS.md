# M44 association evidence and remaining training questions

This note separates three observations from proposed changes. It does not
change the frozen M44 primary, weights, runtime or advancement gates.

## Native evidence survives in three static counterexamples

All three severe geometry-arm static regressions are `egg_indoor` windows.
Both the current and preceding best target candidate are index0, but the
trained identity head selects index1. The correct candidate moves less than
one pixel relative to the preceding default box. Its native response is also
higher than the selected alternative's response.

| Window | Native RGB cosine to preceding target: correct / selected | Native depth cosine: correct / selected | Trained matching-column argmax |
|---|---:|---:|---|
|46 |0.986841 /0.922194 |0.997954 /0.992399 |unmatched |
|779 |0.987414 /0.912732 |0.996990 /0.980941 |correct candidate0 |
|2439 |0.998060 /0.925523 |0.998767 /0.991385 |incorrect candidate1 |

These compare unprojected candidate-own 4x4 tokens to the preceding selected
RoI. They show usable continuity evidence in these specific inputs and an
incorrect final decision. They do not establish that cosine is a safe global
selector, that every failure has this cause, or that the matching branch is
always correct. The initial-template RGB cosine even prefers the incorrect
alternative at window779, so the reference source matters.

Evidence: `egg_correspondence_diagnosis.json` and
`egg_raw_correspondence.json`. These are inspections of sealed static windows;
no new tracker action, fitting or threshold selection was performed.

## Default-policy training and the new policy's states

M44 already trains on **predicted crops** produced by STTrack default. Its
1,511 fitting inputs always mark the preceding default candidate, index0.
Candidate permutation equivariance does not remove the distinction between
selecting the highest-response candidate and selecting a lower-response one.

In the first sealed geometry recursive sequence, `bag05_indoor`,49 of887
association calls have a preceding nondefault choice. The actual preceding
indices include1--7. This proves a fitting-input coverage limitation. It does
not prove that it caused the initial wrong choice: the three egg static
counterexamples above already fail while the preceding choice is0.

Evidence: `input_state_coverage.json`, based on bound trainer source and a
SHA-verified889-frame trajectory, without loading ground truth or computing
partial recursive performance. A controlled follow-up could compare the same
network trained with default-policy states versus a mixture including its own
rollout states, with matched optimization and the same STTrack-default gate.
Such a test would address exposure after a changed decision, not establish a
new visual representation or language contribution.

## What the referenced KeepTrack implementation actually provides

KeepTrack's training combines real consecutive-frame partial target
correspondence with augmented views of one frame, where correspondence among
the retained candidates is known. Unknown real distractor identities are
ignored; augmented-view correspondence is not a new human annotation of
physical object identities.
[Official processing implementation](https://github.com/visionml/pytracking/blob/master/ltr/data/processing.py),
[official training settings](https://github.com/visionml/pytracking/blob/master/ltr/train_settings/keep_track/keep_track.py).

Its instantiated matching network uses two alternating self/cross-attention
pairs and ten Sinkhorn iterations. These settings override the larger generic
defaults in the matcher class. The matcher enforces mutual accepted matches;
the tracker then carries object IDs through a separate candidate collection.
[Network constructor](https://github.com/visionml/pytracking/blob/master/ltr/models/target_candidate_matching/target_candidate_matching.py),
[matcher](https://github.com/visionml/pytracking/blob/master/ltr/models/target_candidate_matching/superglue.py),
[candidate collection](https://github.com/visionml/pytracking/blob/master/pytracking/tracker/keep_track/candidates.py).

M44 has two joint-attention layers, partial target-match supervision and a
separate identity output. Its raw11x11 affinity is an auxiliary output, not a
Sinkhorn assignment or a complete distractor-ID tracker. These implementation
differences should be stated explicitly. They are possible experimental
directions, not proof that copying upstream thresholds or adding every
component will improve RGB-D tracking.

Only one new training or information variable should be tested at a time.
M44's complete recursive result and terminal audit decide its advancement;
this note cannot authorize a post-hoc primary switch or public evaluation.
