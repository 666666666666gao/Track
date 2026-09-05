# Relevant upstream association evidence — read-only

Checked2026-09-05 while M42 collection was running. M42's architecture,
sampling, training and gates were already frozen and were not changed after
this lookup. No upstream model weights or source code were integrated.

KeepTrack explicitly associates the candidate sets of consecutive frames,
including distractors, and permits unmatched candidates. This is more than
scoring the current candidates against one preceding predicted target.
Its paper describes propagation of object identities through those matches.
[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Mayer_Learning_Target_Candidate_Association_To_Keep_Track_of_What_Not_ICCV_2021_paper.pdf)

The official model extracts learned local descriptors, includes coordinates
and response scores, and matches candidates across frames. Its implementation
uses self/cross attention and an assignment with unmatched entries.
[Descriptor model](https://github.com/visionml/pytracking/blob/master/ltr/models/target_candidate_matching/target_candidate_matching.py)
[Matcher](https://github.com/visionml/pytracking/blob/master/ltr/models/target_candidate_matching/superglue.py)

The published training recipe freezes the feature extractor while learning
association, and mixes partial supervision with self-supervision. The
supplement separately identifies failure frames where a correct candidate
exists and failure frames where none exists, and mines these categories.
[Training source](https://github.com/visionml/pytracking/blob/master/ltr/train_settings/keep_track/keep_track.py)
[Supplement](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Mayer_Learning_Target_Candidate_ICCV_2021_supplemental.pdf)

Implication for this project, still a hypothesis: a later experiment could
maintain the identities of multiple similar objects before turning current
left-to-right order into a text attribute. An ordinal is an observed relation,
not a permanent instance identifier. This is relevant to the user's requested
online position description, but it is not evidence that adding that module
will improve STTrack or VOT.

For interpreting M42, inspect the final target-class counts and candidate
oracle coverage. Uniformly spaced low-IoU windows can occur long after the
target is lost; they are not automatically candidate-association positives.
A negative M42 result would concern this frozen observation/head/data recipe.
It would not by itself establish that all local matching or all language is
ineffective. Any future temporal candidate-set model requires a new Train-only
specification and training, without importing the upstream benchmark numbers
or changing this run's thresholds after seeing its result.
