# M105: completed same-final center/size component readout

The exact M104 final was read on 3,039 fixed DepthTrack Train states. The full
correction reproduces M104. Both center and size corrections produce gains and
harms on the fitting split, so neither component can simply be discarded as the
cause of all damage. This readout does not promote a variant into tracking.

## Actual execution and verification

GPU0 read 2,544 fitting events; GPU1 read 495 development events. Both child exits
and the driver exit are 0. Driver time was 14.418560 seconds, from
2026-09-28 06:49:42.888067 UTC to 06:49:57.306602 UTC (14:49 CST).
There were zero optimizer steps and no new checkpoint. M104 refiner final SHA256
is `546bd218a14906beb28033717bb53cc17721e9bb6b3fb53f36a4bac15dff9f6e`;
M101 parent final is bound by both readout JSON files and the M104 result.

All original full rows, ordered keys, strata, selected indices and all 14 original
summary fields replay exactly. Executed decoder checks cover all 30,390 zero-delta
candidate boxes; saved parent/refiner parameters and buffers are unchanged.
`accept_m105_components.py` completed locally: 12,156 selected-box IoUs (four
variants times 3,039) exactly match independently computed float32 dataset-GT
overlaps. All marginal and joint-group summary arithmetic also matches exactly.
The original dataset training-label file is SHA-bound to preparation.json; it is
not a generated identity or semantic reference. `download_receipt.json` records
byte/SHA verification for all 13 runtime files. The local acceptance adds
`acceptance.json`; it is distinct from the original downloaded files.

## Selected candidate localization

Correct means selected IoU >= 0.5. Rescue/break counts are relative to the frozen
M101 selected box. Top10 oracle is a GT diagnostic upper bound, not a deployable
selection. Parent selection is fixed in every variant.

| Split | Readout | Correct | Mean IoU | Rescues | Breaks | Top10 oracle correct |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Fit, 2544 | Parent | 1602 | 0.608313789 | 0 | 0 | 1814 |
| Fit | Full | 1652 | 0.616710538 | 51 | 1 | 1866 |
| Fit | Center only | 1620 | 0.613920198 | 19 | 1 | 1818 |
| Fit | Size only | 1629 | 0.609808391 | 28 | 1 | 1840 |
| Development, 495 | Parent | 272 | 0.530707698 | 0 | 0 | 305 |
| Development | Full | 286 | 0.535297504 | 15 | 1 | 323 |
| Development | Center only | 276 | 0.533739096 | 4 | 0 | 310 |
| Development | Size only | 278 | 0.531034610 | 8 | 2 | 315 |

Healthy correct counts (fit n=1560 / development n=264): parent1559/264,
full1558/263, center1558/264, size1558/263. Transition selected correct counts
(fit n=452 / development n=127): parent22/4, full22/4, center22/4, size22/3.
All stratum rows and their overlapping/joint definitions are preserved in the
raw results; marginal groups must not be summed as disjoint populations.

## Every observed selected-box break

| Split and event | Parent | Full | Center only | Size only |
| --- | ---: | ---: | ---: | ---: |
| Fit pine01_indoor@1276, healthy | 0.500443041 | 0.479231507 | 0.479360998 | 0.501299739 |
| Fit basket_indoor@67, healthy | 0.503328264 | 0.631534517 | 0.581268668 | 0.482185483 |
| Dev mobilephone02_indoor@65, healthy | 0.510501027 | 0.448262632 | 0.510886967 | 0.448261052 |
| Dev ball19_indoor@688, transition | 0.525265217 | 0.518158853 | 0.600251794 | 0.448889107 |

For the pine fit event, the center component alone crosses below .5; for basket,
the size component alone crosses below .5 but the joint output is improved. On
the phone development event, the size component preserves the damage seen in
the full readout. Size-only also damages ball19 whereas the joint output remains
above .5. These are same-state localization observations, not physical-identity
labels or proven causes of complete recursive failures.

Center/size rescue intersections contain one event in each split. Eleven full
fit rescues and four full development rescues occur with neither component alone
passing .5. Three center-only fit rescues and three size-only fit rescues are not
full rescues. Image clipping and the overlap threshold make the components
non-additive; summing their gains or assigning a percentage of total benefit to
each is unjustified.

## Scope and next decision

This is an Empty five-slot visual, fixed-cache, same-final component intervention.
It is not retraining, physical-instance verification, semantic contribution,
template/state submission, recursive tracking, C-module training or official
three-dataset evaluation. Ground truth is used for training-label evaluation and
strata, never as a refiner inference input or candidate-selection gate.

The original M104 healthy protection check still fails. The safer center-only
development count does not authorize choosing it as a deployed strategy.
Fitting events demonstrate a concrete need to train preservation of already
valid selected geometry while retaining useful joint corrections. Any next
training change must be a separately named, fit-defined experiment with fixed
seed/budget/final, and then checked on development. Do not add an event-specific
rule, scan thresholds, or relax the healthy gate. Semantic training continues to
wait for independently confirmed user labels; these visual checks do not.

The fresh completed-result integrity review is recorded separately in
`../M105_COMPLETED_AUDIT.md` and `.json`; model review is same-family/provisional.
All official nine metrics and the active joint target remain unchanged.
