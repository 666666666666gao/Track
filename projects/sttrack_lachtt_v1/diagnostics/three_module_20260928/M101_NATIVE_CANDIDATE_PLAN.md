# M101: preserve reliable native candidates in the new B decoder

M100 actually reread all3039 M99 fixed Train events. Its495 development
selection rows and both split summaries match M99 exactly. The selection head
rescues10/breaks3; the quality head's readout rescues12/breaks5. All three
selection breaks still have a wrong quality argmax, so quality substitution is
not supported. At those selection breaks, residual advantages exceed the
correct native candidate's log-Hann gaps. This observation locates the ranking
reversal; it does not prove the sole cause of the encoder's mistakes.

Use the unchanged M97 A+B architecture, M98 inputs and Empty protocol. This is
an explicit transplantation of M82/M89's reliable decision preservation into
the new candidate decoder, not a novel loss or evidence of semantic identity.
No fourth module, geometry modification, memory policy, new text, public test
or recursive action is introduced.

For fit events only, when native candidate0 has GT-IoU>=.5, preserve its native
log-response advantage over candidates with GT-IoU<=.1:

    relu((native_0-native_k) - (selection_0-selection_k)).

Average over eligible pairs in each batch. A batch with no eligible pair has
zero penalty. The GT-IoU mask is a localization-training label, never a model
input or physical-instance negative label. Wrong native events are not forced
to keep candidate0. Keep both existing BCE terms with weights1.

Run two predeclared arms: preservation weight0 and weight1. No weight sweep.
Seed2027, same zero-residual initialization, cached event order, batch64,
AdamW3e-4, fixed12 epochs/480 updates/final only. Freeze the same semantic and
observation parameters. The weight0 code path must leave the existing loss
expression untouched; it does not call the new function or create an added
loss node. Both arms use the same entry and data, with GPU0 for weight0 and
GPU1 for weight1. A new source review and each arm's two-step fit-only GPU
sanity must pass first. Estimate roughly one minute including input loading,
using M99's measured~21s optimization as the starting estimate.

Afterward compare the weight0 final tensor state, all495development rows and
both selection summaries to the preserved M99 final. Record any difference;
do not rerun automatically or assert bitwise reproducibility from the seed.
Only exact equality permits calling this a reproduction. Retain both arms
even if a protection check fails. Apply the same four predeclared M99 checks:
development correct choices and meanIoU exceed native, healthy breaks0,
transition correctness at least native. Report every gain/harm and fit
preservation loss/pair counts. Quality is trained but not used for selection.

Passing supports a subsequent reviewed semantic A+B experiment, not formal
three-dataset acceptance. Independent phrase/visibility/competitor labels are
still needed; neither Empty nor candidate IoU establishes text contribution.
Do not search a development checkpoint or tune on egg_indoor. Do not add
runtime fallback, threshold, compatibility code or automatic retries.
