# M112 current-candidate phrase evidence: fixed-state experiment

Status: source preparation only; no SSH deployment, GPU sanity, optimizer
updates, checkpoint or new official result. Complete fresh source review and
the actual two-card acceptance before launching the fixed training. This is
the existing Stage1 ModuleA support/conflict/unknown task, not a new C module.

Use the completed M101 visual parent, the unchanged M108 phrase bank and
M90/M95/M98 cached fitting/development observations. The new manifest contains
202 weak model judgments from24 former-fit events, mapped to48 real cached
candidates and their actual zero-based phrase slots. Counts are98supported,
15conflicting and89unknown. No label is optimized from former-development22,
DepthTrack Test, CDTB or VOT. Human confirmation remains pending. Multiframe
Train imagery informed the model teacher but is never an inference feature;
the labels describe current RGB phrase evidence, not physical identity or
Depth correctness. All101 A/B phrase pairs include only10 clear S/C contrasts.

The model and its existing interaction are unchanged. Freeze every visual
parameter and all buffers. Only the same95,683 semantic parameters are trained;
zero the semantic and phrase readout weights at the common M101 initialization.
Keep seed2027,12epochs/480updates,batch64,AdamW3e-4 and the fixed final weight.
The base objective remains selection-logit BCE against candidate IoU plus
native candidate preservation. Do not optimize the old weak A/B identity rank.
Add unweighted three-class CE with coefficient1 at
`phrase_logits[row,actual_candidate_index,actual_phrase_slot]`. Sample32
manifest rows uniformly per update using a separate CPU generator seeded
32027+step. The phrase target is model weak evidence, never an IoU-derived
semantic label. Cached non-Boolean tensors already become FP32 in the shared
input helper, preserving the actual M111 dtype correction.

Reuse the completed M111 weight0 final as the no-current-evidence control;
it exactly reproduced M110. Do not rerun its480updates. GPU1 will load that
fixed control without an optimizer, recompute all stored fitting and development
selection rows, and evaluate the new202 phrase labels. GPU0 runs a fresh
three-update sanity. Only after both exit0 and frozen/Empty invariants pass
does GPU0 begin the one fresh480-update M112 training. Thus both cards perform
necessary work, with no redundant training added just to occupy a card.

Record initial and final202-row CE/confusion/class recalls, the majority-class
reference and10 paired S/C support margins. A read-only intervention replaces
current region/context/support fields with immutable initial fields while
holding text and initialization fixed. Report logit/prediction/CE differences;
this artificial sensitivity probe is not a deployment protocol or transfer
proof. Save actual logits for every label and all selection rows.

Evaluate all495 development states under Empty/generic/weak text, against the
actual training GT for localization. Report each condition, changes relative
to its own Empty, healthy damage and transition results. All3039 fitting plus
development Empty scores/qualities must remain exactly equal to the parent;
frozen parameters and buffers must be exact. The readonly reference must also
reproduce the stored completed control rows. If any required check fails,
preserve logs and stop before further training; do not retry unchanged.

Fitting these202 labels cannot establish semantic transfer: current-phrase
development labels remain0. Old candidate-capacity checks do not constitute
semantic promotion. No recursive action, template change, observation control
or full public evaluation is automatically authorized by this prototype's
completion. The original single-weight three-dataset goal remains unmet.

Runtime estimate after SSH recovery: approximately2–4minutes for the entire
sanity/reference/training sequence, based on the completed77-second M111 pair
and the larger current-candidate CE. This estimate has not been measured for
M112 and excludes server reconnection/deployment. Check near the estimate,
with180–300second long-task intervals; do not poll repeatedly. Keep the remote
master-document sync current before launching. No environment installation,
new feature cache, seed scan or repeated Full152 training is needed.
