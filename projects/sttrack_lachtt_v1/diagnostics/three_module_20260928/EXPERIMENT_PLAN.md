# RGB-D instance-evidence tracker: staged experiment plan (2026-09-28)

## Objective and current boundary

Train one final seed-2027 RGB-D-language tracker on DepthTrack Train and evaluate
that same checkpoint and text/runtime protocol on all of DepthTrack Test50,
CDTB80, and VOT-RGBD2022 full127. The project targets remain P/R/F at least
65.2/64.9/65.1 and 72.9/75.6/74.2, and VOT EAO/ACC/ROB strictly above
77.9/82.1/93.7. No existing M67, M82, or M89 result meets the joint target.

The M89 same-weight Empty/Swapped OPE runs, 57-anchor VOT readout, and
zero-weight training attribution probe are already queued. Preserve their
checkpoints and protocols; this plan does not interrupt or reinterpret them.
No new three-module model or score exists yet.

## Prior experiments that the new method must exceed

| Existing implementation | Measured scope and result | Consequence for the new experiment |
|---|---|---|
| M44 candidate-set association | Train development22: IoU .617098, 8436 low-IoU frames, 87 H10; native .652226/7397/75 | A candidate Transformer and temporal matching alone are not a new solution. |
| M47 multiple IoU-valid correspondence destinations | Same development22: IoU .571038, 9946 low frames, 99 H10; M45 control .684102/6188/80 | The multi-positive matching loss was trained and failed. Candidate slots are not physical instance IDs. |
| M88 local initialization retrieval | Development22: IoU .714801, 5373 low frames, 65 H10 | Search-conditioned local reading exists; new A needs content/visibility supervision. |
| M89 candidate response preservation | Full external evaluation complete; DepthTrack F61.824813, CDTB F70.088471, VOT 75.887986/82.119751/92.533723 | Preserve useful teacher relations as a training constraint, not a new inference module or proof of word meaning. |

## Stage 0: close current attribution work

Record M89 Category/Empty/Swapped with the same final weight on both OPE
datasets. Finish the queued VOT readout and 64-frame zero-weight probe. Report
readout capacity, score/geometry effects, and actual first divergence with their
proper fixed-state or recursive scope. These results guide which bottleneck to
target; they do not select a public-test-specific rule.

## Stage 1: training-only data and fixed-state A+B prototype

Use the former 130 DepthTrack Train fitting sequences for optimization and the
former 22 for development. The latter have already been used repeatedly, so
call them development evidence, never an unseen test. The final frozen method
will be retrained on all 152. Do not use DepthTrack Test, CDTB, or VOT labels to
construct phrases, candidate labels, thresholds, or training-event rules.

Prepare first-frame RGB, Depth, protocol target box, surrounding context,
original category/attribute text, and a review record. Existing M86 screening
(110 supported/18 conflicting/24 uncertain over 152) is assistant judgment,
not approved truth; it cannot directly become a supervised correctness label.
For selected fitting events, record only visually supportable category and
stable attributes, and allow `unknown`. Record current attribute visibility
separately from support/conflict. An unobservable attribute is never a negative
identity label.

Collect a fixed set of own-history RGB-D search states on Train only, starting
with native STTrack; include healthy frames, genuine same-class competitors,
and local failure onsets. Cache all 256 native decoded boxes and score/size/
offset maps; cache candidate-local RGB/Depth features for the 10 NMS peaks,
their grid positions, and the immutable initial instance. These 10 candidate
features provide competition context, while the full grid remains available
for candidate-coverage diagnosis. Text is joined later from reviewed Train
initializations. Preserve the source trajectory. GT is attached
after candidate generation for training labels, never for runtime inputs.
IoU labels supervise localization quality. A distinct physical-object negative
requires separate visual evidence/annotation; low IoU alone is insufficient.

Module A receives immutable initial RGB-D instance tokens, reviewed phrase
tokens, current candidate-local RGB-D features, and surrounding competitor
context. Its explicit output per candidate and phrase is support, conflict, or
unobservable evidence, plus a candidate instance representation. Text-to-vision
finds candidate evidence; vision-to-text assesses whether each phrase applies
now. Current Depth validity may weight current evidence, but missing Depth is
not an identity conflict. This is more than M88's local retrieval or two
unsupervised cross-attention calls.

Module B first keeps the 256 native geometries fixed. It compares a small,
deterministically chosen candidate set using A's evidence, candidate-local
visual features, native response/Hann position, and initial identity. It
outputs one candidate identity score and a separate localization-quality
estimate. No new geometry head, memory writer, search expansion, or threshold
change is introduced in this first experiment. The selected box, score,
feature, and future state must all refer to the same candidate. Compare with
native selection, a parameter-matched visual-only selector, and the old M44
style selector under the same cached states. M47's multi-positive loss is a
historical negative control, not the proposed innovation.

For text attribution, use the *same trained weight and same cached state* with
reviewed correct text, Empty, a semantically equivalent rewrite, and a
reviewed conflicting description. On genuinely discriminative competitors,
measure the target-versus-distractor score margin and target selection. Also
report coverage, unsupported/unknown phrases, and correct native selections
that the new head harms. A different string is not automatically a false
description. Do not reward a model merely for making wrong text harmful.

Stage-1 gate: on the predeclared discriminative Train development events,
reviewed correct text must have more paired rescues than harms against both
same-weight Empty and the visual-only selector. On native-correct events,
report every new wrong choice and require that the total selected-candidate
accuracy still exceeds native. Report event counts and coverage, not only
aggregate accuracy. Failure here means fix evidence/labels or model
responsibility before recursive training.

## Stage 2: A+B complete recursive test

Train on self-predicted crops with frozen STTrack visual backbone and one
seed2027 final checkpoint. Keep the native template schedule and local search
factor fixed. Pair semantic A+B with the same-parameter visual-only control;
keep input bank, initialization, training budget, final selection, and runtime
policy fixed. Test complete first-frame recursion on the 22 development
sequences. Report mean and sequence-equal IoU, IoU<=.1 frames, H10 episodes,
first harmful/helpful divergence, candidate availability, and time per frame.
Compare M44/M47 failures explicitly. Content interventions start from the same
initialization and run their own histories; they are not fixed-state effects.

Only move forward when the semantic model improves mean IoU over native,
does not increase H10 episodes or add an episode to a native-zero-episode
sequence, and correct text beats same-weight Empty on mean IoU without more
H10 episodes. Keep all sequence-level gains and harms visible. A high static
candidate accuracy without recursive gain does not pass. If existing candidates
are often missing, add a visual local box refinement *as a separate B change*
and compare with frozen geometry at matched states and in full recursion.

## Stage 3: C, then new observation only if needed

Keep the initial template immutable. First add a small historical view bank
with fixed, trusted write sources and train only the read choice into the
existing dynamic-template interface. Compare visual-only versus A-conditioned
reading. If read choice helps, separately learn write value using current
identity evidence, localization quality, observability, and view novelty.
The existing response threshold must not simultaneously serve as identity,
box quality, and memory utility. Do not call a current correct box proof of
future memory value.

If the remaining measured failures lack a candidate because the target leaves
the local crop, add a budgeted new-region observer. Train-region proposal and
identity confirmation are separate measurements. Train-only reappearance
geometry already shows that one full-frame 256 crop frequently makes targets
too small and that a uniform tile grid has seam/compute costs; it is capacity
evidence, not a working recovery method. Confirm candidates with the same A+B
instance evidence before committing the frame or memory.

## Final acceptance

Freeze architecture, text protocol, training budget, checkpoint rule, and
runtime policy using Train development only. Retrain once on Full152 with
seed2027. Run the exact same final checkpoint on all three complete datasets,
report all nine metrics, per-sequence gains and harms, VOT failure timing,
content and visual-only controls, and compute cost. Stop only if this one model
passes every target; otherwise locate the remaining observed failure class
before making the next minimal change.
