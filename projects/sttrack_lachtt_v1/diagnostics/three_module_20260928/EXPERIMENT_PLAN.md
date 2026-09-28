# RGB-D instance-evidence tracker: staged experiment plan (2026-09-28)

## Objective and current boundary

Train one final seed-2027 RGB-D-language tracker on DepthTrack Train and evaluate
that same checkpoint and text/runtime protocol on all of DepthTrack Test50,
CDTB80, and VOT-RGBD2022 full127. The project targets remain P/R/F at least
65.2/64.9/65.1 and 72.9/75.6/74.2, and VOT EAO/ACC/ROB strictly above
77.9/82.1/93.7. No existing M67, M82, or M89 result meets the joint target.

The M89 same-weight Empty/Swapped OPE runs, 57-anchor VOT readout, and
zero-weight training attribution probe are complete. Preserve their
checkpoints and protocols; this plan does not interrupt or reinterpret them.
No final trained three-module tracker or new official score exists yet.

M93 has completed a read-only 12-case Train first-frame region-masking pilot
with Qwen bfloat16. The generated category's Yes/No margin was more sensitive
to masking the full-frame target than a nominally same-box unrelated region
in 11/12 cases. This tests image-region dependence only; it is not human
semantic verification, an identity label, or tracking improvement. The
completed protocol, all cases, and limitations are in
`m93_completed/RESULT.md`. A+B training still waits for independent review.
Retrospective audit found unequal effective masking areas from red-border
redrawing and PIL corner clipping, so this is not an equal-area causal control.

M94 then tested Qwen as a read-only identity teacher on all 24 M92 hard
candidate events, with both A/B orders and with/without unverified automatic
category. All 96 first-token decisions chose A: each condition scored 12/24
in each order, zero events were correct in both orders, and zero were
swap-consistent. This fails its predeclared teacher gate; do not use these
outputs as Module A labels. Details are in `m94_completed/RESULT.md`.
The first-event tensor-binding check verifies the image swap; actual short
replies are A in both orders. The M92 answer manifest was omitted, but the
published seeded generator makes its side assignment reconstructible. An
independent human blind-review packet needs a new private assignment.

M95 is a training-only feature-origin diagnostic, not a new selector training.
M90's frozen first-use template pixels are from t0, but their post-TSG encoding
also reads frame1 search. Collect auxiliary t0-template and t0-search RoIs
without committing state, verify first real tracking-frame native parity, and
compare pure cosine on the same candidates. See `M95_INITIAL_ORIGIN_PLAN.md`.
M95 is complete: t0-template cosine changes development correctness8 to9;
t0-search changes it to178, still below native268/495 and breaking91/264
healthy events. Do not deploy cosine selection. M96 now compares two learned
visual controls under the same12epoch budget, changing only the initial
reference source. See `M96_ORIGIN_CONTROL_PLAN.md`.
M96 is now complete: first-use/t0-search development correct choices270/271,
mean IoU.528513/.527995, both healthy breaks0, transition correct4/3.
The first-use control's final parameter tensors exactly reproduce M91.
The one-choice gain does not justify recursive promotion; next semantic A+B
training remains gated on independently reviewed Train evidence.

The human candidate packet has been reissued as private `candidate_review_24_v2`
with the same24 cases, privately permuted IDs and A/B choices. The original
packet and M94 results remain intact. Both human sheets still have0/24 labels.

M97's isolated A+B prototype now passes actual-input wiring checks on eight
Train events: both interaction directions have nonzero localization gradients,
Empty increments are exactly0, quality is text-independent and selected fields
share one index. No semantic labels, checkpoint, recursive action or C module
were used. M98 completes the missing surrounding-context/support fields for the
existing M90 states, rather than duplicating region tokens or changing native
candidates. See `M98_CONTEXT_CACHE_PLAN.md`; it is data preparation only and
does not waive the reviewed-label or content-versus-visual training gate.

## Prior experiments that the new method must exceed

| Existing implementation | Measured scope and result | Consequence for the new experiment |
|---|---|---|
| M44 candidate-set association | Train development22: IoU .617098, 8436 low-IoU frames, 87 H10; native .652226/7397/75 | A candidate Transformer and temporal matching alone are not a new solution. |
| M47 multiple IoU-valid correspondence destinations | Same development22: IoU .571038, 9946 low frames, 99 H10; M45 control .684102/6188/80 | The multi-positive matching loss was trained and failed. Candidate slots are not physical instance IDs. |
| M88 local initialization retrieval | Development22: IoU .714801, 5373 low frames, 65 H10 | Search-conditioned local reading exists; new A needs content/visibility supervision. |
| M89 candidate response preservation | Full external evaluation complete; DepthTrack F61.824813, CDTB F70.088471, VOT 75.887986/82.119751/92.533723 | Preserve useful teacher relations as a training constraint, not a new inference module or proof of word meaning. |

## Stage 0: close current attribution work

M89 Category/Empty/Swapped with the same final weight are recorded on both OPE
datasets. Category minus Empty F is -0.008334 pp on DepthTrack and +0.264581 pp
on CDTB; CDTB Swapped has higher fixed-box recall than Category even though its
F is lower. These results show content sensitivity without stable net semantic
gain. At the 57 new VOT failure onsets, 36 Category states have a correct box
among all 256 cells, but only 26 have one among the 10 NMS peaks; using native
Hann scores gives 13 single-frame recoveries without committing state. The
64-frame old/zero/old probe also shows gradient and weight hashes differ between
two repetitions of the old objective. These are fixed-state capacity and short
recursive diagnostics, not executed VOT recovery or proof of the full-training
control gap. They guide which bottleneck to target without selecting a public-
test-specific rule.

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
Private reviewer packets are now ready: 24 first-frame cases and 24 blinded
current-candidate A/B events from former-fit130. The latter were selected from
28 eligible sequences where native candidate IoU<=.1, another Top-10 candidate
IoU>=.6, and the two boxes have mutual IoU<=.1; A/B placement is balanced at
12/12 with seed2027. Neither packet contains completed human labels, and the
blinded candidate page does not show GT, model choice, IoU, or sequence names.
For selected fitting events, record only visually supportable category and
stable attributes, and allow `unknown`. Record current attribute visibility
separately from support/conflict. An unobservable attribute is never a negative
identity label.

The native STTrack own-history Train cache is complete: 152 sequences and 3,502
preselected events, with all 256 decoded boxes and score/size/offset maps,
candidate-local RGB/Depth features for 10 NMS peaks, their grid positions, and
the immutable initial instance. Former-fit130 valid events have 1,575 native
correct choices, 1,814 Top-10 oracle hits, and 1,882 full-256 oracle hits;
former-development22 has 268/305/324 among 495 valid events. This selected
panel is not a uniform frame sample. Text is joined later from reviewed Train
initializations. Preserve the source trajectory. GT is attached after
candidate generation for training labels, never for runtime inputs.
IoU labels supervise localization quality. A distinct physical-object negative
requires separate visual evidence/annotation; low IoU alone is insufficient.

Before semantic A+B training, run a fixed-state visual selector control to
check whether the cached Top-10 features support any learned candidate choice.
The training partition is former-fit130; former-development22 is held for a
single reported fixed-state diagnostic, with no public-data input or recursive
state action. Train identical zero-residual candidate heads with and without
first-frame/current RGB-D RoI feature pairs. Both retain native candidate
scores, candidate geometry, location and rank. Supervise each candidate's
predicted localization quality with its Train GT IoU; this label is **not** a
physical-instance identity label. Predeclare seed2027, 12 epochs, batch64,
AdamW learning rate 3e-4, and final epoch (no development checkpoint search).
Report native, learned, and Top-10 oracle IoU>=.5 counts, rescues, breaks,
mean IoU, and healthy/transition strata. A direct feature-cosine control has
already failed sharply (development 8/495 vs native 268/495, 262 breaks);
the learned control tests feature/selection capacity, not the new A/B method.
This control is now complete: on 495 former-development valid events, native,
geometry, visual, and Top-10 oracle choose an IoU>=.5 box 268/273/270/305
times. Geometry rescues/breaks 15/10; visual 3/1. The visual arm does not
surpass geometry. Do not promote either static selector into recursion based on
this result. Build independently reviewed semantic evidence before Stage-1 A+B.

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
Report both Top-10 and full-256 candidate coverage on Train: the VOT posthoc
readout shows 10 of 36 correct dense candidate sets at failure onset have no
correct member in the 10 NMS peaks, so Top-10 ranking has a measured ceiling.
On the Train development panel, 37 native misses have a correct Top-10
candidate, 19 more have one only in full-256, and 171 have none in full-256.
At 60 selected sustained-low development onsets, 45 target centers are
already outside the nominal crop. These categories require different later
actions and must not be collapsed into a generic ranking error.

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
candidate accuracy without recursive gain does not pass. If Top-10 misses but
full-256 contains a correct box, test broader proposals as a separate B change.
If full-256 has no correct box while the target remains in-crop, test visual
local box refinement separately. Out-of-crop failures require a new observation
region under Stage 3. Compare each change at matched states and in recursion.

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

## M99 implementation preparation: token/context visual capacity

The M97 A+B wiring smoke passed; M98 is collecting the missing context and
support fields on the same M90 states. Before semantic fitting, prepare the
same-architecture Empty visual control described in
`M99_AB_VISUAL_CONTROL_PLAN.md`: preserved local/context tokens and candidate
interaction, legal M95 t0-search reference, fixed five-slot Empty input,
GT-IoU BCE for visual selection and localization quality, seed2027, batch64,
AdamW3e-4, fixed12 epochs/final only. Semantic-only and observation parameters
are frozen. Report total and optimized counts separately; this preliminary
visual-capacity control does **not** satisfy the effectively parameter-matched
language-ablation requirement above. Later semantic comparisons still need
the corresponding capacity disclosure and same-weight content interventions.

During M98, only a three-event CPU loading/gradient smoke is allowed. Full
input loading and GPU batch64 sanity require M98's successful terminal audit;
the12-epoch run requires that sanity and a fresh source review to pass. No
semantic/identity label is inferred from candidate IoU, no development epoch
selection, and no tracker state or public evaluation is enabled. Record the
predeclared capacity checks and all development rows even if the control fails.
