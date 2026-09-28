# M100: M99 fixed-state candidate score and quality readout

M99 completed its predeclared12epoch/480update final budget. It selected275/495
development events correctly versus native268, but broke three native-correct
events, including two healthy events. All three are in egg_indoor; neither a
sequence-specific rule nor a new checkpoint/epoch/weight scan is allowed.

Read the existing final checkpoint on **all2544fit and495development events**.
This is diagnostic evidence for Module B's responsibilities, not new training,
semantic supervision, recursive action or an official benchmark result. Keep
the source panel, five Empty slots and model parameters unchanged. Verify all
495development rows and both selection summaries against the completed M99
result. Save every event, all10native boxes, native log Hann response, learned
selection logits/residuals, estimated localization quality and GT-IoU labels.
GT-IoU is computed by the existing loader but excluded from model inputs;
only evaluation and output analysis consume those labels.

Answer these distinct questions:

1. On native-correct events, how often does the selection change, and how many
   changes remain correct? Record the native score gap, learned residual
   advantage and final selection gap rather than guessing a mechanism.
2. Does the independently trained quality head agree with the wrong selection,
   or still rank a correct box first? Read its argmax on the whole panel as a
   counterfactual diagnostic only. Do not deploy it, multiply scores or search
   a threshold using development results.
3. Compare fit and development candidate-quality absolute IoU error and all
   pre-existing strata. IoU-supervised quality is not physical identity truth.
   The native cached score is log(clamped Hann response), not log odds; do not
   equate its numerical value or exponent with calibrated identity confidence.

First run one64-fit-event GPU no-grad sanity with no development forward,
checkpoint or optimizer. Full readout requires source review and sanity exit0.
Estimate about20seconds including loading from the previous final reload;
inspect near expected completion, no repeated GPU polling. Use GPU0 because
this is one short readout, not a duplicated two-GPU training experiment.

The output is a separate M100 directory. Preserve M99 sources, checkpoint,
result and failed-original traces. Readout may motivate a future Train-only
controlled change, but neither the quality argmax nor a favorable aggregate
automatically passes M99's failed protection gate. Independent reviewed phrases,
physical competitors and visibility evidence remain required for semantic A+B.
