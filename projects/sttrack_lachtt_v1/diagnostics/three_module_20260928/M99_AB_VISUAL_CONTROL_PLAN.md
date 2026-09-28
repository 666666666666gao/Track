# M99 token/context A+B visual control

M98 is collecting the missing context/support inputs. Prepare the necessary
visual-capacity control for M97's new A+B network while that runs. This is not
a repeat of M91/M96: preserve all16local tokens,12surrounding tokens, role and
position information, modality support, initial/current cross-reading and
candidate-set interaction. Keep the M97 architecture and native frozen boxes.
Use no automatic category, human phrase or physical-identity label.

All five text slots contain the encoded empty vector with a fixed all-active
mask, independent of sequence. Read only that vector from the existing bank;
do not use its generated categories, attributes or sequence-dependent masks.
Train the model's visual selection logits and visual localization-quality
logits against existing Train-GT candidate IoU soft targets (BCE+BCE, weight1
each). These labels do not supervise physical identity, semantic evidence or
observability. Freeze semantic-only and observation parameters, preserve their
initial values, and explicitly report total versus optimized parameter count.
This is a same-architecture Empty control, **not an exactly matched effective
trainable-capacity language ablation**. A later semantic-versus-visual comparison
must disclose this boundary and add the corresponding same-weight content tests.

Use former-fit130/2544valid events for optimization, former-development22/495
valid events for final diagnosis only, same fixed M90 event ordering and M98
fields, legal M95 t0-search reference, seed2027, batch64, AdamW3e-4,12epochs.
Use final epoch only, no best/development checkpoint search. Record every
development event and strata, meanIoU, correct selection, rescues and breaks.
Initial zero-residual output must reproduce native candidate0. Predeclare a
fixed-state capacity check: development correct choices and meanIoU both exceed
native, healthy breaks0, transition correctness at least native. Passing this
only supports the subsequent paired semantic experiment, not recursion or
official acceptance. If it fails, analyze the preserved complete results rather
than sweeping loss weights/seeds or substituting a best epoch.

During M98, run only a CPU three-event loading/gradient sanity using its
completed cube04 smoke [10,12,14]. This has two temporary localization steps,
no checkpoint or development performance. Check finite nonzero gradients on
visual read and candidate-set read, native zero choice, exact Empty/visual
equality, and unchanged frozen parameters. Do not use a GPU needed by M98.

Before any full budget, require M98 queue exit0 and complete152/3502/219194
input audit. Then run the full-data, two-step GPU batch64 capacity sanity
without development evaluation or a checkpoint. Measure time and allocated/
reserved memory; stop and diagnose a real failure, no automatic OOM fallback.
Only a successful source review, CPU actual-input sanity and full GPU batch
sanity permit the12epoch control. Training timing is not yet measured.

No tracker state, new template, C control, public dataset or official nine-metric
evaluation is enabled. Semantic A+B fitting still requires reviewed evidence;
this control does not waive that gate or rename IoU as identity truth.
