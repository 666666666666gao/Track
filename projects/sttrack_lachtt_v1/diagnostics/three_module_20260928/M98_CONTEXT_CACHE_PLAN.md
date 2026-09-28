# M98 complete the fixed-state A+B inputs

M97's eight-case actual-input interface gate passed. The existing M90 cache
contains candidate region tokens but not their surrounding context or native
padding support. Complete these missing input fields for the SAME152Train
sequences and3502preselected events. This is data preparation for the staged
A+B experiment, not semantic training or a new tracker. Do not generate labels
or promote M96/M97 to recursion. Independent phrase/instance review is pending.

Reuse M90's manifest, shard assignment, native checkpoint, all prefix prediction
rows and stored ten proposals. Reuse M95's legal t0 search references. Collect
only t0 surrounding tokens, initial masks/nonzero-depth fraction, and event
candidate context tokens/masks/nonzero-depth fractions using M97's proven
sampling functions. Candidate boxes, region tokens, scores and geometry remain
in M90; initial region references remain in M95. Avoid duplicating those fields.

The collector reads no subsequent GT, text bank or private review answers.
Auxiliary t0 query is not committed. Every real prefix box/score must match
M90 at its existing1e-4px/1e-6tolerances. Each event's proposal boxes and local
half-precision RoIs must exactly match M90, and the t0 region must exactly
match M95. Actual native resize and padding mask are used, as in revised M97.
Preserve these checks rather than weakening them if collection fails.

Run one-case smoke on the first fit sequence and its first three existing
events, then two full shards on GPUs0/1. The full output contains152sequences,
3502events and219194native prefix calls; the smoke is excluded from those
totals. Store each sequence separately, with its original event order and
split; do not change the fit130/development22 division. A CPU completion audit
checks all receipts, original feature linkage, exact event correspondence,
shapes/dtypes/finite values, and total coverage, without loading GT.
In the smoke, the first event's new fields and t0 context/support must also
exactly reproduce the existing M97 cube04 smoke tensors. This links the new
sequence-wise layout to the actual-input prototype without semantic labels.

Current free disk is3.08GB. Event context tensor payload is
3502*10*2*12*768*2=1.291GB (fp16), with initial context/masks and serialization
overhead much smaller. Keep existing caches/checkpoints. No deletion, training
checkpoint, public evaluation or runtime modification is planned.

M90's two shards took6261/5655seconds. With additional event context/support
reads, estimate roughly2hours for the slower full shard. Review code fresh
before deployment, pass the actual smoke, launch both shards with explicit
exit/log receipts, then audit only after both exit0. Monitor about hourly or
near the estimated terminal time; do not repeatedly poll the collection.
Collection speed from completed sequence logs can refine this estimate.

Passing proves complete inputs on these biased Train states, not semantic
correctness, candidate-selection gains, recursive safety or official metrics.
The next training gate remains reviewed semantic/instance labels and a fair
content-versus-visual comparison. C remains staged separately.
