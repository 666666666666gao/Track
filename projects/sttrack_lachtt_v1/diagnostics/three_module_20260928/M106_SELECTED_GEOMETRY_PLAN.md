# M106: preserve the geometry of the actually selected reliable candidate

M105 fit evidence shows both center-only and size-only corrections can damage
correct selected boxes. Keep the joint refiner rather than choose a favorable
component from development. M104 trains on the fitting GT-best candidate, while
deployment uses the frozen M101-selected candidate. This is an observed training
target difference, not a proven unique cause of all damage.

Keep the exact M104 architecture, initial zero refiner, M101 parent, Empty slots,
2033 eligible fitting events, data order, seed2027, AdamW3e-4, batch64 and fixed
12 epochs/384 updates/final. Control weight0 skips the additional forward and
loss. Weight1 adds a differentiable IoU preservation term on the actual selected
candidate for each fitting batch:

    mean(1[IoU_parent >= .5] * relu(IoU_parent - IoU_refined_selected))

The mean is over the original batch, so no empty-reliable branch or new sampling
is needed. Current dataset GT supplies localization labels; it does not establish
physical identity. The original GT-best regression remains unchanged. No GT enters
the refiner input, candidate selection or inference gate. No new capacity, human
prefill, teacher, C module, template, recursive state or public evaluation is used.

GPU0 weight0 and GPU1 weight1 each run a two-update actual-data sanity in distinct
outputs, then full training only if both sanity exits are zero. All existing
finite/zero-init/frozen-parent/full-final-reload assertions remain. The runner
must record actual child statuses; an observation timeout never authorizes a
restart. No retries, seed/weight/epoch scan or best checkpoint. Existing outputs
are rejected. Expect about one minute based on M104's completed32-second pair.

Require weight0 to reproduce M104 final tensors and all3039 eval rows/summaries
exactly before interpreting the paired effect. Report every fit/development gain
and harm. Four unchanged capacity checks are development correctness increase,
mean-IoU increase, zero healthy new breaks, and transition correctness at least
the frozen parent. A failed check remains failed; no automatic C/recursive/public
promotion. Even all four passing is fixed-cache visual evidence, not full tracking,
language contribution or formal nine-metric goal completion.

Reuse the executed STTrack conda Python
`/root/autodl-tmp/envs/sttrack/bin/python`, Torch1.13.1+cu116/CUDA11.6/Python3.8.20.
The recorded M104/M105 invocation and source/deployment receipts in the master
handoff are the existing server notes; no environment rebuild or package change.
Use the authorized server port43811, not the obsolete goal endpoint. Never store
the password in source, commands recorded in plans, logs or reviews.
