# M102: assignment-free single-candidate identity readout

M101's Empty visual-capacity control passes its four predeclared checks.
The two private human sheets were reread on September28: initial categories
and candidate identities remain0/24, with no named reviewer. Do not turn those
blanks, M86 assistant screening or M94's failed responses into supervised truth.

M94's96 A/B/N/U responses all chose A, and the first checked actual replies
remained A after a verified image swap. Test a new read-only protocol on the
same24 M92 fit130 events before relying on Qwen for any evidence. Each query
contains exactly one current candidate, not a choice between A/B labels.

Show four RGB images: a red-marked full frame and crop for each of the two
objects. One pair is the legal initial target, the other is a cached predicted
candidate. Ask whether they are the same physical instance, with Yes/No/Unknown
first-token readouts. Do not show sequence names, GT, candidate scores, A/B
letters, automatic category or the hidden target side. Swap the two complete
image pairs while keeping the question unchanged. Compare both candidates
independently in both pair orders:24events x2candidates x2orders=96queries.

This also adds a marked current full frame and changes the image count and
answer format relative to M94. It is a new protocol, not a single-factor causal
proof that answer format caused M94's failure. Preserve local and full context.
Use the existing Qwen2.5-VL-3B checkpoint, bfloat16, seed2027 and pixel budgets.
The installed tokenizer actually gives one token each for Yes9454, No2753,
Unknown13790. The three-token softmax is uncalibrated; it is not identity truth.

Preflight on the first event of each of two disjoint shards: verify actual
processor pixel/grid blocks swap by2, finite three-token logits, and run an
identical-initial-image self-pair. The self-pair response is recorded as a
basic recognition diagnostic, not a semantic truth label. Full readout requires
both preflights to exit0 and show exact input binding; no favorable semantic
answer is required to hide or stop an informative failure. No checkpoint,
optimization, tracker action or human-label overwrite is permitted.
Sanity's one-event reports omit all GT-derived correctness counts; retain
only binding, finite readouts, predictions and rank consistency.

For each order rank A versus B using their Yes-minus-No margins. Exact ties
are ties, not an A win. Report each order's Train-GT-side accuracy, both-order
correctness, rank consistency/change/ties, and all restricted token choices.
Keep per-event true-side and correctness private; publish only predictions and
aggregate correctness. The original M92 assignment is publicly reconstructible,
so do not pretend these IDs are the independent v2 human packet or publish its
private mapping. No v2 assignment is read.

The inherited teacher-screening threshold is both-order ranking correct>=20/24
and rank changes<=2. Always retain the complete result even on failure.
Passing only supports further independent review; it does not authorize teacher
labels or semantic training. GT-side is a localization proxy, not independently
confirmed physical identity, and this selected panel contains one valid
candidate per event. Neither-valid/unknown calibration is not tested.

Split original rows by even/odd position over GPU0/1,12events each. Sanities
first, then both fixed shards, no prompt/threshold/model/seed scan or retry.
M94 took66.4s for96queries on one GPU; expect roughly1-2minutes with model
loading and two shards. Inspect near completion, no frequent GPU polling.
Keep the three-dataset goal and A+B/C evidence requirements unchanged.
