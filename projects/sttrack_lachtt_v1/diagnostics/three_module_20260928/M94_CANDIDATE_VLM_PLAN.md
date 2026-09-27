# M94 read-only candidate identity probe

Use the fixed M92 packet of 24 selected DepthTrack Train fit130 events. Each
event has one Top-10 candidate with IoU≥0.6 to the training GT, a different
native candidate with IoU≤0.1, and mutual candidate overlap≤0.1. These
conditions make the training-GT target side measurable, but the packet is a
hard-event selection, not a frequency sample of tracking states or a human
semantic review.

Qwen2.5-VL-3B sees five RGB images: initial full frame with the protocol target
red box, initial target crop, current unannotated full frame, candidate A crop,
candidate B crop. It never receives the sequence name, GT, candidate scores,
or the hidden answer mapping. Ask for one letter: A or B for the same physical
instance, N for neither, U for visually undecidable. Read the four first-token
logits rather than parsing generated prose. Use bfloat16 as in M93.

For every event, evaluate both candidate orders and two text conditions:
no category and the existing *unverified automatically generated* category.
The category is explicitly described as possibly inaccurate. Swapping A/B
changes only their crop order and the A/B positions defined in the prompt; the initial and
current full images are unchanged. The category is not a truth label.

Report accuracy for each order, both-order accuracy, mapped-order
consistency, A/B/N/U counts, and paired changes with auto text. Keep the
hidden correct-side manifest private. Publish per-event predictions/logits
and aggregate correctness only; per-event correctness would reveal the side.
A/B bias must be
judged against both orders. No tracking weights, state, training labels,
reporting threshold, or public benchmark metric changes.

Do not promote this VLM as an online teacher unless both-order accuracy is at
least 20/24 and no more than two events disagree under order swapping; even
meeting that pilot gate would still require independent human review of
phrase correctness and candidate identity before Module A training.
