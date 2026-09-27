# M93 read-only initialization region-evidence pilot

The M92 human first-frame and blind candidate sheets are still blank. M93 does
not assign semantic truth or start A+B training. It asks whether the existing
Qwen2.5-VL-3B score for an *automatically generated* category is sensitive to
pixels in the protocol target region, using 12 preselected DepthTrack Train
fit130 first frames from the M92 assistant-screening pilot. The thirteenth
case, `bike01_wild`, has no disjoint same-size corner region in its 640×360
frame, so it cannot enter the balanced masking comparison.

For each unchanged auto category, the same model, prompt, and two-image input
format score the single next-token `Yes` versus `No` response under:

1. Original red-box full frame and target crop.
2. Target pixels filled gray in the full frame, with the original target crop
   kept visible.
3. A same-size nonoverlapping corner region filled gray in the full frame,
   with the same original target crop visible.
4. Target pixels and the target crop both filled gray, as an additional
   sensitivity readout rather than the primary paired control.

The corner with the largest center distance from the target is selected
deterministically. Images, box coordinates, category, model revision, prompt,
and token IDs are logged. Use bfloat16 because the previous server's float16
generation produced non-finite logits. `Yes-No` margins and the two-token
softmax are uncalibrated evidence diagnostics. The primary paired comparison
is target-full-mask change versus unrelated-full-mask change; the crop input
is held fixed for this comparison. Every case is reported.

Gray occlusion causes distribution shift. The target-both condition changes
more input than the unrelated-full control and is interpreted separately. A
large target effect neither proves the
category correct nor proves usable tracking semantics. No public test frames,
candidate labels, STTrack weights, search state, or benchmark metrics enter
this pilot. Human review remains the prerequisite for naming a phrase
`correct`, `conflicting`, or `unobservable` and for training Module A.
