# M108 frozen visual parent, generic versus reviewed semantic increment

M107 updated the existing visual selector while fitting weak textual conditions.
Its same-weight weak-text intervention lowered development mean IoU versus Empty,
and both arms introduced two native-healthy breaks. This comparison isolates a
different hypothesis: preserve the complete M101 visual function while fitting
only the centered text-dependent increment. It does not assume the weak labels
are correct or claim a new tracking architecture.

Both arms start from the same M101 weight1 final. Freeze every parameter except
the text projection/phrase slots, phrase and current-text attention, phrase feature
MLP, and semantic/phrase readout. Zero the two readout matrices. Keep the existing
selection MLP frozen while retaining gradients through it. No entire-model
no_grad wrapping is used for training. The three phrase channels lack independent
support/conflict/visibility labels and must not be named calibrated evidence.

GPU0 uses the frozen CLIP embedding of `object` in every valid slot. GPU1 uses
the unchanged M107 model-first-review category/first-frame supported attributes.
Both have exactly the same per-sequence slot masks, weak A/B labels, fitting
order, seed2027, AdamW3e-4, batch64 and fixed12epochs/480updates. The generic arm
has an active nonempty semantic path; it is not an Empty no-op training control.
The private bank is extended by one generic vector; original reviewed tokens,
Empty vector, masks and labels are preserved. Video-only attributes are excluded.

Losses are selected-candidate IoU-soft-target BCE, the existing reliable-native
relative gap preservation, and softplus pair ranking for the same11 explicit
model-reviewed A/B events. Thirteen uncertain events are ignored. The visual
quality head is frozen and its formerly optimized loss is omitted. No coefficient,
seed, epoch or checkpoint search is performed. All fitting uses the original130
Train sequences/2544 valid cached states;22 Train sequences/495 cached states
remain held out. No CDTB/VOT input review enters this training.

Actual GPU sanity runs3updates per arm before the training pair. Check finite
gradients and a nonzero internal text gradient, no frozen-parameter gradients,
unchanged frozen parameters and buffers, and exact Empty score/quality outputs
for all3039states before/after. Initial495Empty selections must match the saved
M101 parent. Train once, save fixed final, then evaluate each same final under
Empty, generic and reviewed text. Preset four development gates remain unchanged.
All gains/breaks and content interventions are reported; no automatic recursive
or official promotion is included. A semantic claim would require a reliable
benefit over the immutable Empty parent and the generic condition, while meeting
the healthy-state protection gate. Weak labels remain model supervision until
the user supplies independently human-confirmed labels.

Documented warm-reuse invocation (existing Torch1.13.1+cu116 environment):

```bash
/root/autodl-tmp/envs/sttrack/bin/python -u \
 /home/SUTrack_RGBD_L/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m108_frozen_pair.py \
 --output /root/autodl-tmp/sttrack_m108_frozen_semantic_pair_20261002 \
 --labels /home/SUTrack_RGBD_L/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/m107_weak_inputs/weak_labels.json
```

Estimate2–3minutes from M107's80-second actual run, including preparation, sanity,
fixed training and content readouts. Inspect near the estimated finish rather
than repeating GPU polls. Keep both historical M107 finals and all new logs.
