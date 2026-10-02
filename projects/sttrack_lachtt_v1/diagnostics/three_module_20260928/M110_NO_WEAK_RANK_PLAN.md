# M110: same semantic pair, without optimizing the provisional A/B term

M109 recomputed raw gradients at the initial and M108 final weights using only
the fitting states. The provisional pair term was larger than localization plus
preservation in many paired batches at initialization. Its final reviewed-text
gradients were mostly aligned with those terms. This is not historical-gradient
replay and does not establish the cause of held-out degradation.

Test one factor: exclude the A/B softplus term from optimization. Keep the M108
trainer's architecture, initialization, frozen parameter list, seed2027, ordered
12 epochs, batch64, AdamW3e-4, and 480 updates per final. Continue computing the
pair term solely for logging; it is absent from loss.backward(). Keep the same
private bank, slot masks, and labels for comparison. The two arms use generic
`object` and model-reviewed first-frame-supported descriptions, respectively.
No human-confirmed labels have been supplied. Do not select a new coefficient,
seed, epoch, or checkpoint from the development results.

Only the original130 DepthTrack Train fitting sequences /2544 cached states
are optimized. The remaining22 Train sequences /495 states are held out. CDTB,
VOT, their review videos, and later-frame-only attributes are excluded from
training. This is fixed-state prototype training, not Full152 recursive training
or official benchmark evaluation.

Reuse the existing Torch1.13.1+cu116 Python3.8 environment; install nothing.
Use GPU0 for generic and GPU1 for reviewed text. Each arm performs a separate
three-update GPU sanity run before its final training. Assert finite gradients,
nonzero text gradients, exact frozen parameters/buffers, and identical Empty
score/quality outputs over all3039 states before and after training. The existing
four development gates are unchanged. Report both finals and all three content
conditions; no automatic promotion to recursive or public evaluation.

The historical M108 finals, bank and sources are immutable. M110 uses a new
trainer and output directory. Its only optimization change is removal of the
weak rank term. CUDA numerical and recursive-state reproducibility limitations
remain; this fixed-state test does not estimate seed variance.

Documented one-shot invocation:

```bash
/root/autodl-tmp/envs/sttrack/bin/python -u \
 /home/SUTrack_RGBD_L/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m110_no_rank_pair.py \
 --output /root/autodl-tmp/sttrack_m110_no_rank_pair_20261002 \
 --labels /home/SUTrack_RGBD_L/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/m107_weak_inputs/weak_labels.json
```

Estimated runtime about one minute from M108's60.6-second completed pair. Allow
three minutes before the first completion check; do not poll GPUs repeatedly.
